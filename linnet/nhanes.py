"""Patient profiles from NHANES 2017-2018 (CDC; public, de-identified).  [deterministic]

    python -m linnet.nhanes        writes data/patients/nhanes-*.json and data/nhanes/routines.json

Raw files (download once into data/nhanes/raw/, gitignored):
  RXQ_RX_J.xpt   prescription medicines taken in the past 30 days (generic names)
  SLQ_J.xpt      usual sleep and wake time on weekdays (SLQ300, SLQ310)
  DR1IFF_J.xpt   24-hour dietary recall, day 1: time (DR1_020) and name (DR1_030Z) of each eating occasion
  DEMO_J.xpt     age and sex
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/<file>.xpt

What is real: the person's prescription list, usual sleep and wake time, and the
times they ate breakfast, lunch and dinner on the recall day.
What is not: NHANES does not record how many times a day a drug is taken, so dose
counts are each label's usual regimen. Variants marked "hypothetical" add a drug
the person was not recorded taking, to simulate a new prescription.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RAW = Path("data/nhanes/raw")
OUT = Path("data/patients")

# NHANES generic name -> (library product, usual doses/day from its label)
LIBRARY = {
    "LEVOTHYROXINE": ("levothyroxine-100mcg", 1),
    "CIPROFLOXACIN": ("ciprofloxacin-500mg", 2),
    "COLESTIPOL": ("colestipol-1g-tab", 2),
    "NIMODIPINE": ("nimodipine-30mg-cap", 6),
    "ACETAMINOPHEN; CODEINE": ("apap-codeine-300-30", 3),
}
# DR1_030Z eating-occasion codes (NHANES 2017-18 codebook)
MEALS = {"breakfast": {1, 10}, "lunch": {2, 12}, "dinner": {3, 4, 14}}
HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


def _read(name: str):
    import pandas as pd
    return pd.read_sas(RAW / f"{name}.xpt", format="xport", encoding="latin-1")


def _hhmm(seconds: float) -> str:
    m = int(seconds) // 60 % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def people() -> list[dict]:
    """Adults with a usable routine (sleep, wake, breakfast, lunch, dinner) and at least one prescription."""
    rx, slq, dr, demo = _read("RXQ_RX_J"), _read("SLQ_J"), _read("DR1IFF_J"), _read("DEMO_J")
    drugs: dict[int, list[str]] = {}
    for seqn, name in rx[rx["RXDUSE"] == 1][["SEQN", "RXDDRUG"]].itertuples(index=False):
        if isinstance(name, str) and name.strip() and not name.startswith("99999"):
            drugs.setdefault(int(seqn), []).append(name.strip().upper())

    sleep = {int(s): (b, w) for s, b, w in slq[["SEQN", "SLQ300", "SLQ310"]].itertuples(index=False)
             if isinstance(b, str) and isinstance(w, str) and HHMM.match(b) and HHMM.match(w)}

    meals: dict[int, dict[str, float]] = {}
    for seqn, t, code in dr[["SEQN", "DR1_020", "DR1_030Z"]].dropna().itertuples(index=False):
        for meal, codes in MEALS.items():
            if int(code) in codes:
                cur = meals.setdefault(int(seqn), {})
                cur[meal] = min(cur.get(meal, t), t)  # first time they ate that meal

    out = []
    for seqn, age, sex in demo[["SEQN", "RIDAGEYR", "RIAGENDR"]].itertuples(index=False):
        s = int(seqn)
        if age < 18 or s not in drugs or s not in sleep or set(meals.get(s, {})) != set(MEALS):
            continue
        out.append({"seqn": s, "age": int(age), "sex": "M" if sex == 1 else "F", "drugs": sorted(set(drugs[s])),
                    "bed": sleep[s][0], "wake": sleep[s][1],
                    "meals": {k: _hhmm(meals[s][k]) for k in ("breakfast", "lunch", "dinner")}})
    return out


def _sane(p: dict) -> bool:
    """Meals in waking order and inside the waking day (some recall days are unusual)."""
    def m(h):
        a, b = h.split(":")
        return int(a) * 60 + int(b)
    wake, bed = m(p["wake"]), m(p["bed"])
    span = (bed - wake) % 1440
    ts = [(m(p["meals"][k]) - wake) % 1440 for k in ("breakfast", "lunch", "dinner")]
    return ts == sorted(ts) and ts[-1] < span and 12 * 60 <= span <= 20 * 60


def profile(p: dict, extra: list[dict] = (), extra_note: str = "", tag: str = "") -> dict:
    items, in_lib, other = [], [], []
    for d in p["drugs"]:
        if d in LIBRARY:
            prod, n = LIBRARY[d]
            items.append({"product": prod, "prescribed_count": n})
            in_lib.append(d.lower())
        else:
            other.append(d.lower())
    items += list(extra)
    # Shown name is just the ID (plus a tag for hypothetical variants); details stay in `note`.
    name = f"NHANES #{p['seqn']}" + (f" (hypothetical: {tag})" if tag else "")
    note = (f"Routine (usual sleep and wake, and meal times on the recall day) and prescription list from NHANES "
            f"2017-18 participant SEQN {p['seqn']} (CDC, public, de-identified). Doses per day are each label's usual "
            f"regimen; NHANES does not record frequency."
            + (f" Also takes, not in LINNET's library so not checked: {', '.join(other)}." if other else "")
            + (f" HYPOTHETICAL addition, not recorded for this person: {extra_note}." if extra_note else ""))
    return {"name": name, "constructed": bool(extra), "note": note, "items": items,
            "routine": {"meals": p["meals"], "sleep": {"start": p["bed"], "end": p["wake"]}}}


def build() -> list[Path]:
    ps = [p for p in people() if _sane(p)]
    written = []
    for old in OUT.glob("nhanes-*.json"):
        old.unlink()

    def write(stem: str, prof: dict):
        f = OUT / f"{stem}.json"
        f.write_text(json.dumps(prof, indent=1) + "\n", encoding="utf-8")
        written.append(f)

    # Everyone on a library drug other than levothyroxine (rare drugs, all real), then three
    # levothyroxine users, chosen deterministically: the ones with the most prescriptions.
    rare = [p for p in ps if any(d in LIBRARY and d != "LEVOTHYROXINE" for d in p["drugs"])]
    levo = sorted([p for p in ps if "LEVOTHYROXINE" in p["drugs"] and p not in rare],
                  key=lambda p: (-len(p["drugs"]), p["seqn"]))[:3]
    for p in rare + levo:
        write(f"nhanes-{p['seqn']}", profile(p))

    # Hypothetical new prescriptions on real routines.
    for p in levo[:2]:
        write(f"nhanes-{p['seqn']}-uti", profile(
            p, [{"product": "ciprofloxacin-500mg", "prescribed_count": 2, "every_hours": 12},
                {"product": "tums-es-750", "reported_count": 2}],
            "ciprofloxacin twice daily for a UTI, and Tums for heartburn", "UTI"))
    for p in [p for p in rare if "COLESTIPOL" in p["drugs"]][:1]:
        write(f"nhanes-{p['seqn']}-sah", profile(
            p, [{"product": "nimodipine-30mg-cap", "prescribed_count": 6, "every_hours": 4}],
            "nimodipine every 4 hours after a subarachnoid hemorrhage", "brain bleed"))

    # Routines only (no drugs, no ids): for evaluating the solver on real days.
    Path("data/nhanes").mkdir(parents=True, exist_ok=True)
    routines = [{"bed": p["bed"], "wake": p["wake"], "meals": p["meals"]} for p in ps]
    Path("data/nhanes/routines.json").write_text(json.dumps(routines) + "\n", encoding="utf-8")
    print(f"{len(ps)} adults with a usable routine and prescriptions; {len(rare)} on a rare library drug, "
          f"{sum('LEVOTHYROXINE' in p['drugs'] for p in ps)} on levothyroxine")
    return written


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for f in build():
        d = json.loads(f.read_text(encoding="utf-8"))
        print(f"{f}: {d['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
