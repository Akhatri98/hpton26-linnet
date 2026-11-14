"""Patient profiles from NHANES 2017-2018 (CDC; public, de-identified).  [deterministic]

    python -m linnet.nhanes        writes data/patients/nhanes-*.json and data/nhanes/routines.json

Raw files (download once into data/nhanes/raw/, gitignored):
  RXQ_RX_J.xpt   prescription medicines taken in the past 30 days (generic names)
  SLQ_J.xpt      usual sleep and wake time on weekdays (SLQ300, SLQ310)
  DR1IFF_J.xpt   24-hour dietary recall, day 1: time (DR1_020) and name (DR1_030Z) of each eating occasion
  DEMO_J.xpt     age and sex
  DSQIDS_J.xpt   supplements and antacids taken in the past 30 days (product name, how many a day)
  P_*.xpt        the same five files for 2017-March 2020 (optional; adds the 2019-20 participants)
  https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/<file>.xpt

What is real: the person's prescription list, usual sleep and wake time, and the
times they ate breakfast, lunch and dinner on the recall day.
What is not: NHANES does not record how many times a day a drug is taken, so dose
counts are each label's usual regimen. Variants marked "hypothetical" add a drug
the person was not recorded taking, to simulate a new prescription.

Who is written (every choice is a fixed rule, not a hand pick):
  - everyone on a library drug other than levothyroxine (all real);
  - three levothyroxine users with the most prescriptions;
  - four levothyroxine users per release who ate breakfast under 30 minutes after their usual
    wake time (the label asks for 30-60 minutes before breakfast), longest such gaps first;
  - every levothyroxine user who reported Tums Extra Strength 750 (calcium carbonate);
  - hypothetical: ciprofloxacin + Tums on two of those levothyroxine users; nimodipine
    (brain bleed) on every colestipol user; colestipol on every nimodipine user;
    Tylenol Extra Strength on two acetaminophen/codeine users.
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
# Two releases. The 2017-March 2020 file re-numbers everyone (SEQN 109263+) and also holds the
# 2017-18 participants, so its people are kept only if they match no 2017-18 person exactly.
CYCLES = {
    "J": {"files": ("RXQ_RX_J", "SLQ_J", "DR1IFF_J", "DEMO_J", "DSQIDS_J"), "label": "NHANES 2017-18"},
    "P": {"files": ("P_RXQ_RX", "P_SLQ", "P_DR1IFF", "P_DEMO", "P_DSQIDS"),
          "label": "NHANES 2017-March 2020 pre-pandemic file"},
}


def _read(name: str):
    import pandas as pd
    return pd.read_sas(RAW / f"{name}.xpt", format="xport", encoding="latin-1")


def _have(cycle: str) -> bool:
    return all((RAW / f"{f}.xpt").exists() for f in CYCLES[cycle]["files"][:4])


def _tums(name: str) -> dict[int, dict]:
    """Tums Extra Strength 750 use from the 30-day supplement and antacid questionnaire: the
    same product as the library's. Other strengths have other label maximums, so are left out."""
    if not (RAW / f"{name}.xpt").exists():
        return {}
    ds = _read(name)
    out = {}
    for seqn, prod, qty, unit, days in ds[["SEQN", "DSDSUPP", "DSD122Q", "DSD122U", "DSD103"]].itertuples(index=False):
        prod = str(prod).upper()
        # DSD122U 1 = tablets, 20 = chewables; DSD122Q = how many on a day it was taken
        if "TUMS" in prod and "750" in prod and unit in (1, 20) and qty == qty and 0 < qty < 50:
            out[int(seqn)] = {"product": prod.strip(), "per_day": int(qty), "days_of_30": int(days) if days == days else None}
    return out


def _hhmm(seconds: float) -> str:
    m = int(seconds) // 60 % (24 * 60)
    return f"{m // 60:02d}:{m % 60:02d}"


def people(cycle: str = "J") -> list[dict]:
    """Adults with a usable routine (sleep, wake, breakfast, lunch, dinner) and at least one prescription."""
    rx_f, slq_f, dr_f, demo_f, ds_f = CYCLES[cycle]["files"]
    rx, slq, dr, demo = _read(rx_f), _read(slq_f), _read(dr_f), _read(demo_f)
    tums = _tums(ds_f)
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
                    "meals": {k: _hhmm(meals[s][k]) for k in ("breakfast", "lunch", "dinner")},
                    "cycle": cycle, "tums": tums.get(s)})
    return out


def _same_person(p: dict) -> tuple:
    return (p["age"], p["sex"], tuple(p["drugs"]), p["bed"], p["wake"], tuple(sorted(p["meals"].items())))


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
    t = p.get("tums")
    if t:  # reported, so real: one dose of what they took on a day they used it (label: 2-4 tablets a dose)
        doses = -(-t["per_day"] // 4)
        units = t["per_day"] // doses if t["per_day"] % doses == 0 else f"{t['per_day']}/{doses}"
        items.append({"product": "tums-es-750", "reported_count": doses, "units_per_dose": units})
    items += list(extra)
    # Shown name is just the ID (plus a tag for hypothetical variants); details stay in `note`.
    name = f"NHANES #{p['seqn']}" + (f" (hypothetical: {tag})" if tag else "")
    note = (f"Routine (usual sleep and wake, and meal times on the recall day) and prescription list from "
            f"{CYCLES[p.get('cycle', 'J')]['label']} participant SEQN {p['seqn']} (CDC, public, de-identified). "
            f"Doses per day are each label's usual regimen; NHANES does not record frequency."
            + (f" Reported in the supplement and antacid questionnaire: {t['product']}, {t['per_day']} on a day used"
               + (f", on {t['days_of_30']} of the past 30 days" if t["days_of_30"] else "") + "." if t else "")
            + (f" Also takes, not in LINNET's library so not checked: {', '.join(other)}." if other else "")
            + (f" HYPOTHETICAL addition, not recorded for this person: {extra_note}." if extra_note else ""))
    return {"name": name, "constructed": bool(extra), "note": note, "items": items,
            "routine": {"meals": p["meals"], "sleep": {"start": p["bed"], "end": p["wake"]}}}


def build() -> list[Path]:
    ps_j = [p for p in people("J") if _sane(p)]
    ps = list(ps_j)
    if _have("P"):
        seen = {_same_person(p) for p in ps_j}
        ps_p = [p for p in people("P") if _sane(p)]
        new = [p for p in ps_p if _same_person(p) not in seen]
        ps += new
        print(f"2017-March 2020 file: {len(ps_p)} usable adults, {len(ps_p) - len(new)} match a 2017-18 person "
              f"exactly (skipped), {len(new)} added")
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
    levo_all = [p for p in ps if "LEVOTHYROXINE" in p["drugs"] and p not in rare]
    levo = sorted([p for p in levo_all if p["cycle"] == "J"], key=lambda p: (-len(p["drugs"]), p["seqn"]))[:3]
    # Levothyroxine users who ate breakfast less than 30 minutes after their usual wake time: the
    # label asks for it 30 to 60 minutes before breakfast. Unmodified; the four longest such gaps per file.
    early = []
    for cyc in CYCLES:
        early += sorted([p for p in levo_all if p["cycle"] == cyc and p not in levo and _gap(p) < 30],
                        key=lambda p: (-_gap(p), p["seqn"]))[:4]
    # Levothyroxine users who also reported Tums Extra Strength 750 (calcium carbonate): all of them.
    tums = [p for p in levo_all if p.get("tums") and p not in levo + early]
    for p in rare + levo + early + tums:
        write(f"nhanes-{p['seqn']}", profile(p))

    # Hypothetical new prescriptions on real routines.
    for p in levo[:2]:
        write(f"nhanes-{p['seqn']}-uti", profile(
            p, [{"product": "ciprofloxacin-500mg", "prescribed_count": 2, "every_hours": 12},
                {"product": "tums-es-750", "reported_count": 2}],
            "ciprofloxacin twice daily for a UTI, and Tums for heartburn", "UTI"))
    for p in [p for p in rare if "COLESTIPOL" in p["drugs"]]:
        write(f"nhanes-{p['seqn']}-sah", profile(
            p, [{"product": "nimodipine-30mg-cap", "prescribed_count": 6, "every_hours": 4}],
            "nimodipine every 4 hours after a subarachnoid hemorrhage", "brain bleed"))
    for p in [p for p in rare if "NIMODIPINE" in p["drugs"] and "COLESTIPOL" not in p["drugs"]]:
        write(f"nhanes-{p['seqn']}-colestipol", profile(
            p, [{"product": "colestipol-1g-tab", "prescribed_count": 2}],
            "colestipol twice daily for high cholesterol", "colestipol"))
    codeine = [p for p in rare if "ACETAMINOPHEN; CODEINE" in p["drugs"] and "NIMODIPINE" not in p["drugs"]]
    for p in codeine[:2]:
        write(f"nhanes-{p['seqn']}-tylenol", profile(
            p, [{"product": "acetaminophen-500-otc", "reported_count": 4}],
            "Tylenol Extra Strength 4 times a day for back pain", "Tylenol"))

    # Routines only (no drugs, no ids): for evaluating the solver on real days. 2017-18 only, so the
    # published solver evaluation stays reproducible.
    Path("data/nhanes").mkdir(parents=True, exist_ok=True)
    routines = [{"bed": p["bed"], "wake": p["wake"], "meals": p["meals"]} for p in ps_j]
    Path("data/nhanes/routines.json").write_text(json.dumps(routines) + "\n", encoding="utf-8")
    print(f"{len(ps)} adults with a usable routine and prescriptions; {len(rare)} on a rare library drug, "
          f"{sum('LEVOTHYROXINE' in p['drugs'] for p in ps)} on levothyroxine, "
          f"{sum('LEVOTHYROXINE' in p['drugs'] and _gap(p) < 30 for p in ps)} of whom ate breakfast within 30 min of waking")
    return written


def _gap(p: dict) -> int:
    """Minutes from usual wake time to breakfast on the recall day."""
    def m(h):
        a, b = h.split(":")
        return int(a) * 60 + int(b)
    return (m(p["meals"]["breakfast"]) - m(p["wake"])) % 1440


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for f in build():
        d = json.loads(f.read_text(encoding="utf-8"))
        print(f"{f}: {d['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
