"""Extraction evaluation: score several models' answers side by side.  [deterministic]

    python -m library.compare            every answer set under data/drafts/responses/

An "answer set" is a folder of <ingredient>.<pass>.json files: one folder per
model run (the API run, simulated runs by subagents, ...). Each answer goes
through exactly the checks the real pipeline uses, in memory; nothing in the
library is written. Writes data/eval/extraction.json.

Measures, per answer set:
  timing rules   exact / partial / missed against the curated library rules that
                 cite the same label (any ingredient file), plus extras
  provenance     quoted vs interpreted agreement on matched rules
  checks         how many proposed rows the deterministic checks rejected, and why
  safety net     of the gold rules the model missed, how many sit at priority 1 in
                 the review queue (a pharmacist would still be shown them)
  members        group memberships found / missed / extra vs groups.json
                 (an element member such as "calcium" counts for its salts)
  rows           dosing and limits rows accepted (no gold yet: counts only)
And across answer sets: repeatability = how often two runs give the same
timing-rule set for a label.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from .evaluate import _key, _nums
from .extract import PASSES, _check_pass, label_record, numbered_sentences
from .moieties import element_of, is_element
from .passes import MASS_UNITS, uncited

DATA = Path("data")
RESP = DATA / "drafts" / "responses"
LABELS = {
    "nimodipine": "a323ebd8-ef1b-439b-93d1-ba73a5b0127d",
    "colestipol": "03ca5800-4211-4383-a332-be88bd9b5304",
    "levothyroxine": "008de8fd-150f-4022-8ccb-c8bfa94875c7",
    "ciprofloxacin": "0c355077-7361-41d0-9e82-eb31beaf5daa",
    "calcium_carbonate": "036ef1ed-32ef-46b8-8817-b3f63a014280",
    "acetaminophen": "00146b91-008d-4b16-95b0-6077f98821be",
    "acetaminophen-codeine": "02386806-02d4-47e3-9c8b-7810146ee795",
}
NAMES = {".": "stand-in Haiku run 0 (subagent)", "claude-haiku-5-5": "Haiku 5.5 via API (with retries)",
         "sim-haiku-a": "Haiku run A (subagent)", "sim-haiku-b": "Haiku run B (subagent)",
         "sim-sonnet": "Sonnet run (subagent)"}


def _gold_rules(spl: str) -> list[dict]:
    """Curated rules citing this label, from ANY ingredient file (a combination
    label's rules are filed under each ingredient), tagged with that ingredient."""
    out = []
    for f in sorted((DATA / "ingredients").glob("*.json")):
        out += [{**r, "ingredient": f.stem} for r in json.loads(f.read_text(encoding="utf-8"))["rules"]
                if r["source"]["spl_set_id"] == spl]
    return out


def _ing(name) -> str:
    return str(name or "").strip().lower().replace(" ", "_")


def _same_ing(gold: dict, draft: dict) -> bool:
    """A draft that names its ingredient must name the gold rule's; one that names none can match any."""
    return not draft.get("ingredient") or _ing(draft["ingredient"]) == _ing(gold.get("ingredient"))


def _candidates(admin: list[dict], limits: list[dict]) -> list[dict]:
    """Draft rules to score: the administration pass, plus daily maxima from the limits pass (which
    names the ingredient on a combination label). A maximum found by both passes counts once."""
    out = [{"family": "daily_max", "amount": r["amount"], "unit": r["unit"], "ingredient": r.get("ingredient"),
            "provenance": r["provenance"]} for r in limits if r.get("kind") == "per_day" and r.get("unit") in MASS_UNITS]
    seen = {(_key(r), tuple(sorted(_nums(r).items()))) for r in out}
    return out + [r for r in admin if (_key(r), tuple(sorted(_nums(r).items()))) not in seen]


def _gold_members(spl: str) -> set[str]:
    groups = json.loads((DATA / "groups.json").read_text(encoding="utf-8"))["groups"]
    return {m["ingredient"] for b in groups.values() for m in b["members"] if m.get("source_spl") == spl}


def _sig(rules: list[dict]) -> frozenset:
    return frozenset((_key(r), tuple(sorted((k, v) for k, v in _nums(r).items()))) for r in rules)


def score_set(folder: Path) -> dict:
    res = {"labels": {}, "exact": 0, "partial": 0, "missed": 0, "extra": 0, "prov_agree": 0, "prov_total": 0,
           "proposed": 0, "rejected": 0, "reject_reasons": {}, "missed_surfaced": 0, "parse_errors": 0,
           "members_found": 0, "members_missed": 0, "members_extra": 0, "dosing_rows": 0, "limits_rows": 0,
           "answers": 0}
    for ing, spl in LABELS.items():
        sents = numbered_sentences(label_record(spl))
        text_to_sid = {s.text: s.sid for s in sents}
        checked = {}
        for ps in PASSES:
            f = folder / f"{ing}.{ps}.json"
            if not f.exists():
                continue
            try:
                ans = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                res["parse_errors"] += 1
                continue
            res["answers"] += 1
            c = _check_pass(ps, ans, ing, spl, sents)
            checked[ps] = c
            res["proposed"] += len(c.rules) + len(c.memberships) + len(c.rejected)
            res["rejected"] += len(c.rejected)
            for r in c.rejected:
                why = r["reason"].split(":")[0].split(" in S")[0].split(" S")[0]
                res["reject_reasons"][why] = res["reject_reasons"].get(why, 0) + 1
        if "administration" not in checked:
            continue
        adm = checked["administration"]
        res["dosing_rows"] += len(checked["dosing"].rules) if "dosing" in checked else 0
        res["limits_rows"] += len(checked["limits"].rules) if "limits" in checked else 0

        gold = _gold_rules(spl)
        drafts = _candidates(adm.rules, checked["limits"].rules if "limits" in checked else [])
        g_by, d_by = {}, {}
        for r in gold:
            g_by.setdefault(_key(r), []).append(r)
        for r in drafts:
            d_by.setdefault(_key(r), []).append(r)
        lab = {"exact": [], "partial": [], "missed": [], "extra": [], "signature": _sig(drafts)}
        for k in set(g_by) | set(d_by):  # one draft row per gold rule, never shared
            gs, ds = list(g_by.get(k, [])), list(d_by.get(k, []))
            pairs = []
            for g in list(gs):  # exact matches first
                d = next((d for d in ds if _nums(d) == _nums(g) and _same_ing(g, d)), None)
                if d is not None:
                    pairs.append(("exact", g, d))
                    gs.remove(g)
                    ds.remove(d)
            for g in list(gs):
                d = next((d for d in ds if _same_ing(g, d)), None)
                if d is not None:
                    pairs.append(("partial", g, d))
                    gs.remove(g)
                    ds.remove(d)
            for how, g, d in pairs:
                lab[how].append(g["id"])
                res["prov_total"] += 1
                res["prov_agree"] += g["provenance"] == d["provenance"]
            lab["missed"] += [g["id"] for g in gs]
            lab["extra"] += [d.get("id") or f"{d['family']}-{d.get('amount', '')}" for d in ds]

        # safety net: is each missed gold rule's sentence at priority 1 in the review queue?
        cited = {text_to_sid.get(r["source"]["sentence"], "") for c in checked.values() for r in c.rules}
        cited |= {text_to_sid.get(m["basis"], "") for m in adm.memberships}
        queue = {u["sentence"]: u["priority"] for u in uncited(sents, cited)}
        for gid in lab["missed"]:
            g = next(r for r in gold if r["id"] == gid)
            # the gold sentence, or any sentence restating it, at priority 1
            if queue.get(g["source"]["sentence"]) == 1 or any(
                    p == 1 and _same_rule_text(t, g["source"]["sentence"]) for t, p in queue.items()):
                res["missed_surfaced"] += 1

        gm = _gold_members(spl)
        prop = {m["ingredient"] for m in adm.memberships}
        covered = {g for g in gm if g in prop or any(is_element(p) and element_of(g) == p for p in prop)}
        res["members_found"] += len(covered)
        res["members_missed"] += len(gm - covered)
        res["members_extra"] += len({p for p in prop if p not in gm and not any(element_of(g) == p for g in gm)})

        for k in ("exact", "partial", "missed", "extra"):
            res[k] += len(lab[k])
        res["labels"][ing] = lab
    return res


def _same_rule_text(a: str, b: str) -> bool:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    return len(wa & wb) / max(1, len(wa | wb)) >= 0.5


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sets = {}
    if any(RESP.glob("*.administration.json")):
        sets["."] = RESP
    for d in sorted(p for p in RESP.iterdir() if p.is_dir()):
        sets[d.name] = d
    results = {name: score_set(path) for name, path in sets.items()}

    print("EXTRACTION EVALUATION  (7 labels; gold = curated library rules citing the same label)\n")
    hdr = f"{'answer set':38} {'exact':>6} {'part':>5} {'miss':>5} {'extra':>6} {'prov':>6} {'rej/prop':>9} {'net':>5} {'memb f/m/x':>11} {'dose':>5} {'lim':>4}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results.items():
        gold_n = r["exact"] + r["partial"] + r["missed"]
        print(f"{NAMES.get(name, name):38} {r['exact']:>3}/{gold_n:<2} {r['partial']:>5} {r['missed']:>5} {r['extra']:>6} "
              f"{r['prov_agree']:>3}/{r['prov_total']:<2} {r['rejected']:>4}/{r['proposed']:<4} "
              f"{r['missed_surfaced']:>2}/{r['missed']:<2} {r['members_found']:>3}/{r['members_missed']}/{r['members_extra']:<5} "
              f"{r['dosing_rows']:>5} {r['limits_rows']:>4}")
    print("\nnet = missed gold rules that still appear at priority 1 in the review queue (the safety net)")

    print("\nRejections by reason (all answer sets):")
    reasons = {}
    for r in results.values():
        for k, v in r["reject_reasons"].items():
            reasons[k] = reasons.get(k, 0) + v
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {v:3}  {k}")

    haiku = [n for n in results if "haiku" in n or n == "."]
    if len(haiku) >= 2:
        same = total = 0
        for a, b in combinations(haiku, 2):
            for ing in LABELS:
                la, lb = results[a]["labels"].get(ing), results[b]["labels"].get(ing)
                if la and lb:
                    total += 1
                    same += la["signature"] == lb["signature"]
        print(f"\nRepeatability (Haiku runs, {len(haiku)} sets): identical timing-rule set for a label in "
              f"{same}/{total} pairwise comparisons ({100 * same / max(1, total):.0f}%)")

    out = {n: {k: v for k, v in r.items() if k != "labels"} | {"labels": {i: {k: v for k, v in l.items() if k != "signature"}
           for i, l in r["labels"].items()}} for n, r in results.items()}
    (DATA / "eval").mkdir(parents=True, exist_ok=True)
    (DATA / "eval" / "extraction.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print("\nwrote data/eval/extraction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
