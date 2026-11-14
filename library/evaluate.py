"""Score model-drafted rows against the curated library (deterministic).

    python -m library.evaluate                 every ingredient with a draft
    python -m library.evaluate nimodipine ...  just these

Gold = data/ingredients/<ing>.json rules that cite the SAME label the draft
was made from. Gold rows not yet human-verified are counted but flagged:
until they are verified, the score measures agreement with Claude's drafts,
not with a pharmacist.

Per rule key (family + target/mode/meal):
  exact      key and every number match
  partial    key matches, a number differs
  missed     in gold, not in draft
  extra      in draft, not in gold (could be a real rule gold missed: read it)
Also: provenance (quoted/interpreted) agreement on matched keys, and group
memberships proposed vs the curated groups.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

DATA = Path("data")
NUM = ("hours", "before", "after", "amount", "before_meal", "after_meal", "earliest", "latest", "within")


def _key(r: dict) -> tuple:
    return (r["family"], r.get("target") or r.get("mode") or "", r.get("meal") or "")


def _nums(r: dict) -> dict:
    return {k: Fraction(str(r[k])) for k in NUM if k in r and Fraction(str(r[k])) != 0}


def score(ingredient: str) -> dict:
    draft = json.loads((DATA / "drafts" / f"{ingredient}.json").read_text(encoding="utf-8"))
    report = json.loads((DATA / "drafts" / f"{ingredient}.report.json").read_text(encoding="utf-8"))
    spl = draft["primary_spl"]
    gold_file = DATA / "ingredients" / f"{ingredient}.json"
    gold_all = json.loads(gold_file.read_text(encoding="utf-8"))["rules"] if gold_file.exists() else []
    gold = [r for r in gold_all if r["source"]["spl_set_id"] == spl]

    g_by, d_by = {}, {}
    for r in gold:
        g_by.setdefault(_key(r), []).append(r)
    for r in draft["rules"]:
        d_by.setdefault(_key(r), []).append(r)

    res = {"ingredient": ingredient, "model": report.get("model"), "exact": [], "partial": [], "missed": [],
           "extra": [], "provenance_agree": 0, "provenance_total": 0,
           "rejected_by_checks": len(report.get("rejected", [])),
           "gold_unverified": sum(1 for r in gold if not r.get("verified"))}
    for k, gs in g_by.items():
        ds = d_by.get(k, [])
        if not ds:
            res["missed"] += [r["id"] for r in gs]
            continue
        g, d = gs[0], ds[0]
        same = _nums(g) == _nums(d)
        res["exact" if same else "partial"].append(
            g["id"] if same else f"{g['id']}: gold {_fmt(_nums(g))} vs draft {_fmt(_nums(d))}")
        res["provenance_total"] += 1
        res["provenance_agree"] += g["provenance"] == d["provenance"]
    res["extra"] = [f"{r['id']} {_fmt(_nums(r))}" for k, ds in d_by.items() if k not in g_by for r in ds]

    groups = json.loads((DATA / "groups.json").read_text(encoding="utf-8"))["groups"]
    gold_members = {(g, m["ingredient"]) for g, b in groups.items() for m in b["members"] if m.get("source_spl") == spl}
    prop = {(m["group"], m["ingredient"]) for m in report.get("accepted_memberships", [])}
    gold_ings, prop_ings = {i for _, i in gold_members}, {i for _, i in prop}
    res["members_found"] = sorted(gold_ings & prop_ings)
    res["members_missed"] = sorted(gold_ings - prop_ings)
    res["members_extra"] = sorted(prop_ings - gold_ings)
    return res


def _fmt(d: dict) -> str:
    return "{" + ", ".join(f"{k}={float(v):g}" for k, v in d.items()) + "}"


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="library.evaluate")
    ap.add_argument("ingredients", nargs="*")
    args = ap.parse_args(argv)
    names = args.ingredients or sorted(p.stem for p in (DATA / "drafts").glob("*.json")
                                       if not p.stem.endswith(".report"))
    tot = {"exact": 0, "partial": 0, "missed": 0, "extra": 0, "pa": 0, "pt": 0, "rej": 0, "gu": 0}
    for n in names:
        r = score(n)
        print(f"== {n}  (model: {r['model']})")
        for k in ("exact", "partial", "missed", "extra"):
            if r[k]:
                print(f"   {k:8} " + "\n            ".join(r[k]))
        print(f"   provenance agrees {r['provenance_agree']}/{r['provenance_total']}  |  "
              f"rejected by checks: {r['rejected_by_checks']}"
              + (f"  |  gold rows unverified: {r['gold_unverified']}" if r["gold_unverified"] else ""))
        if r["members_found"] or r["members_missed"] or r["members_extra"]:
            print(f"   memberships  found {r['members_found']}  missed {r['members_missed']}  extra {r['members_extra']}")
        for k, t in (("exact", "exact"), ("partial", "partial"), ("missed", "missed"), ("extra", "extra")):
            tot[t] += len(r[k])
        tot["pa"] += r["provenance_agree"]
        tot["pt"] += r["provenance_total"]
        tot["rej"] += r["rejected_by_checks"]
        tot["gu"] += r["gold_unverified"]
    gold_n = tot["exact"] + tot["partial"] + tot["missed"]
    print(f"\nTOTAL  exact {tot['exact']}/{gold_n} gold rules, partial {tot['partial']}, missed {tot['missed']}, "
          f"extra {tot['extra']}, provenance {tot['pa']}/{tot['pt']}, rejected by checks {tot['rej']}")
    if tot["gu"]:
        print(f"NOTE: {tot['gu']} gold rule(s) are not yet human-verified; scores are provisional.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
