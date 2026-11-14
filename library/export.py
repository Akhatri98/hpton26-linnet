"""JSON (source of truth) -> CSV tables (for reading, sorting, filtering).

    python -m library.export        writes data/tables/*.csv

Never edit the CSVs: they are regenerated from the JSON every run.

Tables:
  library_rules.csv     curated rules the solver uses (data/ingredients)
  products.csv          dispensed products (data/products)
  group_members.csv     ingredient groups (data/groups.json)
  dosing.csv            extracted dosing regimens per label (data/extracted)
  limits.csv            extracted limits per label
  administration.csv    extracted timing rules per label (drafts)
  memberships.csv       extracted group-membership proposals
  unaccounted.csv       label sentences with numbers that NO extracted row cites
  rejected.csv          model output the deterministic checks refused, with reasons
  sentences.csv         every numbered sentence the extractor saw (the evidence)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

DATA = Path("data")
OUT = DATA / "tables"
NUM = ("hours", "before", "after", "amount", "unit", "before_meal", "after_meal", "earliest", "latest", "meal",
       "target", "mode")


def _write(name: str, rows: list[dict], cols: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})
    return len(rows)


def _src(r: dict) -> dict:
    s = r.get("source", {})
    return {"spl_set_id": s.get("spl_set_id", ""), "section": s.get("section", ""),
            "sentence_id": s.get("sentence_id", ""), "sentence": s.get("sentence", ""), "label_url": s.get("url", "")}


def export() -> dict[str, int]:
    counts = {}

    rows = []
    for p in sorted((DATA / "ingredients").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for r in d["rules"]:
            rows.append({"ingredient": d["ingredient"], "id": r["id"], "family": r["family"],
                         **{k: r.get(k) for k in NUM}, "provenance": r["provenance"],
                         "loosest": json.dumps(r["loosest"]) if r.get("loosest") else "",
                         "verified": r.get("verified", False), **_src(r), "note": r.get("note", "")})
    counts["library_rules.csv"] = _write("library_rules.csv", rows,
        ["ingredient", "id", "family", *NUM, "provenance", "loosest", "verified",
         "spl_set_id", "section", "sentence", "label_url", "note"])

    rows = []
    for p in sorted((DATA / "products").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d.get("otc_max_doses") or {}
        rows.append({"name": d["name"], "display_name": d["display_name"],
                     "ingredients_per_dose": "; ".join(f"{k} {v['amount']} {v['unit']}" for k, v in d["ingredients"].items()),
                     "dose_note": d.get("dose_note", ""), "otc_max_doses": m.get("count", ""),
                     "ndc": " ".join(d.get("ndc", [])), "verified": d.get("verified", False),
                     "spl_set_id": d["spl_set_id"], "label_url": d["url"]})
    counts["products.csv"] = _write("products.csv", rows,
        ["name", "display_name", "ingredients_per_dose", "dose_note", "otc_max_doses", "ndc", "verified",
         "spl_set_id", "label_url"])

    rows = []
    g = DATA / "groups.json"
    if g.exists():
        for name, body in json.loads(g.read_text(encoding="utf-8"))["groups"].items():
            for m in body["members"]:
                rows.append({"group": name, **m})
    counts["group_members.csv"] = _write("group_members.csv", rows,
        ["group", "ingredient", "provenance", "verified", "source_spl", "basis"])

    tables = {k: [] for k in ("dosing", "limits", "administration", "memberships", "unaccounted", "rejected", "sentences")}
    for p in sorted((DATA / "extracted").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        base = {"ingredient": d["ingredient"], "model": d["model"]}
        for r in d["dosing"]:
            tables["dosing"].append({**base, **{k: v for k, v in r.items() if k != "source"}, **_src(r)})
        for r in d["limits"]:
            tables["limits"].append({**base, **{k: v for k, v in r.items() if k != "source"}, **_src(r)})
        for r in d["administration"]:
            tables["administration"].append({**base, "id": r["id"], "family": r["family"],
                                             **{k: r.get(k) for k in NUM}, "provenance": r["provenance"],
                                             "loosest": json.dumps(r["loosest"]) if r.get("loosest") else "",
                                             **_src(r)})
        for m in d["memberships"]:
            tables["memberships"].append({**base, **m})
        for u in d["unaccounted_for"]:
            tables["unaccounted"].append({**base, "spl_set_id": d["spl_set_id"], **u, "label_url": d["label_url"]})
        for r in d["rejected"]:
            tables["rejected"].append({**base, "pass": r.get("pass", "administration"), "reason": r["reason"],
                                       "model_output": json.dumps(r["item"], ensure_ascii=False)})
        for s in d["sentences"]:
            tables["sentences"].append({**base, "spl_set_id": d["spl_set_id"], **s})

    src_cols = ["spl_set_id", "section", "sentence_id", "sentence", "label_url"]
    counts["dosing.csv"] = _write("dosing.csv", tables["dosing"],
        ["ingredient", "condition", "population", "dose", "dose_max", "unit", "every_hours", "times_per_day",
         "duration_days", "as_needed", "provenance", "verified", "model", *src_cols])
    counts["limits.csv"] = _write("limits.csv", tables["limits"],
        ["ingredient", "kind", "amount", "unit", "population", "provenance", "verified", "model", *src_cols])
    counts["administration.csv"] = _write("administration.csv", tables["administration"],
        ["ingredient", "id", "family", *NUM, "provenance", "loosest", "model", *src_cols])
    counts["memberships.csv"] = _write("memberships.csv", tables["memberships"],
        ["ingredient", "group", "provenance", "matched", "verified", "model", "source_spl", "basis"])
    counts["unaccounted.csv"] = _write("unaccounted.csv", tables["unaccounted"],
        ["ingredient", "priority", "kind", "note", "sentence_id", "section", "sentence", "spl_set_id", "label_url", "model"])
    counts["rejected.csv"] = _write("rejected.csv", tables["rejected"],
        ["ingredient", "pass", "reason", "model_output", "model"])
    counts["sentences.csv"] = _write("sentences.csv", tables["sentences"],
        ["ingredient", "spl_set_id", "sentence_id", "section", "text"])
    return counts


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for name, n in export().items():
        print(f"{OUT / name}: {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
