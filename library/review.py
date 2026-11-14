"""Review and sign off library rows.

    python -m library.review                 list everything not yet verified
    python -m library.review --all           list everything
    python -m library.review --approve ID..  mark items verified (edits the JSON)

IDs:
  rule id               e.g. nimodipine-q4h
  product:<name>        e.g. product:nimodipine-30mg-cap
  max:<product>         the as-needed maximum on a product
  member:<group>:<ing>  e.g. member:polyvalent_cations:calcium_carbonate

Approving is the human step: only do it after reading the label sentence.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path


def _wrap(s: str, indent: int = 6) -> str:
    # Never wrap a line that contains a link: a split URL is a broken URL.
    if "http://" in s or "https://" in s:
        return " " * indent + s
    return textwrap.fill(s, 96, initial_indent=" " * indent, subsequent_indent=" " * indent,
                         break_long_words=False, break_on_hyphens=False)


def _items(root: Path):
    """Yield (id, path, json-pointer-ish getter, display lines)."""
    for path in sorted((root / "ingredients").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for i, r in enumerate(doc["rules"]):
            params = {k: v for k, v in r.items() if k not in
                      ("id", "family", "provenance", "source", "loosest", "note", "verified")}
            lines = [f"{r['family']}  {params}  [{r['provenance']}]"
                     + (f"  loosest={r['loosest']}" if r.get("loosest") else ""),
                     f"\"{r['source']['sentence']}\"",
                     f"section: {r['source']['section']}",
                     f"label: {r['source']['url']}"]
            if r.get("note"):
                lines.append(f"note: {r['note']}")
            yield r["id"], path, ("rules", i), r.get("verified", False), lines
    for path in sorted((root / "products").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        lines = [f"{doc['display_name']}  ingredients/dose={doc['ingredients']}",
                 f"dose: {doc.get('dose_note', '')}", f"label: {doc['url']}"]
        yield f"product:{doc['name']}", path, (), doc.get("verified", False), lines
        if doc.get("otc_max_doses"):
            m = doc["otc_max_doses"]
            yield (f"max:{doc['name']}", path, ("otc_max_doses",), m.get("verified", False),
                   [f"max {m['count']} doses/day [{m['provenance']}]", f"\"{m['source']['sentence']}\"",
                    f"section: {m['source']['section']}", f"label: {m['source']['url']}",
                    f"note: {m.get('note', '')}"])
    gpath = root / "groups.json"
    if gpath.exists():
        doc = json.loads(gpath.read_text(encoding="utf-8"))
        for g, body in doc["groups"].items():
            for i, m in enumerate(body["members"]):
                yield (f"member:{g}:{m['ingredient']}", gpath, ("groups", g, "members", i),
                       m.get("verified", False),
                       [f"[{m['provenance']}] basis: \"{m['basis']}\"",
                        f"label: https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={m.get('source_spl', '?')}"])


def _get(doc, ptr):
    for k in ptr:
        doc = doc[k]
    return doc


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="library.review")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--approve", nargs="+", metavar="ID")
    args = ap.parse_args(argv)

    items = list(_items(args.data))
    if args.approve:
        wanted, done = set(args.approve), set()
        for iid, path, ptr, _, _ in items:
            if iid in wanted:
                doc = json.loads(path.read_text(encoding="utf-8"))
                _get(doc, ptr)["verified"] = True
                path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                done.add(iid)
        for iid in sorted(done):
            print(f"verified: {iid}")
        for iid in sorted(wanted - done):
            print(f"NOT FOUND: {iid}")
        return 0 if wanted == done else 1

    shown = [it for it in items if args.all or not it[3]]
    for iid, path, _, ok, lines in shown:
        print(f"{'[x]' if ok else '[ ]'} {iid}    ({path.as_posix()})")
        for line in lines:
            print(_wrap(line))
        print()
    print(f"{sum(1 for it in items if not it[3])} of {len(items)} items unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
