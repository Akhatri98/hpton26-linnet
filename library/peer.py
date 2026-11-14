"""Peer review: a second reviewer's opinions, kept apart from sign-off.  [deterministic]

    python -m library.peer list [--kind K] [--ingredient I] [--limit N] [--all]
    python -m library.peer show <card_id>
    python -m library.peer record <card_id> agree|disagree|unsure "<reason>" [--by NAME]
    python -m library.peer note <ingredient> "<something no card covers>" [--by NAME]
    python -m library.peer summary

Opinions are appended to data/reviews/peer-<by>.jsonl, one JSON line each. They
never set `verified` and never touch the library: only a human's decision in
the Review screen does that. The Review screen shows the latest opinion on each
card. Each record keeps a snapshot of the card, so it stays readable after the
rows are re-extracted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .review_queue import queue

DATA = Path("data")
VERDICTS = ("agree", "disagree", "unsure")


def _file(root: Path, by: str) -> Path:
    return root / "reviews" / f"peer-{by}.jsonl"


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
    with path.open("a", encoding="utf-8") as f:  # one write per opinion: never half a line
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def opinions(root: Path) -> dict[str, dict]:
    """card id -> latest opinion, across all peer files."""
    out: dict[str, dict] = {}
    d = root / "reviews"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("peer-*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("kind") != "note":
                    out[rec["id"]] = rec
    return out


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _card_text(c: dict, full: bool = False) -> str:
    agree = "in library" if c["origin"] == "library" else f"{c['agree']}/{c['runs']} model runs agree"
    if c["kind"] == "doseset":  # every dosing and limit row of one label, judged together
        lines = [f"[{c['id']}]  dosing and limits  {c['ingredient']}  (lowest agreement {c['agree']}/{c['runs']})",
                 f"  strengths on the label (read by code): {', '.join(c['strengths']) or 'none found'}",
                 f"  active ingredients: {', '.join(c['substances'])}",
                 f"  label:    {c['url']}"]
        if c["uncovered"]:
            lines.append(f"  NO REGIMEN FOUND FOR: {', '.join(c['uncovered'])}")
        for kind in ("dosing", "limits"):
            for x in c[kind]:
                row = {k: v for k, v in x.items() if k not in ("sentence", "review", "verified", "agree") and v not in (None, "")}
                lines.append(f"  {kind:7} {json.dumps(row, ensure_ascii=False)}")
                lines.append(f"          sentence: {x['sentence']}")
        return "\n".join(lines)
    rec = {k: v for k, v in c["record"].items() if k not in ("source", "review")}
    lines = [f"[{c['id']}]  {c['kind']}  {c['ingredient'] or ''}  ({agree})",
             f"  sentence: {c['sentence'] or '(none)'}",
             f"  section:  {c['section']}",
             f"  label:    {c['url']}",
             f"  row:      {json.dumps(rec, ensure_ascii=False)}"]
    if full:
        src = c["record"].get("source") or {}
        spl = src.get("spl_set_id") or c["record"].get("source_spl") or c["record"].get("spl_set_id", "")
        if spl:
            lines.append(f"  full label text (cached): data/labels/{spl}.json")
    return "\n".join(lines)


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="library.peer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--kind")
    ls.add_argument("--ingredient")
    ls.add_argument("--limit", type=int, default=0)
    ls.add_argument("--all", action="store_true", help="include cards you already gave an opinion on")
    sh = sub.add_parser("show")
    sh.add_argument("id")
    rc = sub.add_parser("record")
    rc.add_argument("id")
    rc.add_argument("verdict", choices=VERDICTS)
    rc.add_argument("reason")
    rc.add_argument("--by", default="claude")
    nt = sub.add_parser("note")
    nt.add_argument("ingredient")
    nt.add_argument("text")
    nt.add_argument("--by", default="claude")
    sub.add_parser("summary")
    a = ap.parse_args(argv)

    cards = queue(DATA)
    by_id = {c["id"]: c for c in cards}
    done = opinions(DATA)

    if a.cmd == "list":
        shown = [c for c in cards if (a.all or c["id"] not in done)
                 and (not a.kind or c["kind"] == a.kind) and (not a.ingredient or c["ingredient"] == a.ingredient)]
        for c in shown[: a.limit or None]:
            print(_card_text(c))
            print()
        print(f"{len(shown)} card(s) shown; {len(cards)} in the human review queue; {len(done)} with a peer opinion.")
        return 0
    if a.cmd == "show":
        c = by_id.get(a.id)
        if not c:
            print(f"no card {a.id} in the queue (a human may have decided it already)")
            return 1
        print(_card_text(c, full=True))
        if a.id in done:
            print(f"  peer opinion so far: {done[a.id]['verdict']}: {done[a.id]['reason']}")
        return 0
    if a.cmd == "record":
        c = by_id.get(a.id)
        if not c:
            print(f"no card {a.id} in the queue (a human may have decided it already); nothing recorded")
            return 1
        if len(a.reason.strip()) < 10:
            print("give a reason that cites the label wording")
            return 1
        _append(_file(DATA, a.by), {"id": a.id, "verdict": a.verdict, "reason": a.reason.strip(), "by": a.by,
                                    "at": _now(), "card": {k: c[k] for k in ("kind", "ingredient", "sentence", "record")}})
        print(f"recorded: {a.id} {a.verdict}")
        return 0
    if a.cmd == "note":
        _append(_file(DATA, a.by), {"kind": "note", "ingredient": a.ingredient, "text": a.text.strip(),
                                    "by": a.by, "at": _now()})
        print("noted")
        return 0
    if a.cmd == "summary":
        counts: dict[str, int] = {}
        for o in done.values():
            counts[o["verdict"]] = counts.get(o["verdict"], 0) + 1
        print(f"peer opinions: {counts}  |  cards still without one: {sum(1 for c in cards if c['id'] not in done)}")
        for o in done.values():
            if o["verdict"] != "agree":
                print(f"- {o['verdict']:9} {o['id']}: {o['reason']}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
