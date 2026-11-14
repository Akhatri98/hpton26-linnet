"""Review queue: one card per thing a human must sign off.  [deterministic]

    python -m library.review_queue flagged      list everything flagged for a fix

Cards:
  library    curated items the solver already uses but nobody has reviewed
             (rules, products, as-needed maxima, group members), one card each
  extracted  model-proposed rows (data/extracted):
               timing rules and group members, one card each;
               dosing and limits, ONE card per label, holding every strength,
               regimen and limit together, so the whole parse is judged at once

Decisions: approved / rejected / flagged ("right, but needs a fix", with a note).
Each is written to disk at once and atomically (temp file, then rename): stopping
the server never loses a decision or leaves half a file. Decided items leave the
queue. Every decision records who, their role, when, and any note.

Effects:
  approved  library item -> verified. Extracted timing rule -> added to the
            library (or the identical rule from the same label is verified).
            Extracted member -> added to groups.json. Dosing/limits -> recorded.
  rejected  library rule or member -> no longer applied by the solver (it stays
            in the file as an audit trail). A rejected product is still needed to
            check a patient who takes it, so results that use it say so loudly.
  flagged   nothing changes in what the solver applies; the item stays
            unverified and goes on the fix list.

Agreement: how many independent model runs (folders under data/drafts/responses)
produced the same row. Rows the runs disagree on come first.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from .evaluate import _key, _nums
from .extract import PASSES, _check_pass, label_record, numbered_sentences
from .review import _get, _items

DECISIONS = ("approved", "rejected", "flagged")
_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(mg|mcg|g)\b", re.I)


def write_json(path: Path, doc) -> None:
    """Atomic: a crash leaves either the old file or the new one, never half of one."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _f(v):
    try:
        return None if v is None or v == "" else Fraction(str(v))
    except (ValueError, ZeroDivisionError):
        return str(v)


def _strength_key(s) -> tuple:
    return tuple(sorted((Fraction(a.replace(",", "")), u.lower()) for a, u in _AMOUNT.findall(s or "")))


def _ing_key(s) -> str:
    words = [w for w in re.split(r"[\s_]+", str(s or "").lower()) if w and w not in ("hydrochloride", "sodium", "phosphate", "usp")]
    return " ".join(words)


def signature(kind: str, r: dict) -> tuple:
    if kind == "administration":
        return (_key(r), tuple(sorted(_nums(r).items())))
    if kind == "dosing":
        return (_strength_key(r.get("strength")),) + tuple(
            _f(r.get(k)) for k in ("dose", "dose_max", "every_hours", "times_per_day", "duration_days")) + (
            str(r.get("unit", "")).lower(), bool(r.get("as_needed")))
    if kind == "limits":
        return (r.get("kind"), _f(r.get("amount")), str(r.get("unit", "")).lower(),
                _ing_key(r.get("ingredient")), _strength_key(r.get("strength")))
    return (r.get("group"), r.get("ingredient"))


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _row_id(ing: str, kind: str, r: dict) -> str:
    return f"x:{ing}:{kind}:{_hash({k: v for k, v in r.items() if k != 'review'})}"


def _set_id(ing: str, d: dict) -> str:
    rows = [{k: v for k, v in r.items() if k != "review"} for kind in ("dosing", "limits") for r in d.get(kind, [])]
    return f"x:{ing}:doseset:{_hash(rows)}"


def carry_reviews(old: dict, new: dict) -> int:
    """After re-extraction, keep decisions on rows that came out the same. Returns how many."""
    n = 0
    for kind in ("administration", "memberships", "dosing", "limits"):
        stamps = {signature(kind, r): r["review"] for r in old.get(kind, []) if r.get("review")}
        for r in new.get(kind, []):
            if not r.get("review") and signature(kind, r) in stamps:
                r["review"] = stamps[signature(kind, r)]
                n += 1
    return n


# --- agreement across independent model runs ---------------------------------------------

_AGREE: dict = {}  # root -> result; model answers do not change while reviewing


def agreement(root: Path) -> tuple[dict, int]:
    """{(ingredient, kind, signature): number of runs that produced it}, number of runs."""
    resp = root / "drafts" / "responses"
    # Recount when any model answer or extracted record changes (a new run lands mid-review).
    files = list(resp.rglob("*.json")) + list((root / "extracted").glob("*.json")) if resp.is_dir() else []
    key = (str(root), len(files), max((f.stat().st_mtime for f in files), default=0))
    if _AGREE.get(str(root), (None,))[0] == key:
        return _AGREE[str(root)][1]
    folders = []
    if resp.is_dir():
        folders = ([resp] if any(resp.glob("*.administration.json")) else []) + sorted(p for p in resp.iterdir() if p.is_dir())
    counts: dict = {}
    labels = {}
    for f in (root / "extracted").glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        labels[d["ingredient"]] = d["spl_set_id"]
    for ing, spl in labels.items():
        sents = numbered_sentences(label_record(spl))
        for folder in folders:
            seen = set()
            for ps in PASSES:
                f = folder / f"{ing}.{ps}.json"
                if not f.exists():
                    continue
                try:
                    c = _check_pass(ps, json.loads(f.read_text(encoding="utf-8")), ing, spl, sents)
                except Exception:
                    continue
                for r in c.rules:
                    seen.add((ing, ps, signature(ps, r)))
                for m in c.memberships:
                    seen.add((ing, "memberships", signature("memberships", m)))
            for k in seen:
                counts[k] = counts.get(k, 0) + 1
    _AGREE[str(root)] = (key, (counts, len(folders)))
    return counts, len(folders)


# --- the queue ----------------------------------------------------------------------------

def _uncovered(strengths: list[str], dosing: list[dict]) -> list[str]:
    """Strengths the label comes in that no dosing row is about (only when rows name strengths)."""
    named = {_strength_key(r.get("strength")) for r in dosing if r.get("strength")}
    if len(strengths) < 2 or not named:
        return []
    return [s for s in strengths if _strength_key(s) not in named]


def queue(root: Path) -> list[dict]:
    root = Path(root)
    cards = []
    for iid, path, ptr, ok, _ in _items(root):
        rec = _get(json.loads(path.read_text(encoding="utf-8")), ptr)
        if ok or rec.get("review"):
            continue
        kind = iid.split(":", 1)[0] if ":" in iid else "rule"
        src = rec.get("source") or {}
        cards.append({"id": iid, "origin": "library", "kind": kind, "record": rec,
                      "ingredient": "", "sentence": src.get("sentence") or rec.get("basis", ""),
                      "section": src.get("section", ""), "url": src.get("url") or rec.get("url", ""),
                      "agree": None, "runs": None})

    counts, runs = agreement(root)
    for f in sorted((root / "extracted").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ing = d["ingredient"]
        for kind in ("administration", "memberships"):
            for r in d.get(kind, []):
                if r.get("review"):
                    continue
                src = r.get("source") or {}
                cards.append({"id": _row_id(ing, kind, r), "origin": "extracted", "kind": kind, "record": r,
                              "ingredient": ing, "sentence": src.get("sentence") or r.get("basis", ""),
                              "section": src.get("section", ""), "url": src.get("url") or d.get("label_url", ""),
                              "agree": counts.get((ing, kind, signature(kind, r)), 0), "runs": runs, "model": d["model"]})
        open_rows = [r for kind in ("dosing", "limits") for r in d.get(kind, []) if not r.get("review")]
        if open_rows:
            def row(kind, r):
                src = r.get("source") or {}
                return {**{k: v for k, v in r.items() if k not in ("source",)}, "sentence_id": src.get("sentence_id", ""),
                        "sentence": src.get("sentence", ""), "agree": counts.get((ing, kind, signature(kind, r)), 0)}
            dosing = [row("dosing", r) for r in d.get("dosing", [])]
            limits = [row("limits", r) for r in d.get("limits", [])]
            cards.append({"id": _set_id(ing, d), "origin": "extracted", "kind": "doseset", "ingredient": ing,
                          "record": {}, "strengths": d.get("strengths", []), "substances": d.get("substances", []),
                          "dosing": dosing, "limits": limits, "uncovered": _uncovered(d.get("strengths", []), d.get("dosing", [])),
                          "sentence": "", "section": "", "url": d.get("label_url", ""),
                          "agree": min([x["agree"] for x in dosing + limits] or [0]), "runs": runs, "model": d["model"]})
    order = {"administration": 0, "memberships": 1, "doseset": 2}
    cards.sort(key=lambda c: (c["origin"] != "library", c["agree"] if c["agree"] is not None else 0,
                              order.get(c["kind"], 0), c["ingredient"]))
    return cards


# --- decisions ----------------------------------------------------------------------------

def _stamp(decision: str, by: str, role: str, note: str) -> dict:
    s = {"decision": decision, "by": by, "role": role,
         "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}
    if note:
        s["note"] = note
    return s


def decide(root: Path, card_id: str, decision: str, by: str, role: str, note: str = "") -> dict:
    if decision not in DECISIONS:
        raise ValueError("decision must be approved, rejected or flagged")
    if not by.strip():
        raise ValueError("enter the reviewer's name first")
    root = Path(root)
    stamp = _stamp(decision, by.strip(), role.strip(), note.strip())

    if not card_id.startswith("x:"):  # a library item
        for iid, path, ptr, _, _ in _items(root):
            if iid == card_id:
                doc = json.loads(path.read_text(encoding="utf-8"))
                rec = _get(doc, ptr)
                rec["verified"] = decision == "approved"
                rec["review"] = stamp
                write_json(path, doc)
                return {"id": card_id, "result": f"{decision}: {iid}"}
        raise KeyError(card_id)

    _, ing, kind, _ = card_id.split(":", 3)
    f = root / "extracted" / f"{ing}.json"
    d = json.loads(f.read_text(encoding="utf-8"))

    if kind == "doseset":
        if _set_id(ing, d) != card_id:
            raise KeyError(card_id)  # the rows changed since the card was shown
        n = 0
        for k in ("dosing", "limits"):
            for r in d.get(k, []):
                if not r.get("review"):
                    r["review"] = {**stamp, "result": f"{decision} with the whole dosing and limits set"}
                    n += 1
        write_json(f, d)
        return {"id": card_id, "result": f"{decision}: {n} dosing and limit rows for {ing}"}

    row = next((r for r in d.get(kind, []) if _row_id(ing, kind, r) == card_id), None)
    if row is None:
        raise KeyError(card_id)
    result = f"{decision} (recorded in extracted/{ing}.json)"
    if decision == "approved" and kind == "administration":
        result = _promote_rule(root, ing, d["spl_set_id"], row, stamp)
    elif decision == "approved" and kind == "memberships":
        result = _promote_member(root, row, stamp)
    row["review"] = {**stamp, "result": result}
    write_json(f, d)
    return {"id": card_id, "result": result}


def _promote_rule(root: Path, ing: str, spl: str, row: dict, stamp: dict) -> str:
    # Same rule, same label, already in the library (any ingredient file)? Verify that one.
    for p in sorted((root / "ingredients").glob("*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        for r in doc["rules"]:
            if r["source"]["spl_set_id"] == spl and signature("administration", r) == signature("administration", row):
                r["verified"] = True
                r["review"] = stamp
                write_json(p, doc)
                return f"verified existing library rule {r['id']}"
    p = root / "ingredients" / f"{ing}.json"
    if not p.exists():
        return "approved; not added (no library file for this ingredient)"
    doc = json.loads(p.read_text(encoding="utf-8"))
    new = {k: v for k, v in row.items() if k != "review"}
    new["verified"] = True
    new["review"] = stamp
    new["note"] = (new.get("note", "") + " | approved from model draft").strip(" |")
    ids = {r["id"] for r in doc["rules"]}
    while new["id"] in ids:
        new["id"] += "-b"
    doc["rules"].append(new)
    from .schema import IngredientFile
    IngredientFile.model_validate(doc)
    write_json(p, doc)
    return f"added to library as {new['id']}"


def _promote_member(root: Path, row: dict, stamp: dict) -> str:
    p = root / "groups.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    g = doc["groups"].get(row["group"])
    if g is None:
        return f"approved; not added (unknown group {row['group']})"
    for m in g["members"]:
        if m["ingredient"] == row["ingredient"]:
            m["verified"] = True
            m["review"] = stamp
            write_json(p, doc)
            return f"verified existing member {row['ingredient']} of {row['group']}"
    g["members"].append({k: row[k] for k in ("ingredient", "provenance", "basis", "source_spl", "matched") if k in row}
                        | {"verified": True, "review": stamp})
    from .schema import GroupsFile
    GroupsFile.model_validate(doc)
    write_json(p, doc)
    return f"added {row['ingredient']} to {row['group']}"


# --- the fix list -------------------------------------------------------------------------

def flagged(root: Path) -> list[dict]:
    root = Path(root)
    out = []
    for iid, path, ptr, _, _ in _items(root):
        rv = _get(json.loads(path.read_text(encoding="utf-8")), ptr).get("review") or {}
        if rv.get("decision") == "flagged":
            out.append({"item": iid, "file": path.relative_to(root).as_posix(), **rv})
    for f in sorted((root / "extracted").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        seen = set()
        for kind in ("administration", "memberships", "dosing", "limits"):
            for r in d.get(kind, []):
                rv = r.get("review") or {}
                if rv.get("decision") != "flagged":
                    continue
                label = "dosing and limits set" if kind in ("dosing", "limits") else (r.get("id") or r.get("ingredient"))
                key = (d["ingredient"], label, rv.get("at"))
                if key not in seen:
                    seen.add(key)
                    out.append({"item": f"{d['ingredient']}: {label}", "file": f.relative_to(root).as_posix(), **rv})
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["flagged"]:
        items = flagged(Path("data"))
        for x in items:
            print(f"- {x['item']}  ({x['file']})  by {x.get('by')} {x.get('at')}: {x.get('note', '(no note)')}")
        print(f"{len(items)} flagged")
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
