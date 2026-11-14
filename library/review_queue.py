"""Review queue: one card per item a human must sign off.  [deterministic]

Two sources:
  library    curated items the solver already uses but nobody has verified
             (rules, products, as-needed maxima, group members)
  extracted  rows a model proposed (data/extracted), not yet decided

Each decision is written to disk immediately and atomically (temp file, then
rename), so stopping the server at any moment loses nothing and never leaves a
half-written file. Decided items leave the queue.

Approving an extracted timing rule puts it in the library (data/ingredients),
or, if the library already has the same rule from the same label, verifies that
one instead. Approving a membership adds it to data/groups.json. Dosing and
limit rows are recorded as reviewed in data/extracted (they are reference data,
not solver rules). Every decision records who made it, their role, and when.

Agreement: how many independent model runs (folders under
data/drafts/responses) produced the same row. Rows the runs disagree on come
first; that is where a reviewer's attention matters most.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

from .evaluate import _key, _nums
from .extract import PASSES, _check_pass, label_record, numbered_sentences
from .review import _get, _items

PASS_LIST = {"administration": "administration", "dosing": "dosing", "limits": "limits", "memberships": "memberships"}


def write_json(path: Path, doc) -> None:
    """Atomic: a crash leaves either the old file or the new one, never half of one."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _f(v):
    return None if v is None or v == "" else Fraction(str(v))


def signature(kind: str, r: dict) -> tuple:
    if kind == "administration":
        return (_key(r), tuple(sorted(_nums(r).items())))
    if kind == "dosing":
        return tuple(_f(r.get(k)) for k in ("dose", "dose_max", "every_hours", "times_per_day", "duration_days")) + (
            str(r.get("unit", "")).lower(), bool(r.get("as_needed")))
    if kind == "limits":
        return (r.get("kind"), _f(r.get("amount")), str(r.get("unit", "")).lower())
    return (r.get("group"), r.get("ingredient"))


def _row_id(ing: str, kind: str, r: dict) -> str:
    body = json.dumps({k: v for k, v in r.items() if k != "review"}, sort_keys=True, default=str)
    return f"x:{ing}:{kind}:{hashlib.sha1(body.encode()).hexdigest()[:10]}"


_AGREE: dict = {}  # root -> result; model answers do not change while reviewing


def agreement(root: Path) -> tuple[dict, int]:
    """{(ingredient, kind, signature): number of runs that produced it}, number of runs."""
    if str(root) in _AGREE:
        return _AGREE[str(root)]
    resp = root / "drafts" / "responses"
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
                except (json.JSONDecodeError, Exception):
                    continue
                for r in c.rules:
                    seen.add((ing, ps, signature(ps, r)))
                for m in c.memberships:
                    seen.add((ing, "memberships", signature("memberships", m)))
            for k in seen:
                counts[k] = counts.get(k, 0) + 1
    _AGREE[str(root)] = (counts, len(folders))
    return _AGREE[str(root)]


def queue(root: Path) -> list[dict]:
    root = Path(root)
    cards = []
    for iid, path, ptr, ok, _ in _items(root):
        if ok:
            continue
        rec = _get(json.loads(path.read_text(encoding="utf-8")), ptr)
        if rec.get("review", {}).get("decision") == "rejected":
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
        for kind in ("administration", "memberships", "limits", "dosing"):
            for r in d.get(kind, []):
                if r.get("review"):
                    continue
                src = r.get("source") or {}
                cards.append({"id": _row_id(ing, kind, r), "origin": "extracted", "kind": kind, "record": r,
                              "ingredient": ing, "sentence": src.get("sentence") or r.get("basis", ""),
                              "section": src.get("section", ""), "url": src.get("url") or d.get("label_url", ""),
                              "agree": counts.get((ing, kind, signature(kind, r)), 0), "runs": runs, "model": d["model"]})
    order = {"administration": 0, "memberships": 1, "limits": 2, "dosing": 3}
    cards.sort(key=lambda c: (c["origin"] != "library",
                              c["agree"] if c["agree"] is not None else 0,
                              order.get(c["kind"], 0), c["ingredient"]))
    return cards


def _stamp(decision: str, by: str, role: str) -> dict:
    return {"decision": decision, "by": by, "role": role,
            "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")}


def decide(root: Path, card_id: str, decision: str, by: str, role: str) -> dict:
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be 'approved' or 'rejected'")
    if not by.strip():
        raise ValueError("enter the reviewer's name first")
    root = Path(root)
    stamp = _stamp(decision, by.strip(), role.strip())

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
