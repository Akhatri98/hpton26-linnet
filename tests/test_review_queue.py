"""Review queue: each decision is on disk at once, and rejected rules stop being applied."""

import json
import shutil
from pathlib import Path

import pytest

from library import review_queue as rq
from library.load import load


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "data"
    for sub in ("ingredients", "products", "extracted", "labels"):
        shutil.copytree(Path("data") / sub, r / sub)
    shutil.copy(Path("data") / "groups.json", r / "groups.json")
    # Start from no human decisions, whatever the live data holds.
    for f in list((r / "ingredients").glob("*.json")) + list((r / "products").glob("*.json")) + list((r / "extracted").glob("*.json")) + [r / "groups.json"]:
        d = json.loads(f.read_text(encoding="utf-8"))
        def strip(x):
            if isinstance(x, dict):
                x.pop("review", None)
                for v in x.values():
                    strip(v)
            elif isinstance(x, list):
                for v in x:
                    strip(v)
        strip(d)
        f.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return r


def test_decision_is_written_immediately_and_leaves_the_queue(root):
    q = rq.queue(root)
    card = next(c for c in q if c["kind"] == "doseset")  # all dosing + limits of one label, one card
    assert card["dosing"] or card["limits"]
    rq.decide(root, card["id"], "flagged", "Test Reviewer", "pharmacist", "strength missing on the tablet rows")
    # on disk now, on every row of the set, with who, when and the note
    d = json.loads((root / "extracted" / f"{card['ingredient']}.json").read_text(encoding="utf-8"))
    rows = d["dosing"] + d["limits"]
    assert rows and all(r["review"]["by"] == "Test Reviewer" and r["review"]["note"] for r in rows)
    assert card["id"] not in {c["id"] for c in rq.queue(root)}
    assert any(f["item"].startswith(card["ingredient"]) for f in rq.flagged(root))
    assert not list(root.rglob("*.tmp"))  # atomic write left nothing behind


def test_reextraction_keeps_decisions_on_unchanged_rows(root):
    f = root / "extracted" / "acetaminophen.json"
    old = json.loads(f.read_text(encoding="utf-8"))
    old["dosing"][0]["review"] = {"decision": "approved", "by": "Test Reviewer"}
    new = json.loads(json.dumps(old))
    for r in new["dosing"]:
        r.pop("review", None)
    assert rq.carry_reviews(old, new) >= 1 and new["dosing"][0]["review"]["by"] == "Test Reviewer"


def test_approving_a_rule_the_library_has_verifies_it(root):
    lib_rules = {r["id"]: r for f in (root / "ingredients").glob("*.json")
                 for r in json.loads(f.read_text(encoding="utf-8"))["rules"]}
    card = next(c for c in rq.queue(root) if c["kind"] == "administration" and c["ingredient"] == "colestipol")
    out = rq.decide(root, card["id"], "approved", "Test Reviewer", "pharmacist")
    assert "existing library rule colestipol-other-drugs" in out["result"]
    assert lib_rules["colestipol-other-drugs"]["verified"]  # was already verified; still is


def test_rejected_library_rule_is_not_applied(root):
    assert any(r.id == "cipro-cations" for r in load(root).ingredients["ciprofloxacin"].rules)
    rq.decide(root, "cipro-cations", "rejected", "Test Reviewer", "pharmacist")
    assert not any(r.id == "cipro-cations" for r in load(root).ingredients["ciprofloxacin"].rules)
    assert "cipro-cations" not in {c["id"] for c in rq.queue(root)}


def test_reviewer_name_required(root):
    card = rq.queue(root)[0]
    with pytest.raises(ValueError):
        rq.decide(root, card["id"], "approved", "  ", "pharmacist")
