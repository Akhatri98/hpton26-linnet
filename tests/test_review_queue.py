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
    return r


def test_decision_is_written_immediately_and_leaves_the_queue(root):
    q = rq.queue(root)
    card = next(c for c in q if c["origin"] == "extracted" and c["kind"] == "dosing")
    rq.decide(root, card["id"], "rejected", "Test Reviewer", "pharmacist")
    # on disk now, with who and when
    d = json.loads((root / "extracted" / f"{card['ingredient']}.json").read_text(encoding="utf-8"))
    stamped = [r for r in d["dosing"] if r.get("review")]
    assert len(stamped) == 1 and stamped[0]["review"]["by"] == "Test Reviewer" and stamped[0]["review"]["at"]
    assert card["id"] not in {c["id"] for c in rq.queue(root)}
    assert not list(root.rglob("*.tmp"))  # atomic write left nothing behind


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
