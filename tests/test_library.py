"""Constraint library: schema validation, fallback/strictest/loosest resolution,
and expansion of ingredient groups to products. All data here is invented."""

import json
from fractions import Fraction as F

import pytest
from pydantic import ValidationError

from library.load import Item, build_regimen, load, resolve
from library.schema import IngredientFile
from solver.encode import solve
from solver.verify import violations

PRIMARY, FALLBACK = "spl-primary-0001", "spl-fallback-0002"


def src(spl=PRIMARY, sentence="Invented sentence for testing only."):
    return {"spl_set_id": spl, "url": f"https://example.invalid/{spl}", "section": "7", "sentence": sentence}


def write(root, rel, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def lib_root(tmp_path):
    # drug_a: binds cations. Primary says 2 h before / 6 h after;
    # a fallback label says 4 h before / 6 h after -> disagreement.
    write(tmp_path, "ingredients/drug_a.json", {
        "ingredient": "drug_a",
        "primary_spl": PRIMARY,
        "fallback_spls": [FALLBACK],
        "rules": [
            {"id": "a-sep-cation", "family": "separation", "target": "group:cations",
             "before": 2, "after": 6, "provenance": "quoted", "source": src()},
            {"id": "a-sep-cation-fb", "family": "separation", "target": "group:cations",
             "before": 4, "after": 6, "provenance": "quoted", "source": src(FALLBACK)},
            # only the fallback states an interval -> used, tagged fallback
            {"id": "a-gap-fb", "family": "min_interval", "hours": 8,
             "provenance": "interpreted", "loosest": {"hours": 0},
             "source": src(FALLBACK, "Invented: take twice daily.")},
        ],
    })
    write(tmp_path, "ingredients/mineral_c.json", {
        "ingredient": "mineral_c", "primary_spl": PRIMARY,
        "rules": [{"id": "c-max", "family": "daily_max", "amount": 2000, "unit": "mg",
                   "provenance": "quoted", "source": src()}],
    })
    write(tmp_path, "groups.json", {"groups": {"cations": {"members": [
        {"ingredient": "mineral_c", "provenance": "quoted", "basis": "Invented: products containing mineral C."},
    ]}}})
    write(tmp_path, "products/rx-a.json", {
        "name": "rx-a", "display_name": "Drug A 100 mg", "spl_set_id": PRIMARY, "url": "u",
        "ingredients": {"drug_a": {"amount": 100, "unit": "mg"}},
    })
    write(tmp_path, "products/otc-c.json", {
        "name": "otc-c", "display_name": "Mineral C chewable", "spl_set_id": "spl-otc-0003", "url": "u",
        "ingredients": {"mineral_c": {"amount": 500, "unit": "mg"}},
        "otc_max_doses": {"count": 4, "provenance": "quoted", "source": src("spl-otc-0003")},
    })
    return tmp_path


def test_strict_takes_strictest_and_flags_disagreement(lib_root):
    lib = load(lib_root)
    res = {r.key: r for r in resolve(lib.ingredients["drug_a"], "strict")}
    sep = res[("drug_a", "separation", "group:cations")]
    assert sep.values == {"before": 4, "after": 6}
    assert sep.rule.id == "a-sep-cation"  # primary is still the cited rule
    assert sep.disagreement is not None and not sep.used_fallback

    gap = res[("drug_a", "min_interval", "")]
    assert gap.used_fallback and gap.values == {"hours": 8}


def test_loose_takes_loosest_including_interpretations(lib_root):
    lib = load(lib_root)
    res = {r.key: r for r in resolve(lib.ingredients["drug_a"], "loose")}
    assert res[("drug_a", "separation", "group:cations")].values == {"before": 2, "after": 6}
    assert res[("drug_a", "min_interval", "")].values == {"hours": 0}


def test_group_expands_to_products_and_solves(lib_root):
    lib = load(lib_root)
    built = build_regimen(lib, [Item("rx-a", prescribed_count=2), Item("otc-c")], "strict")
    ids = {r.id for r in built.regimen.rules}
    assert "a-sep-cation@rx-a~otc-c" in ids
    assert "c-max@mineral_c" in ids
    assert built.origin["a-sep-cation@rx-a~otc-c"][0] == "a-sep-cation"

    r = solve(built.regimen)
    assert r.status == "sat"
    assert violations(built.regimen, r.schedule) == []


def test_missing_ingredient_is_reported_not_silent(lib_root):
    write(lib_root, "products/mystery.json", {
        "name": "mystery", "display_name": "Mystery", "spl_set_id": "spl-myst-0004", "url": "u",
        "ingredients": {"unknown_x": {"amount": 1, "unit": "mg"}},
    })
    built = build_regimen(load(lib_root), [Item("mystery", 1)], "strict")
    assert any("unknown_x" in s for s in built.skipped)


# --- schema strictness -------------------------------------------------------------

BASE = {"ingredient": "x", "primary_spl": PRIMARY}


@pytest.mark.parametrize("rule, msg", [
    ({"id": "r", "family": "min_interval", "hours": 8, "provenance": "interpreted", "source": src()},
     "needs `loosest`"),
    ({"id": "r", "family": "min_interval", "hours": 8, "provenance": "quoted", "loosest": {"hours": 0},
      "source": src()}, "must not carry"),
    ({"id": "r", "family": "min_interval", "hours": 8, "provenance": "quoted", "source": src("spl-other-9999")},
     "undeclared SPL"),
    ({"id": "r", "family": "min_interval", "hours": 8, "provenance": "quoted", "source": src(), "typo": 1},
     "Extra inputs"),
    ({"id": "r", "family": "min_interval", "hours": -1, "provenance": "quoted", "source": src()},
     "greater than or equal"),
    ({"id": "r", "family": "min_interval", "hours": 8, "provenance": "quoted",
      "source": {**src(), "sentence": "short"}}, "at least 10"),
])
def test_schema_rejects_bad_rules(rule, msg):
    with pytest.raises(ValidationError, match=msg):
        IngredientFile.model_validate({**BASE, "rules": [rule]})
