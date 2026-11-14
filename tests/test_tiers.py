"""M5/M6: tier assignment end to end, conflict shrinking, multiple conflicts,
and meal rules. Uses the draft library in data/ for the demo patients."""

from fractions import Fraction as F
from pathlib import Path

import pytest

from library.load import load
from linnet.conflicts import find_conflicts, shrink
from linnet.patient import load_patient
from linnet.render import render
from linnet.tiers import run
from solver.encode import build, check, solve
from solver.model import MealGap, MealWindow, MinInterval, Product, Regimen, Separation, Window
from solver.verify import violations

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def lib():
    return load(DATA)


def run_patient(lib, name):
    return run(lib, load_patient(DATA / "patients" / f"{name}.json"))


# --- demo patients -----------------------------------------------------------------

def test_tier0_patients(lib):
    for name in ("tier0-levo-cipro", "tier0-levo-cipro-tums"):
        r = run_patient(lib, name)
        assert r.outcome == "tier0", (name, r.solves)


def test_tier1_duplicate_acetaminophen(lib):
    r = run_patient(lib, "tier1-duplicate-acetaminophen")
    assert r.outcome == "tier1"
    first = r.options[0]
    assert [(c.kind, c.what) for c in first.changes] == [("dose", "acetaminophen-500-otc")]
    # alternatives change a different set of things, never a superset of an earlier one
    sets = [{(c.kind, c.what, c.after) for c in o.changes} for o in r.options]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert not sets[i] <= sets[j]


def test_tier1_is_deterministic(lib):
    a = render(run_patient(lib, "tier1-duplicate-acetaminophen"), lib)
    b = render(run_patient(lib, "tier1-duplicate-acetaminophen"), lib)
    assert a == b


def test_tier2_nimodipine_colestipol(lib):
    r = run_patient(lib, "tier2-nimodipine-colestipol")
    assert r.outcome == "tier2"
    assert len(r.conflicts) == 1
    rules = set(r.conflicts[0].rules)
    assert rules == {
        "nimodipine-q4h@nimodipine",
        "colestipol-other-drugs@colestipol-1g-tab~nimodipine-30mg-cap",
        "count::nimodipine-30mg-cap",
        "count::colestipol-1g-tab",
    }
    text = " ".join(render(r, lib).split())  # the card wraps long lines
    assert "every 4 hours for 21 consecutive days" in text
    assert "Ask the prescriber" in text


# --- conflict shrinking --------------------------------------------------------------

def _two_clashes():
    """A-B clash in the morning, C-D clash in the evening, plus a harmless
    rule. One Z3 core shows only one of them."""
    prods = tuple(Product(n, 1, True) for n in "ABCD")
    rules = (
        Window("a-am", "A", ((8, 8),)), Window("b-am", "B", ((9, 9),)),
        Separation("ab", "A", "B", before=3, after=3),
        Window("c-pm", "C", ((18, 18),)), Window("d-pm", "D", ((19, 19),)),
        Separation("cd", "C", "D", before=3, after=3),
        MinInterval("harmless", "A", 1),
    )
    return Regimen(prods, rules)


def test_shrink_returns_a_minimal_conflict():
    enc = build(_two_clashes())
    r = check(enc)
    assert r.status == "unsat"
    c = shrink(enc, r.core)
    assert c.proven_minimal
    assert set(c.rules) in ({"a-am", "b-am", "ab", "count::A", "count::B"}, {"c-pm", "d-pm", "cd", "count::C", "count::D"})
    # minimal: removing any one rule makes the rest satisfiable
    for rid in c.rules:
        assert check(enc, only=[x for x in c.rules if x != rid]).status == "sat"


def test_find_conflicts_reports_both_independent_clashes():
    enc = build(_two_clashes())
    r = check(enc)
    found = [set(c.rules) for c in find_conflicts(enc, r.core)]
    assert {"a-am", "b-am", "ab", "count::A", "count::B"} in found
    assert {"c-pm", "d-pm", "cd", "count::C", "count::D"} in found
    assert enc.disabled == set()  # left as it was


# --- meal rules --------------------------------------------------------------------

def test_meal_gap_and_window_against_fixed_meals():
    reg = Regimen(
        (Product("L", 1, True), Product("N", 2, True)),
        (MealWindow("l-bkf", "L", "breakfast", 1, F(1, 2)),
         MealGap("n-meals", "N", 1, 2),
         MinInterval("n-gap", "N", 8)),
    )
    meals = {"breakfast": F(8), "lunch": F(13), "dinner": F(19)}
    r = solve(reg, meals=meals)
    assert r.status == "sat"
    assert violations(reg, r.schedule, meals) == []
    assert F(7) <= r.schedule["L"][0] <= F(15, 2)
    # verifier catches a dose 30 min after lunch
    assert violations(reg, {"L": [F(7)], "N": [F(27, 2), F(3)]}, meals) != []


def test_meal_window_wraps_midnight():
    reg = Regimen((Product("L", 1, True),), (MealWindow("l", "L", "breakfast", 1, F(1, 2)),))
    meals = {"breakfast": F(1, 4)}  # 00:15 -> dose between 23:15 and 23:45
    r = solve(reg, meals=meals)
    assert r.status == "sat" and violations(reg, r.schedule, meals) == []
    assert F(93, 4) <= r.schedule["L"][0] <= F(95, 4)


# --- greedy baseline (comparison only) -------------------------------------------------

def test_greedy_false_alarm_on_tums_patient(lib):
    """The greedy heuristic gives up where LINNET finds a valid schedule."""
    from linnet.baseline import run_greedy
    p = load_patient(DATA / "patients" / "tier0-levo-cipro-tums.json")
    assert not run_greedy(lib, p).ok
    assert run(lib, p).outcome == "tier0"


def test_greedy_agrees_when_easy(lib):
    from linnet.baseline import run_greedy
    p = load_patient(DATA / "patients" / "tier0-levo-cipro.json")
    g = run_greedy(lib, p)
    assert g.ok
    built = __import__("library.load", fromlist=["build_regimen"]).build_regimen(lib, p.library_items(), "strict")
    assert violations(built.regimen, g.schedule, p.meal_times()) == []


# --- a solve that hits its time limit is never read as a yes or a no -----------------------

def _time_out_call(monkeypatch, which: int):
    """Make the `which`-th solve in run() report a timeout (1 = A, 2 = B, 3 = C)."""
    import linnet.tiers as tiers
    from solver.encode import Result
    real, n = tiers.check, {"calls": 0}

    def flaky(enc, *a, **k):
        n["calls"] += 1
        return Result("unknown", reason="timeout (test)") if n["calls"] == which else real(enc, *a, **k)
    monkeypatch.setattr(tiers, "check", flaky)


def test_solve_a_timeout_then_no_change_schedule_is_tier0(lib, monkeypatch):
    _time_out_call(monkeypatch, 1)
    r = run_patient(lib, "tier0-levo-cipro")
    assert r.outcome == "tier0" and r.unfinished == ["A"]
    assert r.options[0].changes == []


def test_solve_a_timeout_with_changes_is_tier1_with_a_warning(lib, monkeypatch):
    _time_out_call(monkeypatch, 1)
    r = run_patient(lib, "tier1-duplicate-acetaminophen")
    assert r.outcome == "tier1" and r.unfinished == ["A"]
    assert any("may not all be needed" in n for n in r.notes)


def test_solve_c_timeout_is_unknown_not_a_conflict(lib, monkeypatch):
    _time_out_call(monkeypatch, 3)
    r = run_patient(lib, "tier1-duplicate-acetaminophen")
    assert r.outcome == "unknown" and r.tier is None and r.unfinished == ["C"]
    assert not r.conflicts and not r.options


# --- "upon arising, at least 30 min before the first food, beverage, or medication" (alendronate) ---

def _morning(items, wake, breakfast, bed="22:30"):
    from linnet.patient import Patient
    return Patient.model_validate({"name": "t", "constructed": True, "items": items, "routine": {
        "meals": {"breakfast": breakfast, "lunch": "12:30", "dinner": "18:30"}, "sleep": {"start": bed, "end": wake}}})


ALN = {"product": "alendronate-70mg-tab", "prescribed_count": 1}
LEVO = {"product": "levothyroxine-100mcg", "prescribed_count": 1}


def test_alendronate_then_levothyroxine_needs_an_hour_before_breakfast(lib):
    r = run(lib, _morning([ALN, LEVO], "07:00", "08:30"))
    assert r.outcome == "tier0"
    s = r.options[0].schedule
    assert s["alendronate-70mg-tab"][0] == F(7) and s["levothyroxine-100mcg"][0] >= F(15, 2)  # first, then 30 min
    assert run(lib, _morning([ALN, LEVO], "07:00", "07:45")).outcome == "tier1"


def test_alendronate_never_at_bedtime_even_in_tier1(lib):
    r = run(lib, _morning([ALN], "07:00", "07:20"))
    assert r.outcome == "tier1"
    for o in r.options:  # every suggestion keeps it in the morning, before breakfast
        t, bfast = o.schedule["alendronate-70mg-tab"][0], o.meals["breakfast"]
        assert F(5) <= t <= bfast - F(1, 2)


# --- what holds each dose where it is ---------------------------------------------------

def test_each_dose_says_which_rules_pin_it(lib):
    from linnet.serialize import report_json
    j = report_json(run_patient(lib, "tier0-levo-cipro-tums"), lib, None)
    doses = {(e["product"], e["dose"]): {x["id"] for x in e["pinned_by"]} for e in j["options"][0]["events"] if e["kind"] == "dose"}
    assert "levo-breakfast" in doses[("levothyroxine-100mcg", 1)]       # 30-60 min before breakfast
    assert "prescription" in doses[("ciprofloxacin-500mg", 2)]           # exactly 12 h after dose 1
    assert any("levo-interfering" in ids for (p, _), ids in doses.items() if p == "tums-es-750")  # 4 h after levothyroxine


def test_binding_is_exact_and_skips_slack():
    from solver.model import Product, Regimen, Separation
    from solver.verify import binding
    reg = Regimen((Product("a", 1, True), Product("b", 1, True)), (Separation("sep", "a", "b", F(2), F(6)),))
    assert binding(reg, {"a": [F(8)], "b": [F(10)]}) == {("a", 0): ["sep"], ("b", 0): ["sep"]}  # exactly 2 h before
    assert binding(reg, {"a": [F(7)], "b": [F(10)]}) == {}                                      # 3 h: not at the limit
