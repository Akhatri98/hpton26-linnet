"""M1 gate: the encoding gives sane sat and unsat answers on invented data.

No real drugs here. Product names are placeholders; numbers are arbitrary.
"""

import random
from fractions import Fraction as F

from solver.encode import build, check, solve
from solver.model import IngredientCeiling, IngredientInterval, MinInterval, Product, Regimen, Separation, Window
from solver.verify import violations


def assert_valid(regimen, result):
    assert result.status == "sat", result
    assert violations(regimen, result.schedule) == []


# --- Regression: circular time --------------------------------------------------

def test_midnight_twice_daily_is_not_fooled_by_linear_distance():
    """Two doses, >= 4 h apart, both inside 22:00-02:00.

    A naive linear encoding accepts 01:00 and 23:00 (|1 - 23| = 22 >= 4), but
    on a repeating day they are 2 h apart. The whole window is only 4 h wide,
    so the only valid placement is exactly 22:00 and 02:00.
    """
    reg = Regimen(
        products=(Product("X", 2, prescribed=True),),
        rules=(
            MinInterval("x-gap", "X", 4),
            Window("x-night", "X", ((22, 2),)),
        ),
    )
    r = solve(reg)
    assert_valid(reg, r)
    assert r.schedule["X"] == [2, 22]

    # Tighten to 5 h: impossible inside a 4 h wrapped window.
    reg5 = Regimen(reg.products, (MinInterval("x-gap", "X", 5), reg.rules[1]))
    r5 = solve(reg5)
    assert r5.status == "unsat"
    assert set(r5.core) <= {"x-gap", "x-night", "count::X"}
    assert "x-gap" in r5.core and "x-night" in r5.core


def test_verifier_rejects_the_naive_midnight_schedule():
    reg = Regimen((Product("X", 2, prescribed=True),), (MinInterval("x-gap", "X", 4),))
    assert violations(reg, {"X": [F(1), F(23)]}) != []


def test_separation_wraps_midnight():
    """A at 23:00 and B at 01:00 are 2 h apart, not 22."""
    def window(end):
        return Regimen(
            products=(Product("A", 1, True), Product("B", 1, True)),
            rules=(
                Separation("a-b", "A", "B", before=4, after=4),
                Window("a-late", "A", ((23, 23),)),
                Window("b-early", "B", ((0, end),)),
            ),
        )

    assert solve(window(2)).status == "unsat"
    assert violations(window(2), {"A": [F(23)], "B": [F(1)]}) != []
    # 03:00 is exactly 4 h after 23:00 across midnight: allowed, and the only option.
    r = solve(window(3))
    assert_valid(window(3), r)
    assert r.schedule == {"A": [23], "B": [3]}


# --- Regression: dose ordering ---------------------------------------------------

def test_three_doses_come_back_in_order():
    """Windows are listed late-first so an unordered encoding is free to put
    slot 0 in the evening; ordering forces slot k < slot k+1."""
    reg = Regimen(
        products=(Product("Y", 3, prescribed=True),),
        rules=(
            MinInterval("y-gap", "Y", 6),
            Window("y-day", "Y", ((20, 21), (13, 14), (6, 7))),
        ),
    )
    r = solve(reg)
    assert_valid(reg, r)
    ts = r.schedule["Y"]
    assert ts == sorted(ts) and len(set(ts)) == 3


def test_verifier_rejects_unordered_doses():
    reg = Regimen((Product("Y", 3, True),), ())
    assert violations(reg, {"Y": [F(14), F(7), F(21)]}) != []


# --- Activation / as-needed counts ----------------------------------------------

def test_as_needed_count_is_a_solver_choice():
    """Antacid-like product: max 4, blocked from 4 h around each of 3 fixed doses.
    Fixing the reported count at 4 is unsat; leaving it free is sat with fewer."""
    reg = Regimen(
        products=(Product("RX", 3, True), Product("OTC", 4, False)),
        rules=(
            Window("rx-fixed", "RX", ((8, 8), (14, 14), (20, 20))),
            MinInterval("rx-gap", "RX", 6),
            Separation("rx-otc", "RX", "OTC", before=4, after=4),
            MinInterval("otc-gap", "OTC", 2),
        ),
    )
    fixed = solve(reg, counts={"OTC": 4})
    assert fixed.status == "unsat"
    assert "reported::OTC" in fixed.core

    free = solve(reg)
    assert_valid(reg, free)
    assert len(free.schedule["OTC"]) <= 4


def test_inactive_slots_are_ignored_by_rules():
    """With max 3 and no reported count, an as-needed product may take 0 doses;
    a rule against it then cannot fire."""
    reg = Regimen(
        products=(Product("RX", 1, True), Product("OTC", 3, False)),
        rules=(Separation("never", "RX", "OTC", before=13, after=12),),  # before+after > 24
    )
    r = solve(reg)
    assert_valid(reg, r)
    assert r.schedule["OTC"] == []
    assert solve(reg, counts={"OTC": 1}).status == "unsat"


# --- Family 4: shared ingredient ceiling ---------------------------------------

def test_shared_ingredient_ceiling_spans_products():
    reg = Regimen(
        products=(
            Product("Single", 4, True, ingredients={"ing": 500}),
            Product("Combo", 2, True, ingredients={"ing": 325, "other": 5}),
        ),
        rules=(IngredientCeiling("ing-max", "ing", 2500),),
    )
    r = solve(reg)
    assert r.status == "unsat"
    assert "ing-max" in r.core  # 4*500 + 2*325 = 2650 > 2500

    ok = Regimen(reg.products, (IngredientCeiling("ing-max", "ing", 3000),))
    assert_valid(ok, solve(ok))


def test_ingredient_interval_spans_products():
    """Two products share `ing`. Per-product gaps alone would allow them to be
    stacked; the ingredient interval keeps every dose of `ing` 4 h apart."""
    prods = (
        Product("Single", 2, True, ingredients={"ing": 500}),
        Product("Combo", 2, True, ingredients={"ing": 325, "other": 5}),
    )
    window = Window("combo-morning", "Combo", ((6, 10),))
    per_product = Regimen(prods, (MinInterval("s", "Single", 4), MinInterval("c", "Combo", 4),
                                  Window("single-morning", "Single", ((6, 10),)), window))
    stacked = solve(per_product)
    assert_valid(per_product, stacked)  # allowed: 6,10 and 6,10

    shared = Regimen(prods, per_product.rules + (IngredientInterval("ing-gap", "ing", 4),))
    r = solve(shared)
    assert r.status == "unsat" and "ing-gap" in r.core  # four doses can't fit 4 h apart in 4 h

    roomy = Regimen(prods, (IngredientInterval("ing-gap", "ing", 4), window))
    r2 = solve(roomy)
    assert_valid(roomy, r2)
    all_ing = sorted(r2.schedule["Single"] + r2.schedule["Combo"])
    assert all(b - a >= 4 for a, b in zip(all_ing, all_ing[1:], strict=False))


# --- Optimize (tier 1 will build on this) ----------------------------------------

def test_same_encoding_works_under_optimize():
    """Maximise as-needed doses under the antacid-style regimen: the optimum is
    a concrete count the solver chose, and the schedule passes the verifier."""
    reg = Regimen(
        products=(Product("RX", 3, True), Product("OTC", 4, False)),
        rules=(
            Window("rx-fixed", "RX", ((8, 8), (14, 14), (20, 20))),
            Separation("rx-otc", "RX", "OTC", before=2, after=2),
            MinInterval("otc-gap", "OTC", 2),
        ),
    )
    enc = build(reg, optimize=True)
    enc.solver.maximize_count("OTC")
    r = check(enc)
    assert_valid(reg, r)
    # Free arcs between RX doses: 10-12, 16-18, 22-06 (8 h). Gaps of 2 h fit
    # 1 + 1 + 4 = 6 > 4, so the label max binds.
    assert len(r.schedule["OTC"]) == 4

    enc2 = build(reg, optimize=True, counts={"OTC": 4})
    enc2.add_hard(Window("otc-early", "OTC", ((0, 4),)))  # untracked: [0, 4] fits 0,2,4 only
    r2 = check(enc2)
    assert r2.status == "unsat" and "reported::OTC" in r2.core


# --- Timeouts --------------------------------------------------------------------

def test_timeout_returns_unknown_not_an_answer():
    reg = Regimen((Product("X", 2, True),), (MinInterval("g", "X", 1),))
    enc = build(reg, timeout_ms=1)
    enc.solver = _AlwaysUnknown(enc.solver)
    r = check(enc)
    assert r.status == "unknown" and r.schedule == {} and r.core == []


class _AlwaysUnknown:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def check(self, ids):
        return "unknown"

    def reason_unknown(self):
        return "timeout (simulated)"


# --- Fuzz: no false sat ------------------------------------------------------------

def _random_regimen(rng):
    names = [f"P{i}" for i in range(rng.randint(2, 4))]
    products = tuple(
        Product(n, rng.randint(1, 3), rng.random() < 0.6, {"ing": rng.choice([0, 250, 500])})
        for n in names
    )
    rules = []
    for n in names:
        if rng.random() < 0.7:
            rules.append(MinInterval(f"gap-{n}", n, rng.randint(1, 10)))
        if rng.random() < 0.4:
            s = rng.randint(0, 23)
            rules.append(Window(f"win-{n}", n, ((s, (s + rng.randint(2, 12)) % 24),)))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if rng.random() < 0.5:
                rules.append(Separation(f"sep-{a}-{b}", a, b, rng.randint(0, 6), rng.randint(0, 6)))
    if rng.random() < 0.5:
        rules.append(IngredientCeiling("ceil", "ing", rng.choice([1000, 2000, 4000])))
    if rng.random() < 0.4:
        rules.append(IngredientInterval("ing-gap", "ing", rng.randint(1, 6)))
    return Regimen(products, tuple(rules))


def test_fuzz_every_sat_schedule_passes_independent_check():
    rng = random.Random(1315)
    seen = {"sat": 0, "unsat": 0}
    for _ in range(300):
        reg = _random_regimen(rng)
        r = solve(reg)
        assert r.status in ("sat", "unsat")
        seen[r.status] += 1
        if r.status == "sat":
            assert violations(reg, r.schedule) == [], (reg, r.schedule)
    assert seen["sat"] > 30 and seen["unsat"] > 30, seen
