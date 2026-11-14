"""Python face of the solver: hands a regimen to the native C++ core and reads
the answer back.

The Z3 encoding itself (slots, circular distances, tracking literals, every
rule family) is in `solver/native/linnet_solver.cpp`, written against Z3's C++
API. Nothing here builds a formula: `build` walks the regimen and passes plain
data across, `check` runs the solver and returns exact fractions.

Each label rule, and each dose-count requirement, is tracked under its id. A
check passes the ids it wants enforced, so an unsat core names rules, and
conflict shrinking can re-check any subset without rebuilding the encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Optional, Union

from .model import (
    DAY,
    Hours,
    IngredientCeiling,
    IngredientInterval,
    MealGap,
    MealWindow,
    MinInterval,
    Regimen,
    Separation,
    Window,
)
from .native import Native

DEFAULT_TIMEOUT_MS = 10_000

FREE = object()  # a meal whose time the solver chooses (tier 1), instead of a fixed hour

@dataclass
class Encoding:
    regimen: Regimen
    solver: Native
    disabled: set[str] = field(default_factory=set)  # ids not enforced this solve

    @property
    def tracked(self) -> list[str]:
        """Every tracked rule id, in the order it was added."""
        return self.solver.tracked()

    def assumptions(self, only: Optional[Iterable[str]] = None) -> list[str]:
        ids = self.tracked if only is None else only
        return [i for i in ids if i not in self.disabled]

    def add_hard(self, rule) -> None:
        """Assert `rule` outright: always enforced, never named in a core."""
        _add_rule(self.solver, rule, None)


def build(
    regimen: Regimen,
    *,
    counts: Optional[dict[str, int]] = None,
    meals: Optional[dict[str, Union[Hours, object]]] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    optimize: bool = False,
) -> Encoding:
    """Encode `regimen`.

    counts: optional fixed dose count for as-needed products (e.g. the
    patient's reported usage, for the tier-0 solve). Without it, an as-needed
    count is free in [0, max_doses]. Prescribed products always take every slot.

    meals: meal name -> time, either a number (fixed) or FREE (a solver
    variable, for tier 1). Required if the regimen has meal rules.

    optimize: build on Z3's optimizer instead of a plain solver, so the caller
    can add objectives (tier 1). Same constraints and tracking either way.
    """
    counts = counts or {}
    native = Native(optimize=optimize, timeout_ms=timeout_ms)
    for name, when in (meals or {}).items():
        if when is FREE:
            native.meal_var(name)
        else:
            native.meal_fixed(name, when)
    for p in regimen.products:
        native.product(p.name, p.max_doses, p.prescribed, counts.get(p.name))
        for ingredient, amount in p.ingredients.items():
            native.ingredient(p.name, ingredient, amount)
    for rule in regimen.rules:
        _add_rule(native, rule, rule.id)
    return Encoding(regimen=regimen, solver=native)


def _add_rule(native: Native, rule, rule_id: Optional[str]) -> None:
    if isinstance(rule, MinInterval):
        native.min_interval(rule_id, rule.product, rule.hours)
    elif isinstance(rule, IngredientInterval):
        native.ingredient_interval(rule_id, rule.ingredient, rule.hours)
    elif isinstance(rule, Separation):
        native.separation(rule_id, rule.a, rule.b, rule.before, rule.after)
    elif isinstance(rule, IngredientCeiling):
        native.ingredient_ceiling(rule_id, rule.ingredient, rule.max_amount)
    elif isinstance(rule, Window):
        native.window(rule_id, rule.product, rule.spans)
    elif isinstance(rule, MealGap):
        native.meal_gap(rule_id, rule.product, rule.before, rule.after, rule.meal)
    elif isinstance(rule, MealWindow):
        native.meal_window(rule_id, rule.product, rule.meal, rule.earliest, rule.latest)
    else:
        raise TypeError(f"unknown rule {rule!r}")


@dataclass
class Result:
    status: str  # "sat" | "unsat" | "unknown"
    schedule: dict[str, list[Fraction]] = field(default_factory=dict)
    meals: dict[str, Fraction] = field(default_factory=dict)
    core: list[str] = field(default_factory=list)
    reason: str = ""


def read_model(enc: Encoding) -> Result:
    schedule = {p.name: enc.solver.dose_times(p.name, p.max_doses) for p in enc.regimen.products}
    return Result("sat", schedule=schedule, meals=enc.solver.meal_times())


def check(enc: Encoding, only: Optional[Iterable[str]] = None) -> Result:
    """Run the solver with every tracked rule enforced (or just `only`), minus
    `enc.disabled`. Never raises on timeout: returns status 'unknown' with a
    reason, which callers must surface rather than treat as sat or unsat."""
    status = enc.solver.check(enc.assumptions(only))
    if status == "sat":
        return read_model(enc)
    if status == "unsat":
        return Result("unsat", core=sorted(enc.solver.unsat_core()))
    return Result("unknown", reason=enc.solver.reason_unknown() or "unknown")


def solve(regimen: Regimen, **kw) -> Result:
    return check(build(regimen, **kw))
