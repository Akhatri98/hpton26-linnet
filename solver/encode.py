"""Z3 encoding of a regimen over one repeating day, in linear real arithmetic.

Shape (see the spec, "The solver core"):
  - each product gets `max_doses` slots; slot k has a real time t[k] in [0, 24)
    and a boolean on[k]
  - activation is monotone (on[k+1] -> on[k]), so the active slots are a prefix
  - active slots are strictly ordered (t[k] < t[k+1]), which is also symmetry
    breaking; inactive slots are pinned to 0 so models are deterministic
  - every distance is circular, because the card repeats daily

Structural constraints above are always asserted. Each label rule, and each
dose-count requirement, is guarded by a tracking literal named after its id:
`lit -> formula`. A solve passes the literals it wants enforced as
assumptions, so an unsat core names rules, and conflict shrinking can re-check
any subset without rebuilding the encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterable, Optional, Union

import z3

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

DEFAULT_TIMEOUT_MS = 10_000

Time = Union[Hours, z3.ArithRef]


def _q(x: Hours) -> z3.ArithRef:
    x = Fraction(x)
    return z3.RealVal(f"{x.numerator}/{x.denominator}")


def _t(x: Time) -> z3.ArithRef:
    return x if isinstance(x, z3.ExprRef) else _q(x)


def forward_gap_within(a: z3.ArithRef, b: z3.ArithRef, lo: Hours, hi: Hours) -> z3.BoolRef:
    """The forward gap from a to b, ((b - a) mod 24) in [0, 24), lies in [lo, hi].

    With diff = b - a in (-24, 24):
      forward = diff        when diff >= 0
      forward = diff + 24   when diff <  0   (strict: diff == 0 means forward == 0)
    """
    diff = b - a
    return z3.Or(
        z3.And(diff >= 0, diff >= _q(lo), diff <= _q(hi)),
        z3.And(diff < 0, diff + DAY >= _q(lo), diff + DAY <= _q(hi)),
    )


def forward(a: z3.ArithRef, b: z3.ArithRef) -> z3.ArithRef:
    """(b - a) mod 24 as a term, for a, b in [0, 24)."""
    return z3.If(b >= a, b - a, b - a + DAY)


def circular_distance_at_least(x: z3.ArithRef, y: z3.ArithRef, g: Hours) -> z3.BoolRef:
    """cdist(x, y) = min(|x - y|, 24 - |x - y|) >= g."""
    return forward_gap_within(x, y, g, DAY - Fraction(g))


def outside_window_around(t: z3.ArithRef, anchor: z3.ArithRef, before: Hours, after: Hours) -> z3.BoolRef:
    """t is at least `before` h before and `after` h after `anchor` (circular)."""
    if Fraction(before) + Fraction(after) > DAY:
        return z3.BoolVal(False)
    f = forward_gap_within(t, anchor, before, DAY - Fraction(after))
    # A simultaneous dose is 0 h after the anchor; forward_gap_within alone
    # would accept that when before == 0.
    return z3.And(f, t != anchor) if Fraction(after) > 0 else f


def in_spans(t: z3.ArithRef, spans) -> z3.BoolRef:
    parts = []
    for start, end in spans:
        if Fraction(start) <= Fraction(end):
            parts.append(z3.And(t >= _q(start), t <= _q(end)))
        else:  # wraps midnight
            parts.append(z3.Or(t >= _q(start), t <= _q(end)))
    return z3.Or(*parts)


@dataclass
class Slots:
    t: list[z3.ArithRef]
    on: list[z3.BoolRef]

    def count(self) -> z3.ArithRef:
        return z3.Sum([z3.If(o, 1, 0) for o in self.on]) if self.on else z3.IntVal(0)

    def doses(self):
        return list(zip(self.t, self.on))


@dataclass
class Encoding:
    regimen: Regimen
    solver: Union[z3.Solver, z3.Optimize]
    slots: dict[str, Slots]
    meals: dict[str, z3.ArithRef] = field(default_factory=dict)
    tracked: dict[str, z3.BoolRef] = field(default_factory=dict)
    disabled: set[str] = field(default_factory=set)  # ids not enforced this solve

    def track(self, rule_id: str, formula: z3.BoolRef) -> None:
        if rule_id in self.tracked:
            raise ValueError(f"duplicate rule id {rule_id!r}")
        lit = z3.Bool(f"track::{rule_id}")
        self.tracked[rule_id] = lit
        self.solver.add(z3.Implies(lit, formula))

    def assumptions(self, only: Optional[Iterable[str]] = None) -> list[z3.BoolRef]:
        ids = self.tracked if only is None else only
        return [self.tracked[i] for i in ids if i not in self.disabled]

    def id_of(self, lit) -> str:
        return str(lit).removeprefix("track::")


def build(
    regimen: Regimen,
    *,
    counts: Optional[dict[str, int]] = None,
    meals: Optional[dict[str, Time]] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    optimize: bool = False,
) -> Encoding:
    """Encode `regimen`.

    counts: optional fixed dose count for as-needed products (e.g. the
    patient's reported usage, for the tier-0 solve). Without it, an as-needed
    count is free in [0, max_doses]. Prescribed products always take every slot.

    meals: meal name -> time, either a number (fixed) or a z3 real (variable,
    for tier 1). Required if the regimen has meal rules.

    optimize: build into a z3.Optimize instead of a z3.Solver, so the caller
    can add objectives (tier 1). Same constraints and tracking either way.
    """
    counts = counts or {}
    solver = z3.Optimize() if optimize else z3.Solver()
    solver.set("timeout", timeout_ms)
    enc = Encoding(regimen=regimen, solver=solver, slots={},
                   meals={k: _t(v) for k, v in (meals or {}).items()})

    for p in regimen.products:
        t = [z3.Real(f"t::{p.name}::{k}") for k in range(p.max_doses)]
        on = [z3.Bool(f"on::{p.name}::{k}") for k in range(p.max_doses)]
        s = Slots(t, on)
        enc.slots[p.name] = s
        for k in range(p.max_doses):
            solver.add(z3.Implies(on[k], z3.And(t[k] >= 0, t[k] < DAY)))
            solver.add(z3.Implies(z3.Not(on[k]), t[k] == 0))
        for k in range(p.max_doses - 1):
            solver.add(z3.Implies(on[k + 1], on[k]))  # activation monotonicity
            solver.add(z3.Implies(z3.And(on[k], on[k + 1]), t[k] < t[k + 1]))  # ordering

        if p.prescribed:
            if p.name in counts and counts[p.name] != p.max_doses:
                raise ValueError(f"{p.name} is prescribed; its count is fixed at {p.max_doses}")
            enc.track(f"count::{p.name}", s.count() == p.max_doses)
        else:
            enc.track(f"max::{p.name}", s.count() <= p.max_doses)
            if p.name in counts:
                enc.track(f"reported::{p.name}", s.count() == counts[p.name])

    for rule in regimen.rules:
        enc.track(rule.id, _encode_rule(enc, rule))
    return enc


def _meal_terms(enc: Encoding, rule) -> list[z3.ArithRef]:
    if rule.meal is None:
        return list(enc.meals.values())
    if rule.meal not in enc.meals:
        raise ValueError(f"{rule.id}: no time given for meal {rule.meal!r}")
    return [enc.meals[rule.meal]]


def _all(parts) -> z3.BoolRef:
    return z3.And(True, *parts)


def _encode_rule(enc: Encoding, rule) -> z3.BoolRef:
    if isinstance(rule, MinInterval):
        doses = enc.slots[rule.product].doses()
        return _all(
            z3.Implies(z3.And(oi, oj), circular_distance_at_least(ti, tj, rule.hours))
            for i, (ti, oi) in enumerate(doses)
            for tj, oj in doses[i + 1:]
        )
    if isinstance(rule, IngredientInterval):
        doses = [
            d for p in enc.regimen.products if rule.ingredient in p.ingredients
            for d in enc.slots[p.name].doses()
        ]
        return _all(
            z3.Implies(z3.And(oi, oj), circular_distance_at_least(ti, tj, rule.hours))
            for i, (ti, oi) in enumerate(doses)
            for tj, oj in doses[i + 1:]
        )
    if isinstance(rule, Separation):
        sa, sb = enc.slots[rule.a], enc.slots[rule.b]
        if not sa.on or not sb.on:
            return z3.BoolVal(True)
        if Fraction(rule.before) + Fraction(rule.after) > DAY:
            # No placement satisfies both gaps: the two products cannot share a day.
            return z3.Not(z3.And(z3.Or(*sa.on), z3.Or(*sb.on)))
        return _all(
            z3.Implies(z3.And(oa, ob), outside_window_around(ta, tb, rule.before, rule.after))
            for ta, oa in sa.doses()
            for tb, ob in sb.doses()
        )
    if isinstance(rule, IngredientCeiling):
        terms = []
        for p in enc.regimen.products:
            amt = p.ingredients.get(rule.ingredient)
            if amt is None:
                continue
            terms += [z3.If(o, _q(amt), _q(0)) for o in enc.slots[p.name].on]
        if not terms:
            return z3.BoolVal(True)
        return z3.Sum(terms) <= _q(rule.max_amount)
    if isinstance(rule, Window):
        return _all(z3.Implies(o, in_spans(t, rule.spans)) for t, o in enc.slots[rule.product].doses())
    if isinstance(rule, MealGap):
        meals = _meal_terms(enc, rule)
        return _all(
            z3.Implies(o, outside_window_around(t, m, rule.before, rule.after))
            for t, o in enc.slots[rule.product].doses()
            for m in meals
        )
    if isinstance(rule, MealWindow):
        (m,) = _meal_terms(enc, rule)
        return _all(
            z3.Implies(o, forward_gap_within(t, m, rule.latest, rule.earliest))
            for t, o in enc.slots[rule.product].doses()
        )
    raise TypeError(f"unknown rule {rule!r}")


@dataclass
class Result:
    status: str  # "sat" | "unsat" | "unknown"
    schedule: dict[str, list[Fraction]] = field(default_factory=dict)
    meals: dict[str, Fraction] = field(default_factory=dict)
    core: list[str] = field(default_factory=list)
    reason: str = ""
    model: object = None


def to_fraction(v: z3.ExprRef) -> Fraction:
    if z3.is_rational_value(v):
        return v.as_fraction()
    if z3.is_int_value(v):
        return Fraction(v.as_long())
    raise ValueError(f"non-rational model value {v}")


def read_model(enc: Encoding, m: z3.ModelRef) -> Result:
    schedule = {}
    for name, s in enc.slots.items():
        schedule[name] = [
            to_fraction(m.eval(t, model_completion=True))
            for t, o in s.doses()
            if z3.is_true(m.eval(o, model_completion=True))
        ]
    meals = {k: to_fraction(m.eval(v, model_completion=True)) for k, v in enc.meals.items()}
    return Result("sat", schedule=schedule, meals=meals, model=m)


def check(enc: Encoding, only: Optional[Iterable[str]] = None) -> Result:
    """Run the solver with every tracked rule enforced (or just `only`), minus
    `enc.disabled`. Never raises on timeout: returns status 'unknown' with a
    reason, which callers must surface rather than treat as sat or unsat."""
    r = enc.solver.check(*enc.assumptions(only))
    if r == z3.sat:
        return read_model(enc, enc.solver.model())
    if r == z3.unsat:
        return Result("unsat", core=sorted(enc.id_of(c) for c in enc.solver.unsat_core()))
    return Result("unknown", reason=enc.solver.reason_unknown() or "unknown")


def solve(regimen: Regimen, **kw) -> Result:
    return check(build(regimen, **kw))
