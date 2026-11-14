"""Z3 encoding of a regimen over one repeating day, in linear real arithmetic.

Shape (see the spec, "The solver core"):
  - each product gets `max_doses` slots; slot k has a real time t[k] in [0, 24)
    and a boolean on[k]
  - activation is monotone (on[k+1] -> on[k]), so the active slots are a prefix
  - active slots are strictly ordered (t[k] < t[k+1]), which is also symmetry
    breaking; inactive slots are pinned to 0 so models are deterministic
  - every distance is circular, because the card repeats daily

Structural constraints above are always asserted. Each label rule, and each
dose-count requirement, is asserted under a tracking literal named after its id
so an unsat core names rules rather than internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Union

import z3

from .model import (
    DAY,
    Hours,
    IngredientCeiling,
    IngredientInterval,
    MinInterval,
    Regimen,
    Separation,
    Window,
)

DEFAULT_TIMEOUT_MS = 10_000


def _q(x: Hours) -> z3.ArithRef:
    x = Fraction(x)
    return z3.RealVal(f"{x.numerator}/{x.denominator}")


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


def circular_distance_at_least(x: z3.ArithRef, y: z3.ArithRef, g: Hours) -> z3.BoolRef:
    """cdist(x, y) = min(|x - y|, 24 - |x - y|) >= g."""
    return forward_gap_within(x, y, g, DAY - Fraction(g))


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
        return z3.Sum([z3.If(o, 1, 0) for o in self.on])


@dataclass
class Encoding:
    regimen: Regimen
    solver: Union[z3.Solver, z3.Optimize]
    slots: dict[str, Slots]
    tracked: dict[str, z3.BoolRef] = field(default_factory=dict)


def _track(enc: Encoding, rule_id: str, formula: z3.BoolRef) -> None:
    if rule_id in enc.tracked:
        raise ValueError(f"duplicate rule id {rule_id!r}")
    lit = z3.Bool(f"track::{rule_id}")
    enc.tracked[rule_id] = lit
    enc.solver.assert_and_track(formula, lit)


def build(
    regimen: Regimen,
    *,
    counts: Optional[dict[str, int]] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    optimize: bool = False,
) -> Encoding:
    """Encode `regimen`.

    counts: optional fixed dose count for as-needed products (e.g. the
    patient's reported usage, for the tier-0 solve). Without it, an as-needed
    count is free in [0, max_doses]. Prescribed products always take every slot.

    optimize: build into a z3.Optimize instead of a z3.Solver, so the caller
    can add objectives (tier 1). Same constraints and tracking either way.
    """
    counts = counts or {}
    solver = z3.Optimize() if optimize else z3.Solver()
    solver.set("timeout", timeout_ms)
    enc = Encoding(regimen=regimen, solver=solver, slots={})

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
            _track(enc, f"count::{p.name}", s.count() == p.max_doses)
        else:
            _track(enc, f"max::{p.name}", s.count() <= p.max_doses)
            if p.name in counts:
                _track(enc, f"reported::{p.name}", s.count() == counts[p.name])

    for rule in regimen.rules:
        _track(enc, rule.id, _encode_rule(enc, rule))
    return enc


def _encode_rule(enc: Encoding, rule) -> z3.BoolRef:
    if isinstance(rule, MinInterval):
        s = enc.slots[rule.product]
        n = len(s.t)
        return z3.And(
            True,
            *[
                z3.Implies(
                    z3.And(s.on[i], s.on[j]),
                    circular_distance_at_least(s.t[i], s.t[j], rule.hours),
                )
                for i in range(n)
                for j in range(i + 1, n)
            ],
        )
    if isinstance(rule, IngredientInterval):
        doses = [
            (t, o)
            for p in enc.regimen.products
            if rule.ingredient in p.ingredients
            for t, o in zip(enc.slots[p.name].t, enc.slots[p.name].on)
        ]
        return z3.And(
            True,
            *[
                z3.Implies(z3.And(oi, oj), circular_distance_at_least(ti, tj, rule.hours))
                for i, (ti, oi) in enumerate(doses)
                for tj, oj in doses[i + 1:]
            ],
        )
    if isinstance(rule, Separation):
        sa, sb = enc.slots[rule.a], enc.slots[rule.b]
        if not sa.on or not sb.on:
            return z3.BoolVal(True)
        if Fraction(rule.before) + Fraction(rule.after) > DAY:
            # No placement satisfies both gaps: the two products cannot share a day.
            return z3.Not(z3.And(z3.Or(*sa.on), z3.Or(*sb.on)))

        def ok(ta, tb):
            f = forward_gap_within(ta, tb, rule.before, DAY - Fraction(rule.after))
            # Simultaneous doses are 0 h after each other; forward_gap_within
            # alone would accept that when before == 0.
            return z3.And(f, ta != tb) if Fraction(rule.after) > 0 else f

        return z3.And(
            True,
            *[
                z3.Implies(z3.And(oa, ob), ok(ta, tb))
                for ta, oa in zip(sa.t, sa.on)
                for tb, ob in zip(sb.t, sb.on)
            ],
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
        s = enc.slots[rule.product]
        return z3.And(True, *[z3.Implies(o, in_spans(t, rule.spans)) for t, o in zip(s.t, s.on)])
    raise TypeError(f"unknown rule {rule!r}")


@dataclass
class Result:
    status: str  # "sat" | "unsat" | "unknown"
    schedule: dict[str, list[Fraction]] = field(default_factory=dict)
    core: list[str] = field(default_factory=list)
    reason: str = ""


def _to_fraction(v: z3.ExprRef) -> Fraction:
    if z3.is_rational_value(v):
        return v.as_fraction()
    if z3.is_algebraic_value(v):  # should not occur in LRA; approximate defensively
        return Fraction(v.approx(20).as_decimal(20).rstrip("?"))
    raise ValueError(f"non-numeric model value {v}")


def check(enc: Encoding) -> Result:
    """Run the solver. Never raises on timeout: returns status 'unknown' with a
    reason, which callers must surface rather than treat as sat or unsat."""
    r = enc.solver.check()
    if r == z3.sat:
        m = enc.solver.model()
        schedule = {}
        for name, s in enc.slots.items():
            times = []
            for t, o in zip(s.t, s.on):
                if z3.is_true(m.eval(o, model_completion=True)):
                    times.append(_to_fraction(m.eval(t, model_completion=True)))
            schedule[name] = times
        return Result("sat", schedule=schedule)
    if r == z3.unsat:
        lit_to_id = {str(lit): rid for rid, lit in enc.tracked.items()}
        core = sorted(lit_to_id[str(c)] for c in enc.solver.unsat_core())
        return Result("unsat", core=core)
    return Result("unknown", reason=enc.solver.reason_unknown() or "unknown")


def solve(regimen: Regimen, **kw) -> Result:
    return check(build(regimen, **kw))
