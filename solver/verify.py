"""Independent checker for a concrete schedule.

Deliberately shares no code with the Z3 encoding: it re-derives every rule in
plain Python with exact fractions. Every test that gets a `sat` result runs the
schedule through here, so an encoding bug that produces a false `sat` (a card
that looks valid but breaks a label) fails loudly.
"""

from __future__ import annotations

from fractions import Fraction

from .model import DAY, IngredientCeiling, IngredientInterval, MinInterval, Regimen, Separation, Window


def cdist(x: Fraction, y: Fraction) -> Fraction:
    d = abs(Fraction(x) - Fraction(y))
    return min(d, DAY - d)


def forward(a: Fraction, b: Fraction) -> Fraction:
    """Hours from a forward to b, in [0, 24)."""
    return (Fraction(b) - Fraction(a)) % DAY


def _in_span(t: Fraction, start, end) -> bool:
    start, end = Fraction(start), Fraction(end)
    if start <= end:
        return start <= t <= end
    return t >= start or t <= end


def violations(regimen: Regimen, schedule: dict[str, list[Fraction]]) -> list[str]:
    out: list[str] = []
    for p in regimen.products:
        times = schedule.get(p.name, [])
        if any(not (0 <= t < DAY) for t in times):
            out.append(f"{p.name}: time outside [0, 24)")
        if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
            out.append(f"{p.name}: doses not strictly ordered")
        if p.prescribed and len(times) != p.max_doses:
            out.append(f"{p.name}: prescribed {p.max_doses}, scheduled {len(times)}")
        if not p.prescribed and len(times) > p.max_doses:
            out.append(f"{p.name}: {len(times)} doses exceeds max {p.max_doses}")

    for r in regimen.rules:
        if isinstance(r, MinInterval):
            ts = schedule.get(r.product, [])
            for i in range(len(ts)):
                for j in range(i + 1, len(ts)):
                    if cdist(ts[i], ts[j]) < r.hours:
                        out.append(f"{r.id}: {r.product} doses {i},{j} only {cdist(ts[i], ts[j])}h apart")
        elif isinstance(r, IngredientInterval):
            doses = [
                (p.name, t)
                for p in regimen.products
                if r.ingredient in p.ingredients
                for t in schedule.get(p.name, [])
            ]
            for i in range(len(doses)):
                for j in range(i + 1, len(doses)):
                    (pi, ti), (pj, tj) = doses[i], doses[j]
                    if cdist(ti, tj) < r.hours:
                        out.append(f"{r.id}: {pi}@{ti} and {pj}@{tj} share {r.ingredient}, {cdist(ti, tj)}h apart")
        elif isinstance(r, Separation):
            for ta in schedule.get(r.a, []):
                for tb in schedule.get(r.b, []):
                    f = forward(ta, tb)  # a -> b
                    if f < r.before or (DAY - f if f else 0) < r.after:
                        out.append(f"{r.id}: {r.a}@{ta} vs {r.b}@{tb}")
        elif isinstance(r, IngredientCeiling):
            total = sum(
                (Fraction(p.ingredients[r.ingredient]) * len(schedule.get(p.name, []))
                 for p in regimen.products if r.ingredient in p.ingredients),
                Fraction(0),
            )
            if total > r.max_amount:
                out.append(f"{r.id}: {r.ingredient} total {total} > {r.max_amount}")
        elif isinstance(r, Window):
            for t in schedule.get(r.product, []):
                if not any(_in_span(t, s, e) for s, e in r.spans):
                    out.append(f"{r.id}: {r.product}@{t} outside window")
        else:
            out.append(f"unchecked rule type {type(r).__name__}")
    return out
