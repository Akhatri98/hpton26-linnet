"""Independent checker for a concrete schedule.

Deliberately shares no code with the Z3 encoding: it re-derives every rule in
plain Python with exact fractions. Every test that gets a `sat` result runs the
schedule through here, so an encoding bug that produces a false `sat` (a card
that looks valid but breaks a label) fails loudly.
"""

from __future__ import annotations

from fractions import Fraction

from .model import (DAY, IngredientCeiling, IngredientInterval, MealGap, MealWindow, MinInterval,
                    Regimen, Separation, Window)


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


def _outside(t, anchor, before, after) -> bool:
    """t is >= before h before anchor and >= after h after it (circular)."""
    f = forward(t, anchor)  # t -> anchor
    back = DAY - f if f else Fraction(0)  # anchor -> t
    return f >= before and back >= after


def violations(regimen: Regimen, schedule: dict[str, list[Fraction]],
               meals: dict[str, Fraction] | None = None, partial: bool = False) -> list[str]:
    """partial=True skips dose-count checks, for checking a schedule that is
    still being built (used by the greedy baseline)."""
    meals = meals or {}
    out: list[str] = []
    for p in regimen.products:
        times = schedule.get(p.name, [])
        if any(not (0 <= t < DAY) for t in times):
            out.append(f"{p.name}: time outside [0, 24)")
        if any(times[i] >= times[i + 1] for i in range(len(times) - 1)):
            out.append(f"{p.name}: doses not strictly ordered")
        if partial:
            continue
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
                    if not _outside(ta, tb, r.before, r.after):
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
        elif isinstance(r, MealGap):
            names = [r.meal] if r.meal else list(meals)
            for name in names:
                if name not in meals:
                    out.append(f"{r.id}: no time for meal {name}")
                    continue
                for t in schedule.get(r.product, []):
                    if not _outside(t, meals[name], r.before, r.after):
                        out.append(f"{r.id}: {r.product}@{t} too close to {name}@{meals[name]}")
        elif isinstance(r, MealWindow):
            if r.meal not in meals:
                out.append(f"{r.id}: no time for meal {r.meal}")
                continue
            for t in schedule.get(r.product, []):
                if not (r.latest <= forward(t, meals[r.meal]) <= r.earliest):
                    out.append(f"{r.id}: {r.product}@{t} not {r.latest}-{r.earliest} h before {r.meal}")
        else:
            out.append(f"unchecked rule type {type(r).__name__}")
    return out


def binding(regimen: Regimen, schedule: dict[str, list[Fraction]], meals: dict[str, Fraction] | None = None,
            wake: Fraction | None = None, bed: Fraction | None = None) -> dict[tuple[str, int], list[str]]:
    """For each dose (product, k): the rules it sits exactly at the limit of, i.e. what keeps it
    from moving (exact fractions, so "exactly" is equality). Also "wake" and "bedtime" when a dose
    is at the edge of the waking day. Zero-length limits are skipped: they pin nothing."""
    meals = meals or {}
    out: dict[tuple[str, int], list[str]] = {}

    def pin(product: str, k: int, rid: str) -> None:
        ids = out.setdefault((product, k), [])
        if rid not in ids:
            ids.append(rid)

    doses = {p.name: list(enumerate(schedule.get(p.name, []))) for p in regimen.products}
    for r in regimen.rules:
        if isinstance(r, MinInterval) and r.hours:
            ts = doses.get(r.product, [])
            for i, ti in ts:
                for j, tj in ts:
                    if i < j and cdist(ti, tj) == r.hours:
                        pin(r.product, i, r.id), pin(r.product, j, r.id)
        elif isinstance(r, IngredientInterval) and r.hours:
            ds = [(p.name, k, t) for p in regimen.products if r.ingredient in p.ingredients for k, t in doses[p.name]]
            for x in range(len(ds)):
                for y in range(x + 1, len(ds)):
                    if cdist(ds[x][2], ds[y][2]) == r.hours:
                        pin(ds[x][0], ds[x][1], r.id), pin(ds[y][0], ds[y][1], r.id)
        elif isinstance(r, Separation):
            for ka, ta in doses.get(r.a, []):
                for kb, tb in doses.get(r.b, []):
                    if (r.before and forward(ta, tb) == r.before) or (r.after and forward(tb, ta) == r.after):
                        pin(r.a, ka, r.id), pin(r.b, kb, r.id)
        elif isinstance(r, MealWindow) and r.meal in meals:
            for k, t in doses.get(r.product, []):
                f = forward(t, meals[r.meal])
                if f == r.latest or (f == r.earliest and not r.first_of_day):
                    pin(r.product, k, r.id)
        elif isinstance(r, MealGap):
            for name in ([r.meal] if r.meal else list(meals)):
                for k, t in doses.get(r.product, []) if name in meals else []:
                    if (r.before and forward(t, meals[name]) == r.before) or (r.after and forward(meals[name], t) == r.after):
                        pin(r.product, k, r.id)
    for p, ts in doses.items():
        for k, t in ts:
            if wake is not None and t == Fraction(wake):
                pin(p, k, "wake")
            if bed is not None and t == Fraction(bed):
                pin(p, k, "bedtime")
    return out
