"""Plain data model for a regimen and its timing rules.

All times are hours on a repeating 24-hour day, in [0, 24). Every rule carries
an `id` so it can be tracked by the solver and named in an unsat core.
Provenance (quoted / interpreted, source sentence, SPL id) is attached later by
the constraint library; the solver only needs the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional, Union

Hours = Union[int, Fraction]

DAY = 24


@dataclass(frozen=True)
class Product:
    """One dispensed product.

    prescribed: every slot is taken, `max_doses` is the prescribed count.
    as-needed:  `max_doses` is the label maximum, the count is a solver choice.
    ingredients: active ingredient -> amount per dose (any consistent unit).
    """

    name: str
    max_doses: int
    prescribed: bool
    ingredients: dict[str, Hours] = field(default_factory=dict)


@dataclass(frozen=True)
class MinInterval:
    """Family 1: any two doses of `product` are at least `hours` apart (circular)."""

    id: str
    product: str
    hours: Hours


@dataclass(frozen=True)
class IngredientInterval:
    """Family 1, across products: any two doses that contain `ingredient`,
    from the same product or different ones, are at least `hours` apart
    (circular). Stops two products sharing an active being stacked together."""

    id: str
    ingredient: str
    hours: Hours


@dataclass(frozen=True)
class Separation:
    """Family 3: each dose of `a` is at least `before` hours before every dose of
    `b`, and at least `after` hours after it. Equivalently, no dose of `a` falls
    in the circular window (b - before, b + after).

    Label form: "take A at least <before> h before or <after> h after B".
    """

    id: str
    a: str
    b: str
    before: Hours
    after: Hours
    first_of_day: bool = False  # `after` = bedtime->waking: nothing of b between bedtime and a


@dataclass(frozen=True)
class IngredientCeiling:
    """Family 4: total daily amount of `ingredient`, across all products."""

    id: str
    ingredient: str
    max_amount: Hours


@dataclass(frozen=True)
class Window:
    """Doses of `product` must fall inside one of `spans`.

    A span (start, end) with start > end wraps midnight: [start, 24) U [0, end].
    Used for sleep/work exclusion now and food anchoring later.
    """

    id: str
    product: str
    spans: tuple[tuple[Hours, Hours], ...]


@dataclass(frozen=True)
class MealGap:
    """Family 2: no dose of `product` from `before` h before to `after` h after
    a meal. Applies to every meal, or only `meal` if named. Meal times are not
    part of the regimen: they are supplied at solve time (fixed or variable)."""

    id: str
    product: str
    before: Hours
    after: Hours
    meal: Optional[str] = None


@dataclass(frozen=True)
class MealWindow:
    """Family 2: each dose of `product` falls between `earliest` and `latest`
    hours before the named `meal` ("one-half to one hour before breakfast")."""

    id: str
    product: str
    meal: str
    earliest: Hours
    latest: Hours
    first_of_day: bool = False  # earliest = bedtime->meal, so with sleep: "after waking, before the meal"


Rule = Union[MinInterval, IngredientInterval, Separation, IngredientCeiling, Window, MealGap, MealWindow]


@dataclass(frozen=True)
class Regimen:
    products: tuple[Product, ...]
    rules: tuple[Rule, ...]

    def product(self, name: str) -> Product:
        for p in self.products:
            if p.name == name:
                return p
        raise KeyError(name)
