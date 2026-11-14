"""Load the library and turn a patient's product list into a solver Regimen.

Resolution policy (agreed 13 Nov):
  - Rules live under ingredients; a rule's "key" is (ingredient, family, target).
  - Several SPLs may state the same key. The primary SPL is preferred, but a
    fallback is used when the primary is silent, and tagged as such.
  - When sources disagree on a number, the disagreement is recorded, and:
        mode "strict" (tiers 0/1): take the strictest value across sources
        mode "loose"  (tier 2):    take the loosest value across sources,
                                   including each interpreted rule's `loosest`
  - Separation targets are expanded through groups to concrete products;
    target "*" means every other product in the regimen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal, Optional

from solver.model import (
    DAY,
    IngredientCeiling,
    IngredientInterval,
    MealGap,
    MealWindow,
    MinInterval,
    Product,
    Regimen,
    Separation,
)

from .moieties import element_of, is_element
from .schema import (
    DailyMaxRule,
    FoodRule,
    GroupsFile,
    IngredientFile,
    MinIntervalRule,
    ProductFile,
    SeparationRule,
)

Mode = Literal["strict", "loose"]


@dataclass
class Library:
    ingredients: dict[str, IngredientFile]
    products: dict[str, ProductFile]
    groups: GroupsFile

    def group_members(self, name: str) -> set[str]:
        """Ingredient names in a group. An element member ("calcium") covers every
        ingredient in the library that is a salt of it (calcium_carbonate, ...)."""
        out = set()
        known = set(self.ingredients) | {i for p in self.products.values() for i in p.ingredients}
        for m in self.groups.groups[name].members:
            out.add(m.ingredient)
            if is_element(m.ingredient):
                out |= {i for i in known if element_of(i) == m.ingredient}
        return out

    def target_ingredients(self, target: str) -> set[str]:
        if target.startswith("group:"):
            return self.group_members(target.removeprefix("group:"))
        return {target}


def _read(path: Path, model):
    with path.open(encoding="utf-8") as f:
        return model.model_validate(json.load(f))


def _rejected(item) -> bool:
    return bool(item.review) and item.review.get("decision") == "rejected"


def load(root: Path) -> Library:
    root = Path(root)
    ingredients = {}
    for p in sorted((root / "ingredients").glob("*.json")):
        f = _read(p, IngredientFile)
        if f.ingredient in ingredients:
            raise ValueError(f"two files for ingredient {f.ingredient}")
        # A rule a reviewer rejected stays in the file (audit trail) but is never applied.
        f = f.model_copy(update={"rules": [r for r in f.rules if not _rejected(r)]})
        ingredients[f.ingredient] = f
    products = {}
    for p in sorted((root / "products").glob("*.json")):
        f = _read(p, ProductFile)
        products[f.name] = f
    groups_path = root / "groups.json"
    groups = _read(groups_path, GroupsFile) if groups_path.exists() else GroupsFile(groups={})
    groups = groups.model_copy(update={"groups": {
        name: g.model_copy(update={"members": [m for m in g.members if not _rejected(m)]})
        for name, g in groups.groups.items()}})
    lib = Library(ingredients, products, groups)
    _cross_check(lib)
    return lib


def _cross_check(lib: Library) -> None:
    for ing in lib.ingredients.values():
        for r in ing.rules:
            if isinstance(r, SeparationRule) and r.target.startswith("group:"):
                if r.target.removeprefix("group:") not in lib.groups.groups:
                    raise ValueError(f"{r.id}: unknown group {r.target}")


# --- resolution ------------------------------------------------------------------


@dataclass
class Resolved:
    """One rule key after choosing values across sources."""

    key: tuple[str, str, str]
    rule: object  # the chosen schema rule (for its id, provenance, source)
    values: dict[str, Fraction]
    used_fallback: bool
    disagreement: Optional[dict[str, dict[str, Fraction]]] = None  # spl -> params
    all_rules: list = field(default_factory=list)

    @property
    def verified(self) -> bool:
        return all(r.verified for r in self.all_rules)


def _key(ingredient: str, r) -> tuple[str, str, str]:
    target = getattr(r, "target", "") or getattr(r, "mode", "")
    if isinstance(r, FoodRule) and r.meal:
        target = f"{target}:{r.meal}"
    return (ingredient, r.family, target)


def resolve(ing: IngredientFile, mode: Mode) -> list[Resolved]:
    by_key: dict[tuple, list] = {}
    for r in ing.rules:
        by_key.setdefault(_key(ing.ingredient, r), []).append(r)

    out = []
    for key, rules in by_key.items():
        primary = [r for r in rules if r.source.spl_set_id == ing.primary_spl]
        chosen = primary[0] if primary else sorted(
            rules, key=lambda r: ing.fallback_spls.index(r.source.spl_set_id)
        )[0]

        per_source = {r.source.spl_set_id: r.numeric_params() for r in rules}
        disagreement = per_source if len({tuple(sorted(v.items())) for v in per_source.values()}) > 1 else None

        values = {}
        for param, direction in chosen.STRICTER.items():
            candidates = []
            for r in rules:
                candidates.append(r.numeric_params()[param])
                if mode == "loose" and r.loosest and param in r.loosest:
                    candidates.append(r.loosest[param])
            stricter_is_max = direction > 0
            if mode == "strict":
                values[param] = max(candidates) if stricter_is_max else min(candidates)
            else:
                values[param] = min(candidates) if stricter_is_max else max(candidates)
        out.append(Resolved(key, chosen, values, used_fallback=not primary,
                            disagreement=disagreement, all_rules=rules))
    return out


# --- regimen building --------------------------------------------------------------


@dataclass(frozen=True)
class Item:
    """One line of the patient's regimen.

    prescribed_count: the prescription's doses/day (quoted from the Rx).
    Leave it None for an as-needed product: the solver chooses the count, up to
    the label's `otc_max_doses`.
    """

    product: str
    prescribed_count: Optional[int] = None
    every_hours: Optional[Fraction] = None  # from the prescription's directions, e.g. "every 12 hours"
    units_per_dose: Optional[Fraction] = None  # from the prescription, e.g. "take 2 tablets"; None = the product's usual


RX_PREFIX = "rx::"  # solver ids for prescription facts (not label rules)


@dataclass
class Built:
    regimen: Regimen
    # solver rule id -> (library rule id, ingredient, resolution details)
    origin: dict[str, tuple[str, str, Resolved]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)  # human-readable reasons
    unverified: list[str] = field(default_factory=list)  # library items not yet human-checked
    rejected: list[str] = field(default_factory=list)  # items a reviewer rejected that this result still depends on
    rx: dict[str, str] = field(default_factory=dict)  # solver id -> description, for prescription facts


def build_regimen(lib: Library, items: list[Item], mode: Mode, *, include_food: bool = True,
                  bed: Optional[Fraction] = None, wake: Optional[Fraction] = None,
                  meals: Optional[dict] = None, shift_margin: Fraction = Fraction(0)) -> Built:
    """include_food=False drops meal rules entirely (tier 2: the routine,
    meals included, is not part of the claim). `bed` and `meals` (the reported
    routine) anchor "after waking, before the first meal" rules; `shift_margin` is how
    far a tier-1 search may move bedtime and meals, so the window stays clear of any bedtime
    it could reach."""
    products = []
    built = Built(Regimen((), ()))
    for it in items:
        pf = lib.products[it.product]
        if _rejected(pf):
            built.rejected.append(f"product {pf.name} (rejected by {pf.review.get('by', '?')})")
        elif not pf.verified:
            built.unverified.append(f"product {pf.name}")
        units = Fraction(it.units_per_dose) if it.units_per_dose is not None else Fraction(pf.units_per_dose)
        scale = units / Fraction(pf.units_per_dose)  # e.g. 2 tablets of a product filed per 1 tablet
        amounts = {k: v * scale for k, v in _amounts(pf).items()}
        if pf.label_max_units_per_dose is not None and units > pf.label_max_units_per_dose:
            built.skipped.append(f"{pf.display_name}: {_n(units)} {pf.unit_name or 'unit'}(s) per dose is above the label's "
                                 f"{_n(pf.label_max_units_per_dose)} for this strength")
        if it.prescribed_count is not None:
            products.append(Product(pf.name, it.prescribed_count, True, amounts))
        elif pf.otc_max_doses is None:
            raise ValueError(f"{pf.name}: as-needed use needs `otc_max_doses` on the product")
        else:
            if _rejected(pf.otc_max_doses):
                built.rejected.append(f"max doses of {pf.name} (rejected by {pf.otc_max_doses.review.get('by', '?')})")
            elif not pf.otc_max_doses.verified:
                built.unverified.append(f"max doses of {pf.name}")
            # The label maximum is in doses of the product's usual size: convert through units.
            max_units = pf.otc_max_doses.count * Fraction(pf.units_per_dose)
            products.append(Product(pf.name, int(max_units // units), False, amounts))

    rx_rules = []
    for it in items:
        if it.every_hours is not None:
            # The prescription chose this frequency; it is a fact of the Rx, not a label rule.
            sid = f"{RX_PREFIX}every::{it.product}"
            rx_rules.append(MinInterval(sid, it.product, it.every_hours))
            built.rx[sid] = f"every {_hours(it.every_hours)} (prescription directions)"

    def ingredients_of(product: str) -> set[str]:
        return set(lib.products[product].ingredients)

    def containing(ingredients: set[str]) -> list[str]:
        return [it.product for it in items if ingredients & ingredients_of(it.product)]

    present = {i for it in items for i in ingredients_of(it.product)}
    rules = []

    def add(sid, rule, lib_rule, ing_name, res):
        rules.append(rule)
        built.origin[sid] = (lib_rule.id, ing_name, res)

    for ing_name in sorted(present):
        ing = lib.ingredients.get(ing_name)
        if ing is None:
            built.skipped.append(f"no library entry for ingredient {ing_name!r}: its constraints are unknown")
            continue
        for res in resolve(ing, mode):
            r, v = res.rule, res.values
            if isinstance(r, FoodRule) and not include_food:
                continue
            if not res.verified:
                built.unverified.append(f"rule {r.id}")
            if isinstance(r, MinIntervalRule):
                # Across every product containing the ingredient, not per product.
                sid = f"{r.id}@{ing_name}"
                add(sid, IngredientInterval(sid, ing_name, v["hours"]), r, ing_name, res)
            elif isinstance(r, SeparationRule):
                for a in containing({ing_name}):
                    if r.target == "*":
                        others = [it.product for it in items if it.product != a]
                    else:
                        others = containing(lib.target_ingredients(r.target))
                    for b in others:
                        if a == b:
                            built.skipped.append(f"{r.id}: {a} contains both sides; not separable")
                            continue
                        sid = f"{r.id}@{a}~{b}"
                        after = v["after"]
                        if r.first_of_day:  # nothing of b from bedtime until a (reported sleep)
                            if bed is None or wake is None:
                                built.skipped.append(f"{r.id}: needs the patient's sleep times")
                                continue
                            after = max(after, (Fraction(wake) - Fraction(bed)) % DAY)
                        add(sid, Separation(sid, a, b, v["before"], after, first_of_day=r.first_of_day),
                            r, ing_name, res)
            elif isinstance(r, DailyMaxRule):
                units = {lib.products[p].ingredients[ing_name].unit for p in containing({ing_name})}
                if units != {r.unit}:
                    raise ValueError(f"{r.id}: unit {r.unit} vs product units {units}")
                sid = f"{r.id}@{ing_name}"
                add(sid, IngredientCeiling(sid, ing_name, v["amount"]), r, ing_name, res)
            elif isinstance(r, FoodRule):
                for p in containing({ing_name}):
                    sid = f"{r.id}@{p}"
                    if r.mode == "empty_stomach":
                        add(sid, MealGap(sid, p, v["before_meal"], v["after_meal"], r.meal), r, ing_name, res)
                    elif r.mode == "before_meal_window":
                        add(sid, MealWindow(sid, p, r.meal, v["earliest"], v["latest"]), r, ing_name, res)
                    elif r.mode == "before_first_meal":
                        # Window opens at bedtime; doses are never placed during sleep, so in effect it
                        # opens at waking. Anchored to the reported routine: if a tier-1 change moves
                        # bedtime later than breakfast moves, the window can open a little before bed.
                        if bed is None or not meals or r.meal not in meals:
                            built.skipped.append(f"{r.id}: needs the patient's bedtime and {r.meal} time")
                            continue
                        # Opens one minute after bedtime (plus any tier-1 margin, never past waking), so
                        # a dose cannot sit at bedtime ("should not be taken at bedtime or before arising").
                        sleep_len = (Fraction(wake) - Fraction(bed)) % DAY if wake is not None else Fraction(0)
                        offset = min(Fraction(shift_margin), sleep_len) + Fraction(1, 60)
                        span = max((Fraction(meals[r.meal]) - Fraction(bed)) % DAY - offset, v["latest"])
                        add(sid, MealWindow(sid, p, r.meal, span, v["latest"], first_of_day=True), r, ing_name, res)
                    else:
                        built.skipped.append(f"{r.id}: food mode {r.mode!r} not encoded yet")
    built.regimen = Regimen(tuple(products), tuple(rx_rules + rules))
    return built


def _hours(h: Fraction) -> str:
    h = Fraction(h)
    return f"{h.numerator} h" if h.denominator == 1 else f"{float(h):g} h"


def _n(v) -> str:
    v = Fraction(v)
    return str(v.numerator) if v.denominator == 1 else f"{float(v):g}"


def _amounts(pf: ProductFile) -> dict[str, Fraction]:
    return {k: a.amount for k, a in pf.ingredients.items()}
