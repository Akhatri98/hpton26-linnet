"""Load the library and turn a patient's product list into a solver Regimen.

Resolution policy (agreed 13 Nov):
  - Rules live under ingredients; a rule's "key" is (ingredient, family, target).
  - Several SPLs may state the same key. The primary SPL is preferred, but a
    fallback is used when the primary is silent, and tagged as such.
  - When sources disagree on a number, the disagreement is recorded, and:
        mode "strict" (tiers 0/1): take the strictest value across sources
        mode "loose"  (tier 2):    take the loosest value across sources,
                                   including each interpreted rule's `loosest`
  - Separation targets are expanded through groups to concrete products.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Literal, Optional

from solver.model import IngredientCeiling, IngredientInterval, Product, Regimen, Separation

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
        return {m.ingredient for m in self.groups.groups[name].members}

    def target_ingredients(self, target: str) -> set[str]:
        if target.startswith("group:"):
            return self.group_members(target.removeprefix("group:"))
        return {target}


def _read(path: Path, model):
    with path.open(encoding="utf-8") as f:
        return model.model_validate(json.load(f))


def load(root: Path) -> Library:
    root = Path(root)
    ingredients = {}
    for p in sorted((root / "ingredients").glob("*.json")):
        f = _read(p, IngredientFile)
        if f.ingredient in ingredients:
            raise ValueError(f"two files for ingredient {f.ingredient}")
        ingredients[f.ingredient] = f
    products = {}
    for p in sorted((root / "products").glob("*.json")):
        f = _read(p, ProductFile)
        products[f.name] = f
    groups_path = root / "groups.json"
    groups = _read(groups_path, GroupsFile) if groups_path.exists() else GroupsFile(groups={})
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


def _key(ingredient: str, r) -> tuple[str, str, str]:
    target = getattr(r, "target", "") or getattr(r, "mode", "")
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
        out.append(Resolved(key, chosen, values, used_fallback=not primary, disagreement=disagreement))
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


@dataclass
class Built:
    regimen: Regimen
    # solver rule id -> (library rule id, ingredient, resolution details)
    origin: dict[str, tuple[str, str, Resolved]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)  # human-readable reasons


def build_regimen(lib: Library, items: list[Item], mode: Mode) -> Built:
    products = []
    for it in items:
        pf = lib.products[it.product]
        if it.prescribed_count is not None:
            products.append(Product(pf.name, it.prescribed_count, True, _amounts(pf)))
        elif pf.otc_max_doses is None:
            raise ValueError(f"{pf.name}: as-needed use needs `otc_max_doses` on the product")
        else:
            products.append(Product(pf.name, pf.otc_max_doses.count, False, _amounts(pf)))

    def containing(ingredients: set[str]) -> list[str]:
        return [it.product for it in items if ingredients & set(lib.products[it.product].ingredients)]

    present = {i for it in items for i in lib.products[it.product].ingredients}
    built = Built(Regimen(tuple(products), ()))
    rules = []

    for ing_name in sorted(present):
        ing = lib.ingredients.get(ing_name)
        if ing is None:
            built.skipped.append(f"no library entry for ingredient {ing_name!r}: its constraints are unknown")
            continue
        for res in resolve(ing, mode):
            r, v = res.rule, res.values
            if isinstance(r, MinIntervalRule):
                # Across every product containing the ingredient, not per product.
                sid = f"{r.id}@{ing_name}"
                rules.append(IngredientInterval(sid, ing_name, v["hours"]))
                built.origin[sid] = (r.id, ing_name, res)
            elif isinstance(r, SeparationRule):
                targets = lib.target_ingredients(r.target)
                for a in containing({ing_name}):
                    for b in containing(targets):
                        if a == b:
                            built.skipped.append(f"{r.id}: {a} contains both sides; not separable")
                            continue
                        sid = f"{r.id}@{a}~{b}"
                        rules.append(Separation(sid, a, b, v["before"], v["after"]))
                        built.origin[sid] = (r.id, ing_name, res)
            elif isinstance(r, DailyMaxRule):
                units = {lib.products[p].ingredients[ing_name].unit for p in containing({ing_name})}
                if units != {r.unit}:
                    raise ValueError(f"{r.id}: unit {r.unit} vs product units {units}")
                sid = f"{r.id}@{ing_name}"
                rules.append(IngredientCeiling(sid, ing_name, v["amount"]))
                built.origin[sid] = (r.id, ing_name, res)
            elif isinstance(r, FoodRule):
                built.skipped.append(f"{r.id}: food rules are encoded in M5")
    built.regimen = Regimen(tuple(products), tuple(rules))
    return built


def _amounts(pf: ProductFile) -> dict[str, Fraction]:
    return {k: a.amount for k, a in pf.ingredients.items()}
