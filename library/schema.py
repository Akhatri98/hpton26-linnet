"""JSON schema for the constraint library, as pydantic models.

Three kinds of file under data/:

  ingredients/<name>.json   label rules for one chemical, each citing its SPL
  products/<name>.json      one dispensed product: SPL, ingredients per dose,
                            label max doses for as-needed products
  groups.json               named sets of ingredients that a rule can target
                            ("polyvalent_cations", "all_oral", ...)

Every number traces to a verbatim sentence on a named SPL. A rule is either
"quoted" (the number is stated) or "interpreted" (a judgement was needed); an
interpreted rule must say what its loosest defensible reading is.

Validation is strict: unknown fields are errors, so a typo cannot silently
drop a constraint.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Annotated, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Source(Strict):
    """Where a number came from. `spl_set_id` names one document; "the
    levothyroxine label" does not, because every manufacturer has its own."""

    spl_set_id: str = Field(min_length=8)
    url: str
    section: str  # e.g. "7.2 Multivalent Cation-Containing Products"
    sentence: str = Field(min_length=10)  # verbatim


Provenance = Literal["quoted", "interpreted"]
Number = Annotated[Fraction, Field(ge=0)]


class _RuleBase(Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    provenance: Provenance
    source: Source
    loosest: Optional[dict[str, Number]] = None  # param -> loosest defensible value
    note: str = ""  # reviewer remarks; JSON has no comments

    @model_validator(mode="after")
    def _loosest_matches_provenance(self):
        if self.provenance == "interpreted" and not self.loosest:
            raise ValueError(f"{self.id}: interpreted rule needs `loosest`")
        if self.provenance == "quoted" and self.loosest:
            raise ValueError(f"{self.id}: quoted rule must not carry `loosest`")
        for k in self.loosest or {}:
            if k not in self.numeric_params():
                raise ValueError(f"{self.id}: `loosest` names unknown param {k!r}")
        return self

    def numeric_params(self) -> dict[str, Fraction]:
        raise NotImplementedError

    # Which direction is "stricter" for each param: +1 bigger is stricter.
    STRICTER: ClassVar[dict[str, int]] = {}


class MinIntervalRule(_RuleBase):
    family: Literal["min_interval"]
    hours: Number
    STRICTER: ClassVar[dict[str, int]] = {"hours": +1}

    def numeric_params(self):
        return {"hours": self.hours}


class SeparationRule(_RuleBase):
    """This ingredient must be at least `before` h before and `after` h after
    any dose containing the target. Target is an ingredient name or
    "group:<name>"."""

    family: Literal["separation"]
    target: str
    before: Number
    after: Number
    STRICTER: ClassVar[dict[str, int]] = {"before": +1, "after": +1}

    def numeric_params(self):
        return {"before": self.before, "after": self.after}


class DailyMaxRule(_RuleBase):
    family: Literal["daily_max"]
    amount: Number
    unit: str
    STRICTER: ClassVar[dict[str, int]] = {"amount": -1}

    def numeric_params(self):
        return {"amount": self.amount}


class FoodRule(_RuleBase):
    """Accepted and validated now so curation can record it; the solver
    encodes it in M5, when meal times become soft variables."""

    family: Literal["food"]
    mode: Literal["empty_stomach", "with_food"]
    before_meal: Number = Fraction(0)  # empty_stomach: hours before a meal
    after_meal: Number = Fraction(0)  # empty_stomach: hours after a meal
    within: Number = Fraction(0)  # with_food: hours from a meal
    STRICTER: ClassVar[dict[str, int]] = {"before_meal": +1, "after_meal": +1, "within": -1}

    def numeric_params(self):
        return {"before_meal": self.before_meal, "after_meal": self.after_meal, "within": self.within}


Rule = Annotated[
    Union[MinIntervalRule, SeparationRule, DailyMaxRule, FoodRule],
    Field(discriminator="family"),
]


class IngredientFile(Strict):
    ingredient: str
    primary_spl: str  # spl_set_id of the primary label
    fallback_spls: list[str] = []  # ordered
    rules: list[Rule]

    @model_validator(mode="after")
    def _sources_are_declared(self):
        declared = {self.primary_spl, *self.fallback_spls}
        for r in self.rules:
            if r.source.spl_set_id not in declared:
                raise ValueError(f"{r.id}: cites undeclared SPL {r.source.spl_set_id}")
        ids = [r.id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.ingredient}: duplicate rule ids")
        return self


class Amount(Strict):
    amount: Number
    unit: str


class MaxDoses(Strict):
    count: int = Field(ge=1)
    provenance: Provenance
    source: Source


class ProductFile(Strict):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str
    spl_set_id: str
    url: str
    ingredients: dict[str, Amount]  # per dose
    otc_max_doses: Optional[MaxDoses] = None  # label maximum, for as-needed use

    @field_validator("ingredients")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("product needs at least one ingredient")
        return v


class Membership(Strict):
    ingredient: str
    provenance: Provenance
    basis: str = Field(min_length=10)  # the label wording that puts it in the set


class Group(Strict):
    members: list[Membership]


class GroupsFile(Strict):
    groups: dict[str, Group]
