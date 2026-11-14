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
    verified: bool = False  # set true only after a human checked it against the label
    review: Optional[dict] = None  # who signed it off, their role, when (library/review_queue.py)

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
    any dose containing the target. Target is an ingredient name,
    "group:<name>", or "*" for every other product in the regimen (labels
    that say "other drugs")."""

    family: Literal["separation"]
    target: str
    before: Number
    after: Number
    # "first medication of the day": no target dose between bedtime and this one. Applied with
    # the patient's reported sleep (after = at least the time from bedtime to waking).
    first_of_day: bool = False
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
    """Dose timing relative to the patient's meals.

    empty_stomach:      no dose from `before_meal` h before to `after_meal` h
                        after a meal (every meal, or only `meal` if named)
    before_meal_window: dose falls between `earliest` and `latest` h before the
                        named `meal` (e.g. "one-half to one hour before breakfast")
    before_first_meal:  dose after waking and at least `latest` h before the named
                        `meal`, the first of the day (e.g. "upon arising ... at least
                        one-half hour before the first food ... of the day")
    with_food:          validated but not yet encoded by the solver
    """

    family: Literal["food"]
    mode: Literal["empty_stomach", "before_meal_window", "before_first_meal", "with_food"]
    meal: Optional[str] = None
    before_meal: Number = Fraction(0)
    after_meal: Number = Fraction(0)
    earliest: Number = Fraction(0)
    latest: Number = Fraction(0)
    within: Number = Fraction(0)
    STRICTER: ClassVar[dict[str, int]] = {
        "before_meal": +1, "after_meal": +1, "earliest": -1, "latest": +1, "within": -1,
    }

    @model_validator(mode="after")
    def _window_needs_meal(self):
        if self.mode == "before_meal_window" and (not self.meal or self.earliest < self.latest):
            raise ValueError(f"{self.id}: before_meal_window needs `meal` and earliest >= latest")
        if self.mode == "before_first_meal" and (not self.meal or self.earliest):
            raise ValueError(f"{self.id}: before_first_meal needs `meal` and no `earliest` (waking is the start)")
        return self

    def numeric_params(self):
        return {"before_meal": self.before_meal, "after_meal": self.after_meal,
                "earliest": self.earliest, "latest": self.latest, "within": self.within}


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
    note: str = ""
    verified: bool = False
    review: Optional[dict] = None


class ProductFile(Strict):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    display_name: str
    spl_set_id: str
    url: str
    ingredients: dict[str, Amount]  # per dose of `units_per_dose` units (see `dose_note`)
    unit_name: str = ""  # "tablet", "caplet", "capsule"
    units_per_dose: Number = Fraction(1)  # how many units the `ingredients` amounts are for
    label_max_units_per_dose: Optional[Number] = None  # the label's per-dose maximum for THIS strength
    otc_max_doses: Optional[MaxDoses] = None  # label maximum, for as-needed use (doses of `units_per_dose`)
    ndc: list[str] = []
    dose_note: str = ""  # what one dose is, e.g. "2 caplets", and where that comes from
    note: str = ""
    verified: bool = False
    review: Optional[dict] = None

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
    source_spl: str = ""
    verified: bool = False
    review: Optional[dict] = None
    matched: str = ""  # set when the label names only the element (see library/moieties.py)
    note: str = ""  # reviewer remarks, e.g. why it is interpreted


class Group(Strict):
    members: list[Membership]


class GroupsFile(Strict):
    groups: dict[str, Group]
