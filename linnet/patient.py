"""Patient input: what is dispensed, what they report taking, and their day.

A patient file is JSON. Times are "HH:MM" strings on a 24-hour clock.
Demo patients are constructed; the file says so (`constructed: true`) and the
renderer repeats it.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from library.load import Item


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def parse_hhmm(s: str) -> Fraction:
    h, m = s.split(":")
    v = Fraction(int(h)) + Fraction(int(m), 60)
    if not (0 <= v < 24):
        raise ValueError(f"time out of range: {s}")
    return v


def fmt_hhmm(t: Fraction) -> str:
    total = round(Fraction(t) * 60) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


class PatientItem(Strict):
    """Exactly one of:
    prescribed_count: doses/day on the prescription (fixed);
    reported_count:   as-needed product, how many doses/day the patient says
                      they take (the solver may suggest fewer, never more).
    """

    product: str
    prescribed_count: Optional[int] = Field(default=None, ge=1)
    reported_count: Optional[int] = Field(default=None, ge=0)
    every_hours: Optional[Fraction] = Field(default=None, gt=0)  # Rx directions, e.g. "every 12 hours"

    @model_validator(mode="after")
    def _one_count(self):
        if (self.prescribed_count is None) == (self.reported_count is None):
            raise ValueError(f"{self.product}: give exactly one of prescribed_count / reported_count")
        if self.every_hours is not None and self.prescribed_count is None:
            raise ValueError(f"{self.product}: every_hours comes from a prescription; needs prescribed_count")
        return self


class Sleep(Strict):
    start: str  # bedtime
    end: str  # wake time


class Routine(Strict):
    meals: dict[str, str]  # name -> "HH:MM"; "breakfast" is used by levothyroxine
    sleep: Sleep

    @field_validator("meals")
    @classmethod
    def _times(cls, v):
        for t in v.values():
            parse_hhmm(t)
        return v


class Weights(Strict):
    """Tier-1 costs. These are judgements no label makes; they are shown on
    the card and should be set by the user. Defaults are placeholders."""

    meal_shift_per_hour: Fraction = Fraction(1)
    sleep_shift_per_hour: Fraction = Fraction(2)
    dose_dropped: Fraction = Fraction(3, 2)
    max_meal_shift_hours: Fraction = Fraction(2)
    max_sleep_shift_hours: Fraction = Fraction(2)
    placeholder: bool = True


class Patient(Strict):
    name: str
    constructed: bool = True
    note: str = ""
    items: list[PatientItem]
    routine: Routine
    weights: Weights = Weights()

    def library_items(self, prescribed_only: bool = False) -> list[Item]:
        out = []
        for it in self.items:
            if it.prescribed_count is not None:
                out.append(Item(it.product, it.prescribed_count, it.every_hours))
            elif not prescribed_only:
                out.append(Item(it.product, None))
        return out

    def reported_counts(self) -> dict[str, int]:
        return {it.product: it.reported_count for it in self.items if it.reported_count is not None}

    def meal_times(self) -> dict[str, Fraction]:
        return {k: parse_hhmm(v) for k, v in self.routine.meals.items()}

    def sleep_window(self) -> tuple[Fraction, Fraction]:
        return parse_hhmm(self.routine.sleep.start), parse_hhmm(self.routine.sleep.end)


def load_patient(path: Path) -> Patient:
    with Path(path).open(encoding="utf-8") as f:
        return Patient.model_validate(json.load(f))
