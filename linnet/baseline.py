"""A greedy baseline scheduler, for comparison only.

In the *style* of the one-medication-at-a-time heuristics in Tsai & Liu,
"Algorithms for Scheduling Interaction Medications" (Academia Sinica
TR-IIS-08-001, 2008). This is our reimplementation of the idea, not their
exact algorithm:

  - most-constrained first: products with more doses are placed first
  - each dose goes at the earliest 15-minute slot in the waking day that
    breaks no rule against the doses already placed (as soon as possible)
  - no backtracking: if a dose cannot be placed, the heuristic fails

It answers the same question as LINNET's solve A (everything as reported,
strict reading, meals and sleep fixed). When it fails where solve A finds a
schedule, that is a false alarm: the schedule existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

from library.load import Library, build_regimen
from solver.model import DAY
from solver.verify import violations

from .patient import Patient

STEP = Fraction(1, 4)  # 15 minutes


@dataclass
class BaselineResult:
    ok: bool
    schedule: dict[str, list[Fraction]]
    failed_on: Optional[str] = None  # "product dose k"


def run_greedy(lib: Library, patient: Patient) -> BaselineResult:
    built = build_regimen(lib, patient.library_items(), "strict")
    reg = built.regimen
    meals = patient.meal_times()
    bed, wake = patient.sleep_window()
    awake_len = (bed - wake) % DAY
    slots = [(wake + STEP * i) % DAY for i in range(int(awake_len / STEP) + 1)]

    reported = patient.reported_counts()
    counts = {p.name: (p.max_doses if p.prescribed else reported.get(p.name, p.max_doses))
              for p in reg.products}
    order = sorted(reg.products, key=lambda p: (-counts[p.name], p.name))

    placed: dict[str, list[Fraction]] = {p.name: [] for p in reg.products}
    for p in order:
        start = 0
        for k in range(counts[p.name]):
            for i in range(start, len(slots)):
                trial = {**placed, p.name: sorted(placed[p.name] + [slots[i]])}
                if not violations(reg, trial, meals, partial=True):
                    placed = trial
                    start = i + 1
                    break
            else:
                return BaselineResult(False, placed, f"{lib.products[p.name].display_name}, dose {k + 1} of {counts[p.name]}")
    return BaselineResult(True, placed)
