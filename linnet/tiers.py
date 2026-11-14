"""M5: the three solves and tier assignment.

Solve A, "as they live now": every product at the patient's reported count,
    strict reading of every label, meals and sleep fixed as stated.
    sat -> TIER 0.
Solve B, "prescriptions alone": prescribed products only, no routine (no
    meals, no sleep), loosest defensible reading of every label.
    unsat -> TIER 2. Nothing the patient or pharmacist can rearrange helps.
Solve C, "smallest change": strict labels again, but meals may move, sleep may
    shrink, and as-needed doses may drop, each at a cost (patient weights).
    Minimise total cost -> TIER 1, with up to 3 alternatives that change
    *different things*.
If C finds nothing within the allowed shifts, the regimen conflicts under the
strict reading but not the loosest one: the answer depends on how ambiguous
label text is read. That is reported as such, with its conflicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import z3

from library.load import Built, Library, build_regimen
from solver.encode import DAY, Encoding, _q, build, check, forward, read_model
from solver.verify import violations

from .conflicts import Conflict, find_conflicts
from .patient import Patient

z3.set_param("smt.random_seed", 0)  # deterministic re-runs
EPS = Fraction(1, 60)  # one minute: anything smaller is "unchanged"


@dataclass
class Change:
    kind: str  # "meal" | "sleep" | "dose"
    what: str  # meal name, "bedtime"/"wake", or product
    before: Fraction
    after: Fraction


@dataclass
class Option:
    cost: Fraction
    changes: list[Change]
    schedule: dict[str, list[Fraction]]
    meals: dict[str, Fraction]
    sleep: tuple[Fraction, Fraction]


@dataclass
class Report:
    tier: Optional[int]  # 0, 1, 2, or None (see `outcome`)
    outcome: str  # "tier0" | "tier1" | "tier2" | "strict_only_conflict" | "unknown"
    patient: Patient
    options: list[Option] = field(default_factory=list)  # tier 0: one option with no changes
    conflicts: list[Conflict] = field(default_factory=list)
    built: Optional[Built] = None  # the regimen the shown result came from
    notes: list[str] = field(default_factory=list)
    solves: list[str] = field(default_factory=list)  # audit trail: which solves ran and how they ended


def _awake(t, wake: Fraction, bed: Fraction, early, late) -> z3.BoolRef:
    """t lies in the awake arc [wake - early, bed + late] on the circular day."""
    awake_len = (bed - wake) % DAY
    return z3.Or(forward(_q(wake), t) <= _q(awake_len) + late, forward(t, _q(wake)) <= early)


def _add_sleep(enc: Encoding, wake, bed, early=None, late=None) -> None:
    early = _q(0) if early is None else early
    late = _q(0) if late is None else late
    for p, s in enc.slots.items():
        enc.track(f"routine::sleep@{p}",
                  z3.And(True, *[z3.Implies(o, _awake(t, wake, bed, early, late)) for t, o in s.doses()]))


def _canonical(enc: Encoding) -> None:
    """Secondary objective: earliest valid dose times. Many schedules tie; this
    picks one deterministically so a re-run shows the same card."""
    enc.solver.minimize(z3.Sum([z3.If(o, t, _q(0)) for s in enc.slots.values() for t, o in s.doses()] or [_q(0)]))


def _notes(built: Built) -> list[str]:
    out = list(built.skipped)
    if built.unverified:
        out.append(f"{len(built.unverified)} library item(s) not yet verified by a human: "
                   + ", ".join(built.unverified))
    return out


def run(lib: Library, patient: Patient, timeout_ms: int = 10_000) -> Report:
    bed, wake = patient.sleep_window()
    meals0 = patient.meal_times()
    reported = patient.reported_counts()

    # --- Solve A: tier 0 -------------------------------------------------------------
    built_a = build_regimen(lib, patient.library_items(), "strict")
    enc_a = build(built_a.regimen, counts=reported, meals=meals0, timeout_ms=timeout_ms, optimize=True)
    _add_sleep(enc_a, wake, bed)
    _canonical(enc_a)
    ra = check(enc_a)
    report = Report(None, "unknown", patient, notes=_notes(built_a))
    report.solves.append(f"A (as reported, strict): {ra.status}")
    if ra.status == "sat":
        _assert_valid(built_a, ra.schedule, ra.meals)
        report.tier, report.outcome, report.built = 0, "tier0", built_a
        report.options = [Option(Fraction(0), [], ra.schedule, ra.meals, (bed, wake))]
        return report
    if ra.status == "unknown":
        report.notes.append(f"Solve A did not finish: {ra.reason}")

    # --- Solve B: tier 2 -------------------------------------------------------------
    rx_items = patient.library_items(prescribed_only=True)
    built_b = build_regimen(lib, rx_items, "loose", include_food=False)
    enc_b = build(built_b.regimen, timeout_ms=timeout_ms)
    rb = check(enc_b)
    report.solves.append(f"B (prescriptions only, loosest): {rb.status}")
    if rb.status == "unsat":
        report.tier, report.outcome, report.built = 2, "tier2", built_b
        report.conflicts = find_conflicts(enc_b, rb.core)
        report.notes = _notes(built_b)
        return report
    if rb.status == "unknown":
        report.notes.append(f"Solve B did not finish: {rb.reason}; tier 2 could not be ruled in or out")
        return report

    # --- Solve C: tier 1 -------------------------------------------------------------
    w = patient.weights
    built_c = built_a
    meal_vars = {k: z3.Real(f"meal::{k}") for k in meals0}
    enc_c = build(built_c.regimen, meals=meal_vars, timeout_ms=timeout_ms, optimize=True)
    opt = enc_c.solver
    costs, components = [], []  # components: (Change template, z3 magnitude term)

    for k, m in meal_vars.items():
        dev = z3.Real(f"dev::{k}")
        opt.add(m >= 0, m < DAY, dev >= m - _q(meals0[k]), dev >= _q(meals0[k]) - m,
                dev <= _q(w.max_meal_shift_hours))
        costs.append(_q(w.meal_shift_per_hour) * dev)
        components.append((("meal", k), dev))

    early, late = z3.Real("sleep::wake_earlier"), z3.Real("sleep::bed_later")
    opt.add(early >= 0, late >= 0, early <= _q(w.max_sleep_shift_hours), late <= _q(w.max_sleep_shift_hours))
    _add_sleep(enc_c, wake, bed, early, late)
    costs.append(_q(w.sleep_shift_per_hour) * (early + late))
    components += [(("sleep", "wake"), early), (("sleep", "bedtime"), late)]

    for p, n in reported.items():
        cnt = z3.ToReal(enc_c.slots[p].count())
        enc_c.track(f"reported_cap::{p}", cnt <= n)  # suggest fewer, never more
        dropped = _q(n) - cnt
        costs.append(_q(w.dose_dropped) * dropped)
        components.append((("dose", p), dropped))

    total = z3.Sum(costs)
    opt.minimize(total)  # lexicographic: cost first,
    _canonical(enc_c)    # then earliest times among equal-cost schedules

    options: list[Option] = []
    for _ in range(3):
        rc = check(enc_c)
        if rc.status != "sat":
            if not options:
                report.solves.append(f"C (smallest change): {rc.status}")
            break
        m = rc.model
        changes, signature = [], []
        for (kind, what), term in components:
            v = _val(m, term)
            changed = v > EPS
            # Superset test for the block below: continuous shifts count as "also
            # changed" if nonzero; dose drops only if at least as many are dropped.
            signature.append((term >= _q(v)) if kind == "dose" else (term > _q(EPS)) if changed else None)
            if not changed:
                continue
            if kind == "meal":
                changes.append(Change("meal", what, meals0[what], rc.meals[what]))
            elif kind == "sleep" and what == "wake":
                changes.append(Change("sleep", "wake", wake, (wake - v) % DAY))
            elif kind == "sleep":
                changes.append(Change("sleep", "bedtime", bed, (bed + v) % DAY))
            else:
                changes.append(Change("dose", what, Fraction(reported[what]),
                                      Fraction(reported[what]) - v))
        new_bed = next((c.after for c in changes if c.what == "bedtime"), bed)
        new_wake = next((c.after for c in changes if c.what == "wake"), wake)
        _assert_valid(built_c, rc.schedule, rc.meals)
        options.append(Option(_val(m, total), changes, rc.schedule, rc.meals, (new_bed, new_wake)))
        # Next option must leave at least one of this option's changes undone,
        # so it is a genuinely different fix, not this one plus a pointless tweak.
        superset = [c for c, (_, term) in zip(signature, components) if c is not None and _val(m, term) > EPS]
        if not superset:
            break
        opt.add(z3.Not(z3.And(*superset)))

    if options:
        # Ties in cost come back in solver order; fix the order for stable output.
        options.sort(key=lambda o: (o.cost, [(c.kind, c.what, c.after) for c in o.changes]))
        report.solves.append(f"C (smallest change): {len(options)} option(s)")
        report.tier, report.outcome, report.built, report.options = 1, "tier1", built_c, options
        return report

    # Strict reading conflicts, loosest does not: show the strict prescriptions-only conflict.
    built_s = build_regimen(lib, rx_items, "strict", include_food=False)
    enc_s = build(built_s.regimen, timeout_ms=timeout_ms)
    rs = check(enc_s)
    report.solves.append(f"S (prescriptions only, strict): {rs.status}")
    report.outcome, report.built = "strict_only_conflict", built_s
    if rs.status == "unsat":
        report.conflicts = find_conflicts(enc_s, rs.core)
    else:
        report.notes.append("No schedule found within the allowed routine shifts; widen the limits in `weights`.")
    return report


def _val(m: z3.ModelRef, term) -> Fraction:
    v = m.eval(term, model_completion=True)
    if z3.is_rational_value(v):
        return v.as_fraction()
    if z3.is_int_value(v):
        return Fraction(v.as_long())
    raise ValueError(f"non-rational value {v}")


def _assert_valid(built: Built, schedule, meals) -> None:
    v = violations(built.regimen, schedule, meals)
    if v:  # an encoding bug: never show a card that breaks a label
        raise AssertionError("solver schedule failed independent check: " + "; ".join(v))
