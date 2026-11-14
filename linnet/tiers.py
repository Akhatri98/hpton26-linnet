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

from library.load import Built, Library, build_regimen
from solver.encode import DAY, FREE, build, check
from solver.verify import violations

from .conflicts import Conflict, find_conflicts
from .patient import Patient

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
    unfinished: list[str] = field(default_factory=list)  # solves that hit their time limit ("A", "B", "C")


def _notes(built: Built) -> list[str]:
    out = []
    if built.rejected:  # first and loudest: the result rests on facts a reviewer said are wrong
        out.append("REJECTED BY REVIEWER, still used here, so this result is not reliable: " + ", ".join(built.rejected))
    out += list(built.skipped)
    if built.unverified:
        out.append(f"{len(built.unverified)} library item(s) not yet verified by a human: "
                   + ", ".join(built.unverified))
    return out


def run(lib: Library, patient: Patient, timeout_ms: int = 10_000) -> Report:
    bed, wake = patient.sleep_window()
    meals0 = patient.meal_times()
    reported = patient.reported_counts()

    # --- Solve A: tier 0 -------------------------------------------------------------
    built_a = build_regimen(lib, patient.library_items(), "strict", bed=bed, wake=wake, meals=meals0)
    enc_a = build(built_a.regimen, counts=reported, meals=meals0, timeout_ms=timeout_ms, optimize=True)
    awake_len = (bed - wake) % DAY
    enc_a.solver.sleep(wake, awake_len)
    enc_a.solver.minimize_times()  # many schedules tie; earliest times makes a re-run show the same card
    ra = check(enc_a)
    report = Report(None, "unknown", patient, notes=_notes(built_a))
    report.solves.append(f"A (as reported, strict): {ra.status}")
    if ra.status == "sat":
        _assert_valid(built_a, ra.schedule, ra.meals)
        report.tier, report.outcome, report.built = 0, "tier0", built_a
        report.options = [Option(Fraction(0), [], ra.schedule, ra.meals, (bed, wake))]
        return report
    if ra.status == "unknown":  # not a "no": solve C below can still find a schedule
        report.notes.append(f"Solve A did not finish: {ra.reason}")
        report.unfinished.append("A")

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
        report.unfinished.append("B")
        return report

    # --- Solve C: tier 1 -------------------------------------------------------------
    w = patient.weights
    # Same rules as solve A; "after waking" windows keep clear of any bedtime this search can reach.
    built_c = build_regimen(lib, patient.library_items(), "strict", bed=bed, wake=wake, meals=meals0,
                            shift_margin=w.max_meal_shift_hours + w.max_sleep_shift_hours)
    enc_c = build(built_c.regimen, meals={k: FREE for k in meals0}, timeout_ms=timeout_ms, optimize=True)
    opt = enc_c.solver
    components = []  # (kind, what, name of the solver term measuring the change)

    for k in meals0:
        opt.meal_shift(k, meals0[k], w.max_meal_shift_hours, w.meal_shift_per_hour)
        components.append(("meal", k, f"dev::{k}"))

    opt.sleep(wake, awake_len, w.max_sleep_shift_hours, w.sleep_shift_per_hour)
    components += [("sleep", "wake", "sleep::wake_earlier"), ("sleep", "bedtime", "sleep::bed_later")]

    for p, n in reported.items():
        opt.reported_cap(p, n, w.dose_dropped)  # suggest fewer, never more
        components.append(("dose", p, f"dropped::{p}"))

    opt.minimize_cost()   # lexicographic: cost first,
    opt.minimize_times()  # then earliest times among equal-cost schedules

    options: list[Option] = []
    c_unfinished = False
    for _ in range(3):
        rc = check(enc_c)
        if rc.status != "sat":
            if not options:
                report.solves.append(f"C (smallest change): {rc.status}")
            c_unfinished = rc.status == "unknown"
            break
        # Read this option's changes, and make the next option leave at least one
        # of them undone, so it is a genuinely different fix, not this one plus a
        # pointless tweak.
        blocked, size, cost = opt.block_superset(EPS)
        changes = []
        for kind, what, name in components:
            v = size[name]
            if v <= EPS:  # one minute: anything smaller is "unchanged"
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
        options.append(Option(cost, changes, rc.schedule, rc.meals, (new_bed, new_wake)))
        if not blocked:
            break

    if options:
        # Ties in cost come back in solver order; fix the order for stable output.
        options.sort(key=lambda o: (o.cost, [(c.kind, c.what, c.after) for c in o.changes]))
        report.solves.append(f"C (smallest change): {len(options)} option(s)")
        if "A" in report.unfinished and options[0].cost == 0:
            # Solve A timed out, but solve C found a schedule that changes nothing: that is tier 0.
            report.tier, report.outcome, report.built = 0, "tier0", built_c
            report.options = [Option(Fraction(0), [], options[0].schedule, options[0].meals, (bed, wake))]
            report.notes.append("Schedule found by the smallest-change search, with no change needed.")
            return report
        report.tier, report.outcome, report.built, report.options = 1, "tier1", built_c, options
        if "A" in report.unfinished:
            report.notes.append("The routine as reported was not proven impossible (solve A did not finish), "
                                "so these changes may not all be needed.")
        return report
    if c_unfinished:  # no schedule found in time, but none ruled out either: say so, never guess
        report.notes.append(f"Solve C did not finish: {rc.reason}; no smallest change could be found in time")
        report.unfinished.append("C")
        return report

    # C proved no schedule within the allowed shifts (so A is impossible too). Strict reading
    # conflicts, loosest does not: show the strict prescriptions-only conflict.
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


def _assert_valid(built: Built, schedule, meals) -> None:
    v = violations(built.regimen, schedule, meals)
    if v:  # an encoding bug: never show a card that breaks a label
        raise AssertionError("solver schedule failed independent check: " + "; ".join(v))
