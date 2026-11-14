"""Plain-text rendering of a tier report (the CLI card; the web card comes in M7).

Pure presentation: nothing here changes what was solved.
"""

from __future__ import annotations

import textwrap
from fractions import Fraction

from library.load import Library
from solver.model import (IngredientCeiling, IngredientInterval, MealGap, MealWindow, MinInterval,
                          Separation, Window)

from .conflicts import is_count
from .patient import fmt_hhmm
from .tiers import Option, Report

W = 78
RULE = "=" * W
THIN = "-" * W

HEAD = {
    "tier0": "TIER 0  -  schedulable as they live now",
    "tier1": "TIER 1  -  schedulable if something gives",
    "tier2": "TIER 2  -  the prescriptions alone cannot be scheduled",
    "strict_only_conflict": "NO TIER  -  conflicts under the strict reading of the labels, not the loosest",
    "unknown": "UNDETERMINED  -  a solve did not finish; no answer is shown",
}


# --- small helpers -------------------------------------------------------------------

def _dur(h: Fraction) -> str:
    """1 -> '1 h', 0.5 -> '30 min', 1.5 -> '1 h 30 min'."""
    mins = round(Fraction(h) * 60)
    hh, mm = divmod(mins, 60)
    if hh and mm:
        return f"{hh} h {mm} min"
    return f"{hh} h" if hh else f"{mm} min"


def _num(v: Fraction) -> str:
    return str(int(v)) if v == int(v) else f"{float(v):g}"


def _wrap(text: str, indent: int, first: str = "") -> list[str]:
    pad = " " * indent
    return textwrap.wrap(text, width=W, initial_indent=pad + first,
                         subsequent_indent=pad + " " * len(first)) or [pad + first]


def _section(title: str) -> list[str]:
    return ["", f" {title}", " " + "-" * (len(title))]


def _name(lib: Library, product: str) -> str:
    return lib.products[product].display_name


# --- plain-English rule descriptions ---------------------------------------------------

def describe(rule, lib: Library, unit: str = "") -> str:
    if isinstance(rule, IngredientInterval):
        return f"every dose containing {rule.ingredient} at least {_dur(rule.hours)} apart"
    if isinstance(rule, MinInterval):
        return f"doses of {_name(lib, rule.product)} at least {_dur(rule.hours)} apart"
    if isinstance(rule, Separation):
        a, b = _name(lib, rule.a), _name(lib, rule.b)
        if rule.before and rule.after:
            return f"no {a} from {_dur(rule.before)} before to {_dur(rule.after)} after any {b} dose"
        if rule.before:
            return f"{a} at least {_dur(rule.before)} before any {b} dose"
        return f"{a} at least {_dur(rule.after)} after any {b} dose"
    if isinstance(rule, IngredientCeiling):
        return f"at most {_num(rule.max_amount)}{(' ' + unit) if unit else ''} of {rule.ingredient} per day, across all products"
    if isinstance(rule, MealGap):
        meal = rule.meal or "any meal"
        return f"no {_name(lib, rule.product)} from {_dur(rule.before)} before to {_dur(rule.after)} after {meal}"
    if isinstance(rule, MealWindow):
        return (f"{_name(lib, rule.product)} between {_dur(rule.earliest)} and "
                f"{_dur(rule.latest)} before {rule.meal}")
    if isinstance(rule, Window):
        return f"{_name(lib, rule.product)} only within set hours"
    return rule.id


def _flags(res, r) -> str:
    flags = [r.provenance]
    if res.used_fallback:
        flags.append("from fallback label")
    if res.disagreement:
        flags.append("labels disagree, strictest used")
    if not res.verified:
        flags.append("UNVERIFIED")
    return " | ".join(flags)


# --- report ---------------------------------------------------------------------------

def render(report: Report, lib: Library) -> str:
    p = report.patient
    out = [RULE, " LINNET  -  medication timing check"]
    out += _wrap(f"Patient: {p.name}" + ("  (constructed patient)" if p.constructed else ""), 1)
    out += [" Draft for a pharmacist to check. Reasons over stated label rules only.", RULE,
            "", f" RESULT:  {HEAD[report.outcome]}"]

    if report.outcome == "tier1":
        out += _options(report, lib)
    if report.outcome in ("tier0", "tier1"):
        title = "MEDICATION CARD" + ("  (option 1)" if report.outcome == "tier1" else "")
        out += _card(report, lib, report.options[0], title)
        out += _rules_applied(report, lib)
    if report.outcome == "tier1":
        w = p.weights
        out += _section("WEIGHTS" + ("  (placeholders - set these)" if w.placeholder else ""))
        out += [f"   moving a meal {_num(w.meal_shift_per_hour)} per hour  |  shortening sleep "
                f"{_num(w.sleep_shift_per_hour)} per hour  |  dropping a dose {_num(w.dose_dropped)}"]
    if report.conflicts:
        out += _conflicts(report, lib)
    if report.notes:
        out += _section("NOTES")
        for n in report.notes:
            out += _wrap(n, 3, "! ")
    out += ["", THIN, " Audit: " + "  |  ".join(report.solves), THIN]
    return "\n".join(out)


def _change(c, lib) -> str:
    if c.kind == "meal":
        mins = round((c.after - c.before) * 60)
        return f"move {c.what} {fmt_hhmm(c.before)} -> {fmt_hhmm(c.after)} ({mins:+d} min)"
    if c.kind == "sleep":
        verb = "wake" if c.what == "wake" else "go to bed"
        return f"{verb} at {fmt_hhmm(c.after)} instead of {fmt_hhmm(c.before)}"
    return f"{_name(lib, c.what)}: {int(c.after)}x/day instead of {int(c.before)}x"


def _options(report: Report, lib: Library) -> list[str]:
    out = _section("SMALLEST CHANGES  (lower cost = smaller change; weights below)")
    out.append("   #   cost   change")
    for i, o in enumerate(report.options, 1):
        for j, c in enumerate(o.changes):
            lead = f"   {i}   {float(o.cost):4.2f}   " if j == 0 else " " * 16
            out.append(lead + _change(c, lib))
    return out


def _card(report: Report, lib: Library, opt: Option, title: str) -> list[str]:
    rows = []  # (time, order, text)
    for prod, times in opt.schedule.items():
        for k, t in enumerate(times):
            dose = f"   dose {k + 1} of {len(times)}" if len(times) > 1 else ""
            rows.append((t, 1, f"*  {_name(lib, prod)}{dose}"))
    for meal, t in opt.meals.items():
        rows.append((t, 0, f"   ({meal})"))
    bed, wake = opt.sleep
    rows.append((wake, -1, "   (wake)"))
    rows.append((bed, 2, "   (bedtime)"))
    rows.sort(key=lambda r: (r[0], r[1]))
    out = _section(title)
    out += [f"   {fmt_hhmm(t)}  {text}" for t, _, text in rows]
    return out


def _rules_applied(report: Report, lib: Library) -> list[str]:
    by_sid = {r.id: r for r in report.built.regimen.rules}
    out = _section("RULES APPLIED")
    for sid, text in report.built.rx.items():
        prod = by_sid[sid].product
        out += _wrap(f"{_name(lib, prod)}: {text}", 3, "- ")
        out.append("       [prescription]")
    seen = set()
    for sid, (rid, ing, res) in report.built.origin.items():
        if rid in seen:
            continue
        seen.add(rid)
        out += _wrap(describe(by_sid[sid], lib, getattr(res.rule, "unit", "")), 3, "- ")
        out.append(f"       [{_flags(res, res.rule)}]  {rid}")
    return out


def _conflicts(report: Report, lib: Library) -> list[str]:
    built = report.built
    by_sid = {r.id: r for r in built.regimen.rules}
    out = []
    n = len(report.conflicts)
    for i, c in enumerate(report.conflicts, 1):
        out += _section(f"CONFLICT {i} of {n}  -  these {len(c.rules)} requirements cannot all hold "
                        f"on any 24-hour day")
        if not c.proven_minimal:
            out.append("   (minimality not proven: a re-check timed out)")
        counts = [s for s in c.rules if is_count(s)]
        labels = [s for s in c.rules if not is_count(s)]
        products = set()
        if counts:
            out.append("   Prescriptions:")
            for sid in counts:
                if sid in built.rx:
                    prod = by_sid[sid].product
                    out.append(f"     - {_name(lib, prod)}, {built.rx[sid]}")
                else:
                    prod = sid.split("::", 1)[1]
                    k = next(pr.max_doses for pr in built.regimen.products if pr.name == prod)
                    out.append(f"     - {_name(lib, prod)}, {k}x daily")
                products.add(_name(lib, prod))
        out.append("   Label rules:")
        for j, sid in enumerate(labels, 1):
            rid, ing, res = built.origin[sid]
            r = res.rule
            out += _wrap(describe(by_sid[sid], lib, getattr(r, "unit", "")), 5, f"{j}. ")
            out.append(f"        [{_flags(res, r)}]  {rid}")
            out += _wrap(f"\"{r.source.sentence}\"", 8)
            out.append(f"        SPL {r.source.spl_set_id}  ({r.source.section})")
        if report.outcome == "tier2":
            out += ["", "   Checked at the loosest defensible reading of every label: still no schedule."]
            out += _section("ACTION")
            out += _wrap(" and ".join(sorted(products)) + " cannot both be given as labelled. Ask the "
                         "prescriber whether one can be held, replaced, or re-timed.", 3)
    return out


def render_baseline(g, report: Report, lib: Library) -> str:
    """Side-by-side verdict: greedy heuristic vs LINNET, same question (solve A)."""
    out = _section("COMPARISON  -  greedy heuristic (Tsai & Liu 2008 style, our reimplementation)")
    if g.ok:
        out.append("   Greedy: found a schedule.")
    else:
        out.append(f"   Greedy: GAVE UP - could not place {g.failed_on}.")
        placed = ", ".join(f"{_name(lib, k)} {' '.join(fmt_hhmm(t) for t in v)}" for k, v in g.schedule.items() if v)
        out += _wrap(f"placed before failing: {placed}", 11)
    verdict = {
        "tier0": "LINNET: a schedule exists as they live now (tier 0).",
        "tier1": "LINNET: a schedule exists if something gives (tier 1).",
        "tier2": "LINNET: no schedule exists for the prescriptions at all (tier 2), with the conflict named above.",
    }.get(report.outcome, f"LINNET: {report.outcome}.")
    out.append("   " + verdict)
    if not g.ok and report.outcome == "tier0":
        out += _wrap("FALSE ALARM: the heuristic said this regimen cannot be scheduled; it can, as the card above shows.", 3)
    return "\n".join(out)
