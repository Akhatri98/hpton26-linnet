"""Report -> JSON for the web UI.  [pure presentation]

Same content as the text card (render.py), as data. Nothing here changes
what was solved. Times are sent both as "HH:MM" and as minutes after
midnight (for drawing the timeline).
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional

from library.load import Library

from .conflicts import is_count
from .patient import fmt_hhmm
from .render import HEAD, _change, _name, describe
from .tiers import Option, Report


def _min(t: Fraction) -> int:
    return round(Fraction(t) * 60) % (24 * 60)


def _time(t: Fraction) -> dict:
    return {"hhmm": fmt_hhmm(t), "min": _min(t)}


def _source(rule) -> dict:
    s = rule.source
    return {"sentence": s.sentence, "spl_set_id": s.spl_set_id, "section": s.section, "url": s.url}


def _flags(res) -> dict:
    return {"provenance": res.rule.provenance, "verified": bool(res.verified),
            "from_fallback_label": bool(res.used_fallback), "labels_disagree": bool(res.disagreement)}


def _option(o: Option, lib: Library) -> dict:
    events = []
    for prod, times in o.schedule.items():
        for k, t in enumerate(times):
            events.append({**_time(t), "kind": "dose", "product": prod, "label": _name(lib, prod),
                           "dose": k + 1, "of": len(times)})
    for meal, t in o.meals.items():
        events.append({**_time(t), "kind": "meal", "label": meal})
    bed, wake = o.sleep
    events.append({**_time(wake), "kind": "wake", "label": "wake"})
    events.append({**_time(bed), "kind": "bed", "label": "bedtime"})
    order = {"wake": -1, "meal": 0, "dose": 1, "bed": 2}
    events.sort(key=lambda e: (e["min"], order[e["kind"]]))
    return {
        "cost": float(o.cost),
        "changes": [{"kind": c.kind, "what": c.what, "before": _time(c.before) if c.kind != "dose" else int(c.before),
                     "after": _time(c.after) if c.kind != "dose" else int(c.after), "text": _change(c, lib)}
                    for c in o.changes],
        "events": events,
        "sleep": {"bed": _time(bed), "wake": _time(wake)},
    }


def _rules_applied(report: Report, lib: Library) -> list[dict]:
    built = report.built
    by_sid = {r.id: r for r in built.regimen.rules}
    out = []
    for sid, text in built.rx.items():
        out.append({"kind": "prescription", "id": sid, "text": f"{_name(lib, by_sid[sid].product)}: {text}"})
    seen = set()
    for sid, (rid, ing, res) in built.origin.items():
        if rid in seen:
            continue
        seen.add(rid)
        out.append({"kind": "label", "id": rid, "ingredient": ing,
                    "text": describe(by_sid[sid], lib, getattr(res.rule, "unit", "")),
                    **_flags(res), "source": _source(res.rule)})
    return out


def _conflicts(report: Report, lib: Library) -> list[dict]:
    built = report.built
    by_sid = {r.id: r for r in built.regimen.rules}
    out = []
    for c in report.conflicts:
        rx, labels = [], []
        for sid in c.rules:
            if is_count(sid):
                if sid in built.rx:
                    rx.append(f"{_name(lib, by_sid[sid].product)}, {built.rx[sid]}")
                else:
                    prod = sid.split("::", 1)[1]
                    k = next(p.max_doses for p in built.regimen.products if p.name == prod)
                    rx.append(f"{_name(lib, prod)}, {k}x daily")
            else:
                rid, ing, res = built.origin[sid]
                labels.append({"id": rid, "ingredient": ing,
                               "text": describe(by_sid[sid], lib, getattr(res.rule, "unit", "")),
                               **_flags(res), "source": _source(res.rule)})
        out.append({"proven_minimal": c.proven_minimal, "prescriptions": rx, "label_rules": labels})
    return out


def _baseline(g, report: Report, lib: Library) -> dict:
    return {
        "ok": g.ok, "failed_on": g.failed_on,
        "placed": {_name(lib, k): [fmt_hhmm(t) for t in v] for k, v in g.schedule.items() if v},
        "false_alarm": (not g.ok) and report.outcome == "tier0",
        "method": "greedy, one dose at a time, no backtracking (Tsai & Liu 2008 style; our reimplementation)",
    }


def report_json(report: Report, lib: Library, baseline=None) -> dict:
    p = report.patient
    w = p.weights
    return {
        "tier": report.tier,
        "outcome": report.outcome,
        "headline": HEAD[report.outcome],
        "patient": {"name": p.name, "constructed": p.constructed},
        "options": [_option(o, lib) for o in report.options],
        "rules_applied": _rules_applied(report, lib) if report.built and report.outcome in ("tier0", "tier1") else [],
        "conflicts": _conflicts(report, lib) if report.conflicts else [],
        "weights": {"meal_shift_per_hour": float(w.meal_shift_per_hour), "sleep_shift_per_hour": float(w.sleep_shift_per_hour),
                    "dose_dropped": float(w.dose_dropped), "placeholder": w.placeholder},
        "notes": report.notes,
        "audit": report.solves,
        "baseline": _baseline(baseline, report, lib) if baseline is not None else None,
    }
