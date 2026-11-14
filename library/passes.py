"""The dosing and limits extraction passes, and the coverage check.

Each label is read in THREE focused passes, each with its own short prompt, so
one kind of fact cannot crowd out another:

  dosing          dose and frequency per condition. Reference data, NOT a hard
                  rule: the prescription chooses a regimen; LINNET checks the
                  prescription against these options.
  limits          max per dose, max per day, max duration.
  administration  separations and food/meal rules (library.extract.PROMPT);
                  this is what the solver mainly consumes.

Coverage (deterministic): any sentence that states a number and talks about
timing or limits, but that NO accepted row from any pass cites, is listed as
"unaccounted for" for the pharmacist. The model may skip things; the coverage
list makes every skip visible.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .extract import Checked, Sentence, numbers_in

DOSING_PROMPT = """You are extracting DOSING directions from an FDA drug label for a pharmacist's tool.

Ingredient: {ingredient}
Label set id: {spl}

Numbered sentences from the label:
{sentences}

Return ONLY a JSON object {{"dosing": [ ... ]}}. One row per distinct regimen the label gives (per condition,
per population, per formulation). Each row:
  {{"condition": "what it is for, or 'general'", "population": "adults" | "children" | "renal impairment" | ...,
   "dose": N, "dose_max": N or null, "unit": "mg" | "g" | "mcg" | "tablets" | "caplets" | "capsules" | "mL",
   "every_hours": H or null, "times_per_day": T or null, "duration_days": D or null,
   "as_needed": true | false, "sentence": "Sk", "provenance": "quoted" | "interpreted"}}
Use only numbers stated in the cited sentence. A range like "500-750 mg" is dose=500, dose_max=750.
"twice daily" is times_per_day=2; "every 12 hours" is every_hours=12. Cite exactly one sentence id per row; do not copy text.
Include every regimen you find, even if there are many. If unsure, leave it out.
"""

LIMITS_PROMPT = """You are extracting LIMITS from an FDA drug label for a pharmacist's tool.

Ingredient: {ingredient}
Label set id: {spl}

Numbered sentences from the label:
{sentences}

Return ONLY a JSON object {{"limits": [ ... ]}}. One row per limit the label states. Each row:
  {{"kind": "per_dose" | "per_day" | "duration_days",
   "amount": N, "unit": "mg" | "g" | "mcg" | "tablets" | "caplets" | "capsules" | "doses" | "days",
   "population": "adults" | "children" | "pregnant" | "general" | ...,
   "sentence": "Sk", "provenance": "quoted" | "interpreted"}}
Examples: "do not take more than 6 caplets in 24 hours" -> per_day, 6, caplets.
"more than 4,000 mg of acetaminophen in 24 hours" -> per_day, 4000, mg.
"do not use for more than 10 days" -> duration_days, 10, days.
Use only numbers stated in the cited sentence. Cite exactly one sentence id per row; do not copy text.
"""

UNITS = {"mg", "g", "mcg", "tablets", "caplets", "capsules", "ml", "doses", "days"}
WORD_COUNTS = {"once": 1, "twice": 2, "thrice": 3}
COVER = re.compile(r"\b(hours?|minutes?|daily|a day|per day|24 hours|maximum|more than|exceed|every|"
                   r"before|after|apart|times|meals?|breakfast|food|days?)\b", re.I)


def _grounded(values: dict, s: Sentence) -> list[str]:
    stated = numbers_in(s.text) | {Fraction(v) for w, v in WORD_COUNTS.items()
                                   if re.search(rf"\b{w}\b", s.text, re.I)}
    bad = []
    for k, v in values.items():
        if v is None:
            continue
        try:
            f = Fraction(str(v))
        except (ValueError, ZeroDivisionError):
            bad.append(f"{k}={v} (not a number)")
            continue
        if f != 0 and f not in stated:
            bad.append(f"{k}={v}")
    return bad


def _src(spl: str, s: Sentence) -> dict:
    return {"spl_set_id": spl, "section": s.section, "sentence_id": s.sid, "sentence": s.text,
            "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={spl}"}


def check_dosing(answer: dict, spl: str, sents: list[Sentence]) -> Checked:
    by_id, out = {s.sid: s for s in sents}, Checked()
    for raw in answer.get("dosing", []) or []:
        s = by_id.get(str(raw.get("sentence", "")))
        if s is None:
            out.rejected.append({"pass": "dosing", "item": raw, "reason": "cites a sentence id that does not exist"})
            continue
        nums = {k: raw.get(k) for k in ("dose", "dose_max", "every_hours", "times_per_day", "duration_days")}
        bad = _grounded(nums, s)
        unit = str(raw.get("unit", "")).lower()
        if bad:
            out.rejected.append({"pass": "dosing", "item": raw, "reason": f"number(s) not stated in {s.sid}: {', '.join(bad)}"})
        elif raw.get("dose") is None or unit not in UNITS:
            out.rejected.append({"pass": "dosing", "item": raw, "reason": "needs a dose and a known unit"})
        elif raw.get("provenance") not in ("quoted", "interpreted"):
            out.rejected.append({"pass": "dosing", "item": raw, "reason": "provenance must be quoted or interpreted"})
        else:
            row = {k: raw.get(k) for k in ("condition", "population", "dose", "dose_max", "unit", "every_hours",
                                            "times_per_day", "duration_days", "as_needed", "provenance")}
            out.rules.append({**row, "unit": unit, "source": _src(spl, s), "verified": False})
    return out


def check_limits(answer: dict, spl: str, sents: list[Sentence]) -> Checked:
    by_id, out = {s.sid: s for s in sents}, Checked()
    for raw in answer.get("limits", []) or []:
        s = by_id.get(str(raw.get("sentence", "")))
        if s is None:
            out.rejected.append({"pass": "limits", "item": raw, "reason": "cites a sentence id that does not exist"})
            continue
        bad = _grounded({"amount": raw.get("amount")}, s)
        unit, kind = str(raw.get("unit", "")).lower(), raw.get("kind")
        if bad:
            out.rejected.append({"pass": "limits", "item": raw, "reason": f"number not stated in {s.sid}: {', '.join(bad)}"})
        elif kind not in ("per_dose", "per_day", "duration_days") or unit not in UNITS or raw.get("amount") is None:
            out.rejected.append({"pass": "limits", "item": raw, "reason": "needs kind, amount and a known unit"})
        elif raw.get("provenance") not in ("quoted", "interpreted"):
            out.rejected.append({"pass": "limits", "item": raw, "reason": "provenance must be quoted or interpreted"})
        else:
            out.rules.append({"kind": kind, "amount": raw["amount"], "unit": unit,
                              "population": raw.get("population", "general"), "provenance": raw["provenance"],
                              "source": _src(spl, s), "verified": False})
    return out


# Triage for uncited sentences, highest priority first. Deterministic keyword
# rules; they only ORDER the list for a human, they never drop a sentence.
TRIAGE = [
    # (priority, kind, what it means for the reviewer)
    (1, "possible_timing_rule", "says when to take it relative to something else - check for a missed rule"),
    (2, "review", "states a number about dosing or limits - check for a missed row"),
    (3, "monitoring", "lab monitoring, titration or testing schedule - not a dose time"),
    (4, "population", "applies to a specific population (children, pregnancy, impairment)"),
    (5, "weight_based", "dose per kg of body weight - not expressible in the dosing table yet"),
    (6, "restates", "repeats a sentence a row already cites"),
    (7, "background", "animal, lab or pharmacokinetic data - not a direction"),
]
_SEPARATION = re.compile(r"\b(before|after|apart|separat\w*|prior to|within)\b.*\b(hours?|minutes?)\b|"
                         r"\b(hours?|minutes?)\b.*\b(before|after|apart|separat\w*|prior)\b", re.I)
_MONITORING = re.compile(r"\b(monitor\w*|titrat\w*|assess\w*|testing|serologic|follow-up|evaluate)\b", re.I)
_BACKGROUND = re.compile(r"\b(rats?|mice|rodents?|carcinogen\w*|clastogen\w*|mutagen\w*|fertility|in vitro|"
                         r"AUC|Cmax|plasma (concentrations?|levels?)|exposure|half-life|bioavailability|"
                         r"absorption .* (decreased|reduced)|MHDD|body surface area)\b", re.I)
_WEIGHT = re.compile(r"\b(mg|mcg|g)\s*/\s*kg\b|\bper kg\b|/kg/", re.I)
_POPULATION = re.compile(r"\b(pediatric|children|child|infants?|neonat\w*|newborns?|pregnan\w*|geriatric|"
                         r"elderly|renal|hepatic|impairment|dialysis|years of age|months of age)\b", re.I)
_WORD = re.compile(r"[a-z]+")


def _words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2}


def _restates(s: Sentence, cited: list[Sentence]) -> Sentence | None:
    """A cited sentence with the same numbers and mostly the same words."""
    nums, words = numbers_in(s.text), _words(s.text)
    for c in cited:
        cw = _words(c.text)
        overlap = len(words & cw) / max(1, len(words | cw))
        if nums <= numbers_in(c.text) and overlap >= 0.4:
            return c
    return None


def triage(s: Sentence, cited: list[Sentence]) -> tuple[int, str, str]:
    """(priority, kind, note). The order of checks is deliberate: a repeat of a
    cited sentence is low priority even if it reads like a timing rule."""
    hit = _restates(s, cited)
    if hit:
        return 6, "restates", f"repeats {hit.sid}"
    if _BACKGROUND.search(s.text):
        return 7, "background", ""
    if _WEIGHT.search(s.text):
        return 5, "weight_based", ""
    if _SEPARATION.search(s.text):
        return 1, "possible_timing_rule", ""
    if _MONITORING.search(s.text):
        return 3, "monitoring", ""
    if _POPULATION.search(s.text):
        return 4, "population", ""
    return 2, "review", ""


def uncited(sents: list[Sentence], cited: set[str]) -> list[dict]:
    """Sentences that state a number about timing/limits but no accepted row cites,
    sorted so the ones most likely to hide a missed rule come first."""
    used = [s for s in sents if s.sid in cited]
    out = []
    for s in sents:
        if s.sid in cited or not (COVER.search(s.text) and numbers_in(s.text)):
            continue
        prio, kind, note = triage(s, used)
        out.append({"priority": prio, "kind": kind, "note": note,
                    "sentence_id": s.sid, "section": s.section, "sentence": s.text})
    out.sort(key=lambda u: (u["priority"], int(u["sentence_id"][1:])))
    return out
