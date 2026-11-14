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
Active ingredients on this label: {substances}
Strengths this label comes in (read from the label by code): {strengths}

Numbered sentences from the label:
{sentences}

Return ONLY a JSON object {{"dosing": [ ... ]}}. One row per distinct regimen the label gives: per condition,
per population, and PER STRENGTH when the label gives different directions for different strengths.
Each row:
  {{"strength": "the strength this row is for, written like the list above, e.g. '300 mg/30 mg'", or null if the
              dose is given as an amount (mg, mcg, g) that does not depend on the strength,
   "dose_type": "usual" | "starting" | "maintenance" | "adjusted" (for a population such as renal or hepatic impairment),
   "condition": "what it is for, or 'general'", "population": "adults" | "children" | "renal impairment" | ...,
   "dose": N, "dose_max": N or null,
   "unit": "mg" | "g" | "mcg" | "mg/kg" | "mcg/kg" | "tablets" | "caplets" | "capsules" | "mL",
   "every_hours": H or null, "times_per_day": T or null, "duration_days": D or null,
   "as_needed": true | false, "sentence": "Sk", "provenance": "quoted" | "interpreted"}}
Rules:
- If the dose is a count of tablets, caplets or capsules and the label comes in more than one strength, "strength" is REQUIRED.
- Weight-based doses use "mg/kg" or "mcg/kg" (e.g. "15 mg/kg" -> dose 15, unit "mg/kg").
- A range like "500-750 mg" is dose=500, dose_max=750. "1 to 2 tablets" is dose=1, dose_max=2.
- "twice daily" is times_per_day=2; "every 12 hours" is every_hours=12.
- Not doses: maximums ("do not take more than 6 tablets") belong to limits; increases ("increase by 12.5 to 25 mcg")
  are titration steps, not doses; "do not use for more than 10 days" is a limit, not a course length.
- Read the sentence before a dose: it often says who the dose is for (e.g. patients with liver cirrhosis).
- Use only numbers stated in the cited sentence. Cite exactly one sentence id per row; do not copy text.
Include every regimen you find, even if there are many. If unsure, leave it out.
"""

LIMITS_PROMPT = """You are extracting LIMITS from an FDA drug label for a pharmacist's tool.

Ingredient: {ingredient}
Label set id: {spl}
Active ingredients on this label: {substances}
Strengths this label comes in (read from the label by code): {strengths}

Numbered sentences from the label:
{sentences}

Return ONLY a JSON object {{"limits": [ ... ]}}. One row per limit the label states. Each row:
  {{"kind": "per_dose" | "per_day" | "duration_days",
   "amount": N, "unit": "mg" | "g" | "mcg" | "tablets" | "caplets" | "capsules" | "doses" | "days",
   "ingredient": "which active ingredient the amount is of (one of the list above)", or null when there is only one,
   "strength": "the strength the limit is for, e.g. '300 mg/60 mg'", or null if it applies to all strengths,
   "population": "adults" | "children" | "pregnant" | "general" | ...,
   "condition": "what it is for, if the limit depends on it (e.g. 'complicated UTI')", or null,
   "sentence": "Sk", "provenance": "quoted" | "interpreted"}}
Rules:
- Starting doses, titration steps and per-kg doses are not limits.
- On a label with more than one active ingredient, every limit in mg, mcg or g MUST name its "ingredient".
- A limit counted in tablets, caplets or capsules on a label with more than one strength MUST name its "strength".
Examples: "do not take more than 6 caplets in 24 hours" -> per_day, 6, caplets.
"more than 4,000 mg of acetaminophen in 24 hours" -> per_day, 4000, mg, ingredient "acetaminophen".
"do not use for more than 10 days" -> duration_days, 10, days.
Use only numbers stated in the cited sentence. Cite exactly one sentence id per row; do not copy text.
"""

UNITS = {"mg", "g", "mcg", "mg/kg", "mcg/kg", "tablets", "caplets", "capsules", "ml", "doses", "days"}
COUNT_UNITS = {"tablets", "caplets", "capsules"}
MASS_UNITS = {"mg", "g", "mcg"}
WORD_COUNTS = {"once": 1, "twice": 2, "thrice": 3}
COVER = re.compile(r"\b(hours?|minutes?|daily|a day|per day|24 hours|maximum|more than|exceed|every|"
                   r"before|after|apart|times|meals?|breakfast|food|days?)\b", re.I)
_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(mg|mcg|g)\b", re.I)


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


def _label_facts(spl: str) -> tuple[list[str], list[str]]:
    """(strengths, active ingredients) read from the cached label by code; empty if not cached."""
    from .extract import DATA, label_record, label_strengths, label_substances
    if not (DATA / "labels" / f"{spl}.json").exists():
        return [], []
    rec = label_record(spl)
    return label_strengths(rec), label_substances(rec)


def _amounts(strength) -> list[tuple[Fraction, str]]:
    return sorted((Fraction(a.replace(",", "")), u.lower()) for a, u in _AMOUNT.findall(strength or ""))


def _strength_problem(strength, unit: str, s: Sentence, strengths: list[str]):
    """None if fine, else why the row cannot be trusted to name the right strength."""
    if strength:
        if strengths and _amounts(strength) not in [_amounts(x) for x in strengths]:
            return f"strength '{strength}' is not one this label comes in ({', '.join(strengths)})"
        if not _amounts(strength):
            return f"strength '{strength}' has no amount in mg, mcg or g"
        missing = [str(a) for a, _ in _amounts(strength) if a not in numbers_in(s.text)]
        if missing:
            return f"strength '{strength}' is not stated in {s.sid}"
        return None
    if unit in COUNT_UNITS and len(strengths) > 1:
        return f"counted in {unit} but the label comes in {len(strengths)} strengths ({', '.join(strengths)}): say which one"
    return None


_SALT_WORDS = ("hydrochloride", "sodium", "phosphate", "usp")


LIMIT_WORDS = re.compile(r"(more than|exceed\w*|maximum|no more than|up to|not to exceed|greater than)\W+(?:\w+\W+){0,3}$", re.I)
STEP_WORDS = re.compile(r"(increase\w*|decrease\w*|increments?|titrat\w*|adjust\w*)\W+(?:\w+\W+){0,4}$", re.I)


def _num_strings(v) -> list[str]:
    f = Fraction(str(v))
    out = [f"{float(f):g}"]
    if f.denominator == 1 and f >= 1000:
        out.append(f"{int(f):,}")
    return out


def _occurrences(v, text: str) -> list[int]:
    starts = []
    for n in _num_strings(v):
        starts += [m.start() for m in re.finditer(rf"(?<![\d.,]){re.escape(n)}(?![\d]|[.,]\d)", text)]
    return starts


def _only_after(words, v, text: str) -> bool:
    """Every place the number appears is right after `words` (e.g. 'more than', 'increase by')."""
    starts = _occurrences(v, text)
    return bool(starts) and all(words.search(text[max(0, i - 60):i]) for i in starts)


def _per_kg(v, unit: str, text: str) -> bool:
    """The label gives this number per kilogram ('1.6 mcg/kg') but the row dropped the /kg."""
    if unit not in MASS_UNITS:
        return False
    return any(re.search(rf"(?<![\d.,]){re.escape(n)}\s*{unit}\s*/\s*kg", text, re.I) for n in _num_strings(v))


def _dose_problem(raw: dict, unit: str, s: Sentence):
    for k in ("dose", "dose_max"):
        v = raw.get(k)
        if v is None:
            continue
        try:
            Fraction(str(v))
        except (ValueError, ZeroDivisionError):
            continue
        if _per_kg(v, unit, s.text):
            return f"{s.sid} gives {k}={v} per kg ('{v} {unit}/kg'); use unit '{unit}/kg'"
        if _only_after(LIMIT_WORDS, v, s.text):
            return f"{s.sid} states {v} as a maximum, not a dose; it belongs in limits"
        if _only_after(STEP_WORDS, v, s.text):
            return f"{s.sid} states {v} as an increase or titration step, not a dose"
    if raw.get("duration_days") is not None and _only_after(LIMIT_WORDS, raw["duration_days"], s.text):
        return f"{s.sid} states {raw['duration_days']} days as a limit ('do not use for more than'), not a course length"
    return None


def _named(ingredient: str, text: str) -> bool:
    words = [w for w in re.split(r"[\s_]+", ingredient.lower()) if w and w not in _SALT_WORDS]
    return bool(words) and all(re.search(rf"\b{re.escape(w)}\b", text, re.I) for w in words)


def check_dosing(answer: dict, spl: str, sents: list[Sentence], facts=None) -> Checked:
    strengths, _ = facts if facts is not None else _label_facts(spl)
    by_id, out = {s.sid: s for s in sents}, Checked()
    for raw in answer.get("dosing", []) or []:
        s = by_id.get(str(raw.get("sentence", "")))
        if s is None:
            out.rejected.append({"pass": "dosing", "item": raw, "reason": "cites a sentence id that does not exist"})
            continue
        nums = {k: raw.get(k) for k in ("dose", "dose_max", "every_hours", "times_per_day", "duration_days")}
        bad = _grounded(nums, s)
        unit = str(raw.get("unit", "")).lower()
        problem = _strength_problem(raw.get("strength"), unit, s, strengths) or _dose_problem(raw, unit, s)
        if bad:
            out.rejected.append({"pass": "dosing", "item": raw, "reason": f"number(s) not stated in {s.sid}: {', '.join(bad)}"})
        elif raw.get("dose") is None or unit not in UNITS:
            out.rejected.append({"pass": "dosing", "item": raw, "reason": "needs a dose and a known unit"})
        elif problem:
            out.rejected.append({"pass": "dosing", "item": raw, "reason": problem})
        elif raw.get("provenance") not in ("quoted", "interpreted"):
            out.rejected.append({"pass": "dosing", "item": raw, "reason": "provenance must be quoted or interpreted"})
        else:
            row = {k: raw.get(k) for k in ("strength", "dose_type", "condition", "population", "dose", "dose_max", "unit",
                                            "every_hours", "times_per_day", "duration_days", "as_needed", "provenance")}
            row = {**row, "unit": unit, "source": _src(spl, s), "verified": False}
            if _dup(out.rules, "dosing", row):
                out.rejected.append({"pass": "dosing", "item": raw, "reason": "duplicate of an earlier row"})
            else:
                out.rules.append(row)
    return out


def _dup(rows: list[dict], kind: str, row: dict) -> bool:
    from .review_queue import signature
    sig = signature(kind, row)
    return any(signature(kind, r) == sig and r.get("population") == row.get("population")
               and r.get("condition") == row.get("condition") for r in rows)


def check_limits(answer: dict, spl: str, sents: list[Sentence], facts=None) -> Checked:
    strengths, substances = facts if facts is not None else _label_facts(spl)
    by_id, out = {s.sid: s for s in sents}, Checked()
    for raw in answer.get("limits", []) or []:
        s = by_id.get(str(raw.get("sentence", "")))
        if s is None:
            out.rejected.append({"pass": "limits", "item": raw, "reason": "cites a sentence id that does not exist"})
            continue
        bad = _grounded({"amount": raw.get("amount")}, s)
        unit, kind = str(raw.get("unit", "")).lower(), raw.get("kind")
        ing = str(raw.get("ingredient") or "").strip().lower()
        problem = _strength_problem(raw.get("strength"), unit, s, strengths)
        amt = raw.get("amount")
        if not problem and amt is not None and _per_kg(amt, unit, s.text):
            problem = f"{s.sid} gives {amt} per kg ('{amt} {unit}/kg'), which is a dose guide, not a flat limit"
        if not problem and amt is not None and _only_after(STEP_WORDS, amt, s.text):
            problem = f"{s.sid} states {amt} as an increase or titration step, not a limit"
        if not problem and unit in MASS_UNITS and len(substances) > 1:
            if not ing:
                problem = (f"the label has {len(substances)} active ingredients ({', '.join(substances)}): "
                           f"say which one this {raw.get('amount')} {unit} is of")
            elif not any(_named(ing, sub) or _named(sub, ing) for sub in substances):
                problem = f"ingredient '{ing}' is not an active ingredient of this label"
            elif not _named(ing, s.text):
                problem = f"ingredient '{ing}' is not named in {s.sid}"
        if bad:
            out.rejected.append({"pass": "limits", "item": raw, "reason": f"number not stated in {s.sid}: {', '.join(bad)}"})
        elif kind not in ("per_dose", "per_day", "duration_days") or unit not in UNITS or raw.get("amount") is None:
            out.rejected.append({"pass": "limits", "item": raw, "reason": "needs kind, amount and a known unit"})
        elif problem:
            out.rejected.append({"pass": "limits", "item": raw, "reason": problem})
        elif raw.get("provenance") not in ("quoted", "interpreted"):
            out.rejected.append({"pass": "limits", "item": raw, "reason": "provenance must be quoted or interpreted"})
        else:
            row = {"kind": kind, "amount": raw["amount"], "unit": unit, "ingredient": ing or None,
                   "strength": raw.get("strength"), "population": raw.get("population", "general"),
                   "condition": raw.get("condition"), "provenance": raw["provenance"], "source": _src(spl, s),
                   "verified": False}
            if _dup(out.rules, "limits", row):
                out.rejected.append({"pass": "limits", "item": raw, "reason": "duplicate of an earlier row"})
            else:
                out.rules.append(row)
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
