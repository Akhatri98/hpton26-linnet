"""Label text -> DRAFT library rows. Model proposes; deterministic code disposes.

    python -m library.extract prepare <ingredient> <spl_set_id>      write the prompt (files backend)
    python -m library.extract ingest  <ingredient> <spl_set_id>      check a model's answer, write drafts
    python -m library.extract run     <ingredient> <spl_set_id> --model M   (OpenRouter, with retries)

Pipeline, with the kind of each step:
  1. label record from data/labels/ or openFDA                      [deterministic]
  2. split sections into numbered sentences S1..Sn, keep timing-ish  [deterministic]
  3. model drafts rules + group memberships, citing sentence ids     [PROBABILISTIC]
  4. checks: sentence id exists, quote copied by code (never by the
     model), schema valid, every number in a rule appears in its
     cited sentence, ids unique                                      [deterministic]
  5. accepted rows -> data/drafts/<ingredient>.json, verified=false;
     rejected rows + reasons -> data/drafts/<ingredient>.report.json  [deterministic]
  6. a human approves                                                [human]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from pydantic import ValidationError

from .moieties import element_of, named_in
from .schema import IngredientFile, Membership

DATA = Path("data")
DRAFTS = DATA / "drafts"
SECTIONS = ["dosage_and_administration", "directions", "drug_interactions", "precautions",
            "warnings_and_cautions", "warnings", "boxed_warning", "do_not_use"]
TIMING = re.compile(
    r"\b(hours?|minutes?|before|after|apart|prior|daily|a day|per day|24 hours|maximum|exceed|"
    r"more than|meals?|breakfast|food|empty stomach|every|times|separat\w*|interval)\b", re.I)
MAX_SENTENCES = 60  # timing sentences shown to the administration pass
# Dosing and limits passes also see strength sections, and any sentence stating a dose amount
# (the timing filter alone hid sentences like "given as 30 mg capsules").
STRENGTH_SECTIONS = ["dosage_forms_and_strengths", "how_supplied", "description", "active_ingredient"]
DOSE = re.compile(r"\d[\d,.]*\s*(mg|mcg|g|grams?|mL|tablets?|caplets?|capsules?)\b|\b(mg|mcg)/kg\b", re.I)
MAX_DOSE_SENTENCES = 150
# A stated minimum gap, or as-needed use (where "every N hours" is a ceiling on frequency).
SPACING = re.compile(r"\bat least\b|\bapart\b|\bno (?:more|sooner)\b|\bnot (?:more|sooner)\b|\bminimum\b|"
                     r"\binterval\b|\bbetween doses\b|\bprior to the next\b", re.I)
AS_NEEDED = re.compile(r"\bas needed\b|\bif needed\b|\bwhile symptoms\b|\bas symptoms\b|\bPRN\b|\bfor pain\b", re.I)


# --- 1-2: label -> numbered sentences -----------------------------------------------

def label_record(spl: str) -> dict:
    f = DATA / "labels" / f"{spl}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    url = "https://api.fda.gov/drug/label.json?" + urllib.parse.urlencode({"search": f'set_id:"{spl}"', "limit": 1})
    with urllib.request.urlopen(url, timeout=30) as r:
        rec = json.load(r)["results"][0]
    keep = {k: rec[k] for k in ["set_id", "id", "version", "effective_time", "openfda", *SECTIONS,
                                 *STRENGTH_SECTIONS] if k in rec}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(keep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return keep


@dataclass
class Sentence:
    sid: str
    section: str
    text: str  # verbatim substring of the section text
    timing: bool = True  # shown to the administration (timing) pass; dosing/limits passes see all


def numbered_sentences(rec: dict) -> list[Sentence]:
    """One numbering for the whole label, so a sentence id means the same thing in every pass.
    Timing sentences come first (the administration pass sees those); then sentences that
    state a dose amount, including the strength sections (dosing and limits passes see all)."""
    out, seen, n_timing, n_dose = [], set(), 0, 0

    def add(part: str, sec: str, timing: bool):
        seen.add(part)
        out.append(Sentence(f"S{len(out) + 1}", sec.replace("_", " "), part, timing))

    for sec in SECTIONS + STRENGTH_SECTIONS:
        text = " ".join(rec.get(sec, []))
        parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", text)]
        for i, part in enumerate(parts):
            if len(part) < 12 or part in seen:
                continue
            is_timing = sec in SECTIONS and bool(TIMING.search(part)) and n_timing < MAX_SENTENCES
            is_dose = bool(DOSE.search(part)) and n_dose < MAX_DOSE_SENTENCES
            if not (is_timing or is_dose):
                continue
            assert part in text
            # The sentence before a dose often says who it is for ("Severely disturbed liver
            # function ... ."): show it too, so the dose is not read as applying to everyone.
            prev = parts[i - 1] if i > 0 else ""
            if is_dose and len(prev) >= 12 and prev not in seen:
                prev_timing = sec in SECTIONS and bool(TIMING.search(prev)) and n_timing < MAX_SENTENCES
                add(prev, sec, prev_timing)
                n_timing += prev_timing
                n_dose += not prev_timing
            add(part, sec, is_timing)
            n_timing += is_timing
            n_dose += not is_timing
    return out


# --- strengths, read from the label by code (never by the model) -------------------------

_COMBO = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*mg\s*/\s*(\d[\d,]*(?:\.\d+)?)\s*mg", re.I)
_AMT = r"(\d[\d,]*(?:\.\d+)?)\s*(mg|mcg|grams?|g)\b"
_SINGLE = [
    re.compile(r"strengths?\s*:?\s*((?:\d[\d,.]*\s*(?:mg|mcg|g)\b[\s,]*(?:(?:and|or)\s+)?)+)", re.I),       # "strengths: 25 mcg, 50 mcg"
    re.compile(r"available in\s+((?:\d[\d,.]*\s*(?:mg|mcg|g)\b[\s,]*(?:(?:and|or)\s+)?)+)", re.I),          # "available in 250 mg, 500 mg and 750 mg"
    re.compile(r"(?:tablets?|capsules?|caplets?)\s+(?:USP\s+)?" + _AMT, re.I),                         # "tablets USP 500 mg"
    re.compile(_AMT + r"\s+(?:film-coated\s+|soft gelatin\s+|chewable\s+)?(?:tablets?|capsules?|caplets?)\b", re.I),
    re.compile(r"\((?:in each|per) (?:tablet|caplet|capsule)\)\s*[A-Za-z ,]*?" + _AMT, re.I),         # OTC "Active ingredient (in each caplet) X 500 mg"
    re.compile(r"each (?:[a-z-]+ )*?(?:tablet|caplet|capsule) contains\s+" + _AMT, re.I),            # "Each tablet contains 1 gram"
]


def _norm_amount(num: str, unit: str) -> str:
    unit = unit.lower()
    unit = "g" if unit.startswith("gram") else unit
    return f"{num.replace(',', '')} {unit}"


def label_strengths(rec: dict) -> list[str]:
    """Strengths the label says it comes in, e.g. ["300 mg/15 mg", "300 mg/30 mg"] or ["250 mg", "500 mg"]."""
    text = " ".join(" ".join(rec.get(k, [])) for k in STRENGTH_SECTIONS)
    combos = {f"{a.replace(',', '')} mg/{b.replace(',', '')} mg" for a, b in _COMBO.findall(text)}
    if combos:
        return sorted(combos, key=lambda s: [float(x.split()[0]) for x in s.split("/")])
    found = set()
    for pat in _SINGLE:
        for m in pat.finditer(text):
            if pat.groups == 1:  # a list of amounts
                found |= {_norm_amount(a, u) for a, u in re.findall(_AMT, m.group(1), re.I)}
            else:
                found.add(_norm_amount(m.group(1), m.group(2)))
    return sorted(found, key=lambda s: float(s.split()[0]))


def label_substances(rec: dict) -> list[str]:
    return [s.lower() for s in (rec.get("openfda") or {}).get("substance_name", [])]


# --- 3: the prompt ---------------------------------------------------------------------

PROMPT = """You are extracting medication TIMING rules from an FDA drug label, for a scheduling tool used by pharmacists.

Ingredient: {ingredient}
Label set id: {spl}

Numbered sentences from the label:
{sentences}

Known ingredient groups (you may propose members for these, or a new group):
{groups}

Return ONLY a JSON object:
{{
  "rules": [ ... ],
  "memberships": [ ... ]
}}

Each rule is one of these shapes. Times are in HOURS (30 minutes = 0.5). Use only numbers stated in the cited sentence.
  {{"id": "...", "family": "min_interval", "hours": H, "sentence": "Sk", "provenance": "quoted"|"interpreted", "loosest": {{"hours": X}}?, "note": "..."}}
      any two doses of this ingredient at least H hours apart ("every 6 hours")
  {{"id": "...", "family": "separation", "target": "<ingredient>" | "group:<name>" | "*", "before": B, "after": A, "sentence": "Sk", ...}}
      THIS ingredient is taken at least B hours BEFORE and at least A hours AFTER any dose of the target.
      "*" means every other drug. Careful with direction: if the label says "take OTHER drugs 1 hour before
      or 4 hours after THIS drug", then THIS drug is 4 hours before and 1 hour after the other: before=4, after=1.
  {{"id": "...", "family": "daily_max", "amount": N, "unit": "mg"|"g"|"mcg", "sentence": "Sk", ...}}
  {{"id": "...", "family": "food", "mode": "empty_stomach", "before_meal": B, "after_meal": A, "meal": null|"breakfast", "sentence": "Sk", ...}}
      no dose from B hours before to A hours after a meal
  {{"id": "...", "family": "food", "mode": "before_meal_window", "meal": "breakfast", "earliest": E, "latest": L, "sentence": "Sk", ...}}
      dose between E and L hours before the meal (E >= L)

provenance: "quoted" if the number is stated plainly and firmly; "interpreted" if you had to judge
(soft words like "preferably", "if possible", "consider", a range, or a derived number). Every
"interpreted" rule MUST give "loosest": the most permissive defensible value of each number you judged.

Each membership: {{"group": "<name>", "ingredient": "<lowercase_underscore_name>", "sentence": "Sk", "provenance": "quoted"|"interpreted"}}
  Propose a membership when the label names an ingredient as one of the things to separate from.

Ids: lowercase letters, digits, dashes, starting with the ingredient name. Cite exactly one sentence id per item.
Do not copy sentence text; cite its id. Only include rules the label actually states. If unsure, leave it out.
"""


def build_prompt(ingredient: str, spl: str, sents: list[Sentence], groups: dict) -> str:
    lines = "\n".join(f"{s.sid} [{s.section}] {s.text}" for s in sents if s.timing)
    g = "\n".join(f"- {name}: {sorted(m['ingredient'] for m in body['members'])}"
                  for name, body in groups.items()) or "(none)"
    return PROMPT.format(ingredient=ingredient, spl=spl, sentences=lines, groups=g)


# --- 4: deterministic checks ----------------------------------------------------------

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "twenty-four": 24, "thirty": 30,
         "half": Fraction(1, 2), "one-half": Fraction(1, 2)}
ANY_OTHER = re.compile(r"\b(other|any|all)\s+(oral\s+)?(drugs?|medications?|medicines?)\b", re.I)
PARAMS = ("hours", "before", "after", "amount", "before_meal", "after_meal", "earliest", "latest", "within")


def numbers_in(text: str) -> set[Fraction]:
    """Every number the sentence states, plus each read as minutes -> hours."""
    found: set[Fraction] = set()
    for m in re.finditer(r"\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", text):
        tok = m.group().replace(",", "")
        found.add(Fraction(tok))
    for w, v in WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", text, re.I):
            found.add(Fraction(v))
    return found | {n / 60 for n in found}


@dataclass
class Checked:
    rules: list[dict] = field(default_factory=list)
    memberships: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)  # {"item": ..., "reason": ...}


def check(answer: dict, ingredient: str, spl: str, sents: list[Sentence]) -> Checked:
    by_id = {s.sid: s for s in sents}
    url = f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={spl}"
    out, ids = Checked(), set()

    for raw in answer.get("rules", []) or []:
        item = dict(raw)
        s = by_id.get(str(item.pop("sentence", "")))
        if s is None:
            out.rejected.append({"item": raw, "reason": "cites a sentence id that does not exist"})
            continue
        item["source"] = {"spl_set_id": spl, "url": url, "section": s.section, "sentence": s.text}
        item["verified"] = False
        item["note"] = (str(item.get("note", "")) + " | DRAFT: proposed by a language model.").strip(" |")
        for k in [k for k, v in item.items() if v is None]:
            del item[k]
        stated = numbers_in(s.text)
        ungrounded = [f"{k}={item[k]}" for k in PARAMS
                      if k in item and Fraction(str(item[k])) != 0 and Fraction(str(item[k])) not in stated]
        if ungrounded:
            out.rejected.append({"item": raw, "reason": f"number(s) not stated in {s.sid}: {', '.join(ungrounded)}"})
            continue
        if item.get("target") == "*" and not ANY_OTHER.search(s.text):
            out.rejected.append({"item": raw, "reason": f"target '*' (every other drug) but {s.sid} does not "
                                 "refer to other drugs in general; use a specific ingredient or group"})
            continue
        if item.get("family") == "min_interval" and not (SPACING.search(s.text) or AS_NEEDED.search(s.text)):
            out.rejected.append({"item": raw, "reason": f"{s.sid} gives 'every {item.get('hours')} hours' as a regimen, not a "
                                 "minimum gap or an as-needed limit; the prescription sets the regimen"})
            continue
        if item.get("family") == "daily_max" and str(item.get("unit", "")).lower() not in ("mg", "mcg", "g"):
            out.rejected.append({"item": raw, "reason": f"a daily maximum in {item.get('unit')} counts one product's units; "
                                 "an ingredient maximum must be in mg, mcg or g (product caps belong on the product)"})
            continue
        if item.get("id") in ids:
            out.rejected.append({"item": raw, "reason": "duplicate id"})
            continue
        try:
            IngredientFile.model_validate({"ingredient": ingredient, "primary_spl": spl, "rules": [item]})
        except ValidationError as e:
            out.rejected.append({"item": raw, "reason": "schema: " + "; ".join(x["msg"] for x in e.errors())})
            continue
        ids.add(item["id"])
        out.rules.append(item)

    for raw in answer.get("memberships", []) or []:
        s = by_id.get(str(raw.get("sentence", "")))
        if s is None:
            out.rejected.append({"item": raw, "reason": "membership cites a missing sentence id"})
            continue
        m = {"ingredient": str(raw.get("ingredient", "")).lower(), "provenance": raw.get("provenance"),
             "basis": s.text, "source_spl": spl, "verified": False}
        try:
            Membership.model_validate(m)
        except ValidationError as e:
            out.rejected.append({"item": raw, "reason": "schema: " + "; ".join(x["msg"] for x in e.errors())})
            continue
        how = named_in(m["ingredient"], s.text)
        if how is None:
            out.rejected.append({"item": raw, "reason": f"ingredient '{m['ingredient']}' not named in {s.sid}"})
            continue
        if how == "element":
            # The label names only the element; the salt link is our inference.
            m["provenance"] = "interpreted"
            m["matched"] = f"label names '{element_of(m['ingredient'])}'; {m['ingredient']} is a salt of it"
        out.memberships.append({"group": str(raw.get("group", "")), **m})
    return out


def write_drafts(ingredient: str, spl: str, checked: Checked, model: str) -> Path:
    DRAFTS.mkdir(parents=True, exist_ok=True)
    draft = {"ingredient": ingredient, "primary_spl": spl, "rules": checked.rules}
    IngredientFile.model_validate(draft)  # the whole file must also be valid
    (DRAFTS / f"{ingredient}.json").write_text(json.dumps(draft, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {"model": model, "accepted_rules": len(checked.rules), "accepted_memberships": checked.memberships,
              "rejected": checked.rejected}
    p = DRAFTS / f"{ingredient}.report.json"
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


# --- CLI ---------------------------------------------------------------------------------

def _groups() -> dict:
    g = DATA / "groups.json"
    return json.loads(g.read_text(encoding="utf-8"))["groups"] if g.exists() else {}


PASSES = ("administration", "dosing", "limits")


def _prompt_for(pass_name: str, ingredient: str, spl: str, sents: list[Sentence]) -> str:
    from .passes import DOSING_PROMPT, LIMITS_PROMPT
    if pass_name == "administration":
        return build_prompt(ingredient, spl, sents, _groups())
    lines = "\n".join(f"{s.sid} [{s.section}] {s.text}" for s in sents)
    rec = label_record(spl)
    tpl = DOSING_PROMPT if pass_name == "dosing" else LIMITS_PROMPT
    return tpl.format(ingredient=ingredient, spl=spl, sentences=lines,
                      strengths=", ".join(label_strengths(rec)) or "(none found)",
                      substances=", ".join(label_substances(rec)) or ingredient)


def _check_pass(pass_name: str, answer: dict, ingredient: str, spl: str, sents: list[Sentence]) -> Checked:
    from .passes import check_dosing, check_limits
    if pass_name == "administration":
        return check(answer, ingredient, spl, sents)
    return (check_dosing if pass_name == "dosing" else check_limits)(answer, spl, sents)


def write_record(ingredient: str, spl: str, sents: list[Sentence], results: dict, model: str) -> Path:
    """One JSON record per label: every pass, the sentences, and what nobody cited."""
    from .passes import uncited
    text_to_sid = {s.text: s.sid for s in sents}
    cited = set()
    for name, c in results.items():
        for r in c.rules:
            cited.add(r["source"].get("sentence_id") or text_to_sid.get(r["source"]["sentence"], ""))
        for m in c.memberships:
            cited.add(text_to_sid.get(m["basis"], ""))
    label = label_record(spl)
    rec = {
        "ingredient": ingredient, "spl_set_id": spl, "model": model,
        "label_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={spl}",
        "strengths": label_strengths(label),      # read by code, not by the model
        "substances": label_substances(label),
        "dosing": results["dosing"].rules if "dosing" in results else [],
        "limits": results["limits"].rules if "limits" in results else [],
        "administration": results["administration"].rules if "administration" in results else [],
        "memberships": results["administration"].memberships if "administration" in results else [],
        "rejected": [r for c in results.values() for r in c.rejected],
        "unaccounted_for": uncited(sents, cited),
        "sentences": [{"sentence_id": s.sid, "section": s.section, "text": s.text} for s in sents],
        "passes_run": sorted(results),
    }
    out = DATA / "extracted" / f"{ingredient}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():  # keep human decisions on rows that came out the same this time
        from .review_queue import carry_reviews, write_json
        carry_reviews(json.loads(out.read_text(encoding="utf-8")), rec)
        write_json(out, rec)
    else:
        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    if "administration" in results:  # keep the administration draft + report used by library.evaluate
        write_drafts(ingredient, spl, results["administration"], model)
    return out


def main(argv=None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="library.extract")
    ap.add_argument("cmd", choices=["prepare", "ingest", "run"])
    ap.add_argument("ingredient")
    ap.add_argument("spl")
    ap.add_argument("--model", default="")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--passes", nargs="+", choices=PASSES, default=list(PASSES))
    args = ap.parse_args(argv)

    sents = numbered_sentences(label_record(args.spl))
    pdir, rdir = DRAFTS / "prompts", DRAFTS / "responses"

    if args.cmd == "prepare":
        pdir.mkdir(parents=True, exist_ok=True)
        for ps in args.passes:
            f = pdir / f"{args.ingredient}.{ps}.txt"
            f.write_text(_prompt_for(ps, args.ingredient, args.spl, sents), encoding="utf-8")
            print(f"wrote {f}  ->  answer goes in {rdir / f'{args.ingredient}.{ps}.json'}")
        return 0

    results = {}
    for ps in args.passes:
        prompt = _prompt_for(ps, args.ingredient, args.spl, sents)
        if args.cmd == "ingest":
            f = rdir / f"{args.ingredient}.{ps}.json"
            if not f.exists():
                print(f"(no answer for pass {ps}: {f})")
                continue
            results[ps] = _check_pass(ps, json.loads(f.read_text(encoding="utf-8")), args.ingredient, args.spl, sents)
        else:
            from .llm import DEFAULT_MODEL, complete
            args.model = args.model or DEFAULT_MODEL
            save = rdir / re.sub(r"[^A-Za-z0-9._-]", "_", args.model)  # raw answers, one folder per model
            save.mkdir(parents=True, exist_ok=True)
            p, c = prompt, None
            for attempt in range(args.retries + 1):
                print(f"  {args.ingredient} / {ps}  attempt {attempt + 1}")
                raw = complete(p, args.model)
                (save / f"{args.ingredient}.{ps}.json").write_text(raw + "\n", encoding="utf-8")
                try:
                    answer = json.loads(raw)
                except json.JSONDecodeError as e:
                    p = prompt + f"\n\nYour previous answer was not valid JSON ({e}). Reply with one JSON object only."
                    continue
                c = _check_pass(ps, answer, args.ingredient, args.spl, sents)
                if not c.rejected:
                    break
                p = (prompt + "\n\nYour previous answer had these problems; fix them and answer again:\n"
                     + "\n".join(f"- {r['reason']}: {json.dumps(r['item'])}" for r in c.rejected))
            if c is not None:
                results[ps] = c
    model = args.model or "files backend"
    rec = write_record(args.ingredient, args.spl, sents, results, model)
    data = json.loads(rec.read_text(encoding="utf-8"))
    print(f"{args.ingredient}: dosing {len(data['dosing'])}, limits {len(data['limits'])}, "
          f"administration {len(data['administration'])}, memberships {len(data['memberships'])}, "
          f"rejected {len(data['rejected'])}, unaccounted-for sentences {len(data['unaccounted_for'])}  ->  {rec}")
    if args.cmd == "run":
        from .llm import SPENT
        print(f"spent this run: {SPENT['calls']} calls, {SPENT['input_tokens']} in / {SPENT['output_tokens']} out tokens, "
              f"~${SPENT['usd']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
