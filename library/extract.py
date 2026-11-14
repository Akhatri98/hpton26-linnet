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
MAX_SENTENCES = 60


# --- 1-2: label -> numbered sentences -----------------------------------------------

def label_record(spl: str) -> dict:
    f = DATA / "labels" / f"{spl}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    url = "https://api.fda.gov/drug/label.json?" + urllib.parse.urlencode({"search": f'set_id:"{spl}"', "limit": 1})
    with urllib.request.urlopen(url, timeout=30) as r:
        rec = json.load(r)["results"][0]
    keep = {k: rec[k] for k in ["set_id", "id", "version", "effective_time", "openfda", *SECTIONS,
                                 "active_ingredient", "description"] if k in rec}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(keep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return keep


@dataclass
class Sentence:
    sid: str
    section: str
    text: str  # verbatim substring of the section text


def numbered_sentences(rec: dict) -> list[Sentence]:
    out, seen = [], set()
    for sec in SECTIONS:
        text = " ".join(rec.get(sec, []))
        for part in re.split(r"(?<=[.;])\s+", text):
            part = part.strip()
            if len(part) < 12 or part in seen or not TIMING.search(part):
                continue
            assert part in text
            seen.add(part)
            out.append(Sentence(f"S{len(out) + 1}", sec.replace("_", " "), part))
            if len(out) >= MAX_SENTENCES:
                return out
    return out


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
    lines = "\n".join(f"{s.sid} [{s.section}] {s.text}" for s in sents)
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
    tpl = DOSING_PROMPT if pass_name == "dosing" else LIMITS_PROMPT
    return tpl.format(ingredient=ingredient, spl=spl, sentences=lines)


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
    rec = {
        "ingredient": ingredient, "spl_set_id": spl, "model": model,
        "label_url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={spl}",
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
