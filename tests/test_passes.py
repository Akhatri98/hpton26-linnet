"""Dosing / limits passes and the coverage check (all deterministic)."""

from library.extract import Sentence
from library.passes import check_dosing, check_limits, uncited

SPL = "spl-test-0002"
SENTS = [
    Sentence("S1", "dosage and administration", "Urinary Tract 250-500mg every 12 hours 7 to 14 days"),
    Sentence("S2", "dosage and administration", "take 2 caplets every 6 hours while symptoms last"),
    Sentence("S3", "warnings", "do not take more than 6 caplets in 24 hours, unless directed by a doctor"),
    Sentence("S4", "warnings", "Severe liver damage may occur if you take more than 4,000 mg of acetaminophen in 24 hours"),
    Sentence("S5", "dosage and administration", "Take twice daily with water."),
    Sentence("S6", "precautions", "Store at 20 to 25 degrees."),
]


def test_dosing_row_with_range_and_frequency():
    out = check_dosing({"dosing": [
        {"condition": "urinary tract", "population": "adults", "dose": 250, "dose_max": 500, "unit": "mg",
         "every_hours": 12, "duration_days": 14, "as_needed": False, "sentence": "S1", "provenance": "quoted"},
        # "Take twice daily" states a frequency but no dose: a dose of 1 would be invented.
        {"condition": "general", "dose": 1, "unit": "tablets", "times_per_day": 2, "sentence": "S5",
         "provenance": "quoted"},
    ]}, SPL, SENTS)
    assert len(out.rules) == 1 and "dose=1" in out.rejected[0]["reason"]
    assert out.rules[0]["source"]["sentence"] == SENTS[0].text and out.rules[0]["verified"] is False


def test_dosing_invented_frequency_rejected():
    out = check_dosing({"dosing": [{"condition": "x", "dose": 250, "unit": "mg", "every_hours": 8,
                                    "sentence": "S1", "provenance": "quoted"}]}, SPL, SENTS)
    assert not out.rules and "every_hours=8" in out.rejected[0]["reason"]


def test_limits_per_day_in_units_and_mg():
    out = check_limits({"limits": [
        {"kind": "per_day", "amount": 6, "unit": "caplets", "sentence": "S3", "provenance": "quoted"},
        {"kind": "per_day", "amount": 4000, "unit": "mg", "sentence": "S4", "provenance": "quoted"},
        {"kind": "per_day", "amount": 3000, "unit": "mg", "sentence": "S4", "provenance": "quoted"},
        {"kind": "weekly", "amount": 6, "unit": "caplets", "sentence": "S3", "provenance": "quoted"},
    ]}, SPL, SENTS)
    assert [r["amount"] for r in out.rules] == [6, 4000]
    assert len(out.rejected) == 2


def test_uncited_lists_numeric_timing_sentences_nobody_used():
    left = {u["sentence_id"] for u in uncited(SENTS, cited={"S2", "S3"})}
    assert left == {"S1", "S4"}  # S5 has no digit-number? 'twice' -> no; S6 not timing


# --- multi-strength and combination labels (acetaminophen/codeine) ---------------------
COMBO_FACTS = (["300 mg/15 mg", "300 mg/30 mg", "300 mg/60 mg"], ["acetaminophen", "codeine phosphate"])
CS = [
    Sentence("S1", "dosage and administration",
             "Acetaminophen and Codeine Phosphate Tablets (codeine 30 mg and acetaminophen 300 mg): Take 1 to 2 tablet every 4 hours as needed for pain."),
    Sentence("S2", "dosage and administration",
             "Single Doses (Range) Maximum 24-Hour Dose Codeine Phosphate 15 mg to 60 mg 360 mg Acetaminophen 300 mg to 1,000 mg 4,000 mg"),
]


def test_tablet_dose_on_multi_strength_label_must_name_its_strength():
    row = {"dose": 1, "dose_max": 2, "unit": "tablets", "every_hours": 4, "as_needed": True, "sentence": "S1", "provenance": "quoted"}
    out = check_dosing({"dosing": [row, {**row, "strength": "300 mg/30 mg"}, {**row, "strength": "300 mg/60 mg"}]},
                       SPL, CS, facts=COMBO_FACTS)
    assert [r["strength"] for r in out.rules] == ["300 mg/30 mg"]
    reasons = [r["reason"] for r in out.rejected]
    assert "say which one" in reasons[0]               # no strength at all
    assert "not stated in S1" in reasons[1]            # 60 mg is not what S1 is about


def test_combination_limit_must_name_its_ingredient():
    out = check_limits({"limits": [
        {"kind": "per_day", "amount": 360, "unit": "mg", "sentence": "S2", "provenance": "quoted"},
        {"kind": "per_day", "amount": 360, "unit": "mg", "ingredient": "codeine", "sentence": "S2", "provenance": "quoted"},
        {"kind": "per_day", "amount": 4000, "unit": "mg", "ingredient": "acetaminophen", "sentence": "S2", "provenance": "quoted"},
        {"kind": "per_day", "amount": 360, "unit": "mg", "ingredient": "ibuprofen", "sentence": "S2", "provenance": "quoted"},
    ]}, SPL, CS, facts=COMBO_FACTS)
    assert [(r["ingredient"], r["amount"]) for r in out.rules] == [("codeine", 360), ("acetaminophen", 4000)]
    assert "say which one" in out.rejected[0]["reason"] and "not an active ingredient" in out.rejected[1]["reason"]


def test_fable_findings_per_kg_maximum_and_titration_are_not_doses():
    sents = [
        Sentence("S1", "dosage and administration", "New onset hypothyroidism 1.6 mcg/kg/day"),
        Sentence("S2", "warnings", "if pregnant do not take more than 6 tablets in 24 hours"),
        Sentence("S3", "dosage and administration", "Increase levothyroxine sodium dosage by 12.5 to 25 mcg per day."),
        Sentence("S4", "dosage and administration", "Chew 2 to 4 tablets as symptoms occur."),
    ]
    rows = [
        {"dose": 1.6, "unit": "mcg", "times_per_day": None, "sentence": "S1", "provenance": "quoted"},
        {"dose": 6, "unit": "tablets", "sentence": "S2", "provenance": "quoted"},
        {"dose": 12.5, "dose_max": 25, "unit": "mcg", "sentence": "S3", "provenance": "quoted"},
        {"dose": 1.6, "unit": "mcg/kg", "sentence": "S1", "provenance": "quoted"},
        {"dose": 2, "dose_max": 4, "unit": "tablets", "sentence": "S4", "provenance": "quoted"},
        {"dose": 2, "dose_max": 4, "unit": "tablets", "sentence": "S4", "provenance": "quoted"},
    ]
    out = check_dosing({"dosing": rows}, SPL, sents, facts=([], []))
    assert [(r["dose"], r["unit"]) for r in out.rules] == [(1.6, "mcg/kg"), (2, "tablets")]
    reasons = [r["reason"] for r in out.rejected]
    assert "per kg" in reasons[0] and "maximum" in reasons[1] and "titration" in reasons[2] and "duplicate" in reasons[3]
