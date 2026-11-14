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
