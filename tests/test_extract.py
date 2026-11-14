"""The deterministic gate around the model: whatever a model says, only rows
whose quote is real and whose numbers are on the label get through."""

from fractions import Fraction as F

from library.extract import Sentence, check, numbers_in

SPL = "spl-test-0001"
SENTS = [
    Sentence("S1", "dosage and administration",
             "The recommended oral dose is 60 mg (two 30 mg capsules) every 4 hours for 21 consecutive days."),
    Sentence("S2", "dosage and administration",
             "Patients should take other drugs at least one hour before or four hours after colestipol tablets."),
    Sentence("S3", "warnings", "Do not take more than 4,000 mg of acetaminophen in 24 hours."),
    Sentence("S4", "dosage and administration", "Take one-half to one hour before breakfast."),
]


def test_numbers_in_reads_digits_words_commas_and_minutes():
    assert {F(60), F(30), F(4), F(21), F(2)} <= numbers_in(SENTS[0].text)
    assert {F(1), F(4)} <= numbers_in(SENTS[1].text)
    assert F(4000) in numbers_in(SENTS[2].text)
    assert F(1, 2) in numbers_in(SENTS[3].text)
    assert F(1, 2) in numbers_in("at least 30 minutes before")  # 30 min -> 0.5 h


def test_good_rule_is_accepted_and_quote_is_copied_by_code():
    out = check({"rules": [{"id": "x-q4h", "family": "min_interval", "hours": 4, "sentence": "S1",
                            "provenance": "quoted"}]}, "x", SPL, SENTS)
    assert not out.rejected and len(out.rules) == 1
    r = out.rules[0]
    assert r["source"]["sentence"] == SENTS[0].text and r["verified"] is False


def test_invented_number_is_rejected():
    out = check({"rules": [{"id": "x-q6h", "family": "min_interval", "hours": 6, "sentence": "S1",
                            "provenance": "quoted"}]}, "x", SPL, SENTS)
    assert not out.rules and "not stated in S1" in out.rejected[0]["reason"]


def test_missing_sentence_and_bad_schema_are_rejected():
    out = check({"rules": [
        {"id": "x-a", "family": "min_interval", "hours": 4, "sentence": "S99", "provenance": "quoted"},
        {"id": "x-b", "family": "min_interval", "hours": 4, "sentence": "S1", "provenance": "interpreted"},
    ]}, "x", SPL, SENTS)
    reasons = [r["reason"] for r in out.rejected]
    assert len(out.rules) == 0
    assert any("does not exist" in r for r in reasons) and any("loosest" in r for r in reasons)


def test_separation_from_other_drugs_and_daily_max():
    out = check({"rules": [
        {"id": "c-other", "family": "separation", "target": "*", "before": 4, "after": 1,
         "sentence": "S2", "provenance": "quoted"},
        {"id": "a-max", "family": "daily_max", "amount": 4000, "unit": "mg", "sentence": "S3", "provenance": "quoted"},
    ]}, "x", SPL, SENTS)
    assert not out.rejected and len(out.rules) == 2


def test_membership_must_name_the_ingredient():
    out = check({"memberships": [
        {"group": "g", "ingredient": "acetaminophen", "sentence": "S3", "provenance": "quoted"},
        {"group": "g", "ingredient": "zinc", "sentence": "S3", "provenance": "quoted"},
    ]}, "x", SPL, SENTS)
    assert [m["ingredient"] for m in out.memberships] == ["acetaminophen"]
    assert "not named" in out.rejected[0]["reason"]


def test_every_other_drug_target_needs_general_wording():
    sents = SENTS + [Sentence("S5", "dosage and administration",
                              "Administer at least 4 hours before or after drugs known to interfere with absorption.")]
    out = check({"rules": [
        {"id": "l-all", "family": "separation", "target": "*", "before": 4, "after": 4, "sentence": "S5", "provenance": "quoted"},
        {"id": "c-all", "family": "separation", "target": "*", "before": 4, "after": 1, "sentence": "S2", "provenance": "quoted"},
    ]}, "x", SPL, sents)
    assert [r["id"] for r in out.rules] == ["c-all"]
    assert "every other drug" in out.rejected[0]["reason"]
