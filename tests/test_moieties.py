"""Element <-> salt matching (deterministic) and coverage triage."""

from pathlib import Path

from library.extract import Sentence, check
from library.load import load
from library.moieties import element_of, named_in
from library.passes import uncited

SPL = "spl-test-0003"
CATIONS = Sentence("S1", "drug interactions",
                   "Take at least two hours before or six hours after products containing calcium, iron, or zinc.")


def test_element_of_salts():
    assert element_of("calcium_carbonate") == "calcium"
    assert element_of("ferrous_sulfate") == "iron"
    assert element_of("colestipol") is None


def test_named_in_exact_vs_element():
    assert named_in("calcium_carbonate", CATIONS.text) == "element"
    assert named_in("zinc", CATIONS.text) == "exact"
    assert named_in("ferrous_sulfate", CATIONS.text) == "element"
    assert named_in("magnesium_oxide", CATIONS.text) is None
    assert named_in("calcium_carbonate", "Do not take with calciumless water") is None  # whole words only


def test_salt_membership_accepted_but_marked_interpreted():
    out = check({"memberships": [
        {"group": "cations", "ingredient": "calcium_carbonate", "sentence": "S1", "provenance": "quoted"},
        {"group": "cations", "ingredient": "magnesium_oxide", "sentence": "S1", "provenance": "quoted"},
    ]}, "ciprofloxacin", SPL, [CATIONS])
    [m] = out.memberships
    assert m["ingredient"] == "calcium_carbonate" and m["provenance"] == "interpreted"
    assert "calcium" in m["matched"]
    assert "not named" in out.rejected[0]["reason"]


def test_element_member_covers_every_salt_in_library():
    lib = load(Path("data"))
    lib.groups.groups["polyvalent_cations"].members[0] = lib.groups.groups["polyvalent_cations"].members[0].model_copy(
        update={"ingredient": "calcium"})
    assert "calcium_carbonate" in lib.group_members("polyvalent_cations")


def test_triage_puts_missed_timing_rules_first_and_background_last():
    sents = [
        Sentence("S1", "dosage and administration", "Take every 4 hours."),
        Sentence("S2", "dosage and administration", "Administer at least 4 hours before or after drugs known to interfere."),
        Sentence("S3", "precautions", "No carcinogenicity in rats at doses up to 70 mg/kg/day."),
        Sentence("S4", "dosage and administration", "Monitor TSH every 4 weeks."),
        Sentence("S5", "dosage and administration", "Take every 4 hours, as directed."),
        Sentence("S6", "dosage and administration", "Do not exceed 6 tablets in 24 hours."),
    ]
    left = uncited(sents, cited={"S1"})
    assert [(u["sentence_id"], u["kind"]) for u in left] == [
        ("S2", "possible_timing_rule"), ("S6", "review"), ("S4", "monitoring"),
        ("S5", "restates"), ("S3", "background")]
    assert left[3]["note"] == "repeats S1"
