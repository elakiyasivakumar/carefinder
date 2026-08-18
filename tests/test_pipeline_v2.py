"""run_triage with the orchestrator graph wired in."""
import pytest

from agents.contracts import ReviewVerdict, TelehealthOption
from orchestrator import FacilityShortlist, PricedOption, RejectedFacility
from tests.test_gate import FakeLLM
from triage_cli import Deps, run_triage

URGENT = """{"care_setting":"Urgent Care","care_settings":["Urgent Care"],
 "visit_steps":["Vitals"],"labs":["influenza PCR"],"interventions":[],
 "likely_conditions":["influenza"],"telehealth_coverage":"maybe",
 "reasoning":"Systemic viral illness.","recommendation":"Go to urgent care."}"""

FOUND = [{"name": "K+STAT Urgent Care", "facility_type": "Urgent Care",
          "distance_miles": 1.2, "latitude": 39.19, "longitude": -96.57,
          "place_id": "osm-1", "address": "1 Main St"}]


class SequencedLLM:
    def __init__(self, *replies):
        self._replies = list(replies)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self._replies.pop(0)


class ScriptedGemini:
    def __init__(self, *answers):
        self._answers = list(answers)

    def structured(self, prompt, schema, search=False, thinking_budget=None):
        return self._answers.pop(0)


class ExplodingGemini:
    def structured(self, *a, **k):
        raise RuntimeError("orchestrator is down")


class FakeMaps:
    def find_facilities(self, zip_code, facility_type=None):
        return [type("F", (), f)() for f in []] or _as_objects(FOUND)


def _as_objects(rows):
    from agents.schemas import FacilityInfo
    return [FacilityInfo(name=r["name"], address=r["address"], place_id=r["place_id"],
                         facility_type=r["facility_type"], distance_miles=r["distance_miles"],
                         latitude=r["latitude"], longitude=r["longitude"]) for r in rows]


class FakeTelehealth:
    def get_options_for_care_level(self, care_level, force=False):
        return [TelehealthOption(name="Amazon Clinic", estimated_cost="$35–75", url="x")]


def deps(gate_and_assess, gemini):
    return Deps(llm=gate_and_assess, maps=FakeMaps(), costs=None,
                telehealth=FakeTelehealth(), gemini=gemini)


def test_full_pipeline_returns_places_and_priced_bands():
    d = deps(
        SequencedLLM('{"is_emergency": false}', URGENT),
        ScriptedGemini(
            FacilityShortlist(keep_place_ids=["osm-1"], rejected=[]),
            PricedOption(low_usd=125, high_usd=275, basis="visit plus flu test"),
            type("S", (), {"picks": [{"name": "Amazon Clinic", "why": "flu is treatable online"}]})(),
            ReviewVerdict(complete=True),
        ),
    )

    result = run_triage(22, "F", "high fever 103F", "66502", deps=d)

    assert result["care_setting"] == "Urgent Care"
    assert result["facilities"][0]["cost"] == "$125–$275"
    assert result["facilities"][0]["latitude"] == 39.19
    assert result["online_providers"][0]["name"] == "Amazon Clinic"


def test_emergency_still_answers_when_the_orchestrator_is_dead():
    """The gate must never be swallowed by an orchestrator failure — this is the
    one path where a hidden error is dangerous."""
    d = deps(FakeLLM('{"is_emergency": true}'), ExplodingGemini())

    result = run_triage(55, "M", "crushing chest pain", "66502", deps=d)

    assert result["is_emergency"] is True
    assert "error" not in result


def test_orchestrator_failure_degrades_but_still_names_the_care_level():
    """If Flash dies after the gate, the clinical answer survives."""
    d = deps(SequencedLLM('{"is_emergency": false}', URGENT), ExplodingGemini())

    result = run_triage(22, "F", "high fever", "66502", deps=d)

    assert result["care_setting"] == "Urgent Care"
    assert result.get("is_emergency") is False


def test_review_notes_are_surfaced_for_evaluation():
    d = deps(
        SequencedLLM('{"is_emergency": false}', URGENT),
        ScriptedGemini(
            FacilityShortlist(keep_place_ids=[], rejected=[RejectedFacility(place_id="osm-1", reason="chiropractor, not acute care")]),
            type("S", (), {"picks": [{"name": "Amazon Clinic", "why": "nothing nearby; treatable online"}]})(),
            ReviewVerdict(complete=True),
        ),
    )

    result = run_triage(22, "F", "high fever", "66502", deps=d)

    assert any("chiropractor" in n for n in result["review_notes"])
