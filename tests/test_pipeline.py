"""End-to-end routing: what run_triage does with each verdict, using fake deps."""
import pytest

from agents.schemas import FacilityInfo, TelehealthOption
from tests.test_gate import FakeLLM
from triage_cli import Deps, run_triage

URGENT = """{"care_setting": "Urgent Care", "care_settings": ["Urgent Care"],
 "visit_steps": ["Ankle exam", "X-ray"], "labs": [], "interventions": ["Splinting"],
 "telehealth_coverage": "no", "reasoning": "Needs imaging.",
 "recommendation": "Go to urgent care."}"""

HOME = """{"care_setting": "Home Care", "care_settings": ["Home Care"],
 "visit_steps": [], "labs": [], "interventions": [], "telehealth_coverage": "yes",
 "reasoning": "Self-limiting.", "recommendation": "Rest and fluids."}"""


class FakeMaps:
    def __init__(self):
        self.calls = []

    def find_facilities(self, zip_code, facility_type=None):
        self.calls.append((zip_code, facility_type))
        return [FacilityInfo(
            name="Aurora Urgent Care", address="1 Main St", place_id="p1",
            facility_type="Urgent Care", distance_miles=1.2,
        )]


class FakeCosts:
    def __init__(self):
        self.calls = []

    def get_facility_cost(self, name, facility_type, zip_code, procedures=None):
        self.calls.append({"name": name, "procedures": procedures})
        return {"cost": "$210", "cost_source": "perplexity"}


class FakeTelehealth:
    def get_options_for_care_level(self, care_level, force=False):
        return [TelehealthOption(
            name="Teladoc", conditions=[], estimated_cost="$75", url="https://x"
        )]


def build(reply, error=None):
    deps = Deps(
        llm=FakeLLM(reply, error), maps=FakeMaps(),
        costs=FakeCosts(), telehealth=FakeTelehealth(),
    )
    return deps


class SequencedLLM:
    """Answers the gate, then the assessment."""

    def __init__(self, *replies):
        self._replies = list(replies)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self._replies.pop(0)


def test_emergency_stops_immediately():
    deps = build('{"is_emergency": true}')

    result = run_triage(61, "M", "crushing chest pain", "10001", deps=deps)

    assert result["is_emergency"] is True
    assert deps.maps.calls == [], "no facility lookup should happen for a 911 case"
    assert deps.costs.calls == []


def test_gate_failure_does_not_route_to_a_clinic():
    """An unreachable model has cleared nobody — it must not produce a care plan."""
    deps = build(None, error=RuntimeError("endpoint down"))

    result = run_triage(61, "M", "chest pain", "10001", deps=deps)

    assert result.get("care_setting") is None
    assert "error" in result
    assert deps.costs.calls == []


def test_home_care_does_no_lookups():
    deps = build(None)
    deps.llm = SequencedLLM('{"is_emergency": false}', HOME)

    result = run_triage(30, "F", "mild cold", "10001", deps=deps)

    assert result["care_level"] == 1
    assert result["facilities"] == []
    assert deps.maps.calls == []
    assert deps.costs.calls == []


def test_invalid_zip_is_rejected_before_any_model_call():
    deps = build('{"is_emergency": false}')

    result = run_triage(30, "F", "cough", "not-a-zip", deps=deps)

    assert "error" in result
    assert deps.llm.prompts == []
