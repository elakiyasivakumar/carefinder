"""Phase 2 turns model text into a care level. Every unrecognised value used to
become level 2 (Retail Clinic) via `.get(setting, 2)` — including "ER".
"""
import pytest

from tests.test_gate import FakeLLM
from triage import AssessmentError, assess

VALID = """{
  "care_setting": "Urgent Care",
  "care_settings": ["Urgent Care"],
  "visit_steps": ["Ankle exam", "X-ray"],
  "likely_conditions": ["ankle sprain"],
  "labs": [],
  "interventions": ["Splinting"],
  "telehealth_coverage": "no",
  "reasoning": "Weight-bearing pain after a twist needs imaging.",
  "recommendation": "Go to urgent care for an x-ray."
}"""


def test_valid_assessment_is_parsed():
    result = assess(FakeLLM(VALID), 30, "Female", "twisted ankle")

    assert result.care_setting == "Urgent Care"
    assert result.care_level == 3
    assert result.interventions == ["Splinting"]
    assert result.telehealth_coverage == "no"


def test_er_is_never_downgraded_to_a_retail_clinic():
    """If Phase 2 says ER, the answer is not 'go to CVS'."""
    reply = '{"care_setting": "ER", "reasoning": "", "recommendation": ""}'

    with pytest.raises(AssessmentError):
        assess(FakeLLM(reply), 61, "Male", "chest pressure")


def test_unrecognised_care_setting_is_an_error_not_a_default():
    reply = '{"care_setting": "Space Station", "reasoning": "", "recommendation": ""}'

    with pytest.raises(AssessmentError):
        assess(FakeLLM(reply), 30, "Female", "headache")


def test_model_failure_is_an_error_not_a_retail_clinic():
    with pytest.raises(AssessmentError):
        assess(FakeLLM(error=RuntimeError("endpoint down")), 30, "Female", "headache")


def test_missing_optional_lists_default_to_empty():
    reply = """{
      "care_setting": "Home Care",
      "reasoning": "Self-limiting.",
      "recommendation": "Rest and fluids."
    }"""

    result = assess(FakeLLM(reply), 30, "Female", "mild cold")

    assert result.care_level == 1
    assert result.labs == []
    assert result.visit_steps == []


def test_symptoms_are_delimited_and_capped_in_the_prompt():
    """Patient text is data. It must not read as instructions, or crowd them out."""
    llm = FakeLLM(VALID)

    assess(llm, 30, "Female", "x" * 5000)

    prompt = llm.prompts[0]
    assert "<symptoms>" in prompt and "</symptoms>" in prompt

    reported = prompt.split("<symptoms>")[1].split("</symptoms>")[0].strip()
    assert len(reported) == 600
