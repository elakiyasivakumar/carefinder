"""Model output is text. Getting the object out of it is where runs silently break."""
import pytest

from triage import extract_json


def test_bare_json():
    assert extract_json('{"is_emergency": true}') == {"is_emergency": True}


def test_fenced_json():
    assert extract_json('```json\n{"care_setting": "Urgent Care"}\n```') == {
        "care_setting": "Urgent Care"
    }


def test_prose_before_the_object():
    raw = 'Here is my assessment.\n{"care_setting": "Home Care"}'
    assert extract_json(raw) == {"care_setting": "Home Care"}


def test_braces_in_the_reasoning_do_not_capture_the_wrong_span():
    """A {word} in the model's preamble used to swallow the real object."""
    raw = 'Reasoning {step 1} then the answer:\n{"care_setting": "Urgent Care"}'
    assert extract_json(raw) == {"care_setting": "Urgent Care"}


def test_fenced_scratchpad_does_not_beat_the_fenced_answer():
    """The prompt asks for a scratchpad; the answer is the last block, not the first."""
    raw = (
        "```\nthinking out loud about {the patient}\n```\n"
        '```json\n{"care_setting": "Retail Clinic"}\n```'
    )
    assert extract_json(raw) == {"care_setting": "Retail Clinic"}


def test_nested_objects_survive():
    raw = '{"care_setting": "Urgent Care", "meta": {"confidence": "high"}}'
    assert extract_json(raw)["meta"] == {"confidence": "high"}


def test_unclosed_fence_still_parses():
    assert extract_json('```json\n{"is_emergency": false}') == {"is_emergency": False}


def test_no_object_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        extract_json("I cannot help with that request.")


def test_truncated_object_raises_rather_than_returning_a_partial():
    """max_tokens cutoffs are real; half an assessment must not look like a whole one."""
    with pytest.raises(ValueError):
        extract_json('{"care_setting": "Urgent Care", "visit_steps": ["exam"')
