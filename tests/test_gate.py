"""The emergency gate decides whether someone is told to call 911.

Its failure mode matters more than its accuracy: a gate that answers
"not an emergency" when it actually failed sends chest pain to a retail clinic.
"""
import pytest

from triage import GateVerdict, emergency_gate


class FakeLLM:
    """Stands in for the clinical model at the inference boundary."""

    def __init__(self, reply=None, error=None):
        self._reply = reply
        self._error = error
        self.prompts = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self._error:
            raise self._error
        return self._reply


def test_model_says_emergency():
    verdict = emergency_gate(FakeLLM('{"is_emergency": true}'), 61, "Male", "crushing chest pain")
    assert verdict is GateVerdict.EMERGENCY


def test_model_says_not_emergency():
    verdict = emergency_gate(FakeLLM('{"is_emergency": false}'), 24, "Female", "mild sore throat")
    assert verdict is GateVerdict.NOT_EMERGENCY


def test_model_failure_is_not_reported_as_safe():
    """An unreachable model has not cleared anybody. It must not answer 'no'."""
    verdict = emergency_gate(FakeLLM(error=RuntimeError("endpoint down")), 61, "Male", "chest pain")
    assert verdict is GateVerdict.UNKNOWN


def test_unparseable_reply_is_not_reported_as_safe():
    verdict = emergency_gate(FakeLLM("I cannot help with that."), 61, "Male", "chest pain")
    assert verdict is GateVerdict.UNKNOWN


def test_reply_missing_the_field_is_not_reported_as_safe():
    verdict = emergency_gate(FakeLLM('{"foo": "bar"}'), 61, "Male", "chest pain")
    assert verdict is GateVerdict.UNKNOWN
