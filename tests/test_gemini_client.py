"""The Flash client behind every orchestrator node.

Structure is guaranteed by the API (response_schema), not by parsing hopefully,
so the client's job is to pass the schema through and fail loudly when the call
does not come back usable.
"""
import pytest
from pydantic import BaseModel

from services.gemini_client import GeminiClient, GeminiError


class Answer(BaseModel):
    verdict: str
    score: int


class FakeModels:
    """Stands in for google-genai at the SDK boundary."""

    def __init__(self, text=None, error=None, parsed=None):
        self._text, self._error, self._parsed = text, error, parsed
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self._error:
            raise self._error
        return type("R", (), {"text": self._text, "parsed": self._parsed})()


def client(models):
    c = GeminiClient(project="p", location="global", model="gemini-3.6-flash")
    c._models = models
    return c


def test_structured_call_returns_a_validated_model():
    models = FakeModels(text='{"verdict": "ok", "score": 7}')
    out = client(models).structured("rate this", Answer)

    assert isinstance(out, Answer)
    assert out.verdict == "ok" and out.score == 7


def test_schema_is_handed_to_the_api_not_the_prompt():
    """The guarantee comes from response_schema; asking nicely in the prompt does not."""
    models = FakeModels(text='{"verdict": "ok", "score": 1}')
    client(models).structured("rate this", Answer)

    config = models.calls[0]["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_schema"] is Answer


def test_search_grounding_is_off_by_default():
    models = FakeModels(text='{"verdict": "ok", "score": 1}')
    client(models).structured("rate this", Answer)

    assert "tools" not in models.calls[0]["config"]


def test_search_grounding_can_be_enabled():
    """Pricing needs the live web; classification does not, and paying for it there is waste."""
    models = FakeModels(text='{"verdict": "ok", "score": 1}')
    client(models).structured("price this", Answer, search=True)

    assert models.calls[0]["config"].get("tools")


def test_api_failure_raises_rather_than_returning_empty():
    models = FakeModels(error=RuntimeError("503 overloaded"))

    with pytest.raises(GeminiError):
        client(models).structured("x", Answer)


def test_output_that_does_not_match_the_schema_raises():
    models = FakeModels(text='{"verdict": "ok"}')  # score missing

    with pytest.raises(GeminiError):
        client(models).structured("x", Answer)


def test_prose_around_the_json_is_tolerated():
    """Grounded calls sometimes prepend commentary despite the mime type."""
    models = FakeModels(text='Here you go:\n{"verdict": "ok", "score": 3}')

    assert client(models).structured("x", Answer).score == 3


def test_sdk_parsed_object_is_used_when_present():
    models = FakeModels(text=None, parsed=Answer(verdict="direct", score=9))

    assert client(models).structured("x", Answer).verdict == "direct"


def test_thinking_budget_is_passed_through():
    """Review is a verdict, not a deliberation — capped thinking keeps it cheap."""
    models = FakeModels(text='{"verdict": "ok", "score": 1}')
    client(models).structured("x", Answer, thinking_budget=0)

    assert models.calls[0]["config"]["thinking_config"]["thinking_budget"] == 0
