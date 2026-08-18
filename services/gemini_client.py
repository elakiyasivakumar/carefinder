"""Gemini Flash client for the orchestrator nodes.

Structure is enforced by the API: the schema is passed as `response_schema`, so
generation is constrained to it rather than us parsing and hoping. Search
grounding is opt-in per call, because only pricing needs the live web and
paying for a search on a classification call is waste.

Runs on Vertex rather than the Gemini API key: the key is free-tier and returns
RESOURCE_EXHAUSTED under eval load, which would fail mid-run.
"""
import json
import logging
import os
import re
from typing import Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_LOCATION = "global"


class GeminiError(Exception):
    """The call did not produce a usable, schema-valid answer.

    Raised rather than returning a partial object: a half-filled answer is
    indistinguishable from a real one downstream.
    """


class GeminiClient:
    """Structured, optionally search-grounded calls to Gemini Flash."""

    def __init__(
        self,
        project: Optional[str] = None,
        location: str = DEFAULT_LOCATION,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
    ):
        self.project = project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.location = location
        self.model = model
        self.temperature = temperature
        self._client = None
        self._models = None

    @property
    def models(self):
        """Build the SDK client on first use, not at import."""
        if self._models is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True, project=self.project, location=self.location
            )
            self._models = self._client.models
        return self._models

    def structured(
        self,
        prompt: str,
        schema: Type[T],
        search: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> T:
        """Return an instance of `schema`, or raise GeminiError."""
        config = {
            "temperature": self.temperature,
            "response_mime_type": "application/json",
            "response_schema": schema,
        }
        if search:
            config["tools"] = [{"google_search": {}}]
        if thinking_budget is not None:
            config["thinking_config"] = {"thinking_budget": thinking_budget}

        try:
            response = self.models.generate_content(
                model=self.model, contents=prompt, config=config
            )
        except Exception as e:
            raise GeminiError(f"Gemini call failed: {e}") from e

        parsed = getattr(response, "parsed", None)
        if isinstance(parsed, schema):
            return parsed

        text = getattr(response, "text", None)
        if not text:
            raise GeminiError("Gemini returned no content")

        try:
            return schema.model_validate(_first_json_object(text))
        except (ValidationError, ValueError) as e:
            raise GeminiError(f"Gemini output did not match {schema.__name__}: {e}") from e


def _first_json_object(text: str) -> dict:
    """Extract the JSON object from a response.

    response_mime_type should make this unnecessary, but grounded calls
    occasionally prepend commentary, and a stray sentence must not lose the answer.
    """
    text = text.strip()
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    fenced = re.findall(r"```(?:json)?\s*(.+?)```", text, re.S)
    for block in reversed(fenced):
        try:
            loaded = json.loads(block.strip())
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            continue

    start, end = text.find("{"), text.rfind("}") + 1
    if start != -1 and end > start:
        return json.loads(text[start:end])

    raise ValueError(f"no JSON object in response: {text[:160]!r}")
