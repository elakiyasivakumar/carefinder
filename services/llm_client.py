"""Clinical model transport.

The Vertex SDK is imported lazily inside the client so the rest of the app —
and its tests — do not need the cloud SDK or credentials to be importable.
"""
import json
import logging
from typing import Any, List

logger = logging.getLogger(__name__)

# Keys that serving containers commonly use for the generated text.
TEXT_KEYS = ("generated_text", "content", "text", "output", "prediction")


def prediction_to_text(predictions: List[Any]) -> str:
    """Flatten a Vertex prediction payload to the model's text output.

    Containers return a string, a dict keyed by one of TEXT_KEYS, a dict that
    *is* the structured answer, or any of those nested one level deep.
    """
    if not predictions:
        return ""
    return _coerce(predictions[0])


def _coerce(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _coerce(value[0]) if value else ""
    if isinstance(value, dict):
        for key in TEXT_KEYS:
            if key in value:
                return _coerce(value[key])
        # The dict is the answer itself. Emit JSON, never str() — a Python repr
        # uses single quotes and True/False and will never parse.
        return json.dumps(value)
    return str(value)


def apply_chat_template(prompt: str) -> str:
    """Wrap a prompt in Gemma turn markers.

    MedGemma-*-it is instruction-tuned. Sent as a raw completion prompt it
    emits EOS immediately and returns an empty string — a silent zero-token
    reply that looks like a parse failure rather than a formatting mistake.
    """
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def strip_echoed_prompt(text: str, prompt: str) -> str:
    """Drop the prompt from a completion that echoes it back.

    vLLM's /generate route returns prompt + completion. Both triage prompts
    contain JSON format examples, so an un-stripped echo lets the parser read
    the example instead of the model's answer.
    """
    if prompt and text.startswith(prompt):
        return text[len(prompt):].strip()

    # Model Garden's vLLM container wraps replies as "Prompt:\n<prompt>\nOutput:\n<answer>".
    # Everything before the final "Output:" is the echoed prompt, which carries
    # the JSON format examples.
    if text.lstrip().startswith("Prompt:") and "Output:" in text:
        return text.rsplit("Output:", 1)[1].strip()

    return text


class VertexLLMClient:
    """Calls a deployed MedGemma endpoint on Vertex AI."""

    def __init__(
        self,
        project_id: str,
        region: str,
        endpoint_id: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        top_p: float = 0.95,
    ):
        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT is not set")
        if not endpoint_id:
            raise ValueError("MEDGEMMA_ENDPOINT_ID is not set")

        self.project_id = project_id
        self.region = region
        self.endpoint_id = endpoint_id
        # Model Garden vLLM containers read sampling params from the instance.
        # Sent as a separate `parameters` dict they are silently ignored, and
        # the endpoint quietly runs at its own defaults.
        self.sampling = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        self._endpoint = None

    @property
    def endpoint(self):
        """Initialise Vertex on first use, not at import."""
        if self._endpoint is None:
            from google.cloud import aiplatform

            aiplatform.init(project=self.project_id, location=self.region)
            self._endpoint = aiplatform.Endpoint(
                endpoint_name=(
                    f"projects/{self.project_id}/locations/{self.region}"
                    f"/endpoints/{self.endpoint_id}"
                )
            )
        return self._endpoint

    def complete(self, prompt: str) -> str:
        wrapped = apply_chat_template(prompt)
        response = self.endpoint.predict(
            instances=[{"prompt": wrapped, **self.sampling}]
        )
        return strip_echoed_prompt(prediction_to_text(response.predictions), wrapped)
