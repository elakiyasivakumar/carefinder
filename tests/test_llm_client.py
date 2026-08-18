"""Vertex endpoints return predictions in several shapes depending on the
serving container. `str()` on a dict yields a Python repr with single quotes,
which no JSON parser will ever accept — so every reply would look malformed.
"""
import pytest

from services.llm_client import prediction_to_text


def test_plain_string_prediction():
    assert prediction_to_text(['{"is_emergency": true}']) == '{"is_emergency": true}'


def test_dict_prediction_with_generated_text():
    predictions = [{"generated_text": '{"is_emergency": false}'}]
    assert prediction_to_text(predictions) == '{"is_emergency": false}'


def test_dict_prediction_with_content_key():
    predictions = [{"content": "Output: yes"}]
    assert prediction_to_text(predictions) == "Output: yes"


def test_nested_list_prediction():
    predictions = [[{"generated_text": "nested reply"}]]
    assert prediction_to_text(predictions) == "nested reply"


def test_structured_prediction_without_a_text_key_is_kept_as_json():
    """A dict that *is* the answer must survive as parseable JSON, not a repr."""
    import json

    predictions = [{"is_emergency": True}]

    result = prediction_to_text(predictions)

    assert json.loads(result) == {"is_emergency": True}


def test_empty_predictions():
    assert prediction_to_text([]) == ""


def test_echoed_prompt_is_stripped():
    """vLLM's /generate returns prompt + completion. The gate prompt itself
    contains {"is_emergency": true} as a format example — left in, the parser
    can read the example instead of the model's actual answer.
    """
    from services.llm_client import strip_echoed_prompt

    prompt = 'Respond with ONLY this JSON:\n{"is_emergency": true} or {"is_emergency": false}'
    raw = prompt + '\n{"is_emergency": false}'

    assert strip_echoed_prompt(raw, prompt) == '{"is_emergency": false}'


def test_response_without_an_echo_is_untouched():
    from services.llm_client import strip_echoed_prompt

    assert strip_echoed_prompt('{"is_emergency": true}', "some prompt") == '{"is_emergency": true}'


def test_sampling_params_travel_inside_the_instance():
    """Model Garden vLLM containers read sampling params from the instance.
    Passing them as a separate `parameters` dict silently does nothing.
    """
    from services.llm_client import VertexLLMClient

    client = VertexLLMClient("proj", "us-central1", "123", temperature=0.1, max_tokens=512)

    captured = {}

    class FakeEndpoint:
        def predict(self, instances, **kwargs):
            captured["instances"] = instances
            captured["kwargs"] = kwargs
            class R:
                predictions = ['{"ok": true}']
            return R()

    client._endpoint = FakeEndpoint()
    client.complete("hello")

    instance = captured["instances"][0]
    assert instance["temperature"] == 0.1
    assert instance["max_tokens"] == 512
    assert captured["kwargs"] == {}, "params must not be sent as a separate dict"


def test_prompt_is_wrapped_in_gemma_turn_markers():
    """MedGemma-*-it returns an empty string for un-templated prompts."""
    from services.llm_client import apply_chat_template

    wrapped = apply_chat_template("hello")

    assert wrapped.startswith("<start_of_turn>user\n")
    assert wrapped.endswith("<end_of_turn>\n<start_of_turn>model\n")
    assert "hello" in wrapped


def test_vllm_prompt_output_wrapper_is_stripped():
    """The serving container replies 'Prompt:\\n<echo>\\nOutput:\\n<answer>'."""
    from services.llm_client import strip_echoed_prompt

    raw = 'Prompt:\nRespond with {"is_emergency": true} or {"is_emergency": false}\nOutput:\n{"is_emergency": false}'

    assert strip_echoed_prompt(raw, "unrelated") == '{"is_emergency": false}'
