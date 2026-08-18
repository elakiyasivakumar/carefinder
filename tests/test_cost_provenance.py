"""Every price shown to a patient must have been retrieved. None may be invented."""
import pytest
import requests

from services.perplexity_service import PerplexityService


def test_no_api_key_yields_no_price(monkeypatch):
    """Without a key nothing can be retrieved, so no dollar figure may be returned."""
    monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
    service = PerplexityService()

    result = service.get_facility_cost("CVS MinuteClinic", "Retail Clinic", "10001")

    assert result["cost"] is None
    assert result["cost_source"] == "unavailable"


def test_network_failure_yields_no_price(monkeypatch):
    """A live lookup that fails retrieved nothing, so it may not report a price."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    service = PerplexityService()

    def explode(*args, **kwargs):
        raise requests.exceptions.ConnectionError("network down")

    monkeypatch.setattr(requests, "post", explode)

    result = service.get_facility_cost("Local Urgent Care", "Urgent Care", "10001")

    assert result["cost"] is None
    assert result["cost_source"] == "unavailable"


def _stub_perplexity(monkeypatch, content):
    """Stand in for the Perplexity HTTP endpoint at the network boundary."""
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": content}}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: Response())


def test_retrieved_price_is_labelled_as_retrieved(monkeypatch):
    """A price that came back from a live search is attributed to that search."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    service = PerplexityService()
    _stub_perplexity(monkeypatch, "A visit typically runs $125-$185 for self-pay patients.")

    result = service.get_facility_cost("Aurora Urgent Care", "Urgent Care", "10001")

    assert result["cost"] == "$125–$185"
    assert result["cost_source"] == "perplexity"


def test_cost_query_asks_about_the_procedures_the_model_identified(monkeypatch):
    """Pricing 'an urgent care visit' ignores the clinical assessment entirely.
    The point of the system is to price what MedGemma said this patient needs.
    """
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    service = PerplexityService()

    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "About $210."}}]}

    def capture(url, **kwargs):
        sent.update(kwargs.get("json", {}))
        return Response()

    monkeypatch.setattr(requests, "post", capture)

    service.get_facility_cost(
        "Aurora Urgent Care", "Urgent Care", "10001",
        procedures=["ankle x-ray", "splinting"],
    )

    prompt = " ".join(m["content"] for m in sent["messages"])
    assert "ankle x-ray" in prompt
    assert "splinting" in prompt


def test_cost_query_works_without_procedures(monkeypatch):
    """Home Care and unknown-procedure cases still need a plain visit price."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    service = PerplexityService()
    _stub_perplexity(monkeypatch, "Typically $90.")

    result = service.get_facility_cost("Corner Clinic", "Retail Clinic", "10001")

    assert result["cost"] == "$90"


def test_retrieved_price_keeps_the_text_it_came_from(monkeypatch):
    """A price is only trustworthy if you can point at the sentence it came from."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    service = PerplexityService()
    _stub_perplexity(monkeypatch, "Self-pay visits run $125-$185 at this location.")

    result = service.get_facility_cost("Aurora Urgent Care", "Urgent Care", "10001")

    assert result["cost"] == "$125–$185"
    assert "125" in result["source_text"]
    assert result["verified_in_source"] is True


def test_price_not_present_in_the_source_is_flagged(monkeypatch):
    """If the figure we report cannot be found in what we retrieved, say so
    rather than presenting it as sourced.
    """
    from services.perplexity_service import PerplexityService as P

    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    service = P()
    _stub_perplexity(monkeypatch, "Pricing varies by service.")

    # Force an extraction that the source does not support.
    monkeypatch.setattr(service, "_extract_cost", lambda text: "$999")

    result = service.get_facility_cost("Aurora Urgent Care", "Urgent Care", "10001")

    assert result["verified_in_source"] is False


def test_answer_without_a_price_is_unavailable(monkeypatch):
    """A live answer that quotes no figure still leaves us with no price to show."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")
    service = PerplexityService()
    _stub_perplexity(monkeypatch, "Pricing is not published for this location.")

    result = service.get_facility_cost("Aurora Urgent Care", "Urgent Care", "10001")

    assert result["cost"] is None
    assert result["cost_source"] == "unavailable"
