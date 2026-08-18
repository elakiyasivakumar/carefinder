"""The Pydantic contracts every graph node reads and writes.

These double as Gemini response_schema definitions, so a field the model
cannot fill is a field the schema should not require.
"""
import pytest
from pydantic import ValidationError

from agents.contracts import (
    CareOption,
    PriceEstimate,
    ReviewVerdict,
    TriageAnswer,
)


def test_price_estimate_keeps_the_whole_band():
    """The old extractor reported $350 from '$350-$700+', showing the floor
    of a wide band as if it were the estimate."""
    p = PriceEstimate(low_usd=125, high_usd=275, basis="base visit", source="gemini_search")

    assert p.low_usd == 125 and p.high_usd == 275
    assert p.display == "$125–$275"


def test_single_number_price_displays_once():
    p = PriceEstimate(low_usd=90, high_usd=90, basis="flat fee", source="gemini_search")

    assert p.display == "$90"


def test_price_rejects_an_inverted_band():
    with pytest.raises(ValidationError):
        PriceEstimate(low_usd=300, high_usd=100, basis="x", source="gemini_search")


def test_unpriced_option_is_allowed_and_says_so():
    """No price retrieved is a valid answer. An invented one is not."""
    o = CareOption(name="Tonopah Primary Care", facility_type="Walk-in Clinic",
                   distance_miles=1.5, latitude=38.06, longitude=-117.23,
                   place_id="osm-node-1")

    assert o.price is None
    assert o.price_display == "price not available"


def test_care_option_rejects_an_unknown_facility_type():
    with pytest.raises(ValidationError):
        CareOption(name="X", facility_type="Chiropractor", distance_miles=1.0,
                   latitude=1.0, longitude=1.0, place_id="p")


def test_review_verdict_can_demand_a_retry_with_a_reason():
    """The review node's requests are the eval instrument, so the reason is required."""
    v = ReviewVerdict(complete=False, missing=["no price on any option"],
                      retry_tool="pricing")

    assert v.complete is False
    assert v.retry_tool == "pricing"
    assert v.missing


def test_review_verdict_complete_needs_no_retry():
    v = ReviewVerdict(complete=True)

    assert v.retry_tool is None
    assert v.missing == []


def test_triage_answer_carries_the_three_promised_endpoints():
    """Care level, where to go, and what it costs — the product's promise."""
    a = TriageAnswer(
        care_setting="Urgent Care",
        care_level=3,
        reasoning="Needs imaging.",
        recommendation="Go to urgent care.",
        options=[CareOption(name="K+STAT Urgent Care", facility_type="Urgent Care",
                            distance_miles=1.2, latitude=39.19, longitude=-96.57,
                            place_id="osm-node-9",
                            price=PriceEstimate(low_usd=125, high_usd=275,
                                                basis="base visit", source="gemini_search"))],
    )

    assert a.care_level == 3
    assert a.options[0].price_display == "$125–$275"
    assert a.telehealth == []
