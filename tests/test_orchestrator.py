"""Orchestrator nodes. Each takes graph state, calls Flash, returns state."""
import pytest

from agents.contracts import CareOption, PriceEstimate, ReviewVerdict, TelehealthOption
from orchestrator import (
    FacilityShortlist,
    RejectedFacility,
    PricedOption,
    filter_facilities,
    price_options,
    review_answer,
    select_telehealth,
)


class FakeGemini:
    """Returns queued answers; records the prompts it was given."""

    def __init__(self, *answers):
        self._answers = list(answers)
        self.prompts = []
        self.searched = []

    def structured(self, prompt, schema, search=False, thinking_budget=None):
        self.prompts.append(prompt)
        self.searched.append(search)
        return self._answers.pop(0)


RAW = [
    {"name": "K+STAT Urgent Care", "facility_type": "Urgent Care", "distance_miles": 1.2,
     "latitude": 39.19, "longitude": -96.57, "place_id": "osm-node-1", "address": ""},
    {"name": "William Shen, Chiropractor", "facility_type": "Walk-in Clinic",
     "distance_miles": 0.4, "latitude": 39.19, "longitude": -96.57,
     "place_id": "osm-node-2", "address": ""},
]


def test_orchestrator_drops_facilities_that_cannot_treat_the_case():
    """OSM has no 'is urgent care' tag, so a chiropractor arrives looking like a
    clinic. Rejecting it is the orchestrator's job, not a keyword list's."""
    llm = FakeGemini(FacilityShortlist(
        keep_place_ids=["osm-node-1"],
        rejected=[RejectedFacility(place_id="osm-node-2",
                                   reason="chiropractor cannot treat a fever")]))

    kept, notes = filter_facilities(llm, RAW, "high fever and body aches", "Urgent Care")

    assert [f["name"] for f in kept] == ["K+STAT Urgent Care"]
    assert "chiropractor" in notes[0].lower()


def test_facility_filter_does_not_use_web_search():
    """Classification needs judgement, not the live web. Searching here is waste."""
    llm = FakeGemini(FacilityShortlist(keep_place_ids=["osm-node-1"], rejected=[]))

    filter_facilities(llm, RAW, "fever", "Urgent Care")

    assert llm.searched == [False]


def test_filter_never_invents_a_facility():
    """A place_id the model returns that we never sent it must be ignored."""
    llm = FakeGemini(FacilityShortlist(keep_place_ids=["osm-node-1", "made-up-id"], rejected=[]))

    kept, _ = filter_facilities(llm, RAW, "fever", "Urgent Care")

    assert [f["place_id"] for f in kept] == ["osm-node-1"]


def test_pricing_uses_web_search_and_returns_a_band():
    llm = FakeGemini(PricedOption(low_usd=125, high_usd=275, basis="base visit + labs"))

    priced = price_options(llm, [RAW[0]], "66502", ["influenza PCR"])

    assert priced[0].price.display == "$125–$275"
    assert llm.searched == [True]


def test_pricing_names_the_procedures_in_the_query():
    llm = FakeGemini(PricedOption(low_usd=100, high_usd=200, basis="x"))

    price_options(llm, [RAW[0]], "66502", ["ankle x-ray", "splinting"])

    assert "ankle x-ray" in llm.prompts[0] and "splinting" in llm.prompts[0]


def test_a_failed_price_lookup_leaves_the_option_unpriced():
    """No price is a valid answer. An invented one is not."""
    class Failing:
        def structured(self, *a, **k):
            raise RuntimeError("search unavailable")

    priced = price_options(Failing(), [RAW[0]], "66502", [])

    assert priced[0].price is None
    assert priced[0].price_display == "price not available"


def test_telehealth_selection_explains_each_pick():
    llm = FakeGemini(type("S", (), {"picks": [{"name": "Amazon Clinic", "why": "treats pink eye online"}]})())
    providers = [TelehealthOption(name="Amazon Clinic", estimated_cost="$35–75", url="x"),
                 TelehealthOption(name="Teladoc", estimated_cost="$75", url="y")]

    picked = select_telehealth(llm, providers, "pink eye")

    assert [p.name for p in picked] == ["Amazon Clinic"]
    assert picked[0].why


def test_review_flags_an_answer_with_no_prices():
    llm = FakeGemini(ReviewVerdict(complete=False, missing=["no option has a price"],
                                   retry_tool="pricing"))
    option = CareOption(name="X", facility_type="Urgent Care", distance_miles=1.0,
                        latitude=1.0, longitude=1.0, place_id="p")

    verdict = review_answer(llm, "Urgent Care", [option], [])

    assert verdict.complete is False
    assert verdict.retry_tool == "pricing"


def test_review_uses_no_thinking_budget():
    """Review is a verdict, not a deliberation."""
    llm = FakeGemini(ReviewVerdict(complete=True))
    captured = {}
    orig = llm.structured

    def spy(prompt, schema, search=False, thinking_budget=None):
        captured["tb"] = thinking_budget
        return orig(prompt, schema, search, thinking_budget)

    llm.structured = spy
    review_answer(llm, "Home Care", [], [])

    assert captured["tb"] == 0


def test_a_rejection_without_a_reason_is_invalid():
    """The reasons ARE the eval instrument. A dict of arbitrary keys cannot be
    expressed in Gemini's schema subset, so it silently returned nothing —
    the reason must be a required field on a real object."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        RejectedFacility(place_id="osm-node-2")
