"""The LangGraph: assess -> facilities -> price -> telehealth -> review (loop).

The emergency gate sits OUTSIDE this graph on purpose: it must not depend on
the orchestrator being healthy.
"""
import pytest

from agents.contracts import ReviewVerdict, TelehealthOption
from graph import build_graph, TriageState
from orchestrator import FacilityShortlist, PricedOption, RejectedFacility, TelehealthSelection
from triage import Assessment

ASSESSMENT = Assessment(
    care_setting="Urgent Care", care_level=3, care_settings=["Urgent Care"],
    care_levels=[3], labs=["influenza PCR"], interventions=[],
    likely_conditions=["influenza"], reasoning="Systemic viral illness.",
    recommendation="Go to urgent care.",
)

FOUND = [{"name": "K+STAT Urgent Care", "facility_type": "Urgent Care",
          "distance_miles": 1.2, "latitude": 39.19, "longitude": -96.57,
          "place_id": "osm-1", "address": ""}]


class ScriptedGemini:
    def __init__(self, *answers):
        self._answers = list(answers)
        self.seen = []

    def structured(self, prompt, schema, search=False, thinking_budget=None):
        self.seen.append(schema.__name__)
        return self._answers.pop(0)


def run(llm, facilities=FOUND, providers=None):
    app = build_graph(
        llm=llm,
        find_facilities=lambda zip_code, setting: list(facilities),
        telehealth_providers=lambda: providers if providers is not None else [],
    )
    return app.invoke(TriageState(
        symptoms="high fever 103F", zip_code="66502", assessment=ASSESSMENT,
    ))


def test_happy_path_produces_a_complete_answer():
    llm = ScriptedGemini(
        FacilityShortlist(keep_place_ids=["osm-1"], rejected=[]),
        PricedOption(low_usd=125, high_usd=275, basis="visit plus flu test"),
        ReviewVerdict(complete=True),
    )

    answer = run(llm)["answer"]

    assert answer.care_setting == "Urgent Care"
    assert answer.options[0].name == "K+STAT Urgent Care"
    assert answer.options[0].price.display == "$125–$275"


def test_rejected_facilities_are_recorded_as_review_notes():
    llm = ScriptedGemini(
        FacilityShortlist(keep_place_ids=[], rejected=[RejectedFacility(place_id="osm-1", reason="chiropractor")]),
        ReviewVerdict(complete=True),
    )

    answer = run(llm)["answer"]

    assert answer.options == []
    assert any("chiropractor" in n for n in answer.review_notes)


def test_review_can_send_the_graph_back_to_pricing_once():
    """The retry loop is the point of using a graph rather than a chain."""
    llm = ScriptedGemini(
        FacilityShortlist(keep_place_ids=["osm-1"], rejected=[]),
        PricedOption(low_usd=0, high_usd=0, basis="unclear"),
        ReviewVerdict(complete=False, missing=["price looks wrong"], retry_tool="pricing"),
        PricedOption(low_usd=150, high_usd=250, basis="corrected"),
        ReviewVerdict(complete=True),
    )

    state = run(llm)

    assert state["answer"].options[0].price.display == "$150–$250"
    assert state["retries"] == 1


def test_the_retry_loop_is_bounded():
    """A review that always says incomplete must not spin forever."""
    always_bad = ReviewVerdict(complete=False, missing=["still wrong"], retry_tool="pricing")
    llm = ScriptedGemini(
        FacilityShortlist(keep_place_ids=["osm-1"], rejected=[]),
        PricedOption(low_usd=1, high_usd=2, basis="x"), always_bad,
        PricedOption(low_usd=1, high_usd=2, basis="x"), always_bad,
        PricedOption(low_usd=1, high_usd=2, basis="x"), always_bad,
        PricedOption(low_usd=1, high_usd=2, basis="x"), always_bad,
    )

    state = run(llm)

    assert state["retries"] <= 2
    assert state["answer"] is not None


def test_home_care_skips_every_lookup():
    home = ASSESSMENT.model_copy(update={"care_setting": "Home Care", "care_level": 1}) \
        if hasattr(ASSESSMENT, "model_copy") else ASSESSMENT
    llm = ScriptedGemini(ReviewVerdict(complete=True))

    app = build_graph(
        llm=llm,
        find_facilities=lambda z, s: pytest.fail("must not search for Home Care"),
        telehealth_providers=lambda: [],
    )
    state = app.invoke(TriageState(symptoms="mild cold", zip_code="66502",
                                   assessment=_home_assessment()))

    assert state["answer"].care_level == 1
    assert state["answer"].options == []


def _home_assessment():
    return Assessment(
        care_setting="Home Care", care_level=1, care_settings=["Home Care"],
        care_levels=[1], reasoning="Self-limiting.", recommendation="Rest and fluids.",
    )


def test_telehealth_is_offered_and_explained():
    providers = [TelehealthOption(name="Amazon Clinic", estimated_cost="$35–75", url="x")]
    llm = ScriptedGemini(
        FacilityShortlist(keep_place_ids=["osm-1"], rejected=[]),
        PricedOption(low_usd=125, high_usd=275, basis="visit"),
        TelehealthSelection(picks=[{"name": "Amazon Clinic", "why": "handles flu remotely"}]),
        ReviewVerdict(complete=True),
    )

    answer = run(llm, providers=providers)["answer"]

    assert answer.telehealth[0].name == "Amazon Clinic"
    assert "remotely" in answer.telehealth[0].why


def _assessment(setting, level, coverage="maybe"):
    return Assessment(care_setting=setting, care_level=level, care_settings=[setting],
                      care_levels=[level], telehealth_coverage=coverage,
                      reasoning="r", recommendation="rec")


def test_home_care_is_not_reviewed_as_if_it_were_missing_places():
    """Level 1 means no visit is needed. Reviewing it against 'where to go and
    what it costs' produces gaps that are not gaps, and floods the eval log."""
    llm = ScriptedGemini(ReviewVerdict(complete=True))
    app = build_graph(llm=llm, find_facilities=lambda z, s: [],
                      telehealth_providers=lambda: [])

    state = app.invoke(TriageState(symptoms="mild cold", zip_code="66502",
                                   assessment=_assessment("Home Care", 1)))

    assert state["answer"].review_notes == []
    assert "ReviewVerdict" not in llm.seen, "level 1 needs no review of places"


def test_telehealth_is_offered_for_home_care():
    """Home care is where an online consult is most useful: cheap advice, no
    travel, and often the only thing available in a sparse area."""
    providers = [TelehealthOption(name="Amazon Clinic", estimated_cost="$35–75", url="x")]
    llm = ScriptedGemini(
        TelehealthSelection(picks=[{"name": "Amazon Clinic", "why": "can advise on a cold"}]),
    )
    app = build_graph(llm=llm, find_facilities=lambda z, s: [],
                      telehealth_providers=lambda: providers)

    state = app.invoke(TriageState(symptoms="mild cold", zip_code="66502",
                                   assessment=_assessment("Home Care", 1)))

    assert [t.name for t in state["answer"].telehealth] == ["Amazon Clinic"]


def test_orchestrator_judges_telehealth_fit_not_medgemma_alone():
    """MedGemma's telehealth_coverage was a hard gate that suppressed every
    online option. The orchestrator sees the case and decides."""
    providers = [TelehealthOption(name="Amazon Clinic", estimated_cost="$35–75", url="x")]
    # No facilities found, so filter and price short-circuit without calling Flash.
    llm = ScriptedGemini(
        TelehealthSelection(picks=[{"name": "Amazon Clinic", "why": "pink eye is treatable online"}]),
        ReviewVerdict(complete=True),
    )
    app = build_graph(llm=llm, find_facilities=lambda z, s: [],
                      telehealth_providers=lambda: providers)

    state = app.invoke(TriageState(symptoms="pink eye", zip_code="66502",
                                   assessment=_assessment("Retail Clinic", 2, coverage="no")))

    assert state["answer"].telehealth, "orchestrator should still weigh telehealth"


def test_the_orchestrator_sees_every_candidate_before_the_cap_applies():
    """Capping to 3 before filtering hands Flash a pre-truncated list, so a
    chiropractor in the top 3 is never rejected and a real clinic at #4 is
    never seen. The cap belongs after judgement, not before it."""
    many = [
        {"name": f"Place {i}", "facility_type": "Urgent Care", "distance_miles": float(i),
         "latitude": 39.0 + i / 100, "longitude": -96.0, "place_id": f"osm-{i}", "address": ""}
        for i in range(8)
    ]
    seen = {}

    class Watcher:
        def structured(self, prompt, schema, search=False, thinking_budget=None):
            if schema.__name__ == "FacilityShortlist":
                seen["candidates"] = prompt.count("place_id=")
                return FacilityShortlist(
                    keep_place_ids=[f"osm-{i}" for i in range(8)],
                    rejected=[],
                )
            if schema.__name__ == "PricedOption":
                return PricedOption(low_usd=100, high_usd=200, basis="x")
            if schema.__name__ == "TelehealthSelection":
                return TelehealthSelection(picks=[])
            return ReviewVerdict(complete=True)

    app = build_graph(llm=Watcher(),
                      find_facilities=lambda z, s: list(many),
                      telehealth_providers=lambda: [])
    state = app.invoke(TriageState(symptoms="fever", zip_code="66502", assessment=ASSESSMENT))

    assert seen["candidates"] == 8, "the filter must see every candidate found"
    assert len(state["answer"].options) <= 3, "but only 3 may be priced and shown"
