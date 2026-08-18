"""LangGraph orchestration for everything after the emergency gate.

    assess (MedGemma, already done) -> facilities -> price -> telehealth -> review
                                            ^                                 |
                                            +------- retry, bounded ----------+

The emergency gate deliberately sits outside this graph. It decides whether
someone should call 911, and that must not depend on the orchestrator being
healthy — a graph failure should never be able to swallow an emergency.

A cyclic retry is why this is a graph and not a chain.
"""
import logging
from typing import Any, Callable, List, Optional

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.contracts import CARE_SETTING_TO_LEVEL, CareOption, TelehealthOption, TriageAnswer
from orchestrator import filter_facilities, price_options, review_answer, select_telehealth
from triage import Assessment

logger = logging.getLogger(__name__)

# A review that always reports a gap must not spin. Two retries is enough to
# recover a transient tool failure and cheap enough to run over an eval set.
MAX_RETRIES = 2

# How many options a patient is shown, and therefore how many prices we pay for.
# Applied AFTER the orchestrator has judged the candidates: capping first hands
# it a pre-truncated list, so a specialty practice in the top few is never
# rejected and a real clinic just outside them is never seen.
MAX_PRICED = 3


class TriageState(BaseModel):
    """State threaded through the graph."""

    symptoms: str
    zip_code: str
    assessment: Assessment

    raw_facilities: List[dict] = Field(default_factory=list)
    options: List[CareOption] = Field(default_factory=list)
    telehealth: List[TelehealthOption] = Field(default_factory=list)
    review_notes: List[str] = Field(default_factory=list)
    retries: int = 0
    answer: Optional[TriageAnswer] = None

    model_config = {"arbitrary_types_allowed": True}


def build_graph(
    llm: Any,
    find_facilities: Callable[[str, str], List[dict]],
    telehealth_providers: Callable[[], List[TelehealthOption]],
):
    """Compile the orchestration graph.

    Tools are injected so the whole graph runs offline in tests: no network, no
    model endpoint, no credentials.
    """

    def needs_a_visit(state: TriageState) -> bool:
        return state.assessment.care_level != 1

    def find_node(state: TriageState) -> dict:
        if not needs_a_visit(state):
            return {"raw_facilities": []}
        try:
            found = find_facilities(state.zip_code, state.assessment.care_setting)
        except Exception as e:
            logger.error(f"Facility search failed: {e}")
            found = []
        return {"raw_facilities": found}

    def filter_node(state: TriageState) -> dict:
        if not state.raw_facilities:
            return {}
        kept, notes = filter_facilities(
            llm, state.raw_facilities, state.symptoms, state.assessment.care_setting
        )
        kept.sort(key=lambda f: f.get("distance_miles") or 999)
        dropped = len(kept) - MAX_PRICED
        if dropped > 0:
            notes = notes + [f"{dropped} further option(s) found but not priced (showing nearest {MAX_PRICED})"]
        return {"raw_facilities": kept[:MAX_PRICED], "review_notes": state.review_notes + notes}

    def price_node(state: TriageState) -> dict:
        if not state.raw_facilities:
            return {"options": []}
        return {"options": price_options(
            llm, state.raw_facilities, state.zip_code, state.assessment.procedures()
        )}

    def telehealth_node(state: TriageState) -> dict:
        # MedGemma's telehealth_coverage used to be a hard gate here, and it
        # suppressed every online option in the live run. It is now advice the
        # orchestrator weighs: the orchestrator sees the case and returns an
        # empty list itself when nothing fits.
        #
        # Home care is included deliberately. A cheap online consult with no
        # travel is most useful precisely when no visit is required, and in a
        # sparse area it is often the only thing available.
        try:
            providers = telehealth_providers()
        except Exception as e:
            logger.error(f"Telehealth lookup failed: {e}")
            return {"telehealth": []}
        return {"telehealth": select_telehealth(
            llm, providers, state.symptoms, state.assessment.telehealth_coverage
        )}

    def review_node(state: TriageState) -> dict:
        # Level 1 needs no visit, so judging it against "where to go and what it
        # costs" invents gaps that are not gaps and floods the eval log.
        if not needs_a_visit(state):
            return {"answer": _assemble(state, state.review_notes)}

        verdict = review_answer(
            llm, state.assessment.care_setting, state.options, state.telehealth
        )
        notes = list(state.review_notes)
        if not verdict.complete:
            # These strings are the eval instrument: a per-case record of where
            # the pipeline fell short, produced without a separate analysis pass.
            notes += [f"review: {m}" for m in verdict.missing]

        if not verdict.complete and state.retries < MAX_RETRIES and verdict.retry_tool:
            return {"review_notes": notes, "retries": state.retries + 1,
                    "_retry": verdict.retry_tool}
        return {"review_notes": notes, "answer": _assemble(state, notes)}

    def after_review(state: TriageState) -> str:
        return END if state.answer is not None else "price"

    graph = StateGraph(TriageState)
    graph.add_node("find", find_node)
    graph.add_node("filter", filter_node)
    graph.add_node("price", price_node)
    graph.add_node("telehealth", telehealth_node)
    graph.add_node("review", review_node)

    graph.set_entry_point("find")
    graph.add_edge("find", "filter")
    graph.add_edge("filter", "price")
    graph.add_edge("price", "telehealth")
    graph.add_edge("telehealth", "review")
    graph.add_conditional_edges("review", after_review, {END: END, "price": "price"})

    return graph.compile()


def _assemble(state: TriageState, notes: List[str]) -> TriageAnswer:
    a = state.assessment
    return TriageAnswer(
        care_setting=a.care_setting,
        care_level=CARE_SETTING_TO_LEVEL.get(a.care_setting, a.care_level),
        reasoning=a.reasoning,
        recommendation=a.recommendation,
        likely_conditions=a.likely_conditions,
        labs=a.labs,
        interventions=a.interventions,
        options=state.options,
        telehealth=state.telehealth,
        review_notes=notes,
    )
