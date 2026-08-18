#!/usr/bin/env python3
"""Triage pipeline wiring.

MedGemma decides *what care* (triage.py). Python decides *what to look up*.
Nothing here talks to a network at import time — dependencies are built on
demand by build_deps(), so the routing rules below are testable offline.
"""
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, List, Optional

from dotenv import load_dotenv

from agents.contracts import TelehealthOption
from triage import AssessmentError, Assessment, GateVerdict, assess, emergency_gate

logger = logging.getLogger(__name__)

load_dotenv()

REGION = "us-central1"
ZIP_CODE_RE = re.compile(r'^\d{5}$')

# Levels that correspond to a physical place we can search for.
LEVEL_TO_FACILITY_TYPE = {2: "Retail Clinic", 3: "Urgent Care"}

# How many options are shown is decided inside the graph (graph.MAX_PRICED),
# after the orchestrator has judged the candidates.

GATE_UNAVAILABLE_MESSAGE = (
    "We could not complete a safety check. Seek in-person evaluation, and call 911 "
    "now if you have chest pain, trouble breathing, weakness on one side, confusion, "
    "or heavy bleeding."
)
ASSESSMENT_UNAVAILABLE_MESSAGE = (
    "Assessment temporarily unavailable. Please try again, or consult a provider directly."
)


@dataclass
class Deps:
    """Everything run_triage talks to. Swap any of it in tests."""

    llm: Any          # MedGemma — emergency gate and clinical assessment
    maps: Any         # Overpass-backed facility search
    telehealth: Any   # static provider list
    gemini: Any = None  # Flash — orchestrator, pricing, review
    costs: Any = None   # retired: pricing now runs through gemini + search


def build_deps(
    project_id: Optional[str] = None,
    endpoint_id: Optional[str] = None,
    region: str = REGION,
) -> Deps:
    """Construct the real dependencies. Raises if the environment is incomplete."""
    from services.gemini_client import GeminiClient
    from services.llm_client import VertexLLMClient
    from services.maps_service import MapsService
    from services.telehealth import TelehealthService

    project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    endpoint_id = endpoint_id or os.getenv("MEDGEMMA_ENDPOINT_ID", "")

    return Deps(
        llm=VertexLLMClient(project_id, region, endpoint_id),
        maps=MapsService(),
        telehealth=TelehealthService(),
        gemini=GeminiClient(project=project_id),
    )


def run_triage(
    age: int, gender: str, symptoms: str, zip_code: str, deps: Optional[Deps] = None
) -> dict:
    """Main entry point. Returns a structured dict for CLI or API consumers."""
    zip_code = str(zip_code).strip()
    if not ZIP_CODE_RE.match(zip_code):
        return {"error": "Invalid zip code. Please enter a valid 5-digit US zip code (e.g. 10001)."}

    if not str(symptoms).strip():
        return {"error": "Please describe your symptoms."}

    deps = deps or build_deps()

    verdict = emergency_gate(deps.llm, age, gender, symptoms)

    if verdict is GateVerdict.EMERGENCY:
        return {"is_emergency": True}

    if verdict is GateVerdict.UNKNOWN:
        # Never fall through to Phase 2: the patient has not been cleared.
        return {"error": GATE_UNAVAILABLE_MESSAGE, "seek_care": True}

    try:
        assessment = assess(deps.llm, age, gender, symptoms)
    except AssessmentError as e:
        logger.error(f"MedGemma assessment failed: {e}")
        return {"error": ASSESSMENT_UNAVAILABLE_MESSAGE}

    try:
        answer = _orchestrate(deps, assessment, symptoms, zip_code)
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}")
        return _degraded_result(assessment)

    if answer is None:
        return _degraded_result(assessment)
    return _to_result(answer)


def _orchestrate(deps: Deps, assessment: Assessment, symptoms: str, zip_code: str):
    """Run the LangGraph orchestrator. Returns a TriageAnswer."""
    from graph import build_graph, TriageState

    def find(zip_code_: str, setting: str) -> List[dict]:
        facility_type = "Retail Clinic" if setting == "Retail Clinic" else "Urgent Care"
        found = deps.maps.find_facilities(zip_code_, facility_type=facility_type)
        return [
            {
                "name": f.name, "facility_type": f.facility_type,
                "distance_miles": f.distance_miles or 0.0,
                "latitude": f.latitude, "longitude": f.longitude,
                "place_id": f.place_id, "address": f.address,
            }
            for f in found
            if f.latitude is not None and f.longitude is not None
        ]

    def providers() -> List[TelehealthOption]:
        raw = deps.telehealth.get_options_for_care_level(assessment.care_level, force=True)
        return [
            TelehealthOption(name=p.name, estimated_cost=p.estimated_cost, url=p.url)
            for p in raw
        ]

    app = build_graph(llm=deps.gemini, find_facilities=find, telehealth_providers=providers)
    state = app.invoke(TriageState(
        symptoms=symptoms, zip_code=zip_code, assessment=assessment
    ))
    return state["answer"]


def _to_result(answer) -> dict:
    return {
        "is_emergency": False,
        "care_setting": answer.care_setting,
        "care_level": answer.care_level,
        "recommendation": answer.recommendation,
        "reasoning": answer.reasoning,
        "likely_conditions": answer.likely_conditions,
        "labs": answer.labs,
        "interventions": answer.interventions,
        "review_notes": answer.review_notes,
        "facilities": [
            {
                "name": o.name, "type": o.facility_type,
                "distance_miles": o.distance_miles,
                "latitude": o.latitude, "longitude": o.longitude,
                "place_id": o.place_id, "address": o.address,
                "cost": o.price.display if o.price else None,
                "cost_source": o.price.source if o.price else "unavailable",
                "cost_basis": o.price.basis if o.price else "",
            }
            for o in answer.options
        ],
        "online_providers": [
            {"name": t.name, "url": t.url, "estimated_cost": t.estimated_cost, "why": t.why}
            for t in answer.telehealth
        ],
    }


def _degraded_result(assessment: Assessment) -> dict:
    """Orchestrator unavailable. The clinical answer still stands on its own —
    the patient learns what level of care they need, just not where or what it costs."""
    from agents.contracts import TriageAnswer
    return _to_result(TriageAnswer(
        care_setting=assessment.care_setting,
        care_level=assessment.care_level,
        reasoning=assessment.reasoning,
        recommendation=assessment.recommendation,
        likely_conditions=assessment.likely_conditions,
        labs=assessment.labs,
        interventions=assessment.interventions,
        review_notes=["orchestrator unavailable — no places or prices retrieved"],
    ))
