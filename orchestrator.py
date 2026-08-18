"""Orchestrator nodes: Gemini Flash reasoning over tool results.

Each function takes raw tool output and returns something the graph can put in
state. The division of labour throughout: the tool supplies facts, Flash
supplies judgement, and Python supplies the guarantee that Flash cannot invent
anything the tool did not return.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from agents.contracts import CareOption, PriceEstimate, ReviewVerdict, TelehealthOption

logger = logging.getLogger(__name__)


class RejectedFacility(BaseModel):
    """A facility the orchestrator ruled out, and why.

    A real object rather than a dict: Gemini's response_schema subset cannot
    express arbitrary-key objects, so a Dict[str, str] came back silently empty
    and the reasons — which are the eval instrument — were lost.
    """

    place_id: str
    reason: str = Field(description="One line: why this cannot treat the presentation")


class FacilityShortlist(BaseModel):
    """Which of the facilities we found can actually treat this patient."""

    keep_place_ids: List[str] = Field(
        default_factory=list, description="place_ids that can treat this presentation"
    )
    rejected: List[RejectedFacility] = Field(
        default_factory=list, description="Every facility ruled out, each with a reason"
    )


class PricedOption(BaseModel):
    """A self-pay price band for one facility."""

    low_usd: float = Field(ge=0)
    high_usd: float = Field(ge=0)
    basis: str = Field(description="What the figure covers, one sentence")


class TelehealthPick(BaseModel):
    name: str
    why: str = Field(description="Why this provider fits this presentation")


class TelehealthSelection(BaseModel):
    picks: List[TelehealthPick] = Field(default_factory=list)


FILTER_PROMPT = """You are triaging where an uninsured patient should go.

Patient presentation: {symptoms}
Recommended care setting: {setting}

These facilities were found near the patient by an OpenStreetMap proximity
search. OSM has no reliable "urgent care" tag, so the list contains whatever
medical features exist nearby — including practices that cannot treat this case.

{listing}

Keep only the facilities that could actually treat this presentation as a
walk-in. Reject specialty practices that could not: chiropractors, dentists,
optometrists, dermatology, physical therapy, mental health, aesthetics,
veterinary, imaging-only and laboratory-only sites.

Keep primary care and family practice — in rural areas they are often the only
option, and they treat routine acute complaints.

Return keep_place_ids for those you keep, and in `rejected` an entry for EVERY
facility you exclude, each with its place_id and a one-line reason. Do not leave
`rejected` empty if you excluded anything."""

PRICING_PROMPT = """What is the typical self-pay (uninsured, cash price) cost of a
visit at {name}, a {facility_type} near zip code {zip_code}?

{procedures}

Search for actual published or reported prices for this facility or, failing
that, for comparable facilities in the same area. Return the realistic range a
self-pay patient would be quoted, not the best case alone. State in one sentence
what the figure covers."""

TELEHEALTH_PROMPT = """Patient presentation: {symptoms}

Available telehealth providers:
{listing}

Decide which of these could genuinely handle this presentation remotely.

Telehealth routinely handles: conjunctivitis, urinary tract infections, rashes
and skin complaints, colds and flu, sore throats, sinus infections, allergies,
pink eye, medication questions and prescription refills. A provider can examine
by video, take a history, and prescribe.

Telehealth cannot handle anything needing hands-on care: wounds requiring
sutures, imaging, IV fluids, injections, or a physical examination that decides
the diagnosis.

For context, a separate clinical model suggested {coverage_hint}. Treat that as
one opinion, not a rule — it has been over-cautious about remote care, and it
did not see this provider list.

The patient is uninsured and paying list price, so a suitable remote option at
$35-99 may be the best answer available. Offer every provider that genuinely
fits, with one line each on why. Return an empty list only if hands-on care is
truly required."""

REVIEW_PROMPT = """Review this triage answer for an uninsured patient.

Care setting: {setting}
Places to go: {options}
Online options: {telehealth}

The product promises three things: what level of care is needed, where to go,
and what it will cost. Judge whether this answer delivers all three.

Mark it incomplete only if something is genuinely missing and could be
recovered by retrying a tool. An empty facility list in a sparse rural area is a
correct answer, not a gap, and a price that could not be retrieved is honest.

If incomplete, name which tool to retry: facilities, pricing, or telehealth."""


def filter_facilities(llm, raw_facilities: List[dict], symptoms: str, setting: str):
    """Drop facilities that cannot treat this presentation. Returns (kept, notes)."""
    if not raw_facilities:
        return [], []

    listing = "\n".join(
        f"- place_id={f['place_id']} | {f['name']} | tagged {f['facility_type']} "
        f"| {f['distance_miles']} mi"
        for f in raw_facilities
    )
    prompt = FILTER_PROMPT.format(symptoms=symptoms, setting=setting, listing=listing)

    try:
        shortlist = llm.structured(prompt, FacilityShortlist, search=False)
    except Exception as e:
        # Judgement unavailable: keep what the tool found rather than dropping
        # real facilities on our own guess.
        logger.error(f"Facility filter failed, keeping all candidates: {e}")
        return raw_facilities, []

    # Only ids we actually supplied may survive — the model cannot add a place.
    offered = {f["place_id"]: f for f in raw_facilities}
    kept = [offered[pid] for pid in shortlist.keep_place_ids if pid in offered]
    notes = [
        f"{offered[r.place_id]['name']}: {r.reason}"
        for r in shortlist.rejected
        if r.place_id in offered
    ]
    return kept, notes


def price_options(
    llm, facilities: List[dict], zip_code: str, procedures: List[str]
) -> List[CareOption]:
    """Attach a self-pay price band to each facility, where one can be found."""
    wanted = [p for p in procedures if str(p).strip()]
    procedure_line = (
        f"The visit is expected to include: {', '.join(wanted)}."
        if wanted else
        "Price a standard walk-in visit."
    )

    def price_one(f: dict) -> CareOption:
        option = CareOption(
            name=f["name"], facility_type=f["facility_type"],
            distance_miles=f["distance_miles"], latitude=f["latitude"],
            longitude=f["longitude"], place_id=f["place_id"],
            address=f.get("address", ""),
        )
        prompt = PRICING_PROMPT.format(
            name=f["name"], facility_type=f["facility_type"],
            zip_code=zip_code, procedures=procedure_line,
        )
        try:
            quote = llm.structured(prompt, PricedOption, search=True)
            option.price = PriceEstimate(
                low_usd=quote.low_usd, high_usd=quote.high_usd,
                basis=quote.basis, source="gemini_search",
            )
        except Exception as e:
            # Leave it unpriced. An invented figure is worse than none.
            logger.error(f"Pricing failed for {f['name']}: {e}")
        return option

    if not facilities:
        return []

    # Each lookup is an independent grounded search taking ~15s; run them
    # together so a three-option answer does not take a minute to appear.
    with ThreadPoolExecutor(max_workers=len(facilities)) as pool:
        return list(pool.map(price_one, facilities))


def select_telehealth(
    llm,
    providers: List[TelehealthOption],
    symptoms: str,
    coverage_hint: str = "maybe",
) -> List[TelehealthOption]:
    """Pick the online providers that fit this presentation.

    `coverage_hint` is MedGemma's view of remote suitability. It informs the
    choice rather than gating it: as a hard gate it suppressed every online
    option, including for cases telehealth handles routinely.
    """
    if not providers:
        return []

    hints = {
        "yes": "this is fully treatable remotely",
        "no": "physical presence may be needed",
        "maybe": "either route could work",
    }
    listing = "\n".join(f"- {p.name} ({p.estimated_cost})" for p in providers)
    prompt = TELEHEALTH_PROMPT.format(
        symptoms=symptoms, listing=listing,
        coverage_hint=hints.get(coverage_hint, hints["maybe"]),
    )

    try:
        selection = llm.structured(prompt, TelehealthSelection, search=False)
    except Exception as e:
        logger.error(f"Telehealth selection failed, offering all: {e}")
        return providers

    by_name = {p.name.lower(): p for p in providers}
    picked = []
    for pick in selection.picks:
        name = pick["name"] if isinstance(pick, dict) else pick.name
        why = pick["why"] if isinstance(pick, dict) else pick.why
        provider = by_name.get(str(name).lower())
        if provider:
            picked.append(provider.model_copy(update={"why": why}))
    return picked


def review_answer(
    llm, setting: str, options: List[CareOption], telehealth: List[TelehealthOption]
) -> ReviewVerdict:
    """Judge whether the answer delivers all three promises.

    Its `missing` strings are logged per case and are the eval instrument: a
    record of where the pipeline fell short, without a separate analysis pass.
    """
    prompt = REVIEW_PROMPT.format(
        setting=setting,
        options="\n".join(
            f"- {o.name} ({o.facility_type}, {o.distance_miles} mi) — {o.price_display}"
            for o in options
        ) or "none",
        telehealth="\n".join(f"- {t.name} ({t.estimated_cost})" for t in telehealth) or "none",
    )

    try:
        return llm.structured(prompt, ReviewVerdict, search=False, thinking_budget=0)
    except Exception as e:
        logger.error(f"Review failed, accepting answer as-is: {e}")
        return ReviewVerdict(complete=True)
