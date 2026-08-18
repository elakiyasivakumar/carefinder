"""Pydantic contracts shared by every graph node.

These are also handed to Gemini as `response_schema`, so the API constrains
generation to them rather than us parsing hopefully. Keep required fields to
what a model can actually know: anything optional here is something we would
rather report as missing than have invented.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# The settings this product routes to. "ER" is the emergency gate's answer and
# never appears here — the graph only runs once a patient has been cleared.
CareSetting = Literal["Home Care", "Retail Clinic", "Urgent Care"]

# What a facility may be. Anything Overpass returns that is not one of these
# (a chiropractor, a medspa, a dentist) is rejected by the orchestrator.
FacilityType = Literal["Retail Clinic", "Urgent Care", "Walk-in Clinic"]

CARE_SETTING_TO_LEVEL = {"Home Care": 1, "Retail Clinic": 2, "Urgent Care": 3}


class PriceEstimate(BaseModel):
    """A self-pay price that was actually retrieved.

    Holds the whole band. Reporting only the floor of "$350–$700+" tells an
    uninsured patient the best case and hides the risk.
    """

    low_usd: float = Field(ge=0, description="Low end of the self-pay range")
    high_usd: float = Field(ge=0, description="High end of the self-pay range")
    basis: str = Field(description="What this figure covers, in one sentence")
    source: str = Field(default="gemini_search", description="How it was retrieved")

    @model_validator(mode="after")
    def _band_must_be_ordered(self):
        if self.high_usd < self.low_usd:
            raise ValueError("high_usd cannot be below low_usd")
        return self

    @property
    def display(self) -> str:
        if self.low_usd == self.high_usd:
            return f"${self.low_usd:,.0f}"
        return f"${self.low_usd:,.0f}–${self.high_usd:,.0f}"


class CareOption(BaseModel):
    """Somewhere the patient can actually go, with coordinates for the map."""

    name: str
    facility_type: FacilityType
    distance_miles: float = Field(ge=0)
    latitude: float
    longitude: float
    place_id: str
    address: str = ""
    price: Optional[PriceEstimate] = None

    @property
    def price_display(self) -> str:
        return self.price.display if self.price else "price not available"


class TelehealthOption(BaseModel):
    """An online option. Geography-independent, so it works where nothing else does."""

    name: str
    estimated_cost: str
    url: str
    why: str = Field(default="", description="Why this fits the patient's case")


class ReviewVerdict(BaseModel):
    """The review node's judgement.

    `missing` is deliberately required when incomplete: those strings are the
    eval instrument, a per-case record of where the pipeline fell short.
    """

    complete: bool
    missing: List[str] = Field(default_factory=list)
    retry_tool: Optional[Literal["facilities", "pricing", "telehealth"]] = None

    @model_validator(mode="after")
    def _incomplete_must_say_why(self):
        if not self.complete and not self.missing:
            raise ValueError("an incomplete verdict must list what is missing")
        return self


class TriageAnswer(BaseModel):
    """The three things this product promises: what level of care, where, and
    what it costs."""

    care_setting: CareSetting
    care_level: int = Field(ge=1, le=3)
    reasoning: str = ""
    recommendation: str = ""
    likely_conditions: List[str] = Field(default_factory=list)
    labs: List[str] = Field(default_factory=list)
    interventions: List[str] = Field(default_factory=list)
    options: List[CareOption] = Field(default_factory=list)
    telehealth: List[TelehealthOption] = Field(default_factory=list)
    review_notes: List[str] = Field(default_factory=list)
