"""Pydantic schemas for the Medical Triage CLI Agent."""
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import IntEnum


class CareLevel(IntEnum):
    """Care level classification based on US Healthcare System."""
    HOME_CARE = 1       # OTC/Self-Care - no provider visit needed
    RETAIL_CLINIC = 2   # Prescription, known conditions (strep, UTI, pink eye)
    URGENT_CARE = 3     # Diagnostic tests, unknown conditions
    ER = 4              # Life-threatening emergencies


class MedGemmaResponse(BaseModel):
    """Response from MedGemma-4B medical assessment."""
    is_emergency: bool = Field(description="True if life-threatening emergency requiring ER")
    care_level: int = Field(ge=1, le=4, description="Primary care level: 1=Home, 2=Retail, 3=Urgent, 4=ER")
    care_levels: List[int] = Field(default_factory=list, description="All applicable care levels (for compound answers like 2,3)")
    facility_type: str = Field(description="Recommended facility: Home, Retail Clinic, Urgent Care, ER")
    symptoms_restated: str = Field(description="Symptoms as understood by the model")
    likely_conditions: List[str] = Field(default_factory=list, description="Possible conditions")
    likely_tests: List[str] = Field(default_factory=list, description="Tests that may be performed")
    care_type: str = Field(description="Type of care: OTC, prescription, IV fluids, sutures, etc.")
    reasoning: str = Field(description="Clinical reasoning for the assessment")
    recommendation: str = Field(description="Plain-language recommendation for the patient")
    
    # Triage-specific fields
    needs_testing: bool = Field(default=False, description="Whether diagnostic tests are needed")
    tests_needed: List[str] = Field(default_factory=list, description="Specific tests if needs_testing is True")
    triage_steps: List[str] = Field(default_factory=list, description="Steps medical staff would take")
    telehealth_recommended: bool = Field(default=False, description="Whether telehealth is a viable option")
    
    # Metadata for best-output selection
    attempt_number: int = Field(default=1, description="Which attempt this response came from")
    is_refusal: bool = Field(default=False, description="True if model refused to answer")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score from selector")


class FacilityInfo(BaseModel):
    """Information about a medical facility from Google Maps."""
    name: str
    address: str
    place_id: str
    rating: Optional[float] = None
    facility_type: str = Field(description="Detected type: Urgent Care, Retail Clinic, Walk-in Clinic")
    distance_miles: Optional[float] = None
    is_open_now: Optional[bool] = None

    # Where the listing actually is. Needed to place it on a map, and to check
    # it against the real world — a listing with no coordinates cannot be verified.
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    
    # Cost info from Perplexity
    estimated_cost: Optional[str] = None
    cost_source: Optional[str] = None
    services_available: List[str] = Field(default_factory=list)


class TelehealthOption(BaseModel):
    """A telehealth provider option."""
    name: str
    conditions: List[str]
    estimated_cost: str
    url: str


class TriageResult(BaseModel):
    """Final triage result returned to the user."""
    is_emergency: bool
    care_level: int
    care_levels: List[int] = Field(default_factory=list, description="All applicable care levels")
    summary: str = Field(description="Plain-language summary for the patient")
    recommendation: str = Field(description="What the patient should do")
    
    # Facility options (if not emergency)
    facilities: List[FacilityInfo] = Field(default_factory=list)
    telehealth_options: List[TelehealthOption] = Field(default_factory=list)
    
    # Metadata
    medgemma_attempts: int = Field(default=1, description="Number of MedGemma attempts needed")


class TriageState(BaseModel):
    """State object passed through the triage pipeline."""
    # Input
    symptoms: str
    zip_code: str
    age: int = 0
    gender: str = "unspecified"
    
    # MedGemma responses (up to 5 attempts)
    medgemma_responses: List[MedGemmaResponse] = Field(default_factory=list)
    best_response: Optional[MedGemmaResponse] = None
    
    # Facility search results
    facilities: List[FacilityInfo] = Field(default_factory=list)
    
    # Telehealth options
    telehealth_options: List[TelehealthOption] = Field(default_factory=list)
    
    # Final result
    result: Optional[TriageResult] = None
    
    # Error handling
    error: Optional[str] = None
