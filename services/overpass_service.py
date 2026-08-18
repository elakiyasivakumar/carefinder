"""Facility search via Overpass, OpenStreetMap's proximity query API.

Nominatim answers "where is the place named X". Asking it for "urgent care near
95134" returned clinics named "Urgent Care" in the Bronx and Los Angeles,
because the zip is not a constraint in a free-text name search — every result
was then correctly discarded by the distance filter, leaving nothing.

Overpass asks the question we actually mean: which features tagged as medical
care lie within R metres of this point.
"""
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

from geopy.distance import geodesic

from agents.schemas import FacilityInfo

logger = logging.getLogger(__name__)

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "med-triage-cli/1.0"
DEFAULT_TIMEOUT = 60

# The public Overpass instance rate-limits under load. A 429 is transient, and
# letting one through as an empty list reads to a patient as "nothing nearby" —
# which is the fabrication problem in reverse: a false negative rather than a
# false positive, but still an untrue answer.
DEFAULT_RETRY_DELAYS = (2.0, 5.0)
RATE_LIMIT_MARKERS = ("429", "too many requests", "rate limit", "504", "gateway")

# A drugstore counts only where it names an actual clinic. Most CVS and
# Walgreens stores are a dispensing counter with no clinician, and sending a
# patient there wastes the trip.
RETAIL_CLINIC_MARKERS = (
    "minuteclinic", "minute clinic", "healthhub", "health hub",
    "healthcare clinic", "little clinic", "walmart health", "target clinic",
    "wellness clinic",
)

URGENT_CARE_HINTS = ("urgent", "immediate", "express care", "walk-in", "walk in")

# Real places that cannot treat an acute non-emergency complaint. Offering a
# chiropractor for a fever is not a hallucination — it is still the wrong answer,
# and it is the failure mode a proximity search introduces.
EXCLUDED_SPECIALTIES = (
    "chiropract", "dental", "dentist", "orthodont", "veterinar", "animal",
    "aesthetic", "medspa", "med spa", "cosmetic", "plastic surgery", "dermatolog",
    "mental health", "psychiat", "psycholog", "counseling", "counselling",
    "physical therapy", "physiotherap", "rehabilitation", "chiropractic",
    "optometr", "optician", "eye center", "eye care", "vision center",
    "digestive", "gastroenter", "cardiolog", "oncolog", "orthoped", "podiatr",
    "fertility", "obstetric", "gynecolog", "dialysis", "imaging", "radiolog",
    "laborator", "blood bank", "hospice", "nursing home", "assisted living",
    "acupunctur", "massage", "wellness spa", "weight loss", "surgery center",
)

# Tag values for specialties we exclude regardless of name.
EXCLUDED_HEALTHCARE_SPECIALITIES = (
    "dentist", "physiotherapist", "psychotherapist", "optometrist",
    "podiatrist", "chiropractor", "alternative", "laboratory", "sample_collection",
    "blood_donation", "rehabilitation", "hospice", "nursing_home", "counselling",
)


def build_query(lat: float, lon: float, radius_meters: int = 32000) -> str:
    """Overpass QL for medical facilities within radius_meters of a point."""
    selectors = (
        f'node["amenity"~"^(clinic|doctors|pharmacy)$"](around:{radius_meters},{lat},{lon});'
        f'way["amenity"~"^(clinic|doctors|pharmacy)$"](around:{radius_meters},{lat},{lon});'
        f'node["healthcare"~"^(clinic|centre|doctor)$"](around:{radius_meters},{lat},{lon});'
        f'way["healthcare"~"^(clinic|centre|doctor)$"](around:{radius_meters},{lat},{lon});'
    )
    return f"[out:json][timeout:25];({selectors});out center 80;"


def classify(tags: dict) -> Optional[str]:
    """Map OSM tags to one of our care settings, or None if it is not an option.

    Returning None matters: hospitals and bare pharmacies are filtered out here
    rather than being offered as somewhere to seek non-emergency care.
    """
    name = (tags.get("name") or "").lower()
    amenity = (tags.get("amenity") or "").lower()
    healthcare = (tags.get("healthcare") or "").lower()
    speciality = (tags.get("healthcare:speciality") or "").lower()

    if amenity == "hospital" or healthcare == "hospital":
        return None

    if any(s in speciality for s in EXCLUDED_HEALTHCARE_SPECIALITIES):
        return None
    if healthcare in EXCLUDED_HEALTHCARE_SPECIALITIES:
        return None
    if any(s in name for s in EXCLUDED_SPECIALTIES):
        return None

    # A named in-store clinic is a retail clinic; the drugstore around it is not.
    if any(marker in name for marker in RETAIL_CLINIC_MARKERS):
        return "Retail Clinic"

    if healthcare == "urgent_care" or any(h in name for h in URGENT_CARE_HINTS):
        return "Urgent Care"

    if amenity == "pharmacy":
        # A dispensing counter with no named clinic cannot treat anybody.
        return None

    if amenity == "clinic" or healthcare in ("clinic", "centre"):
        return "Urgent Care"

    if amenity == "doctors" or healthcare == "doctor":
        return "Walk-in Clinic"

    return None


def parse_elements(
    elements: List[dict],
    origin: Tuple[float, float],
    max_distance_miles: float,
) -> List[FacilityInfo]:
    """Turn raw Overpass elements into facilities, nearest first."""
    found: List[FacilityInfo] = []
    seen = set()

    for el in elements:
        tags = el.get("tags") or {}
        name = tags.get("name")
        if not name:
            # Unnamed nodes cannot be given to a patient as somewhere to go.
            continue

        facility_type = classify(tags)
        if facility_type is None:
            continue

        lat, lon = _coords(el)
        if lat is None or lon is None:
            continue

        miles = geodesic(origin, (lat, lon)).miles
        if miles > max_distance_miles:
            continue

        key = name.strip().lower()
        if key in seen:
            continue
        seen.add(key)

        found.append(FacilityInfo(
            name=name.strip(),
            address=_address(tags) or name.strip(),
            place_id=f"osm-{el.get('type', 'node')}-{el.get('id')}",
            rating=None,
            facility_type=facility_type,
            distance_miles=round(miles, 1),
            is_open_now=None,
            latitude=lat,
            longitude=lon,
        ))

    found.sort(key=lambda f: f.distance_miles if f.distance_miles is not None else 999)
    return found


def _coords(el: dict):
    if el.get("lat") is not None and el.get("lon") is not None:
        return el["lat"], el["lon"]
    center = el.get("center") or {}
    return center.get("lat"), center.get("lon")


def _address(tags: dict) -> str:
    parts = [
        " ".join(p for p in [tags.get("addr:housenumber"), tags.get("addr:street")] if p),
        tags.get("addr:city"),
        tags.get("addr:state"),
        tags.get("addr:postcode"),
    ]
    return ", ".join(p for p in parts if p)


class OverpassService:
    """Queries Overpass for medical facilities near a point."""

    def __init__(
        self,
        url: str = OVERPASS_URL,
        timeout: int = DEFAULT_TIMEOUT,
        retry_delays=DEFAULT_RETRY_DELAYS,
    ):
        self.url = url
        self.timeout = timeout
        self.retry_delays = tuple(retry_delays)

    def find_near(
        self,
        origin: Tuple[float, float],
        max_distance_miles: float,
        radius_meters: Optional[int] = None,
    ) -> List[FacilityInfo]:
        lat, lon = origin
        radius = radius_meters or int(max_distance_miles * 1609.34)
        query = build_query(lat, lon, radius)

        elements = self._post_with_retry(query, origin)
        if elements is None:
            return []
        return parse_elements(elements, origin, max_distance_miles)

    def _post_with_retry(self, query: str, origin) -> Optional[List[dict]]:
        """Retry only rate limits. A malformed query fails identically every
        time, so retrying it just stalls the request."""
        attempts = len(self.retry_delays) + 1
        for attempt in range(attempts):
            try:
                return self._post(query)
            except Exception as e:
                if not _is_rate_limited(e) or attempt == attempts - 1:
                    logger.error(f"Overpass query failed near {origin}: {e}")
                    return None
                delay = self.retry_delays[attempt]
                logger.warning(f"Overpass rate-limited, retrying in {delay}s")
                time.sleep(delay)
        return None

    def _post(self, query: str) -> List[dict]:
        request = urllib.request.Request(
            self.url,
            data=urllib.parse.urlencode({"data": query}).encode(),
            headers={"User-Agent": USER_AGENT},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.load(response).get("elements", [])


def _is_rate_limited(error: Exception) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in RATE_LIMIT_MARKERS)
