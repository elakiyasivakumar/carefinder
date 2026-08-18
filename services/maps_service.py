"""Facility search: Nominatim locates the zip, Overpass finds what is near it.

The split matters. Nominatim is a geocoder — it answers "where is 95134" well
and "what clinics are near 95134" not at all: a free-text search matches place
*names* globally, so it returned clinics in the Bronx and Los Angeles for a San
Jose zip, all of which the distance filter then correctly discarded, leaving
nothing. Overpass is OSM's proximity query and answers the real question.
"""
import logging
from typing import List, Optional

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

from agents.schemas import FacilityInfo
from services.overpass_service import OverpassService

logger = logging.getLogger(__name__)

MAX_DISTANCE_MILES = 20

# Nominatim's usage policy allows at most 1 request per second.
NOMINATIM_MIN_DELAY_SECONDS = 1.0

# Which facility types satisfy a request for a given care setting.
#
# Retail Clinic accepts primary care because OSM rarely tags in-store clinics —
# a strict search returns nothing almost everywhere, and a family practice
# treats pink eye, strep and UTIs perfectly well. It deliberately does NOT
# accept Urgent Care: that is the more expensive tier, and quietly upselling
# into it is the failure this product exists to prevent.
TYPE_FALLBACKS = {
    "Urgent Care": ("Urgent Care", "Walk-in Clinic"),
    "Retail Clinic": ("Retail Clinic", "Walk-in Clinic"),
    "Walk-in Clinic": ("Walk-in Clinic", "Urgent Care"),
}


class MapsService:
    """Finds non-emergency medical facilities near a US zip code."""

    def __init__(
        self,
        min_delay_seconds: float = NOMINATIM_MIN_DELAY_SECONDS,
        overpass: Optional[OverpassService] = None,
    ):
        self.geolocator = Nominatim(user_agent="med-triage-cli")
        # Every Nominatim call goes through this — their usage policy caps us at
        # 1 request/second and will block the user agent otherwise.
        self.geocode = RateLimiter(
            self.geolocator.geocode, min_delay_seconds=min_delay_seconds
        )
        self.overpass = overpass or OverpassService()

    def find_facilities(
        self,
        zip_code: str,
        max_results: int = 10,
        max_distance_miles: float = MAX_DISTANCE_MILES,
        facility_type: Optional[str] = None,
    ) -> List[FacilityInfo]:
        """Return real facilities near a zip, nearest first, or [] if none.

        An empty list is a real answer: sparse areas genuinely have nothing
        within the radius, and saying so beats inventing somewhere to go.
        """
        origin = self._locate(zip_code)
        if origin is None:
            return []

        facilities = self.overpass.find_near(origin, max_distance_miles)

        if facility_type:
            accepted = TYPE_FALLBACKS.get(facility_type, (facility_type,))
            facilities = [f for f in facilities if f.facility_type in accepted]

        if not facilities:
            logger.info(f"No {facility_type or 'medical'} facilities within "
                        f"{max_distance_miles} miles of {zip_code}")

        return facilities[:max_results]

    def _locate(self, zip_code: str):
        """Geocode the zip to a point. This is what Nominatim is actually for."""
        try:
            location = self.geocode(f"{zip_code}, USA")
        except Exception as e:
            logger.error(f"Could not geocode zip {zip_code}: {e}")
            return None

        if not location:
            logger.warning(f"Could not geocode zip: {zip_code}")
            return None

        return (location.latitude, location.longitude)
