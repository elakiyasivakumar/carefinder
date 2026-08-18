"""Audit a triage result for invented facilities and invented prices.

Nothing here trusts the pipeline's own labels alone. Coordinates are re-checked
against the searched location, and every price must be traceable to text that
was actually retrieved.
"""
from typing import Optional, Tuple

from geopy.distance import geodesic

# The only cost_source that represents a real retrieval. Anything else is a
# locally produced number, which is exactly what must never reach a patient.
RETRIEVED_SOURCE = "perplexity"

# Levels that should produce a place to go. Level 1 is self-care; an emergency
# exits before any lookup.
LEVELS_EXPECTING_FACILITIES = (2, 3)


def audit_result(
    result: dict,
    zip_centroid: Optional[Tuple[float, float]],
    radius_miles: float,
) -> dict:
    """Return a per-case audit of map and price trustworthiness."""
    care_level = result.get("care_level")
    is_emergency = bool(result.get("is_emergency"))
    blocked = bool(result.get("error"))
    facilities = result.get("facilities") or []

    maps_expected = (
        not is_emergency
        and not blocked
        and care_level in LEVELS_EXPECTING_FACILITIES
    )

    bad_facilities = []
    bad_prices = []
    unavailable = 0
    verified_prices = 0

    for f in facilities:
        name = f.get("name") or "<unnamed>"

        reason = _facility_problem(f, zip_centroid, radius_miles)
        if reason:
            bad_facilities.append({"name": name, "reason": reason})

        price_reason, was_unavailable, was_verified = _price_problem(f)
        if price_reason:
            bad_prices.append({"name": name, "cost": f.get("cost"), "reason": price_reason})
        unavailable += was_unavailable
        verified_prices += was_verified

    return {
        "maps_expected": maps_expected,
        "maps_returned": bool(facilities),
        "facility_count": len(facilities),
        "hallucinated_facilities": bad_facilities,
        "hallucinated_prices": bad_prices,
        "prices_verified": verified_prices,
        "prices_unavailable": unavailable,
        "clean": not bad_facilities and not bad_prices,
    }


def _facility_problem(f: dict, centroid, radius_miles: float) -> Optional[str]:
    """Why this listing cannot be trusted as a real place, or None."""
    lat, lon = f.get("latitude"), f.get("longitude")

    if lat is None or lon is None:
        return "no coordinates — cannot be located or mapped"
    if not f.get("place_id"):
        return "no OSM place_id — no upstream record"

    if centroid:
        miles = geodesic(centroid, (lat, lon)).miles
        # Allow a little slack over the search radius for centroid imprecision.
        if miles > radius_miles * 1.5:
            return f"{miles:.0f} mi from the searched zip, beyond the {radius_miles} mi radius"

    return None


def _price_problem(f: dict):
    """(reason, counted_unavailable, counted_verified) for this listing's price."""
    cost = f.get("cost")
    source = f.get("cost_source", "unavailable")

    if cost is None:
        # Honest: nothing retrieved, nothing claimed.
        return None, 1, 0

    if source != RETRIEVED_SOURCE:
        return f"price shown with cost_source={source!r}, not a live retrieval", 0, 0

    if not f.get("cost_verified_in_source"):
        return "figure does not appear in the retrieved text", 0, 0

    return None, 0, 1
