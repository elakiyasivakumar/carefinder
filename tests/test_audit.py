"""Audits a triage result for invented facilities and invented prices.

These checks are what let the eval report claim "nothing was hallucinated"
rather than assume it.
"""
import pytest

from audit import audit_result

ZIP_CENTROID = (40.7506, -73.9972)  # 10001


def facility(**over):
    base = dict(
        name="Aurora Urgent Care", type="Urgent Care", distance_miles=1.2,
        latitude=40.7550, longitude=-73.9900, place_id="p1", address="1 Main St",
        cost="$125–$185", cost_source="perplexity",
        cost_source_text="Self-pay runs $125-$185 here.", cost_verified_in_source=True,
    )
    base.update(over)
    return base


def result(**over):
    base = dict(is_emergency=False, care_setting="Urgent Care", care_level=3,
                facilities=[facility()], online_providers=[])
    base.update(over)
    return base


def test_clean_result_passes_every_check():
    a = audit_result(result(), ZIP_CENTROID, radius_miles=20)

    assert a["maps_returned"] is True
    assert a["facility_count"] == 1
    assert a["hallucinated_facilities"] == []
    assert a["hallucinated_prices"] == []
    assert a["clean"] is True


def test_facility_without_coordinates_is_flagged():
    """A listing with no coordinates cannot be shown on a map or verified."""
    a = audit_result(result(facilities=[facility(latitude=None, longitude=None)]),
                     ZIP_CENTROID, radius_miles=20)

    assert "Aurora Urgent Care" in str(a["hallucinated_facilities"])
    assert a["clean"] is False


def test_facility_outside_the_search_radius_is_flagged():
    """Boston is not 'near' 10001 — a result that far out was not really found there."""
    a = audit_result(result(facilities=[facility(latitude=42.3601, longitude=-71.0589)]),
                     ZIP_CENTROID, radius_miles=20)

    assert a["hallucinated_facilities"]
    assert a["clean"] is False


def test_facility_without_a_place_id_is_flagged():
    a = audit_result(result(facilities=[facility(place_id="")]), ZIP_CENTROID, radius_miles=20)

    assert a["hallucinated_facilities"]


def test_price_not_found_in_its_source_is_flagged():
    a = audit_result(result(facilities=[facility(cost_verified_in_source=False)]),
                     ZIP_CENTROID, radius_miles=20)

    assert a["hallucinated_prices"]
    assert a["clean"] is False


def test_price_from_a_non_retrieval_source_is_flagged():
    """Any cost_source other than a live lookup means the number was made up."""
    a = audit_result(result(facilities=[facility(cost="$99", cost_source="estimate")]),
                     ZIP_CENTROID, radius_miles=20)

    assert a["hallucinated_prices"]


def test_missing_price_is_honest_not_hallucinated():
    """'price not available' is the correct answer when nothing was retrieved."""
    a = audit_result(
        result(facilities=[facility(cost=None, cost_source="unavailable",
                                    cost_source_text="", cost_verified_in_source=False)]),
        ZIP_CENTROID, radius_miles=20)

    assert a["hallucinated_prices"] == []
    assert a["prices_unavailable"] == 1
    assert a["clean"] is True


def test_emergency_result_needs_no_maps():
    a = audit_result({"is_emergency": True}, ZIP_CENTROID, radius_miles=20)

    assert a["maps_returned"] is False
    assert a["maps_expected"] is False
    assert a["clean"] is True


def test_home_care_needs_no_maps():
    a = audit_result(result(care_setting="Home Care", care_level=1, facilities=[]),
                     ZIP_CENTROID, radius_miles=20)

    assert a["maps_expected"] is False
    assert a["clean"] is True


def test_urgent_care_with_no_facilities_is_recorded_but_not_a_hallucination():
    """Rural zips legitimately return nothing. That is honest, not invented."""
    a = audit_result(result(facilities=[]), ZIP_CENTROID, radius_miles=20)

    assert a["maps_expected"] is True
    assert a["maps_returned"] is False
    assert a["hallucinated_facilities"] == []
    assert a["clean"] is True
