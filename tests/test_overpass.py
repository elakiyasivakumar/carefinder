"""Overpass finds facilities BY PROXIMITY, which is the question we actually ask.

Nominatim's free-text geocode answers "a place named X" and ignores the zip, so
searching 95134 returned clinics in the Bronx and Los Angeles.
"""
import pytest

from services.overpass_service import (
    build_query,
    classify,
    parse_elements,
)

SAN_JOSE = (37.4060, -121.9375)


def element(**over):
    base = {
        "type": "node", "id": 123, "lat": 37.4100, "lon": -121.9400,
        "tags": {"name": "Valley Urgent Care", "amenity": "clinic"},
    }
    base.update(over)
    return base


def test_query_asks_for_things_near_a_point():
    q = build_query(*SAN_JOSE, radius_meters=32000)

    assert "around:32000,37.406,-121.9375" in q.replace(" ", "")
    assert "amenity" in q


def test_unnamed_features_are_dropped():
    """An OSM node with no name is useless to a patient."""
    out = parse_elements([element(tags={"amenity": "clinic"})], SAN_JOSE, 20)

    assert out == []


def test_named_facility_is_returned_with_coordinates():
    out = parse_elements([element()], SAN_JOSE, 20)

    assert len(out) == 1
    f = out[0]
    assert f.name == "Valley Urgent Care"
    assert f.latitude == 37.4100
    assert f.place_id == "osm-node-123"
    assert f.distance_miles is not None


def test_ways_use_their_center_point():
    """Buildings come back as ways with a computed center, not lat/lon."""
    way = {"type": "way", "id": 55, "center": {"lat": 37.41, "lon": -121.94},
           "tags": {"name": "Northside Clinic", "amenity": "clinic"}}

    out = parse_elements([way], SAN_JOSE, 20)

    assert out[0].latitude == 37.41
    assert out[0].place_id == "osm-way-55"


def test_features_beyond_the_radius_are_dropped():
    far = element(lat=40.7484, lon=-73.9938)  # NYC

    assert parse_elements([far], SAN_JOSE, 20) == []


def test_results_are_sorted_nearest_first():
    near = element(id=1, lat=37.4070, lon=-121.9380, tags={"name": "Near", "amenity": "clinic"})
    far = element(id=2, lat=37.5000, lon=-121.9800, tags={"name": "Far", "amenity": "clinic"})

    out = parse_elements([far, near], SAN_JOSE, 20)

    assert [f.name for f in out] == ["Near", "Far"]


def test_duplicate_names_are_collapsed():
    a = element(id=1, tags={"name": "Same Clinic", "amenity": "clinic"})
    b = element(id=2, lat=37.4110, tags={"name": "Same Clinic", "amenity": "clinic"})

    assert len(parse_elements([a, b], SAN_JOSE, 20)) == 1


@pytest.mark.parametrize("tags,expected", [
    ({"name": "K-Stat Urgent Care", "amenity": "clinic"}, "Urgent Care"),
    ({"name": "Northside Immediate Care", "amenity": "clinic"}, "Urgent Care"),
    ({"name": "Anytown Clinic", "healthcare": "urgent_care"}, "Urgent Care"),
    ({"name": "CVS MinuteClinic", "amenity": "pharmacy"}, "Retail Clinic"),
    ({"name": "Walgreens Healthcare Clinic", "amenity": "clinic"}, "Retail Clinic"),
    ({"name": "Tonopah Primary Care", "amenity": "doctors"}, "Walk-in Clinic"),
])
def test_facility_type_is_derived_from_tags_and_name(tags, expected):
    assert classify(tags) == expected


def test_plain_pharmacy_is_not_a_clinic():
    """A pharmacy counter cannot treat pink eye. Only branded retail clinics count."""
    assert classify({"name": "Joe's Pharmacy", "amenity": "pharmacy"}) is None


@pytest.mark.parametrize("name", [
    "CVS Pharmacy",
    "Walgreens",
    "Rite Aid",
])
def test_a_bare_drugstore_is_not_a_retail_clinic(name):
    """CVS is a retail clinic only where it runs a MinuteClinic. Most stores
    are just a dispensing counter, and sending someone there wastes the trip.
    """
    assert classify({"name": name, "amenity": "pharmacy"}) is None


@pytest.mark.parametrize("name", [
    "CVS MinuteClinic",
    "Walgreens Healthcare Clinic",
    "CVS Pharmacy — HealthHUB",
])
def test_a_drugstore_with_a_named_clinic_counts(name):
    assert classify({"name": name, "amenity": "pharmacy"}) == "Retail Clinic"


@pytest.mark.parametrize("name", [
    "William Shen, Chiropractor",
    "Sapphire Medical Aesthetics",
    "Pawnee Mental Health Services",
    "Bright Smiles Dental",
    "Bozeman Physical Therapy",
    "Valley Veterinary Clinic",
    "Cotton O'Neil Digestive Health",
    "Advanced Dermatology Associates",
])
def test_specialty_practices_are_not_acute_care_options(name):
    """These are real places, but none can treat a fever or a twisted ankle.
    Offering them is not a hallucination — it is still the wrong answer.
    """
    assert classify({"name": name, "amenity": "doctors"}) is None


@pytest.mark.parametrize("tags", [
    {"name": "Northside Family Practice", "amenity": "doctors"},
    {"name": "Tonopah Primary Care", "amenity": "doctors"},
])
def test_primary_care_remains_an_option(tags):
    """A family practice can see an acute non-emergency patient."""
    assert classify(tags) == "Walk-in Clinic"


def test_hospitals_are_excluded():
    """This product routes non-emergencies; an ER is not an option we offer."""
    assert classify({"name": "General Hospital", "amenity": "hospital"}) is None


def test_rate_limited_requests_are_retried():
    """The public Overpass instance returns 429 under load. One 429 must not
    become an empty facility list — that reads to a patient as 'nothing nearby'."""
    from services.overpass_service import OverpassService

    calls = {"n": 0}

    class Svc(OverpassService):
        def _post(self, query):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("HTTP Error 429: Too Many Requests")
            return [element()]

    out = Svc(retry_delays=(0, 0)).find_near(SAN_JOSE, 20)

    assert calls["n"] == 3
    assert len(out) == 1


def test_retries_are_bounded_and_then_it_gives_up_honestly():
    from services.overpass_service import OverpassService

    class AlwaysLimited(OverpassService):
        def _post(self, query):
            raise RuntimeError("HTTP Error 429: Too Many Requests")

    assert AlwaysLimited(retry_delays=(0, 0)).find_near(SAN_JOSE, 20) == []


def test_a_non_rate_limit_error_is_not_retried():
    """A malformed query will fail identically every time; retrying just stalls."""
    from services.overpass_service import OverpassService

    calls = {"n": 0}

    class Broken(OverpassService):
        def _post(self, query):
            calls["n"] += 1
            raise ValueError("bad query syntax")

    assert Broken(retry_delays=(0, 0)).find_near(SAN_JOSE, 20) == []
    assert calls["n"] == 1
