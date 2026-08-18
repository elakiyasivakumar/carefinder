"""Every facility shown to a patient must have been found. None may be invented."""
import pytest

import services.maps_service as maps_module
from services.maps_service import MapsService


class FakeLocation:
    """Stands in for a geopy Location returned by Nominatim."""

    def __init__(self, name, latitude, longitude, place_id, place_type="clinic"):
        self.address = name
        self.latitude = latitude
        self.longitude = longitude
        self.raw = {"place_id": place_id, "type": place_type}

    def __str__(self):
        return self.address


class FakeGeolocator:
    """Stands in for Nominatim at the network boundary."""

    def __init__(self, results=None, fail=False):
        self._results = results or []
        self._fail = fail
        self.queries = []

    def geocode(self, query, **kwargs):
        self.queries.append(query)
        if self._fail:
            raise RuntimeError("nominatim unreachable")
        if kwargs.get("exactly_one") is False:
            return self._results
        return self._results[0] if self._results else None


class FakeOverpass:
    """Stands in for Overpass. Records the point it was asked about."""

    def __init__(self, facilities=None):
        self._facilities = facilities or []
        self.asked = []

    def find_near(self, origin, max_distance_miles, radius_meters=None):
        self.asked.append(origin)
        return list(self._facilities)


@pytest.fixture
def fake_nominatim(monkeypatch):
    def install(results=None, fail=False, overpass=None):
        monkeypatch.setattr(
            maps_module, "Nominatim", lambda **kw: FakeGeolocator(results, fail)
        )
        return MapsService(min_delay_seconds=0, overpass=overpass or FakeOverpass())

    return install


def test_unreachable_search_returns_no_facilities(fake_nominatim):
    """If we could not search, we have found nothing — and must claim nothing."""
    service = fake_nominatim(fail=True)

    facilities = service.find_facilities("10001", facility_type="Urgent Care")

    assert facilities == []


def test_zip_that_cannot_be_located_returns_no_facilities(fake_nominatim):
    """An unlocatable zip yields no real results, so it must yield no listings."""
    service = fake_nominatim(results=[])

    facilities = service.find_facilities("99999", facility_type="Retail Clinic")

    assert facilities == []


def test_facilities_come_from_overpass_near_the_geocoded_zip():
    """The honest path: geocode the zip, then ask what is actually around it."""
    from agents.schemas import FacilityInfo

    overpass = FakeOverpass([
        FacilityInfo(name="Aurora Urgent Care", address="1 Main St",
                     place_id="osm-node-1", facility_type="Urgent Care",
                     distance_miles=1.2, latitude=40.7550, longitude=-73.9900),
    ])
    service = MapsService(min_delay_seconds=0, overpass=overpass)
    import services.maps_service as m
    service.geocode = lambda q, **kw: FakeLocation("10001", 40.7484, -73.9938, "z")

    facilities = service.find_facilities("10001", facility_type="Urgent Care")

    assert [f.name for f in facilities] == ["Aurora Urgent Care"]
    assert overpass.asked == [(40.7484, -73.9938)], "must search around the zip, not by name"


def test_requested_facility_type_filters_the_results():
    from agents.schemas import FacilityInfo

    overpass = FakeOverpass([
        FacilityInfo(name="Corner MinuteClinic", address="a", place_id="osm-node-1",
                     facility_type="Retail Clinic", distance_miles=1.0,
                     latitude=40.75, longitude=-73.99),
        FacilityInfo(name="Aurora Urgent Care", address="b", place_id="osm-node-2",
                     facility_type="Urgent Care", distance_miles=2.0,
                     latitude=40.76, longitude=-73.98),
    ])
    service = MapsService(min_delay_seconds=0, overpass=overpass)
    service.geocode = lambda q, **kw: FakeLocation("10001", 40.7484, -73.9938, "z")

    out = service.find_facilities("10001", facility_type="Retail Clinic")

    assert [f.name for f in out] == ["Corner MinuteClinic"]


def test_retail_clinic_falls_back_to_primary_care():
    """OSM rarely tags in-store clinics, so a strict Retail Clinic search returns
    nothing almost everywhere. A family practice treats pink eye, strep and UTIs
    perfectly well — offering one beats offering nothing.
    """
    from agents.schemas import FacilityInfo

    overpass = FakeOverpass([
        FacilityInfo(name="Northside Family Practice", address="a",
                     place_id="osm-node-1", facility_type="Walk-in Clinic",
                     distance_miles=1.0, latitude=40.75, longitude=-73.99),
    ])
    service = MapsService(min_delay_seconds=0, overpass=overpass)
    service.geocode = lambda q, **kw: FakeLocation("10001", 40.7484, -73.9938, "z")

    out = service.find_facilities("10001", facility_type="Retail Clinic")

    assert [f.name for f in out] == ["Northside Family Practice"]


def test_retail_clinic_search_still_excludes_urgent_care():
    """Urgent care is the more expensive tier. Do not quietly upsell into it."""
    from agents.schemas import FacilityInfo

    overpass = FakeOverpass([
        FacilityInfo(name="Pricey Urgent Care", address="a", place_id="osm-node-2",
                     facility_type="Urgent Care", distance_miles=0.5,
                     latitude=40.75, longitude=-73.99),
    ])
    service = MapsService(min_delay_seconds=0, overpass=overpass)
    service.geocode = lambda q, **kw: FakeLocation("10001", 40.7484, -73.9938, "z")

    assert service.find_facilities("10001", facility_type="Retail Clinic") == []


def test_nominatim_is_used_only_to_locate_the_zip(fake_nominatim):
    """Nominatim is a geocoder. Asking it for 'urgent care near 95134' returned
    clinics in the Bronx, because a free-text name search ignores the zip.
    It now answers exactly one question: where is this zip.
    """
    service = fake_nominatim(results=[
        FakeLocation("10001, New York", 40.7484, -73.9938, "z"),
    ])

    service.find_facilities("10001", facility_type="Urgent Care")

    sent = service.geolocator.queries
    assert sent == ["10001, USA"], f"expected one zip geocode, got {sent}"
