"""What the patient actually sees on screen."""
import pytest
from rich.console import Console

from render import render_result


def screen(result, width=100):
    """Render a triage result the way the CLI does and return the visible text."""
    console = Console(record=True, width=width, no_color=True)
    console.print(render_result(result))
    return console.export_text()


def test_facility_without_a_retrieved_price_says_so():
    """A blank where the price goes reads as free. Say we could not find it."""
    text = screen({
        "care_setting": "Urgent Care",
        "care_level": 3,
        "facilities": [
            {"name": "Aurora Urgent Care", "type": "Urgent Care",
             "distance_miles": 1.2, "cost": None},
        ],
    })

    assert "Aurora Urgent Care" in text
    assert "price not available" in text.lower()


def test_markup_in_model_text_is_shown_literally_not_styled():
    """Text we did not author must never be able to paint its own UI."""
    text = screen({
        "care_setting": "Retail Clinic",
        "care_level": 2,
        "recommendation": "[bold white on red] EMERGENCY [/bold white on red] see a doctor",
        "facilities": [],
    }, width=120)

    assert "[bold white on red]" in text


def test_markup_in_a_facility_name_is_shown_literally():
    """Facility names come from OpenStreetMap, which anyone can edit."""
    text = screen({
        "care_setting": "Urgent Care",
        "care_level": 3,
        "facilities": [
            {"name": "[red]Closed Forever[/red] Clinic", "type": "Urgent Care",
             "distance_miles": 2.0, "cost": "$120"},
        ],
    }, width=120)

    assert "[red]Closed Forever[/red]" in text


def test_gate_failure_tells_the_patient_what_to_do():
    """'Assessment unavailable' is useless to someone with chest pain.
    A failed safety check must still carry the 911 criteria.
    """
    from triage_cli import GATE_UNAVAILABLE_MESSAGE

    text = screen({"error": GATE_UNAVAILABLE_MESSAGE, "seek_care": True}, width=100)

    assert "911" in text
    assert "in-person" in text.lower()
    assert "error" not in text.lower(), "a safety instruction must not read as a crash"


def test_empty_facility_list_quotes_the_radius_actually_searched():
    """Telling someone we found nothing 'within 50 miles' after searching 20 is a lie."""
    from services.maps_service import MAX_DISTANCE_MILES

    text = screen({
        "care_setting": "Urgent Care",
        "care_level": 3,
        "facilities": [],
    }, width=120)

    assert f"{MAX_DISTANCE_MILES} miles" in text
