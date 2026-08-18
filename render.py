"""Presentation layer — turns a run_triage() result dict into Rich renderables.

Deliberately imports nothing from the triage pipeline, so the screen a patient
sees can be tested without credentials, a model endpoint, or a network call.
"""
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from services.maps_service import MAX_DISTANCE_MILES

NO_PRICE = "price not available"


def _safe(value) -> str:
    """Neutralise Rich markup in text we did not author.

    Model output and OpenStreetMap names are untrusted: unescaped, a `[...]`
    sequence lets them restyle the panel or forge an emergency badge.
    """
    return escape(str(value)) if value is not None else ""


def render_result(result: dict):
    """Return the Rich renderable for a run_triage() result dict."""
    if result.get("error"):
        # A failed safety check is guidance, not a crash — title it accordingly.
        if result.get("seek_care"):
            return _seek_care_panel(result["error"])
        return _error_panel(result["error"])
    if result.get("is_emergency"):
        return _emergency_panel()
    return _assessment_panel(result)


def _seek_care_panel(message: str) -> Panel:
    return Panel(
        f"[yellow]{_safe(message)}[/yellow]",
        border_style="yellow",
        title="[bold yellow]Get Checked In Person[/bold yellow]",
    )


def _error_panel(message: str) -> Panel:
    return Panel(
        f"[red]{_safe(message)}[/red]",
        border_style="red",
        title="[bold red]Error[/bold red]",
    )


def _emergency_panel() -> Panel:
    return Panel(
        "[bold white on red] EMERGENCY [/bold white on red]\n\n"
        "[bold red]CALL 911 IMMEDIATELY.[/bold red]\n\n"
        "Do not delay seeking care.",
        border_style="red",
        title="[bold red]URGENT[/bold red]",
    )


def _assessment_panel(result: dict) -> Panel:
    lines = []
    care_level = result.get("care_level")

    lines.append(f"[bold blue]{_safe(result.get('care_setting'))}[/bold blue]\n")

    if result.get("recommendation"):
        lines.append(f"[bold]Recommendation:[/bold] {_safe(result['recommendation'])}\n")

    facilities = result.get("facilities", [])
    if care_level == 1:
        lines.append(
            "[bold]No facility visit needed.[/bold] Rest and OTC treatment recommended.\n"
        )
    elif facilities:
        lines.append("[bold]Nearby facilities (closest first):[/bold]")
        for facility in facilities:
            lines.append(f"  • {_facility_line(facility)}")
        lines.append("")
    else:
        lines.append(
            f"[dim]No in-person facilities found within {MAX_DISTANCE_MILES} miles. "
            f"See online options below.[/dim]\n"
        )

    online = result.get("online_providers", [])
    if online:
        lines.append("[bold]Online alternatives (no travel needed):[/bold]")
        for provider in online:
            lines.append(
                f"  • {_safe(provider['name'])} — {_safe(provider['estimated_cost'])}"
                f"  [dim]{_safe(provider['url'])}[/dim]"
            )
        lines.append("")

    if result.get("reasoning"):
        lines.append(f"[dim]Why this recommendation:[/dim] {_safe(result['reasoning'])}\n")

    if result.get("visit_steps") and care_level != 1:
        lines.append("[bold]What to expect:[/bold]")
        for step in result["visit_steps"][:4]:
            lines.append(f"  • {_safe(step)}")
        lines.append("")

    if result.get("labs") and care_level != 1:
        labs = ", ".join(_safe(lab) for lab in result["labs"][:3])
        lines.append(f"[bold]Labs likely needed:[/bold] {labs}\n")

    if result.get("interventions") and care_level != 1:
        interventions = ", ".join(_safe(item) for item in result["interventions"][:3])
        lines.append(f"[bold]Interventions likely needed:[/bold] {interventions}\n")

    lines.append("[dim]For informational purposes only. Not medical advice.[/dim]")

    return Panel(
        Text.from_markup("\n".join(lines)),
        title="[bold]Triage Result[/bold]",
        border_style="blue",
    )


def _facility_line(facility: dict) -> str:
    distance = facility.get("distance_miles")
    distance_text = f"{distance:.1f}mi" if distance is not None else "distance unknown"
    cost = facility.get("cost")
    cost_text = _safe(cost) if cost else f"[dim]{NO_PRICE}[/dim]"
    return (
        f"{_safe(facility['name'])} ({_safe(facility['type'])}, {distance_text})"
        f" — {cost_text}"
    )
