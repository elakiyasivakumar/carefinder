#!/usr/bin/env python3
"""Medical Triage CLI — Rich terminal UI over MedGemma 27B."""
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.live import Live

from render import render_result
from triage_cli import run_triage, ZIP_CODE_RE

console = Console()


def print_header():
    console.print()
    console.print(Panel.fit(
        "[bold blue]Medical Triage Assistant[/bold blue]\n"
        "[dim]Powered by MedGemma 27B[/dim]",
        border_style="blue"
    ))
    console.print()


def get_user_input() -> tuple[int, str, str, str]:
    """Collect and validate age, gender, symptoms, zip code from the user."""
    console.print("[bold]Patient Information:[/bold]")

    age_str = Prompt.ask("[green]Age[/green]")
    while not age_str.isdigit() or not (0 <= int(age_str) <= 120):
        console.print("[red]Please enter a valid age (0–120)[/red]")
        age_str = Prompt.ask("[green]Age[/green]")
    age = int(age_str)

    gender = Prompt.ask("[green]Gender[/green]", choices=["M", "F", "Other"], default="Other")

    console.print()
    console.print("[bold]Describe your symptoms:[/bold]")
    console.print("[dim](e.g. 'sore throat and fever for 2 days')[/dim]")
    symptoms = Prompt.ask("[green]Symptoms[/green]")

    console.print()
    zip_code = Prompt.ask("[green]Zip code[/green]")
    while not ZIP_CODE_RE.match(zip_code.strip()):
        console.print("[red]Please enter a valid 5-digit US zip code (e.g. 10001)[/red]")
        zip_code = Prompt.ask("[green]Zip code[/green]")

    return age, gender, symptoms, zip_code


def display_result(result: dict) -> None:
    """Render a run_triage() result dict using Rich."""
    console.print()
    console.print(render_result(result))


def main():
    print_header()

    try:
        age, gender, symptoms, zip_code = get_user_input()

        console.print()
        with Live(
            Panel("[bold blue]Assessing symptoms and finding options...[/bold blue]", border_style="blue"),
            console=console,
            refresh_per_second=4,
        ):
            result = run_triage(age, gender, symptoms, zip_code)

        display_result(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("[dim]Please try again or consult a healthcare provider directly.[/dim]")
        sys.exit(1)

    console.print()


if __name__ == "__main__":
    main()
