#!/usr/bin/env python3
"""Drive the CareFinder web UI with Playwright and capture the demo screenshots.

Runs against a live localhost server, so every screenshot shows real model
output, real OpenStreetMap facilities and real retrieved prices. Nothing here
is mocked or staged.

    python demo/capture.py --base-url http://127.0.0.1:8000
"""
import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "screenshots"

CASES = [
    {
        "slug": "01-landing",
        "title": "Landing page",
        "note": "The standing 'not medical advice' notice is visible before any input.",
        "fill": None,
    },
    {
        "slug": "02-urgent-care",
        "title": "Urgent care — fever with systemic symptoms",
        "note": "Three real clinics from OpenStreetMap, mapped, each with a self-pay band.",
        "fill": {"age": "22", "gender": "Female", "zip": "66502",
                 "symptoms": "High fever 103F, severe body aches, extreme fatigue for 2 days"},
    },
    {
        "slug": "03-retail-clinic",
        "title": "Routine complaint — pink eye",
        "note": "Telehealth at $35-75 sits alongside in-person options.",
        "fill": {"age": "8", "gender": "Male", "zip": "10001",
                 "symptoms": "Pink eye with yellow discharge, started this morning"},
    },
    {
        "slug": "04-emergency",
        "title": "Emergency gate fires",
        "note": "MedGemma's first pass stops the pipeline. No facility search, no pricing.",
        "fill": {"age": "55", "gender": "Male", "zip": "10001",
                 "symptoms": "Chest pain, shortness of breath, sweating, pain radiating to left arm"},
    },
    {
        "slug": "05-rural",
        "title": "Rural zip — genuinely sparse",
        "note": "Tonopah NV has two clinics. The app says so rather than inventing more.",
        "fill": {"age": "40", "gender": "Female", "zip": "89049",
                 "symptoms": "Skin rash on arms, red and itchy, spreading since yesterday"},
    },
    {
        "slug": "06-home-care",
        "title": "Home care — no visit needed",
        "note": "Level 1 does no lookups at all, so nothing is searched or priced.",
        "fill": {"age": "30", "gender": "Male", "zip": "95134",
                 "symptoms": "Runny nose, mild cough, slight congestion, no fever"},
    },
]


def capture(base_url: str, timeout_ms: int) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    captured = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1000},
                                device_scale_factor=2)

        for case in CASES:
            print(f"  {case['slug']} ...", flush=True)
            page.goto(base_url, wait_until="networkidle")

            if case["fill"]:
                f = case["fill"]
                page.fill("#age", f["age"])
                page.select_option("#gender", f["gender"])
                page.fill("#symptoms", f["symptoms"])
                page.fill("#zip", f["zip"])
                started = time.time()
                page.click("#go")
                # Wait for the placeholder text to be replaced by a real answer.
                page.wait_for_function(
                    "() => !document.querySelector('#out').textContent.includes('Assessing symptoms')",
                    timeout=timeout_ms,
                )
                page.wait_for_timeout(2500)   # let map tiles settle
                elapsed = round(time.time() - started, 1)
            else:
                elapsed = None

            path = OUT / f"{case['slug']}.png"
            page.screenshot(path=str(path), full_page=True)
            text = page.inner_text("#out")
            captured.append({**case, "file": path.name, "seconds": elapsed,
                             "result_text": text[:1200]})
            print(f"     saved {path.name}" + (f" ({elapsed}s)" if elapsed else ""))

        browser.close()

    (OUT / "captures.json").write_text(json.dumps(captured, indent=2))
    return captured


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout-ms", type=int, default=180000)
    args = ap.parse_args()

    print(f"capturing from {args.base_url}")
    results = capture(args.base_url, args.timeout_ms)
    print(f"\n{len(results)} screenshots -> {OUT}")
