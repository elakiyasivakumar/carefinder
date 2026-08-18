#!/usr/bin/env python3
"""Full-pipeline evaluation: model + facility search + live pricing, audited.

Unlike run_eval.py (which exercises the model alone), this runs run_triage()
end to end so map and price behaviour can be checked. Every case is audited for
invented facilities and invented prices.

Nominatim allows 1 request/second, so facility search is serialised behind a
lock while model and pricing calls run in parallel.
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from audit import audit_result
from services.maps_service import MAX_DISTANCE_MILES
from triage_cli import Deps, build_deps, run_triage

EVAL_DATA = Path(__file__).resolve().parent / "eval_data.json"


class SerialisedMaps:
    """One Nominatim request at a time, whatever the worker count."""

    def __init__(self, inner):
        self._inner = inner
        self._lock = threading.Lock()

    def find_facilities(self, zip_code, facility_type=None):
        with self._lock:
            return self._inner.find_facilities(zip_code, facility_type=facility_type)

    def geocode_zip(self, zip_code):
        with self._lock:
            try:
                loc = self._inner.geocode(f"{zip_code}, USA")
                return (loc.latitude, loc.longitude) if loc else None
            except Exception:
                return None


def run_case(deps, case: Dict, centroid) -> Dict:
    given = case["input"]
    started = time.time()

    result = run_triage(
        given["age"], given["gender"], given["symptoms"], given["zip_code"], deps=deps
    )

    checks = audit_result(result, centroid, radius_miles=MAX_DISTANCE_MILES)
    expected = case["expected"]["care_level"]

    if result.get("is_emergency"):
        actual = "ER (gate fired)"
    elif result.get("seek_care"):
        actual = "BLOCKED — gate unavailable"
    elif result.get("error"):
        actual = "ERROR"
    else:
        actual = result.get("care_setting", "?")

    passed = (
        actual.startswith("ER") if case["category"] == "ER"
        else actual == expected
    )

    return {
        "id": case["id"],
        "category": case["category"],
        "zip": given["zip_code"],
        "zip_label": case.get("zip_label", ""),
        "age": given["age"],
        "gender": given["gender"],
        "symptoms": given["symptoms"],
        "expected": expected,
        "actual": actual,
        "passed": passed,
        "care_level": result.get("care_level"),
        "reasoning": result.get("reasoning", ""),
        "labs": result.get("labs", []),
        "interventions": result.get("interventions", []),
        "facilities": result.get("facilities", []),
        "audit": checks,
        "seconds": round(time.time() - started, 1),
    }


def run_cycle(deps, cases: List[Dict], centroids: Dict[str, object],
              label: str, workers: int) -> List[Dict]:
    print(f"\n=== CYCLE {label}: {len(cases)} cases, {workers} workers ===", flush=True)
    results = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_case, deps, c, centroids.get(c["input"]["zip_code"])): c
            for c in cases
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                r = future.result()
            except Exception as e:
                print(f"  [{case['id']:2}] CRASHED: {type(e).__name__}: {e}", flush=True)
                continue
            results.append(r)
            flag = "ok " if r["passed"] else "MISS"
            dirty = "" if r["audit"]["clean"] else "  <-- AUDIT FLAG"
            print(f"  [{r['id']:2}] {flag} {r['category']:<14} -> {r['actual']:<24}"
                  f" {r['audit']['facility_count']} fac{dirty}", flush=True)

    return sorted(results, key=lambda r: r["id"])


def summarise(results: List[Dict], label: str) -> Dict:
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    by_cat = {}
    for cat in ["Home Care", "Retail Clinic", "Urgent Care", "ER"]:
        subset = [r for r in results if r["category"] == cat]
        if subset:
            by_cat[cat] = f"{sum(1 for r in subset if r['passed'])}/{len(subset)}"

    bad_fac = [f for r in results for f in r["audit"]["hallucinated_facilities"]]
    bad_price = [p for r in results for p in r["audit"]["hallucinated_prices"]]
    expected_maps = [r for r in results if r["audit"]["maps_expected"]]
    got_maps = [r for r in expected_maps if r["audit"]["maps_returned"]]

    return {
        "cycle": label,
        "total": total,
        "passed": passed,
        "accuracy": round(100 * passed / total, 1) if total else 0.0,
        "by_category": by_cat,
        "cases_expecting_maps": len(expected_maps),
        "cases_that_got_maps": len(got_maps),
        "total_facilities": sum(r["audit"]["facility_count"] for r in results),
        "prices_verified": sum(r["audit"]["prices_verified"] for r in results),
        "prices_unavailable": sum(r["audit"]["prices_unavailable"] for r in results),
        "hallucinated_facilities": bad_fac,
        "hallucinated_prices": bad_price,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=Path("cycle_results.json"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = json.loads(EVAL_DATA.read_text())["test_cases"]
    if args.limit:
        cases = cases[:args.limit]

    non_er = [c for c in cases if c["category"] != "ER"]
    model_calls = (len(cases) - len(non_er)) + 2 * len(non_er)
    print(f"{len(cases)} cases x {args.cycles} cycles")
    print(f"  model calls: ~{model_calls * args.cycles}")
    print(f"  price calls: up to ~{3 * len(non_er) * args.cycles} (3 per case, non-Home-Care only)")

    if args.dry_run:
        for c in cases:
            print(f"  [{c['id']:2}] {c['category']:<14} {c['input']['zip_code']} "
                  f"({c.get('zip_label','')[:28]}) | {c['input']['symptoms'][:44]}")
        print("\nDry run — nothing contacted, nothing billed.")
        return 0

    real = build_deps()
    maps = SerialisedMaps(real.maps)
    deps = Deps(llm=real.llm, maps=maps, costs=real.costs, telehealth=real.telehealth)

    zips = sorted({c["input"]["zip_code"] for c in cases})
    print(f"\ngeocoding {len(zips)} zips for the audit baseline...")
    centroids = {z: maps.geocode_zip(z) for z in zips}
    for z, c in centroids.items():
        print(f"  {z}: {c}")

    all_cycles, summaries = [], []
    for n in range(1, args.cycles + 1):
        results = run_cycle(deps, cases, centroids, str(n), args.workers)
        all_cycles.append(results)
        s = summarise(results, str(n))
        summaries.append(s)
        print(f"\n  cycle {n}: {s['passed']}/{s['total']} ({s['accuracy']}%) | "
              f"facilities={s['total_facilities']} prices_verified={s['prices_verified']} "
              f"hallucinations={len(s['hallucinated_facilities']) + len(s['hallucinated_prices'])}")

    args.out.write_text(json.dumps(
        {"summaries": summaries, "cycles": all_cycles}, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
