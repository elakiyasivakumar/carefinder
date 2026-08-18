#!/usr/bin/env python3
"""Calibration check for MedGemma on care-level routing.

ER cases exercise the emergency gate (Phase 1). L1-L3 cases exercise the
assessment (Phase 2). Every model call costs money on a live endpoint, so the
runner reports exactly how many it will make and supports --limit / --category.

Exit code is 1 if any case fails, so this can gate a build.
"""
import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

from triage import AssessmentError, GateVerdict, assess, emergency_gate

EVAL_DATA = Path(__file__).resolve().parent / "eval_data.json"


@dataclass
class EvalResult:
    test_id: int
    category: str
    passed: bool
    expected: str
    actual: str
    symptoms: str = ""
    labs: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    reasoning: str = ""
    error: str = ""


def load_eval_data(path: Path = EVAL_DATA) -> List[Dict]:
    with open(path) as f:
        return json.load(f)["test_cases"]


def run_single_eval(llm, test_case: Dict) -> EvalResult:
    given = test_case["input"]
    category = test_case["category"]
    expected = test_case["expected"]["care_level"]
    common = dict(
        test_id=test_case["id"], category=category,
        expected=expected, symptoms=given["symptoms"],
    )

    if category == "ER":
        verdict = emergency_gate(llm, given["age"], given["gender"], given["symptoms"])
        return EvalResult(
            passed=verdict is GateVerdict.EMERGENCY,
            actual={
                GateVerdict.EMERGENCY: "ER (gate fired)",
                GateVerdict.NOT_EMERGENCY: "MISSED — routed to assessment",
                GateVerdict.UNKNOWN: "UNKNOWN — gate could not decide",
            }[verdict],
            **common,
        )

    # Non-ER cases must clear the gate first, exactly as a real run does.
    verdict = emergency_gate(llm, given["age"], given["gender"], given["symptoms"])
    if verdict is not GateVerdict.NOT_EMERGENCY:
        return EvalResult(
            passed=False,
            actual=f"gate said {verdict.value} — over-triaged",
            error="emergency gate did not clear a non-emergency case",
            **common,
        )

    try:
        result = assess(llm, given["age"], given["gender"], given["symptoms"])
    except AssessmentError as e:
        return EvalResult(passed=False, actual="ERROR", error=str(e), **common)

    return EvalResult(
        passed=result.care_setting == expected,
        actual=result.care_setting,
        labs=result.labs,
        interventions=result.interventions,
        reasoning=result.reasoning,
        **common,
    )


def print_results(results: List[EvalResult]) -> None:
    print("\n" + "=" * 96)
    print("  MedGemma calibration — Phase 1 (ER): gate accuracy | Phase 2 (L1-3): care level")
    print("=" * 96)
    print(f"\n{'ID':<4} {'Category':<15} {'Pass':<6} {'Expected':<16} {'Actual':<30} Note")
    print("-" * 96)
    for r in results:
        print(f"{r.test_id:<4} {r.category:<15} {'PASS' if r.passed else 'FAIL':<6} "
              f"{r.expected:<16} {r.actual:<30} {r.error[:24]}")
    print("-" * 96)


def print_summary(results: List[EvalResult]) -> None:
    total = len(results)
    if not total:
        print("\nNo test cases matched.")
        return

    passed = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 50}\n  SUMMARY\n{'=' * 50}")
    print(f"Total:   {total}")
    print(f"Passed:  {passed}/{total} ({100 * passed / total:.1f}%)")
    print("\n  BY CATEGORY:")
    for cat in ["Home Care", "Retail Clinic", "Urgent Care", "ER"]:
        subset = [r for r in results if r.category == cat]
        if subset:
            print(f"  {cat:<15}: {sum(1 for r in subset if r.passed)}/{len(subset)}")
    print("=" * 50 + "\n")


def select(cases: List[Dict], ids, categories, limit, one_per_category) -> List[Dict]:
    if ids:
        cases = [c for c in cases if c["id"] in ids]
    if categories:
        wanted = {c.lower() for c in categories}
        cases = [c for c in cases if c["category"].lower() in wanted]
    if one_per_category:
        seen, picked = set(), []
        for c in cases:
            if c["category"] not in seen:
                seen.add(c["category"])
                picked.append(c)
        cases = picked
    if limit:
        cases = cases[:limit]
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description="MedGemma calibration eval")
    parser.add_argument("--ids", type=int, nargs="+")
    parser.add_argument("--category", nargs="+", help="e.g. ER 'Urgent Care'")
    parser.add_argument("--limit", type=int, help="cap the number of cases (cost control)")
    parser.add_argument("--one-per-category", action="store_true",
                        help="one case per category — cheapest generalisation check")
    parser.add_argument("--out", type=Path, help="write results JSON here")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the cases and the model-call count, spend nothing")
    args = parser.parse_args()

    cases = select(load_eval_data(), args.ids, args.category,
                   args.limit, args.one_per_category)

    # ER cases cost 1 call; others cost 2 (gate, then assessment).
    calls = sum(1 if c["category"] == "ER" else 2 for c in cases)
    print(f"{len(cases)} case(s) selected → {calls} model call(s).")

    if args.dry_run:
        for c in cases:
            print(f"  [{c['id']:2}] {c['category']:<15} | {c['input']['symptoms'][:56]}")
        print("\nDry run — no endpoint contacted, nothing billed.")
        return 0

    from triage_cli import build_deps
    llm = build_deps().llm

    results = []
    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case['category']}: {case['input']['symptoms'][:56]}")
        result = run_single_eval(llm, case)
        results.append(result)
        print(f"  -> {result.actual} (expected {result.expected}) "
              f"{'PASS' if result.passed else 'FAIL'}")

    print_results(results)
    print_summary(results)

    if args.out:
        args.out.write_text(json.dumps([asdict(r) for r in results], indent=2))
        print(f"Wrote {args.out}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
