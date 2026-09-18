"""Verify the public cases offline or against a running real-LLM API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from gridwise.optimizer import optimize
from gridwise.replay import replay
from gridwise.validation import validate_directives, validate_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Running API base URL; omitted = optimizer-only, reference directives.")
    parser.add_argument("--cases", type=Path, default=Path(__file__).resolve().parents[1] / "data/public_sample_cases.json")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())["cases"]
    print("LIVE LLM + API validation" if args.url else "OFFLINE optimizer validation (public reference directives; no LLM tested)")
    failures = 0
    durations = []
    with httpx.Client(timeout=30, trust_env=False) as client:
        if args.url:
            try:
                health = client.get(args.url.rstrip("/") + "/health")
                health.raise_for_status()
                assert health.json().get("status") == "ok"
            except Exception:
                print("FAIL: /health is not ready. Configure the model and start the API.")
                return 1
        for case in cases:
            start = time.perf_counter()
            try:
                scenario = validate_scenario(case["input"])
                expected = case["expected_output"]
                directives = validate_directives(scenario, expected["directive_interpretation"])
                if args.url:
                    response = client.post(args.url.rstrip("/") + "/optimize-energy", json=scenario)
                    response.raise_for_status()
                    result = response.json()
                else:
                    result = optimize(scenario, directives)
                replay(scenario, result, directives=directives, tolerance=0.01)
                difference = abs(result["total_cost_bdt"] - expected["total_cost_bdt"])
                if difference > 0.01:
                    raise ValueError("Cost differs from public optimum.")
                elapsed = time.perf_counter() - start
                durations.append(elapsed)
                print(f"PASS {case['id']}: BDT {result['total_cost_bdt']:,.2f}, {elapsed:.3f}s")
            except Exception as exc:
                failures += 1
                print(f"FAIL {case['id']}: {type(exc).__name__} (check service configuration or scenario validity)")
    print(f"\n{len(cases) - failures}/{len(cases)} cases passed.")
    if durations:
        print(f"Mean {statistics.mean(durations):.3f}s; maximum {max(durations):.3f}s (not a p95 benchmark).")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
