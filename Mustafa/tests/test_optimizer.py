"""Solver checks use reference objectives and an independent physical replay."""

from copy import deepcopy
import json
from math import fsum, isfinite
from pathlib import Path
import random

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import gridwise.optimizer as optimizer_module
from gridwise.optimizer import InfeasibleSchedule, OptimizationError, compile_constraints, optimize
from gridwise.replay import replay as standalone_replay


SAMPLES = json.loads((Path(__file__).resolve().parents[1] / "data/public_sample_cases.json").read_text())["cases"]


def scenario(demand=10.0, solar=0.0, tariff=1.0, **battery_overrides):
    battery = {
        "capacity_kwh": 20.0, "initial_energy_kwh": 10.0,
        "minimum_energy_kwh": 0.0, "max_charge_kwh_per_hour": 10.0,
        "max_discharge_kwh_per_hour": 10.0,
    }
    battery.update(battery_overrides)
    return {
        "scenario_id": "unit-test", "operator_notes": ["A note for this test."],
        "hours": [{"hour": h, "demand_kwh": demand, "solar_kwh": solar, "tariff_bdt_per_kwh": tariff} for h in range(24)],
        "battery": battery,
    }


def directive(kind, hours=None, index=0, **adjustment):
    return {
        "note_index": index, "applies": kind != "no_op", "directive_type": kind,
        "structured_adjustment": None if kind == "no_op" else {"hours": hours, **adjustment},
        "explanation": "Synthetic test directive.",
    }


def replay(request, directives, response):
    """Check physical correctness independently of the optimizer's constraint compiler."""
    assert response["scenario_id"] == request["scenario_id"]
    assert response["directive_interpretation"] == directives
    assert [row["hour"] for row in response["hourly_plan"]] == list(range(24))
    battery = request["battery"]
    energy = battery["initial_energy_kwh"]
    by_hour = {item["hour"]: item for item in request["hours"]}
    for row in response["hourly_plan"]:
        h = row["hour"]
        hour = by_hour[h]
        active = [d for d in directives if d["applies"] and h in d["structured_adjustment"]["hours"]]
        factor = min([1.0] + [d["structured_adjustment"]["factor"] for d in active if d["directive_type"] == "solar_reduction"])
        reserve = max([battery["minimum_energy_kwh"]] + [d["structured_adjustment"]["minimum_energy_kwh"] for d in active if d["directive_type"] == "minimum_battery_reserve"])
        for key in ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"):
            assert isfinite(row[key]) and row[key] >= 0
        assert row["solar_used_kwh"] <= hour["solar_kwh"] * factor + 1e-7
        amount = row["battery_kwh"]
        action = row["battery_action"]
        assert action in {"charge", "discharge", "idle"}
        if action == "charge":
            assert amount <= battery["max_charge_kwh_per_hour"] + 1e-7
            assert not any(d["directive_type"] == "no_charge_window" for d in active)
            energy += amount
            delta = amount
        elif action == "discharge":
            assert amount <= battery["max_discharge_kwh_per_hour"] + 1e-7
            assert not any(d["directive_type"] == "no_discharge_window" for d in active)
            energy -= amount
            delta = -amount
        else:
            assert amount == 0
            delta = 0
        assert row["grid_kwh"] + row["solar_used_kwh"] - delta == pytest.approx(hour["demand_kwh"], abs=1e-7)
        assert row["battery_energy_after_kwh"] == pytest.approx(energy, abs=1e-7)
        assert reserve - 1e-7 <= energy <= battery["capacity_kwh"] + 1e-7
        for d in active:
            if d["directive_type"] == "max_grid_window":
                assert row["grid_kwh"] <= d["structured_adjustment"]["max_grid_kwh"] + 1e-7
    assert energy == pytest.approx(battery["initial_energy_kwh"], abs=1e-7)
    rows = response["hourly_plan"]
    assert response["total_grid_kwh"] == pytest.approx(fsum(row["grid_kwh"] for row in rows), abs=1e-7)
    assert response["total_cost_bdt"] == pytest.approx(fsum(row["grid_kwh"] * by_hour[row["hour"]]["tariff_bdt_per_kwh"] for row in rows), abs=1e-7)
    assert response["peak_grid_kwh"] == pytest.approx(max(row["grid_kwh"] for row in rows), abs=1e-7)


@pytest.mark.parametrize("case", SAMPLES, ids=[case["id"] for case in SAMPLES])
def test_all_public_reference_optima(case):
    directives = case["expected_output"]["directive_interpretation"]
    response = optimize(case["input"], directives)
    replay(case["input"], directives, response)
    assert response["total_cost_bdt"] == pytest.approx(case["expected_output"]["total_cost_bdt"], abs=0.01)


def test_zero_price_ties_minimize_peak_then_avoid_unnecessary_cycling():
    request = scenario(tariff=0)
    request["hours"][0]["demand_kwh"] = 20
    request["hours"][1]["demand_kwh"] = 0
    directives = [directive("no_op")]
    result = optimize(request, directives)
    replay(request, directives, result)
    assert result["total_cost_bdt"] == 0
    assert result["peak_grid_kwh"] == pytest.approx(10)
    assert sum(row["battery_kwh"] for row in result["hourly_plan"]) == pytest.approx(20)


def test_flat_prices_and_load_need_no_battery_cycles():
    request = scenario()
    directives = [directive("no_op")]
    result = optimize(request, directives)
    replay(request, directives, result)
    assert all(row["battery_action"] == "idle" for row in result["hourly_plan"])
    assert result["total_cost_bdt"] == pytest.approx(240)


def test_no_free_energy_from_initial_battery():
    request = scenario(max_charge_kwh_per_hour=0)
    request["hours"][0]["tariff_bdt_per_kwh"] = 100
    directives = [directive("no_op")]
    result = optimize(request, directives)
    replay(request, directives, result)
    assert all(row["battery_action"] == "idle" for row in result["hourly_plan"])
    assert result["total_cost_bdt"] == pytest.approx(1230)


def test_surplus_solar_can_be_curtailed_without_grid_export():
    request = scenario(demand=0.125, solar=2.75, capacity_kwh=0, initial_energy_kwh=0)
    directives = [directive("no_op")]
    result = optimize(request, directives)
    replay(request, directives, result)
    assert result["total_grid_kwh"] == 0
    assert all(row["solar_used_kwh"] == 0.125 for row in result["hourly_plan"])


def test_overlapping_solar_reductions_use_original_forecast_and_strictest_factor():
    request = scenario(demand=10, solar=10, capacity_kwh=0, initial_energy_kwh=0)
    directives = [directive("solar_reduction", [12, 13], factor=0.5), directive("solar_reduction", [13, 14], index=1, factor=0.2)]
    result = optimize(request, directives)
    replay(request, directives, result)
    assert result["total_grid_kwh"] == pytest.approx(21)
    assert result["hourly_plan"][13]["solar_used_kwh"] == pytest.approx(2)


def test_overlapping_reserves_caps_and_prohibitions_intersect():
    request = scenario()
    directives = [
        directive("minimum_battery_reserve", [2, 3], minimum_energy_kwh=12),
        directive("minimum_battery_reserve", [3, 4], index=1, minimum_energy_kwh=15),
        directive("max_grid_window", [2, 3], index=2, max_grid_kwh=14),
        directive("max_grid_window", [3, 4], index=3, max_grid_kwh=12),
        directive("no_charge_window", [3], index=4),
        directive("no_discharge_window", [3], index=5),
    ]
    rules = compile_constraints(request, directives)
    assert rules["minimum_energy"][2:5] == [12, 15, 15]
    assert rules["max_grid"][2:5] == [14, 12, 12]
    result = optimize(request, directives)
    replay(request, directives, result)
    assert result["hourly_plan"][3]["battery_action"] == "idle"


def test_infeasible_cap_and_discharge_prohibition_is_controlled():
    request = scenario()
    directives = [directive("max_grid_window", [0], max_grid_kwh=0), directive("no_discharge_window", [0], index=1)]
    with pytest.raises(InfeasibleSchedule):
        optimize(request, directives)


def test_final_hour_reserve_cannot_override_battery_neutrality():
    request = scenario()
    directives = [directive("minimum_battery_reserve", [23], minimum_energy_kwh=15)]
    with pytest.raises(InfeasibleSchedule):
        optimize(request, directives)


def test_shuffled_hour_input_is_sorted_and_inputs_are_never_mutated():
    case = deepcopy(SAMPLES[0])
    request = case["input"]
    request["hours"].reverse()
    directives = case["expected_output"]["directive_interpretation"]
    before = deepcopy((request, directives))
    response = optimize(request, directives)
    replay(request, directives, response)
    assert (request, directives) == before
    response["directive_interpretation"][0]["structured_adjustment"]["factor"] = 0
    assert directives == before[1]


def test_fractional_battery_limits_and_directive_values():
    request = scenario(demand=0.77, solar=0.23, tariff=1.09, capacity_kwh=2.13, initial_energy_kwh=0.97, minimum_energy_kwh=0.11, max_charge_kwh_per_hour=0.37, max_discharge_kwh_per_hour=0.29)
    request["hours"][16]["tariff_bdt_per_kwh"] = 13.79
    directives = [directive("minimum_battery_reserve", [16], minimum_energy_kwh=0.59)]
    result = optimize(request, directives)
    replay(request, directives, result)
    assert result["hourly_plan"][16]["battery_action"] == "discharge"
    assert result["hourly_plan"][16]["battery_kwh"] == pytest.approx(0.29)


def integer_dynamic_program(request, directives):
    """Independent shortest path over every integer battery state.

    These generated inputs have integer demands, effective solar, reserves, and
    rate limits. The underlying lossless energy-flow network has an integral
    optimum, so enumerating its integer energy states gives the exact continuous
    optimum too. This oracle does not call the LP or its constraint compiler.
    """
    battery = request["battery"]
    costs = {battery["initial_energy_kwh"]: 0}
    for hour in sorted(request["hours"], key=lambda item: item["hour"]):
        solar = hour["solar_kwh"]
        reserve = battery["minimum_energy_kwh"]
        charge = battery["max_charge_kwh_per_hour"]
        discharge = battery["max_discharge_kwh_per_hour"]
        grid_cap = float("inf")
        for entry in directives:
            adjustment = entry["structured_adjustment"]
            if adjustment is None or hour["hour"] not in adjustment["hours"]:
                continue
            kind = entry["directive_type"]
            if kind == "solar_reduction":
                solar = min(solar, hour["solar_kwh"] * adjustment["factor"])
            elif kind == "minimum_battery_reserve":
                reserve = max(reserve, adjustment["minimum_energy_kwh"])
            elif kind == "no_charge_window":
                charge = 0
            elif kind == "no_discharge_window":
                discharge = 0
            elif kind == "max_grid_window":
                grid_cap = min(grid_cap, adjustment["max_grid_kwh"])
        assert solar == int(solar)
        next_costs = {}
        for before, previous_cost in costs.items():
            for after in range(int(reserve), int(battery["capacity_kwh"]) + 1):
                delta = after - before
                if delta > charge or -delta > discharge:
                    continue
                supply_needed = hour["demand_kwh"] + delta
                if supply_needed < 0:
                    continue  # There is no grid export or battery-energy dumping.
                grid = max(0, supply_needed - solar)
                if grid > grid_cap:
                    continue
                cost = previous_cost + grid * hour["tariff_bdt_per_kwh"]
                next_costs[after] = min(next_costs.get(after, float("inf")), cost)
        costs = next_costs
    return costs.get(battery["initial_energy_kwh"], float("inf"))


def random_integer_cases():
    rng = random.Random(20260918)
    cases = []
    kinds = ["solar_reduction", "minimum_battery_reserve", "no_charge_window", "no_discharge_window", "max_grid_window", "no_op"]
    for attempt in range(1000):
        capacity = rng.randrange(9)
        minimum = rng.randint(0, capacity)
        initial = rng.randint(minimum, capacity)
        request = scenario(
            capacity_kwh=capacity, initial_energy_kwh=initial,
            minimum_energy_kwh=minimum, max_charge_kwh_per_hour=rng.randrange(4),
            max_discharge_kwh_per_hour=rng.randrange(4),
        )
        request["scenario_id"] = f"dp-oracle-{attempt}"
        for hour in request["hours"]:
            hour.update(demand_kwh=rng.randrange(8), solar_kwh=2 * rng.randrange(5), tariff_bdt_per_kwh=rng.randrange(10))
        directives = []
        for index in range(rng.randint(1, 3)):
            kind = rng.choice(kinds)
            start = rng.randrange(24)
            window = list(range(start, min(24, start + rng.randint(1, 8))))
            parameters = {}
            if kind == "solar_reduction":
                parameters["factor"] = rng.choice([0, 0.5, 1])
            elif kind == "minimum_battery_reserve":
                parameters["minimum_energy_kwh"] = rng.randint(0, capacity)
            elif kind == "max_grid_window":
                parameters["max_grid_kwh"] = rng.randrange(9)
            directives.append(directive(kind, window, index=index, **parameters))
        request["operator_notes"] = [f"Synthetic directive {index}." for index in range(len(directives))]
        expected = integer_dynamic_program(request, directives)
        if isfinite(expected):
            cases.append((request, directives, expected))
        if len(cases) == 80:
            break
    assert len(cases) == 80
    assert {entry["directive_type"] for _, entries, _ in cases for entry in entries} == set(kinds)
    return cases


RANDOM_INTEGER_CASES = random_integer_cases()


@pytest.mark.parametrize("request_data,directives,expected", RANDOM_INTEGER_CASES, ids=[item[0]["scenario_id"] for item in RANDOM_INTEGER_CASES])
def test_random_integer_optima_against_independent_dynamic_program(request_data, directives, expected):
    result = optimize(request_data, directives)
    standalone_replay(request_data, result, directives)
    assert result["total_cost_bdt"] == pytest.approx(expected, abs=1e-7, rel=0)


def test_last_hour_zero_grid_cap_reserve_and_neutrality_can_require_early_charge():
    request = scenario(demand=0, capacity_kwh=8, initial_energy_kwh=4, max_charge_kwh_per_hour=4, max_discharge_kwh_per_hour=4)
    request["hours"][23]["demand_kwh"] = 4
    request["operator_notes"] = ["Last-hour cap.", "Last-hour reserve.", "Last-hour charging prohibition."]
    directives = [
        directive("max_grid_window", [23], max_grid_kwh=0),
        directive("minimum_battery_reserve", [23], index=1, minimum_energy_kwh=4),
        directive("no_charge_window", [23], index=2),
    ]
    result = optimize(request, directives)
    standalone_replay(request, result, directives)
    assert integer_dynamic_program(request, directives) == 4
    assert result["total_cost_bdt"] == pytest.approx(4, abs=1e-7, rel=0)
    assert result["hourly_plan"][22]["battery_energy_after_kwh"] == pytest.approx(8, abs=1e-7, rel=0)
    assert result["hourly_plan"][23]["battery_kwh"] == pytest.approx(4, abs=1e-7, rel=0)


@pytest.mark.parametrize("status", [1, 3, 4])
def test_primary_timeout_or_numerical_failure_is_controlled(monkeypatch, status):
    monkeypatch.setattr(optimizer_module, "linprog", lambda *args, **kwargs: OptimizeResult(success=False, status=status))
    with pytest.raises(OptimizationError) as caught:
        optimize(scenario(), [directive("no_op")])
    assert not isinstance(caught.value, InfeasibleSchedule)


def test_nonfinite_primary_solution_is_rejected_before_tie_break(monkeypatch):
    calls = []

    def failed_solver(*args, **kwargs):
        calls.append(1)
        return OptimizeResult(success=True, status=0, x=np.full(121, float("nan")))

    monkeypatch.setattr(optimizer_module, "linprog", failed_solver)
    with pytest.raises(OptimizationError):
        optimize(scenario(), [directive("no_op")])
    assert len(calls) == 1


@pytest.mark.parametrize("failing_stage", [2, 3])
@pytest.mark.parametrize("failure", ["time_limit", "numerical", "nonfinite", "exception"])
def test_optional_solver_failure_retains_valid_cost_optimum(monkeypatch, failing_stage, failure):
    real_solver = optimizer_module.linprog
    calls = []

    def interrupted_solver(*args, **kwargs):
        calls.append(1)
        if len(calls) == failing_stage:
            if failure == "exception":
                raise ValueError("private native solver diagnostics")
            if failure == "nonfinite":
                return OptimizeResult(success=True, status=0, x=np.full(121, float("nan")))
            return OptimizeResult(success=False, status=1 if failure == "time_limit" else 4)
        return real_solver(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "linprog", interrupted_solver)
    case = SAMPLES[0]
    directives = case["expected_output"]["directive_interpretation"]
    result = optimize(case["input"], directives)
    standalone_replay(case["input"], result, directives)
    assert result["total_cost_bdt"] == pytest.approx(case["expected_output"]["total_cost_bdt"], abs=1e-7, rel=0)
    assert len(calls) == failing_stage


def test_native_solver_exception_becomes_controlled_error(monkeypatch):
    def broken_solver(*args, **kwargs):
        raise ValueError("private native solver diagnostics")

    monkeypatch.setattr(optimizer_module, "linprog", broken_solver)
    with pytest.raises(OptimizationError, match="could not find an optimal schedule"):
        optimize(scenario(), [directive("no_op")])


def test_solver_passes_share_one_time_budget(monkeypatch):
    real_solver = optimizer_module.linprog
    times = iter([100, 100.25, 101, 102])
    limits = []

    def timed_solver(*args, **kwargs):
        limits.append(kwargs["options"]["time_limit"])
        return real_solver(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "monotonic", lambda: next(times))
    monkeypatch.setattr(optimizer_module, "linprog", timed_solver)
    optimize(scenario(), [directive("no_op")])
    assert limits == [3.75, 3, 2]


def test_expired_budget_skips_optional_passes(monkeypatch):
    real_solver = optimizer_module.linprog
    times = iter([100, 100.25, 105])
    calls = []

    def timed_solver(*args, **kwargs):
        calls.append(1)
        return real_solver(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "monotonic", lambda: next(times))
    monkeypatch.setattr(optimizer_module, "linprog", timed_solver)
    request = scenario()
    directives = [directive("no_op")]
    result = optimize(request, directives)
    standalone_replay(request, result, directives)
    assert len(calls) == 1
