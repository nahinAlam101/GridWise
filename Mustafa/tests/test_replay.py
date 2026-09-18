"""Replay regression checks using public schedules and deliberate corruption."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from gridwise.replay import ReplayError, replay


PUBLIC_CASES = json.loads(
    (Path(__file__).resolve().parents[1] / "data" / "public_sample_cases.json").read_text()
)["cases"]


def directive(kind="no_op", adjustment=None, index=0):
    return {
        "note_index": index,
        "applies": kind != "no_op",
        "directive_type": kind,
        "structured_adjustment": adjustment,
        "explanation": "Test directive.",
    }


def fixture_pair(directives=None):
    directives = directives or [directive()]
    scenario = {
        "scenario_id": "REPLAY-TEST",
        "operator_notes": ["Synthetic operator note."] * len(directives),
        "hours": [
            {"hour": h, "demand_kwh": 10, "solar_kwh": 2, "tariff_bdt_per_kwh": 3}
            for h in range(24)
        ],
        "battery": {
            "capacity_kwh": 20,
            "initial_energy_kwh": 10,
            "minimum_energy_kwh": 2,
            "max_charge_kwh_per_hour": 5,
            "max_discharge_kwh_per_hour": 5,
        },
    }
    response = {
        "scenario_id": scenario["scenario_id"],
        "directive_interpretation": deepcopy(directives),
        "hourly_plan": [
            {
                "hour": h,
                "grid_kwh": 8,
                "solar_used_kwh": 2,
                "battery_action": "idle",
                "battery_kwh": 0,
                "battery_energy_after_kwh": 10,
            }
            for h in range(24)
        ],
        "total_grid_kwh": 192,
        "total_cost_bdt": 576,
        "peak_grid_kwh": 8,
        "plan_summary": "Use solar and grid to serve the campus.",
    }
    return scenario, response


def refresh_accounting(scenario, response):
    """Keep unrelated accounting valid when mutating one physical constraint."""
    energy = scenario["battery"]["initial_energy_kwh"]
    for entry, hour in zip(response["hourly_plan"], scenario["hours"]):
        delta = {
            "charge": entry["battery_kwh"],
            "discharge": -entry["battery_kwh"],
            "idle": 0,
        }[entry["battery_action"]]
        energy += delta
        entry["battery_energy_after_kwh"] = energy
        entry["grid_kwh"] = hour["demand_kwh"] + delta - entry["solar_used_kwh"]
    response["total_grid_kwh"] = sum(p["grid_kwh"] for p in response["hourly_plan"])
    response["total_cost_bdt"] = sum(
        p["grid_kwh"] * h["tariff_bdt_per_kwh"]
        for p, h in zip(response["hourly_plan"], scenario["hours"])
    )
    response["peak_grid_kwh"] = max(p["grid_kwh"] for p in response["hourly_plan"])


def set_action(response, hour, action, amount):
    response["hourly_plan"][hour].update(battery_action=action, battery_kwh=amount)


@pytest.mark.parametrize("case", PUBLIC_CASES, ids=lambda case: case["id"])
def test_public_reference_schedules_replay(case):
    replay(case["input"], case["expected_output"], case["expected_output"]["directive_interpretation"])


@pytest.mark.parametrize("field", ["grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"])
@pytest.mark.parametrize("value", [True, "1", None, -1e-10, float("inf"), float("nan")])
def test_hourly_numbers_are_finite_nonnegative_json_numbers(field, value):
    scenario, response = fixture_pair()
    response["hourly_plan"][0][field] = value
    with pytest.raises(ReplayError, match=field):
        replay(scenario, response)


@pytest.mark.parametrize("field", ["total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"])
@pytest.mark.parametrize("value", [True, "192", -1, float("inf"), float("nan"), 0])
def test_totals_are_valid_and_recomputed(field, value):
    scenario, response = fixture_pair()
    response[field] = value
    with pytest.raises(ReplayError, match=field):
        replay(scenario, response)


@pytest.mark.parametrize("bad_hour", [True, 0.0, "0", 1, 24])
def test_hours_require_integer_order(bad_hour):
    scenario, response = fixture_pair()
    response["hourly_plan"][0]["hour"] = bad_hour
    with pytest.raises(ReplayError, match="hour order"):
        replay(scenario, response)


def test_plan_requires_all_24_hours():
    scenario, response = fixture_pair()
    response["hourly_plan"].pop()
    with pytest.raises(ReplayError, match="24"):
        replay(scenario, response)


def test_unsorted_input_hours_are_mapped_by_hour():
    scenario, response = fixture_pair()
    scenario["hours"].reverse()
    replay(scenario, response)


def test_scenario_id_and_summary_are_required():
    scenario, response = fixture_pair()
    response["scenario_id"] = "ANOTHER-SCENARIO"
    with pytest.raises(ReplayError, match="scenario_id"):
        replay(scenario, response)
    response["scenario_id"] = scenario["scenario_id"]
    response.pop("plan_summary")
    with pytest.raises(ReplayError, match="plan_summary"):
        replay(scenario, response)


def test_battery_actions_are_one_of_three_supported_strings():
    scenario, response = fixture_pair()
    response["hourly_plan"][0]["battery_action"] = "charge_and_discharge"
    with pytest.raises(ReplayError, match="battery_action"):
        replay(scenario, response)


def test_idle_battery_cannot_move_energy():
    scenario, response = fixture_pair()
    response["hourly_plan"][0]["battery_kwh"] = 1
    with pytest.raises(ReplayError, match="idle battery_kwh"):
        replay(scenario, response)


def test_state_transitions_are_replayed_from_initial_energy():
    scenario, response = fixture_pair()
    response["hourly_plan"][3]["battery_energy_after_kwh"] += 1
    with pytest.raises(ReplayError, match="battery_energy_after_kwh"):
        replay(scenario, response)


def test_small_state_errors_cannot_accumulate_across_hours():
    scenario, response = fixture_pair()
    for h, entry in enumerate(response["hourly_plan"]):
        entry["battery_energy_after_kwh"] += (h + 1) * 0.000006
    with pytest.raises(ReplayError, match="battery_energy_after_kwh"):
        replay(scenario, response)


def test_energy_balance_is_independent_of_reported_totals():
    scenario, response = fixture_pair()
    response["hourly_plan"][0]["grid_kwh"] += 1
    response["total_grid_kwh"] += 1
    response["total_cost_bdt"] += 3
    response["peak_grid_kwh"] += 1
    with pytest.raises(ReplayError, match="energy balance"):
        replay(scenario, response)


@pytest.mark.parametrize("action,other,limit", [
    ("charge", "discharge", "charging"),
    ("discharge", "charge", "discharging"),
])
def test_hourly_battery_rates(action, other, limit):
    scenario, response = fixture_pair()
    set_action(response, 0, action, 6)
    set_action(response, 1, other, 6)
    refresh_accounting(scenario, response)
    with pytest.raises(ReplayError, match=f"hourly {limit} limit"):
        replay(scenario, response)


@pytest.mark.parametrize("action,amount,error", [
    ("charge", 11, "capacity"), ("discharge", 9, "minimum battery reserve")
])
def test_battery_energy_bounds(action, amount, error):
    scenario, response = fixture_pair()
    scenario["battery"]["max_charge_kwh_per_hour"] = 20
    scenario["battery"]["max_discharge_kwh_per_hour"] = 20
    scenario["hours"][0]["demand_kwh"] = 20
    set_action(response, 0, action, amount)
    set_action(response, 1, "discharge" if action == "charge" else "charge", amount)
    refresh_accounting(scenario, response)
    with pytest.raises(ReplayError, match=error):
        replay(scenario, response)


def test_battery_neutrality_is_required():
    scenario, response = fixture_pair()
    set_action(response, 0, "charge", 1)
    refresh_accounting(scenario, response)
    with pytest.raises(ReplayError, match="neutrality"):
        replay(scenario, response)


@pytest.mark.parametrize("kind,action,other", [
    ("no_charge_window", "charge", "discharge"),
    ("no_discharge_window", "discharge", "charge"),
])
def test_action_windows_are_applied(kind, action, other):
    scenario, response = fixture_pair([directive(kind, {"hours": [0]})])
    set_action(response, 0, action, 1)
    set_action(response, 1, other, 1)
    refresh_accounting(scenario, response)
    with pytest.raises(ReplayError, match=kind):
        replay(scenario, response)


def test_effective_solar_uses_strongest_original_forecast_fraction():
    scenario, response = fixture_pair([
        directive("solar_reduction", {"hours": [0], "factor": 0.5}),
        directive("solar_reduction", {"hours": [0], "factor": 0.25}, index=1),
    ])
    response["hourly_plan"][0]["solar_used_kwh"] = 0.5
    refresh_accounting(scenario, response)
    replay(scenario, response)
    response["hourly_plan"][0]["solar_used_kwh"] = 0.6
    refresh_accounting(scenario, response)
    with pytest.raises(ReplayError, match="effective solar"):
        replay(scenario, response)


def test_overlapping_reserves_use_the_highest_requirement():
    scenario, response = fixture_pair([
        directive("minimum_battery_reserve", {"hours": [0], "minimum_energy_kwh": 11}),
        directive("minimum_battery_reserve", {"hours": [0], "minimum_energy_kwh": 9}, index=1),
    ])
    with pytest.raises(ReplayError, match="minimum battery reserve"):
        replay(scenario, response)
    set_action(response, 0, "charge", 1)
    set_action(response, 1, "discharge", 1)
    refresh_accounting(scenario, response)
    replay(scenario, response)


def test_overlapping_grid_caps_use_the_lowest_requirement():
    scenario, response = fixture_pair([
        directive("max_grid_window", {"hours": [0], "max_grid_kwh": 7}),
        directive("max_grid_window", {"hours": [0], "max_grid_kwh": 9}, index=1),
    ])
    with pytest.raises(ReplayError, match="grid cap"):
        replay(scenario, response)
    set_action(response, 0, "discharge", 1)
    set_action(response, 1, "charge", 1)
    refresh_accounting(scenario, response)
    replay(scenario, response)


def test_ground_truth_prevents_self_consistent_wrong_interpretation():
    scenario, response = fixture_pair()
    ground_truth = [directive("no_charge_window", {"hours": [0]})]
    with pytest.raises(ReplayError, match="expected interpretation"):
        replay(scenario, response, ground_truth)


def test_semantic_comparison_ignores_explanation_and_tolerates_rounding():
    ground_truth = [directive("solar_reduction", {"hours": [0], "factor": 0.5})]
    scenario, response = fixture_pair(ground_truth)
    response["directive_interpretation"][0]["explanation"] = "A differently worded explanation."
    response["directive_interpretation"][0]["structured_adjustment"]["factor"] += 1e-6
    response["hourly_plan"][0]["solar_used_kwh"] = 1
    refresh_accounting(scenario, response)
    replay(scenario, response, ground_truth)
    response["directive_interpretation"][0]["structured_adjustment"]["factor"] += 0.01
    with pytest.raises(ReplayError, match="factor"):
        replay(scenario, response, ground_truth)


def test_ground_truth_hours_are_compared_exactly():
    ground_truth = [directive("no_charge_window", {"hours": [0]})]
    scenario, response = fixture_pair(ground_truth)
    response["directive_interpretation"][0]["structured_adjustment"]["hours"] = [1]
    with pytest.raises(ReplayError, match="hours does not match"):
        replay(scenario, response, ground_truth)


def test_ground_truth_controls_physics_even_when_numeric_difference_is_tolerated():
    ground_truth = [directive("solar_reduction", {"hours": [0], "factor": 0.5})]
    scenario, response = fixture_pair(ground_truth)
    scenario["hours"][0]["solar_kwh"] = 100
    scenario["hours"][0]["demand_kwh"] = 100
    response["directive_interpretation"][0]["structured_adjustment"]["factor"] += 1e-6
    response["hourly_plan"][0]["solar_used_kwh"] = 50.0001
    refresh_accounting(scenario, response)
    with pytest.raises(ReplayError, match="effective solar"):
        replay(scenario, response, ground_truth)


def test_directive_schema_is_validated_before_application():
    scenario, response = fixture_pair()
    response["directive_interpretation"][0]["applies"] = True
    with pytest.raises(ReplayError, match="applies"):
        replay(scenario, response)


@pytest.mark.parametrize("tolerance", [True, -1, float("nan"), float("inf")])
def test_invalid_tolerance_cannot_disable_checks(tolerance):
    scenario, response = fixture_pair()
    with pytest.raises(ReplayError, match="tolerance"):
        replay(scenario, response, tolerance=tolerance)


def test_tolerance_is_absolute_not_relative():
    scenario, response = fixture_pair()
    for hour in scenario["hours"]:
        hour["demand_kwh"] = 1_000_000_000
    refresh_accounting(scenario, response)
    response["total_grid_kwh"] += 0.1
    with pytest.raises(ReplayError, match="total_grid_kwh"):
        replay(scenario, response)
