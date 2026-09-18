"""Independently replay a returned schedule before exposing it to a caller.

This module deliberately does not depend on the optimizer or its compiled
constraints. It is also useful for checking an HTTP response against known
directive semantics with ``replay(scenario, response, ground_truth)``.
"""

from __future__ import annotations

import math
from typing import Any

from gridwise.validation import validate_directives


class ReplayError(ValueError):
    """A response does not meet the scenario's accounting or directive rules."""


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(f"{path} must be a finite, non-negative number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ReplayError(f"{path} must be a finite, non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ReplayError(f"{path} must be a finite, non-negative number")
    return number


def _object(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ReplayError(f"{path} must be an object")
    return value


def _equal(actual: float, expected: float, tolerance: float, path: str) -> None:
    if not math.isfinite(expected) or abs(actual - expected) > tolerance:
        raise ReplayError(f"{path}: expected {expected}, received {actual}")


def _directives(scenario: dict, raw: Any, path: str) -> list[dict]:
    try:
        validated = validate_directives(scenario, raw)
    except (ValueError, TypeError, KeyError) as exc:
        raise ReplayError(f"{path}: {exc}") from exc
    return raw if validated is None else validated


def _compare_interpretations(
    actual: list[dict], expected: list[dict], tolerance: float
) -> None:
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        path = f"directive_interpretation[{index}]"
        for field in ("note_index", "applies", "directive_type"):
            if observed[field] != reference[field]:
                raise ReplayError(f"{path}.{field} does not match the expected interpretation")
        left, right = observed["structured_adjustment"], reference["structured_adjustment"]
        if right is None:
            if left is not None:
                raise ReplayError(f"{path}.structured_adjustment must be null")
            continue
        if left is None or left.keys() != right.keys():
            raise ReplayError(f"{path}.structured_adjustment has different fields")
        for field, value in right.items():
            if field == "hours":
                if left[field] != value:
                    raise ReplayError(f"{path}.structured_adjustment.hours does not match")
            else:
                _equal(float(left[field]), float(value), tolerance, f"{path}.{field}")


def replay(
    scenario: dict,
    response: dict,
    directives: list[dict] | None = None,
    tolerance: float = 1e-5,
) -> None:
    """Raise :class:`ReplayError` if any response rule fails.

    ``directives``, when supplied, are independently known ground truth. Both
    their interpretation semantics and their application are checked. Without
    ground truth, this verifies internal consistency with the response's own
    validated interpretation; it cannot establish language understanding.

    Numeric comparisons use absolute tolerance only. Overlapping solar rules
    use the smallest remaining fraction of the original forecast, overlapping
    reserves use the maximum, and overlapping grid caps use the minimum.
    """
    tolerance = _number(tolerance, "tolerance")
    scenario = _object(scenario, "scenario")
    response = _object(response, "response")
    scenario_id = scenario.get("scenario_id")
    if not isinstance(scenario_id, str) or response.get("scenario_id") != scenario_id:
        raise ReplayError("scenario_id must match the request")
    if not isinstance(response.get("plan_summary"), str) or not response["plan_summary"].strip():
        raise ReplayError("plan_summary must be a non-empty string")

    interpretation = _directives(
        scenario, response.get("directive_interpretation"), "directive_interpretation"
    )
    if directives is None:
        applied_directives = interpretation
    else:
        applied_directives = _directives(scenario, directives, "ground_truth_directives")
        if len(interpretation) != len(applied_directives):
            raise ReplayError("directive_interpretation has the wrong number of entries")
        _compare_interpretations(interpretation, applied_directives, tolerance)

    hourly_data = scenario.get("hours")
    if not isinstance(hourly_data, list) or len(hourly_data) != 24:
        raise ReplayError("scenario.hours must contain exactly 24 entries")
    by_hour = {}
    for item in hourly_data:
        item = _object(item, "scenario.hours entry")
        hour = item.get("hour")
        if type(hour) is not int or not 0 <= hour <= 23 or hour in by_hour:
            raise ReplayError("scenario.hours must contain unique integer hours 0 through 23")
        by_hour[hour] = {
            field: _number(item.get(field), f"scenario.hours[{hour}].{field}")
            for field in ("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh")
        }
    battery = _object(scenario.get("battery"), "scenario.battery")
    capacity, initial, minimum, max_charge, max_discharge = (
        _number(battery.get(field), f"scenario.battery.{field}")
        for field in (
            "capacity_kwh", "initial_energy_kwh", "minimum_energy_kwh",
            "max_charge_kwh_per_hour", "max_discharge_kwh_per_hour",
        )
    )
    if not minimum <= initial <= capacity:
        raise ReplayError("scenario.battery must satisfy minimum <= initial <= capacity")

    solar = [by_hour[h]["solar_kwh"] for h in range(24)]
    reserves = [minimum] * 24
    grid_caps = [math.inf] * 24
    no_charge: set[int] = set()
    no_discharge: set[int] = set()
    for directive in applied_directives:
        kind = directive["directive_type"]
        adjustment = directive["structured_adjustment"]
        if kind == "no_op":
            continue
        for hour in adjustment["hours"]:
            if kind == "solar_reduction":
                solar[hour] = min(
                    solar[hour], by_hour[hour]["solar_kwh"] * adjustment["factor"]
                )
            elif kind == "minimum_battery_reserve":
                reserves[hour] = max(reserves[hour], adjustment["minimum_energy_kwh"])
            elif kind == "max_grid_window":
                grid_caps[hour] = min(grid_caps[hour], adjustment["max_grid_kwh"])
            elif kind == "no_charge_window":
                no_charge.add(hour)
            elif kind == "no_discharge_window":
                no_discharge.add(hour)

    plan = response.get("hourly_plan")
    if not isinstance(plan, list) or len(plan) != 24:
        raise ReplayError("hourly_plan must contain exactly 24 entries")
    energy = initial
    grids, costs = [], []
    for hour, raw in enumerate(plan):
        entry = _object(raw, f"hourly_plan[{hour}]")
        path = f"hourly_plan[{hour}]"
        if type(entry.get("hour")) is not int or entry["hour"] != hour:
            raise ReplayError("hourly_plan must be in integer hour order 0 through 23")
        grid, used_solar, amount, reported_energy = (
            _number(entry.get(field), f"{path}.{field}")
            for field in (
                "grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh"
            )
        )
        action = entry.get("battery_action")
        if action not in ("charge", "discharge", "idle"):
            raise ReplayError(f"{path}.battery_action must be charge, discharge, or idle")
        charge = amount if action == "charge" else 0.0
        discharge = amount if action == "discharge" else 0.0
        if action == "idle":
            _equal(amount, 0.0, tolerance, f"{path}.idle battery_kwh")
        if charge > max_charge + tolerance:
            raise ReplayError(f"{path} exceeds the hourly charging limit")
        if discharge > max_discharge + tolerance:
            raise ReplayError(f"{path} exceeds the hourly discharging limit")
        if hour in no_charge and charge > tolerance:
            raise ReplayError(f"{path} violates no_charge_window")
        if hour in no_discharge and discharge > tolerance:
            raise ReplayError(f"{path} violates no_discharge_window")
        if used_solar > solar[hour] + tolerance:
            raise ReplayError(f"{path}.solar_used_kwh exceeds effective solar availability")
        if grid > grid_caps[hour] + tolerance:
            raise ReplayError(f"{path}.grid_kwh exceeds the applicable grid cap")
        energy += charge - discharge
        _equal(reported_energy, energy, tolerance, f"{path}.battery_energy_after_kwh")
        if energy < reserves[hour] - tolerance:
            raise ReplayError(f"{path} violates the minimum battery reserve")
        if energy > capacity + tolerance:
            raise ReplayError(f"{path} exceeds battery capacity")
        _equal(
            grid + used_solar + discharge,
            by_hour[hour]["demand_kwh"] + charge,
            tolerance,
            f"{path} energy balance",
        )
        grids.append(grid)
        costs.append(grid * by_hour[hour]["tariff_bdt_per_kwh"])

    _equal(energy, initial, tolerance, "end-of-day battery neutrality")
    totals = {
        "total_grid_kwh": sum(grids),
        "total_cost_bdt": sum(costs),
        "peak_grid_kwh": max(grids),
    }
    for field, expected in totals.items():
        _equal(_number(response.get(field), field), expected, tolerance, field)
