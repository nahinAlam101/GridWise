"""Exact, deterministic 24-hour dispatch after directive validation.

The battery has unit efficiency under the challenge rules, so one signed energy
change replaces separate charge/discharge variables. This is a linear program,
with no relaxation that could permit simultaneous charging and discharging.
"""

from __future__ import annotations

from copy import deepcopy
from math import fsum, isfinite
from time import monotonic

import numpy as np
from scipy.optimize import OptimizeResult, linprog


# Keep solver work within the HTTP budget, including optional tie-break passes.
_SOLVER_BUDGET_SECONDS = 4.0


class OptimizationError(RuntimeError):
    """The solver could not produce a valid optimal schedule."""


class InfeasibleSchedule(OptimizationError):
    """The supplied scenario and interpreted directives cannot all be satisfied."""


def compile_constraints(scenario: dict, directives: list[dict]) -> dict:
    """Compile validated directives without changing either input.

    Reserves combine by maximum; grid caps combine by minimum; prohibitions
    accumulate. The specification does not explicitly define overlapping solar
    notes. We use the strictest remaining fraction against the ORIGINAL forecast,
    consistently with its original_solar * factor formula, not multiplication of
    successive reductions. Uncapped grid hours have an infinite internal bound.
    """
    hours = sorted(scenario["hours"], key=lambda item: item["hour"])
    battery = scenario["battery"]
    factors = [1.0] * 24
    reserve = [float(battery["minimum_energy_kwh"])] * 24
    grid = [float("inf")] * 24
    charge = [float(battery["max_charge_kwh_per_hour"])] * 24
    discharge = [float(battery["max_discharge_kwh_per_hour"])] * 24
    for directive in directives:
        kind = directive["directive_type"]
        if kind == "no_op":
            continue
        adjustment = directive["structured_adjustment"]
        for hour in adjustment["hours"]:
            if kind == "solar_reduction":
                factors[hour] = min(factors[hour], adjustment["factor"])
            elif kind == "minimum_battery_reserve":
                reserve[hour] = max(reserve[hour], adjustment["minimum_energy_kwh"])
            elif kind == "max_grid_window":
                grid[hour] = min(grid[hour], adjustment["max_grid_kwh"])
            elif kind == "no_charge_window":
                charge[hour] = 0.0
            elif kind == "no_discharge_window":
                discharge[hour] = 0.0
            else:
                raise OptimizationError("Unsupported directive reached the optimizer.")
    return {
        "effective_solar": [float(item["solar_kwh"]) * factors[h] for h, item in enumerate(hours)],
        "minimum_energy": reserve,
        "max_grid": grid,
        "max_charge": charge,
        "max_discharge": discharge,
    }


def optimize(scenario: dict, directives: list[dict]) -> dict:
    """Return the complete official response for already validated dictionaries.

    Three lexicographic LP passes minimize electricity cost, then peak grid
    demand, then battery throughput. Each later pass fixes the earlier optimum
    by equality, so a weighted tie-break never trades away electricity savings.
    If an optional tie-break cannot be solved, retain the preceding optimum.
    Final independent replay is performed by the API's validation layer.
    """
    hours = sorted(scenario["hours"], key=lambda item: item["hour"])
    battery = scenario["battery"]
    rules = compile_constraints(scenario, directives)
    initial = float(battery["initial_energy_kwh"])
    capacity = float(battery["capacity_kwh"])

    # Variables: grid[0:24], solar[24:48], signed battery delta[48:72],
    # energy after hour[72:96], peak[96], absolute battery delta[97:121].
    n = 121
    equalities = []
    equality_rhs = []
    inequalities = []
    inequality_rhs = []
    cost = np.zeros(n)
    bounds = [(0.0, rules["max_grid"][h]) for h in range(24)]
    bounds += [(0.0, rules["effective_solar"][h]) for h in range(24)]
    bounds += [(-rules["max_discharge"][h], rules["max_charge"][h]) for h in range(24)]
    bounds += [(rules["minimum_energy"][h], capacity) for h in range(24)]
    bounds += [(0.0, None)]
    bounds += [(0.0, max(rules["max_charge"][h], rules["max_discharge"][h])) for h in range(24)]

    for h, hour in enumerate(hours):
        cost[h] = float(hour["tariff_bdt_per_kwh"])
        # Grid + solar - battery delta = campus demand.
        row = np.zeros(n)
        row[h], row[24 + h], row[48 + h] = 1.0, 1.0, -1.0
        equalities.append(row)
        equality_rhs.append(float(hour["demand_kwh"]))

        # Energy after = energy before + battery delta.
        row = np.zeros(n)
        row[72 + h], row[48 + h] = 1.0, -1.0
        if h:
            row[72 + h - 1] = -1.0
        equalities.append(row)
        equality_rhs.append(initial if h == 0 else 0.0)

        row = np.zeros(n)
        row[h], row[96] = 1.0, -1.0
        inequalities.append(row)
        inequality_rhs.append(0.0)
        for sign in (1.0, -1.0):
            row = np.zeros(n)
            row[48 + h], row[97 + h] = sign, -1.0
            inequalities.append(row)
            inequality_rhs.append(0.0)

    row = np.zeros(n)
    row[95] = 1.0
    equalities.append(row)
    equality_rhs.append(initial)

    a_eq = np.asarray(equalities)
    b_eq = np.asarray(equality_rhs)
    a_ub = np.asarray(inequalities)
    b_ub = np.asarray(inequality_rhs)
    deadline = monotonic() + _SOLVER_BUDGET_SECONDS

    def solve(objective, eq=a_eq, rhs=b_eq):
        remaining = deadline - monotonic()
        if remaining <= 0:
            return OptimizeResult(success=False, status=1)
        options = {
            "primal_feasibility_tolerance": 1e-9,
            "dual_feasibility_tolerance": 1e-9,
            "time_limit": remaining,
        }
        try:
            return linprog(
                objective, A_ub=a_ub, b_ub=b_ub, A_eq=eq, b_eq=rhs,
                bounds=bounds, method="highs", options=options,
            )
        except (ValueError, RuntimeError, FloatingPointError):
            # Do not expose native solver diagnostics or raw input in API errors.
            return OptimizeResult(success=False, status=4)

    def usable(result):
        return (
            result.success and result.x is not None
            and np.shape(result.x) == (n,)
            and np.all(np.isfinite(result.x))
        )

    result = solve(cost)
    if result.status == 2:
        raise InfeasibleSchedule("The scenario and operator directives have no feasible schedule.")
    if not usable(result):
        raise OptimizationError("The optimization solver could not find an optimal schedule.")

    best = result.x
    optimal_cost = float(cost @ best)
    if not isfinite(optimal_cost):
        raise OptimizationError("The optimization solver returned a non-finite objective.")
    fixed_cost_eq = np.vstack((a_eq, cost))
    fixed_cost_rhs = np.append(b_eq, optimal_cost)
    peak_objective = np.zeros(n)
    peak_objective[96] = 1.0
    peak_result = solve(peak_objective, fixed_cost_eq, fixed_cost_rhs)
    if usable(peak_result):
        best = peak_result.x
        throughput = np.zeros(n)
        throughput[97:] = 1.0
        fixed_peak_eq = np.vstack((fixed_cost_eq, peak_objective))
        fixed_peak_rhs = np.append(fixed_cost_rhs, float(best[96]))
        throughput_result = solve(throughput, fixed_peak_eq, fixed_peak_rhs)
        if usable(throughput_result):
            best = throughput_result.x

    if not all(isfinite(float(value)) for value in best):
        raise OptimizationError("The optimization solver returned non-finite values.")

    def nonnegative(value: float) -> float:
        # HiGHS may produce a negative zero or a negligible bound residual.
        return max(0.0, round(float(value), 10))

    plan = []
    deltas = []
    for h in range(24):
        delta = round(float(best[48 + h]), 10)
        deltas.append(delta)
        action = "charge" if delta > 0 else "discharge" if delta < 0 else "idle"
        plan.append({
            "hour": h,
            "grid_kwh": nonnegative(best[h]),
            "solar_used_kwh": nonnegative(best[24 + h]),
            "battery_action": action,
            "battery_kwh": abs(delta),
            "battery_energy_after_kwh": nonnegative(fsum([initial, *deltas])),
        })

    total_grid = fsum(item["grid_kwh"] for item in plan)
    total_cost = fsum(item["grid_kwh"] * hours[h]["tariff_bdt_per_kwh"] for h, item in enumerate(plan))
    peak = max(item["grid_kwh"] for item in plan)
    charge_total = fsum(item["battery_kwh"] for item in plan if item["battery_action"] == "charge")
    discharge_total = fsum(item["battery_kwh"] for item in plan if item["battery_action"] == "discharge")
    applicable = sum(directive["directive_type"] != "no_op" for directive in directives)
    return {
        "scenario_id": scenario["scenario_id"],
        "directive_interpretation": deepcopy(directives),
        "hourly_plan": plan,
        "total_grid_kwh": total_grid,
        "total_cost_bdt": total_cost,
        "peak_grid_kwh": peak,
        "plan_summary": (
            f"Minimum-cost schedule purchases {total_grid:.2f} kWh for {total_cost:.2f} BDT. "
            f"The battery charges {charge_total:.2f} kWh and discharges {discharge_total:.2f} kWh, "
            f"ending at its initial {initial:.2f} kWh. "
            f"Applied {applicable} operating constraint(s)."
        ),
    }
