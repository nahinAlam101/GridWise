from __future__ import annotations

from .database import SessionLocal
from .models import Battery, Directive, HourlyPlan, OperatorNote, OptimizationRun, ScenarioHour


def save_completed_run(scenario: dict, directives: list[dict], response: dict) -> int | None:
    if SessionLocal is None:
        return None

    run = OptimizationRun(
        scenario_id=scenario["scenario_id"],
        status="completed",
        total_grid_kwh=response["total_grid_kwh"],
        total_cost_bdt=response["total_cost_bdt"],
        peak_grid_kwh=response["peak_grid_kwh"],
        plan_summary=response["plan_summary"],
        request_json=scenario,
        response_json=response,
        battery=Battery(**scenario["battery"]),
        hours=[ScenarioHour(**hour) for hour in scenario["hours"]],
        notes=[OperatorNote(note_index=index, content=note) for index, note in enumerate(scenario["operator_notes"])],
        directives=[Directive(**directive) for directive in directives],
        plans=[HourlyPlan(**plan) for plan in response["hourly_plan"]],
    )
    with SessionLocal() as session:
        session.add(run)
        session.commit()
        return run.id