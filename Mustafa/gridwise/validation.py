"""Strict boundaries for HTTP input and untrusted model output."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator


Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
HourNumber = Annotated[StrictInt, Field(ge=0, le=23)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Hour(StrictModel):
    hour: HourNumber
    demand_kwh: Nonnegative
    solar_kwh: Nonnegative
    tariff_bdt_per_kwh: Nonnegative


class Battery(StrictModel):
    capacity_kwh: Nonnegative
    initial_energy_kwh: Nonnegative
    minimum_energy_kwh: Nonnegative
    max_charge_kwh_per_hour: Nonnegative
    max_discharge_kwh_per_hour: Nonnegative

    @model_validator(mode="after")
    def valid_bounds(self) -> Battery:
        if not self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh:
            raise ValueError("Battery must satisfy minimum <= initial <= capacity.")
        return self


class Scenario(StrictModel):
    scenario_id: Annotated[StrictStr, Field(min_length=1)]
    operator_notes: Annotated[list[StrictStr], Field(min_length=1, max_length=3)]
    hours: Annotated[list[Hour], Field(min_length=24, max_length=24)]
    battery: Battery

    @model_validator(mode="after")
    def valid_scenario(self) -> Scenario:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be blank.")
        if any(not note.strip() for note in self.operator_notes):
            raise ValueError("Operator notes must not be blank.")
        if sorted(hour.hour for hour in self.hours) != list(range(24)):
            raise ValueError("Hours must contain every hour 0 through 23 exactly once.")
        self.hours.sort(key=lambda hour: hour.hour)
        return self


class DirectiveValidationError(ValueError):
    """A model result cannot be safely applied to the optimization problem."""


def validate_scenario(raw: Any) -> dict:
    return Scenario.model_validate(raw).model_dump()


_DIRECTIVE_FIELDS = {
    "note_index", "applies", "directive_type", "structured_adjustment", "explanation"
}
_ADJUSTMENT_FIELDS = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
    "no_op": set(),
}


def _number(value: Any, field: str, maximum: float | None = None) -> None:
    if type(value) not in (int, float):
        raise DirectiveValidationError(f"{field} must be a finite nonnegative number.")
    try:
        valid = math.isfinite(value) and value >= 0
    except (ValueError, OverflowError):
        valid = False
    if not valid or (maximum is not None and value > maximum):
        raise DirectiveValidationError(f"{field} is outside its allowed range.")


def validate_directives(scenario: dict, raw: Any) -> list[dict]:
    """Validate, never guess or silently repair, a model's interpretation list.

    Semantic accuracy still depends on the model. This boundary prevents malformed
    types, invented fields, missing notes, and invalid numeric constraints from
    entering the solver. Error messages contain schema names, never input values.
    """
    if not isinstance(raw, list) or len(raw) != len(scenario["operator_notes"]):
        raise DirectiveValidationError("Return exactly one interpretation per operator note.")
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict) or set(entry) != _DIRECTIVE_FIELDS:
            raise DirectiveValidationError("Each interpretation must have exactly the required fields.")
        if type(entry["note_index"]) is not int or entry["note_index"] != index:
            raise DirectiveValidationError("note_index must follow input order, once per note.")
        kind = entry["directive_type"]
        if not isinstance(kind, str) or kind not in _ADJUSTMENT_FIELDS:
            raise DirectiveValidationError("Unsupported directive_type.")
        if not isinstance(entry["explanation"], str) or not entry["explanation"].strip():
            raise DirectiveValidationError("explanation must be a nonempty string.")
        if type(entry["applies"]) is not bool or entry["applies"] != (kind != "no_op"):
            raise DirectiveValidationError("applies must be false only for no_op.")
        adjustment = entry["structured_adjustment"]
        if kind == "no_op":
            if adjustment is not None:
                raise DirectiveValidationError("no_op must use a null structured_adjustment.")
            continue
        if not isinstance(adjustment, dict) or set(adjustment) != _ADJUSTMENT_FIELDS[kind]:
            raise DirectiveValidationError("structured_adjustment must match its directive type exactly.")
        hours = adjustment["hours"]
        if (
            not isinstance(hours, list)
            or not hours
            or any(type(hour) is not int or not 0 <= hour <= 23 for hour in hours)
            or hours != sorted(set(hours))
        ):
            raise DirectiveValidationError("hours must be nonempty, unique integers 0..23 in ascending order.")
        if kind == "solar_reduction":
            _number(adjustment["factor"], "factor", 1)
        elif kind == "minimum_battery_reserve":
            _number(adjustment["minimum_energy_kwh"], "minimum_energy_kwh", scenario["battery"]["capacity_kwh"])
        elif kind == "max_grid_window":
            _number(adjustment["max_grid_kwh"], "max_grid_kwh")
    return deepcopy(raw)
