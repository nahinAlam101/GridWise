import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gridwise.validation import DirectiveValidationError, validate_directives, validate_scenario


CASES = json.loads((Path(__file__).parents[1] / "data/public_sample_cases.json").read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_public_contract(case):
    scenario = validate_scenario(case["input"])
    assert validate_directives(scenario, case["expected_output"]["directive_interpretation"])


def test_unordered_hours_are_normalized_without_mutating_input():
    scenario = copy.deepcopy(CASES[0]["input"])
    scenario["hours"].reverse()
    validated = validate_scenario(scenario)
    assert [h["hour"] for h in validated["hours"]] == list(range(24))
    assert scenario["hours"][0]["hour"] == 23


@pytest.mark.parametrize("value", [True, "90", -1, float("nan"), float("inf"), None])
def test_reject_non_numeric_or_nonfinite_energy(value):
    scenario = copy.deepcopy(CASES[0]["input"])
    scenario["hours"][0]["demand_kwh"] = value
    with pytest.raises(ValidationError):
        validate_scenario(scenario)


@pytest.mark.parametrize("mutation", [
    lambda s: s["hours"].pop(),
    lambda s: s["hours"][1].update(hour=0),
    lambda s: s["hours"][0].update(hour=True),
    lambda s: s.update(operator_notes=[]),
    lambda s: s.update(operator_notes=[" "]),
    lambda s: s.update(operator_notes=["x"] * 4),
    lambda s: s["battery"].update(initial_energy_kwh=1000),
    lambda s: s["battery"].update(minimum_energy_kwh=120),
    lambda s: s.update(scenario_id=" "),
])
def test_reject_structural_errors(mutation):
    scenario = copy.deepcopy(CASES[0]["input"])
    mutation(scenario)
    with pytest.raises(ValidationError):
        validate_scenario(scenario)


@pytest.mark.parametrize("mutation", [
    lambda d: d.pop(),
    lambda d: d[0].update(note_index=True),
    lambda d: d[0].update(note_index=1),
    lambda d: d[0].update(applies=False),
    lambda d: d[0].update(applies=1),
    lambda d: d[0].update(directive_type="demand_change"),
    lambda d: d[0].update(demand_kwh=0),
    lambda d: d[0].update(explanation=""),
    lambda d: d[1].update(structured_adjustment={}),
    lambda d: d[0]["structured_adjustment"].update(hours=[13, 12]),
    lambda d: d[0]["structured_adjustment"].update(hours=[12, 12]),
    lambda d: d[0]["structured_adjustment"].update(hours=[]),
    lambda d: d[0]["structured_adjustment"].update(hours=[24]),
    lambda d: d[0]["structured_adjustment"].update(hours=[True]),
    lambda d: d[0]["structured_adjustment"].update(factor=1.1),
    lambda d: d[0]["structured_adjustment"].update(factor=float("nan")),
    lambda d: d[0]["structured_adjustment"].update(factor=True),
    lambda d: d[0]["structured_adjustment"].update(tariff_bdt_per_kwh=0),
])
def test_model_output_cannot_bypass_guardrails(mutation):
    directives = copy.deepcopy(CASES[0]["expected_output"]["directive_interpretation"])
    mutation(directives)
    with pytest.raises(DirectiveValidationError):
        validate_directives(CASES[0]["input"], directives)
