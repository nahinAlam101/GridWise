"""HTTP integration with injected model doubles, never a production fallback."""

import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from gridwise.app import create_app
from gridwise.interpreter import InterpreterError
from gridwise.replay import replay


CASES = json.loads((Path(__file__).parents[1] / "data/public_sample_cases.json").read_text())["cases"]


class TestInterpreter:
    __test__ = False
    configured = True

    def __init__(self, directives=None, failure=None):
        self.directives = directives
        self.failure = failure
        self.calls = 0

    async def interpret(self, scenario):
        self.calls += 1
        if self.failure:
            raise self.failure
        return copy.deepcopy(self.directives)

    async def close(self):
        pass


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_http_public_samples_with_injected_reference_interpretation(case):
    model = TestInterpreter(case["expected_output"]["directive_interpretation"])
    with TestClient(create_app(model)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/optimize-energy", json=case["input"])
        assert response.status_code == 200, response.text
        result = response.json()
        replay(case["input"], result, case["expected_output"]["directive_interpretation"])
        assert result["total_cost_bdt"] == pytest.approx(case["expected_output"]["total_cost_bdt"], abs=0.01)
        assert model.calls == 1


@pytest.mark.parametrize("body", ["{", "null", "[]", '{}', '{"scenario_id": "secret-input"}'])
def test_invalid_json_or_schema_is_400_and_never_calls_model(body):
    model = TestInterpreter()
    with TestClient(create_app(model)) as client:
        response = client.post("/optimize-energy", content=body, headers={"Content-Type": "application/json"})
        assert response.status_code == 400
        assert model.calls == 0
        assert "secret-input" not in response.text


def test_unconfigured_model_is_not_ready():
    model = TestInterpreter()
    model.configured = False
    with TestClient(create_app(model)) as client:
        assert client.get("/health").status_code == 503


@pytest.mark.parametrize("failure", [InterpreterError("private-provider-error"), RuntimeError("private-provider-error")])
def test_provider_errors_do_not_leak_or_crash(failure):
    with TestClient(create_app(TestInterpreter(failure=failure))) as client:
        response = client.post("/optimize-energy", json=CASES[0]["input"])
        assert response.status_code == 500
        assert "private-provider-error" not in response.text
        assert client.get("/health").status_code == 200


def test_invalid_interpretation_fails_closed():
    with TestClient(create_app(TestInterpreter([]))) as client:
        response = client.post("/optimize-energy", json=CASES[0]["input"])
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "verification_failed"


def test_impossible_grid_cap_is_controlled():
    scenario = copy.deepcopy(CASES[1]["input"])
    directives = [{"note_index": 0, "applies": True, "directive_type": "max_grid_window",
                   "structured_adjustment": {"hours": list(range(24)), "max_grid_kwh": 0},
                   "explanation": "The grid is unavailable all day."}]
    with TestClient(create_app(TestInterpreter(directives))) as client:
        response = client.post("/optimize-energy", json=scenario)
        assert response.status_code == 422


def test_docs_include_scenario_schema():
    with TestClient(create_app(TestInterpreter())) as client:
        schema = client.get("/openapi.json").json()
        assert "Scenario" in schema["components"]["schemas"]
