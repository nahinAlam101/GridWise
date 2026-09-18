"""Transport-level interpreter tests; no network or paid model requests."""

import asyncio
from copy import deepcopy
import json
import time

import httpx
import pytest

from gridwise.interpreter import InterpreterConfig, InterpreterError, LLMInterpreter


def scenario():
    return {
        "scenario_id": "transport-test",
        "operator_notes": ["Retain one quarter of battery capacity from noon until 2 PM."],
        "battery": {
            "capacity_kwh": 400,
            "initial_energy_kwh": 200,
            "minimum_energy_kwh": 20,
            "max_charge_kwh_per_hour": 100,
            "max_discharge_kwh_per_hour": 100,
        },
    }


def reserve(value=100):
    return [{
        "note_index": 0,
        "applies": True,
        "directive_type": "minimum_battery_reserve",
        "structured_adjustment": {"hours": [12, 13], "minimum_energy_kwh": value},
        "explanation": "Keep a quarter of capacity after the noon and 1 PM slots.",
    }]


def response(directives=None, *, content=None, finish_reason="stop", refusal=None):
    if content is None:
        content = json.dumps({"directive_interpretation": directives if directives is not None else reserve()})
    return httpx.Response(200, json={"choices": [{
        "finish_reason": finish_reason,
        "message": {"content": content, "refusal": refusal},
    }]})


def test_real_request_structure_credentials_and_validated_result():
    seen = []

    def handler(request):
        seen.append(request)
        return response()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test-secret"), client)
            assert await interpreter.interpret(scenario()) == reserve()
            await interpreter.close()
            assert not interpreter.configured
            assert not client.is_closed  # Injected client belongs to its caller.

    asyncio.run(run())
    request = seen[0]
    assert str(request.url) == "https://api.openai.com/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-secret"
    payload = json.loads(request.content)
    assert "test-secret" not in request.content.decode()
    assert payload["model"] == "gpt-4.1-mini"
    assert payload["max_completion_tokens"] == 1800
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert '"capacity_kwh": 400' in payload["messages"][1]["content"]


def test_missing_configuration_never_silently_uses_a_local_parser():
    async def run():
        interpreter = LLMInterpreter(InterpreterConfig())
        assert not interpreter.configured
        with pytest.raises(InterpreterError) as exc:
            await interpreter.interpret(scenario())
        assert exc.value.code == "llm_not_configured"
        assert interpreter._client is None

    asyncio.run(run())


@pytest.mark.parametrize("base_url", ["http://localhost:11434/v1", "http://127.0.0.1:11434/v1", "http://[::1]:11434/v1"])
def test_local_model_can_run_without_api_key(base_url):
    def handler(request):
        assert "Authorization" not in request.headers
        return response()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(base_url=base_url), client)
            assert interpreter.configured
            assert await interpreter.interpret(scenario()) == reserve()

    asyncio.run(run())
    assert not LLMInterpreter(InterpreterConfig(base_url="https://remote.example/v1")).configured


def test_cache_depends_on_notes_and_every_battery_field_not_scenario_id():
    calls = []

    def handler(request):
        calls.append(request)
        context = json.loads(json.loads(request.content)["messages"][1]["content"].split("\n", 1)[1])
        return response(reserve(context["battery"]["capacity_kwh"] / 4))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test"), client)
            original = scenario()
            first = await interpreter.interpret(original)
            first[0]["structured_adjustment"]["hours"].append(20)
            renamed = deepcopy(original)
            renamed["scenario_id"] = "different-id"
            assert await interpreter.interpret(renamed) == reserve()
            assert len(calls) == 1
            resized = deepcopy(original)
            resized["battery"]["capacity_kwh"] = 800
            assert await interpreter.interpret(resized) == reserve(200)
            assert len(calls) == 2
            for field in original["battery"]:
                changed = deepcopy(original)
                changed["battery"][field] += 1
                await interpreter.interpret(changed)
            assert len(calls) == 7
            changed = deepcopy(original)
            changed["operator_notes"] = ["Keep 100 kWh from noon until 2 PM."]
            await interpreter.interpret(changed)
            assert len(calls) == 8

    asyncio.run(run())


def test_invalid_model_result_is_repaired_before_it_can_enter_cache():
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return response(reserve(401) if len(calls) == 1 else reserve())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test"), client)
            assert await interpreter.interpret(scenario()) == reserve()
            assert await interpreter.interpret(scenario()) == reserve()
            assert len(calls) == 2
            assert "minimum_energy_kwh is outside" in calls[1]["messages"][-1]["content"]

    asyncio.run(run())


def test_failed_interpretation_is_not_cached():
    calls = []

    def handler(request):
        calls.append(request)
        return response(content="not JSON") if len(calls) <= 2 else response()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test"), client)
            with pytest.raises(InterpreterError) as exc:
                await interpreter.interpret(scenario())
            assert exc.value.code == "llm_invalid_output"
            assert await interpreter.interpret(scenario()) == reserve()
            assert len(calls) == 3

    asyncio.run(run())


@pytest.mark.parametrize("bad_response", [
    response(content='{"directive_interpretation":[],"directive_interpretation":[]}'),
    response(content='{"directive_interpretation":NaN}'),
    response(content='{"directive_interpretation":[],"unexpected":"test-secret"}'),
    response(finish_reason="length"),
    response(refusal="test-secret"),
    httpx.Response(200, json={"choices": []}),
    httpx.Response(200, content="test-secret"),
    response(content="[" * 2000 + "]" * 2000),
    response(content=json.dumps({"directive_interpretation": [{**reserve()[0], "applies": "true"}]})),
])
def test_malformed_refused_or_truncated_output_is_controlled(bad_response):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: bad_response)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test-secret", max_attempts=1), client)
            with pytest.raises(InterpreterError) as exc:
                await interpreter.interpret(scenario())
            assert exc.value.code == "llm_invalid_output"
            assert "test-secret" not in str(exc.value)

    asyncio.run(run())


def test_authentication_error_is_not_retried_or_exposed():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="Rejected credential test-secret")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test-secret"), client)
            with pytest.raises(InterpreterError) as exc:
                await interpreter.interpret(scenario())
            assert exc.value.code == "llm_provider_rejected"
            assert "test-secret" not in str(exc.value)
            assert len(calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("status", [429, 503])
def test_transient_provider_failure_can_recover(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status) if len(calls) == 1 else response()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test"), client)
            assert await interpreter.interpret(scenario()) == reserve()
            assert len(calls) == 2

    asyncio.run(run())


def test_wall_clock_deadline_cancels_a_stalled_provider():
    async def handler(_):
        await asyncio.sleep(1)
        return response()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test", total_budget_seconds=0.02), client)
            started = time.monotonic()
            with pytest.raises(InterpreterError) as exc:
                await interpreter.interpret(scenario())
            assert exc.value.code == "llm_timeout"
            assert time.monotonic() - started < 0.5

    asyncio.run(run())


def test_json_object_provider_still_uses_validation_and_schema_in_prompt():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert "additionalProperties" in payload["messages"][0]["content"]
        return response()

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            interpreter = LLMInterpreter(InterpreterConfig(api_key="test", json_mode="json_object"), client)
            assert await interpreter.interpret(scenario()) == reserve()

    asyncio.run(run())


def test_configuration_is_bounded_and_secret_repr_is_redacted(monkeypatch):
    assert "test-secret" not in repr(InterpreterConfig(api_key="test-secret"))
    for kwargs in [{"max_attempts": 20}, {"json_mode": "text"}, {"timeout_seconds": float("nan")}, {"total_budget_seconds": 30}, {"base_url": "file:///tmp/provider"}]:
        with pytest.raises(InterpreterError):
            InterpreterConfig(**kwargs)
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "secret-invalid-config")
    with pytest.raises(InterpreterError) as exc:
        InterpreterConfig.from_env()
    assert "secret-invalid-config" not in str(exc.value)
