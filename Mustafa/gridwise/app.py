"""The two judge-facing HTTP endpoints."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .database import close_database, initialize_database
from .interpreter import InterpreterError, LLMInterpreter
from .optimizer import InfeasibleSchedule, OptimizationError, optimize
from .repository import save_completed_run
from .replay import ReplayError, replay
from .validation import DirectiveValidationError, Scenario, validate_directives


logger = logging.getLogger("gridwise")


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(interpreter=None) -> FastAPI:
    llm_required = os.getenv("LLM_REQUIRED", "true").strip().lower() not in {"0", "false", "no"}

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        initialize_database()
        application.state.interpreter = interpreter or LLMInterpreter()
        try:
            yield
        finally:
            await application.state.interpreter.close()
            close_database()

    application = FastAPI(
        title="GridWise",
        description="LLM-assisted operator directives and verified, minimum-cost campus energy scheduling.",
        version="1.0.0",
        lifespan=lifespan,
    )

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request: Request, _exception: RequestValidationError):
        # Validation exceptions may embed input values; do not echo them or log them.
        return error(400, "invalid_request", "Malformed JSON or invalid scenario. Check the schema at /docs.")

    @application.get("/health", responses={503: {"description": "LLM configuration is missing"}})
    async def health(request: Request):
        if llm_required and not request.app.state.interpreter.configured:
            return error(503, "not_ready", "Configure the language model before sending optimization requests.")
        return {"status": "ok"}

    @application.post(
        "/optimize-energy",
        responses={400: {"description": "Invalid scenario"}, 422: {"description": "Infeasible directives"},
                   500: {"description": "Controlled model or optimization failure"}},
    )
    async def optimize_energy(scenario: Scenario, request: Request):
        data = scenario.model_dump()
        try:
            async with asyncio.timeout(28):
                directives = await request.app.state.interpreter.interpret(data)
                directives = validate_directives(data, directives)
                response = await run_in_threadpool(optimize, data, directives)
                replay(data, response)
                try:
                    await run_in_threadpool(save_completed_run, data, directives, response)
                except Exception as exc:
                    logger.error("Database persistence failed (%s)", type(exc).__name__)
                return response
        except InfeasibleSchedule:
            return error(422, "infeasible_schedule", "The interpreted constraints have no feasible 24-hour schedule.")
        except InterpreterError:
            return error(500, "interpretation_failed", "The language model could not produce a validated interpretation. Check provider configuration and availability.")
        except (DirectiveValidationError, ReplayError, OptimizationError):
            return error(500, "verification_failed", "The service could not produce a verified schedule.")
        except TimeoutError:
            return error(500, "request_timeout", "The optimization request exceeded the service time budget.")
        except Exception as exc:
            logger.error("Optimization failed (%s)", type(exc).__name__)
            return error(500, "internal_error", "The service could not complete this request.")

    return application


app = create_app()
