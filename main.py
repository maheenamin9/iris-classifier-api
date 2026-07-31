"""
FastAPI application entrypoint.

Creates and configures the FastAPI app instance, loads the trained
iris model once at startup via a lifespan context manager (stored
on app.state so it's shared across requests instead of being
reloaded per-request), registers the API routes from
`api/routes.py`, and wires up structured request/error logging.
Run with:
    uvicorn main:app --reload
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from api.routes import router
from ml.predict import ModelService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_service = ModelService()
    except FileNotFoundError as e:
        logger.warning("model not loaded at startup: %s", e)
        app.state.model_service = None
    yield


app = FastAPI(title="Iris Classifier API", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    logger.info("request received: method=%s path=%s", request.method, request.url.path)
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request completed: method=%s path=%s status=%s duration_ms=%.1f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    logger.warning("validation error: path=%s errors=%s", request.url.path, errors)
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")
