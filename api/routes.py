"""
API route definitions.

Defines the FastAPI router exposing a health check, model metadata,
and single/batch iris prediction endpoints. Routes retrieve the
shared ModelService instance from app.state (populated once at
startup by main.py's lifespan handler) instead of reloading the
model on every request.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from ml.predict import ModelService
from schemas import IrisBatchInput, IrisBatchOutput, IrisInput, IrisOutput

logger = logging.getLogger(__name__)

router = APIRouter()


def get_model_service(request: Request) -> ModelService:
    model_service = getattr(request.app.state, "model_service", None)
    if model_service is None:
        logger.warning("model unavailable: path=%s", request.url.path)
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Run `python -m ml.train` and restart the API.",
        )
    return model_service


@router.get("/health")
def health(request: Request) -> dict:
    model_loaded = getattr(request.app.state, "model_service", None) is not None
    return {"status": "ok", "model_loaded": model_loaded}


@router.get("/model/info")
def model_info(model_service: ModelService = Depends(get_model_service)) -> dict:
    return {
        "classes": model_service.target_names,
        "feature_names": model_service.metadata.get("feature_names"),
        "accuracy": model_service.metadata.get("accuracy"),
        "model_version": model_service.model_version,
    }


@router.post("/predict", response_model=IrisOutput)
def predict(
    input: IrisInput, model_service: ModelService = Depends(get_model_service)
) -> IrisOutput:
    logger.info("request received: endpoint=/predict input=%s", input.model_dump())
    try:
        result = model_service.predict_single(input)
    except Exception:
        logger.exception("prediction failed: endpoint=/predict")
        raise HTTPException(status_code=500, detail="Inference failed.") from None
    logger.info(
        "prediction made: endpoint=/predict class=%s class_id=%s",
        result.predicted_class,
        result.predicted_class_id,
    )
    return result


@router.post("/predict/batch", response_model=IrisBatchOutput)
def predict_batch(
    batch: IrisBatchInput, model_service: ModelService = Depends(get_model_service)
) -> IrisBatchOutput:
    logger.info("request received: endpoint=/predict/batch size=%d", len(batch.items))
    try:
        outputs = model_service.predict_batch(batch.items)
    except Exception:
        logger.exception(
            "prediction failed: endpoint=/predict/batch size=%d", len(batch.items)
        )
        raise HTTPException(status_code=500, detail="Batch inference failed.") from None
    logger.info(
        "prediction made: endpoint=/predict/batch size=%d", len(outputs)
    )
    return IrisBatchOutput(items=outputs)
