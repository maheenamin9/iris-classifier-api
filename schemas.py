"""
Pydantic schemas for request/response validation.

Defines the data models used by the API: the input features
required for an iris prediction request (sepal/petal length and
width), the shape of a single prediction response, and batch
variants of both for submitting/receiving multiple predictions
in one call.
"""

from pydantic import BaseModel, Field, field_validator

MAX_BATCH_SIZE = 100


class IrisInput(BaseModel):
    sepal_length: float = Field(
        ...,
        ge=4.3,
        le=7.9,
        description="Sepal length in cm (dataset range: 4.3-7.9)",
    )
    sepal_width: float = Field(
        ...,
        ge=2.0,
        le=4.4,
        description="Sepal width in cm (dataset range: 2.0-4.4)",
    )
    petal_length: float = Field(
        ...,
        ge=1.0,
        le=6.9,
        description="Petal length in cm (dataset range: 1.0-6.9)",
    )
    petal_width: float = Field(
        ...,
        ge=0.1,
        le=2.5,
        description="Petal width in cm (dataset range: 0.1-2.5)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }
    }


class IrisOutput(BaseModel):
    predicted_class: str = Field(..., description="Predicted iris species name")
    predicted_class_id: int = Field(..., description="Predicted iris species class index")
    probabilities: dict[str, float] = Field(
        ..., description="Predicted probability for each class"
    )
    model_version: str = Field(..., description="Version identifier of the model used")

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_class": "setosa",
                "predicted_class_id": 0,
                "probabilities": {
                    "setosa": 0.97,
                    "versicolor": 0.02,
                    "virginica": 0.01,
                },
                "model_version": "1.0.0",
            }
        }
    }


class IrisBatchInput(BaseModel):
    items: list[IrisInput] = Field(..., min_length=1)

    @field_validator("items")
    @classmethod
    def enforce_max_batch_size(cls, items: list[IrisInput]) -> list[IrisInput]:
        if len(items) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(items)} exceeds the maximum allowed batch "
                f"size of {MAX_BATCH_SIZE}. Please split your request into "
                "smaller batches."
            )
        return items


class IrisBatchOutput(BaseModel):
    items: list[IrisOutput]
