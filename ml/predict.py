"""
Inference logic.

Loads the trained model artifact from `models/` and exposes a
ModelService class to run predictions on new input data (sepal/
petal measurements), returning the predicted class and
probabilities for use by the API layer.
"""

import json
import logging
from pathlib import Path

import joblib

from schemas import IrisInput, IrisOutput

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "iris_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


class ModelService:
    """Loads the trained iris model and metadata once, then serves predictions."""

    def __init__(
        self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH
    ) -> None:
        if not model_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {model_path} / {metadata_path}. "
                "Run `python -m ml.train` first to train and save the model."
            )

        self.model = joblib.load(model_path)
        with open(metadata_path) as f:
            self.metadata: dict = json.load(f)
        self.target_names: list[str] = self.metadata["target_names"]
        self.model_version: str = self.metadata.get("model_version", "unknown")

        logger.info(
            "model loaded: version=%s classes=%s", self.model_version, self.target_names
        )

    def predict_single(self, input: IrisInput) -> IrisOutput:
        return self.predict_batch([input])[0]

    def predict_batch(self, inputs: list[IrisInput]) -> list[IrisOutput]:
        features = [
            [i.sepal_length, i.sepal_width, i.petal_length, i.petal_width]
            for i in inputs
        ]
        predicted_ids = self.model.predict(features)
        probabilities = self.model.predict_proba(features)

        outputs = []
        for predicted_id, probs in zip(predicted_ids, probabilities):
            outputs.append(
                IrisOutput(
                    predicted_class=self.target_names[predicted_id],
                    predicted_class_id=int(predicted_id),
                    probabilities={
                        name: float(p) for name, p in zip(self.target_names, probs)
                    },
                    model_version=self.model_version,
                )
            )
        return outputs
