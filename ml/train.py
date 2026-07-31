"""
Model training script.

Loads the iris dataset, trains a scikit-learn classifier,
evaluates it, and serializes the fitted model (and metadata) to
the `models/` directory using joblib.
Intended to be run as a standalone script, e.g. `python -m ml.train`.
"""

import json
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "iris_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

MODEL_VERSION = "1.0.1"


def main() -> None:
    iris = load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    print("Feature Importances:")
    for name, importance in zip(iris.feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "feature_names": list(iris.feature_names),
        "target_names": list(iris.target_names),
        "accuracy": accuracy,
        "model_version": MODEL_VERSION,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")


if __name__ == "__main__":
    main()
