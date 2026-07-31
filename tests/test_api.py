from fastapi.testclient import TestClient

from main import app


def test_health_reports_model_loaded():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_setosa_for_known_sample():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "setosa"
    assert body["predicted_class_id"] == 0
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}


def test_predict_rejects_out_of_range_input():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "sepal_length": 100.0,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            },
        )

    assert response.status_code == 422
