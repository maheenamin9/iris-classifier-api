# Iris Classifier API

A FastAPI service that serves predictions from a trained iris species
classifier.

## Project Structure

```
.
├── __init__.py         # Root package marker
├── main.py             # FastAPI app entrypoint
├── schemas.py           # Pydantic request/response models
├── models/              # Trained model artifacts (joblib)
├── ml/
│   ├── train.py          # Model training script
│   └── predict.py        # Inference logic
├── api/
│   └── routes.py          # API route definitions
├── tests/                # Test suite
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
uvicorn main:app --reload
```

## Testing

```bash
pytest
```

## Deployment (Docker)

The image trains the model at build time (deterministic, fixed `random_state`),
so no model artifact needs to be committed or copied in separately.

```bash
docker build -t iris-classifier-api .
docker run -p 8000:8000 iris-classifier-api
```

Or with Compose:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000` (docs at `/docs`).
This image can be deployed as-is to any container host (a VPS with Docker,
Fly.io, AWS ECS/App Runner, Google Cloud Run, etc.).
