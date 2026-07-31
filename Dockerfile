FROM python:3.11-slim

WORKDIR /app

RUN groupadd -r app && useradd -r -g app app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model artifacts are gitignored/not baked in; train deterministically at build time
# (fixed random_state in ml/train.py) so the image is self-contained.
RUN python -m ml.train

RUN chown -R app:app /app
USER app

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
