FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY apps ./apps
COPY pytest.ini .
COPY logo.png ./logo.png

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "uvicorn apps.insurance_claims.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
