FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip && pip install .

EXPOSE 8080

CMD ["uvicorn", "llm_cost_proxy.main:app", "--host", "0.0.0.0", "--port", "8080"]
