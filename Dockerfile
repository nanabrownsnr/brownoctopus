FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --extra service --extra jev --no-dev

COPY src ./src
COPY data/indexes/default ./data/indexes/default
COPY data/mcps.json ./data/mcps.json

EXPOSE 8000
CMD ["uv", "run", "--no-dev", "uvicorn", "brown_octopus.server:app", "--host", "0.0.0.0", "--port", "8000"]
