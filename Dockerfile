# =====================================================
# Stage 1 — Builder
# =====================================================
FROM python:3.11-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Build dependencies ONLY here
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip \
    && pip install poetry

# Copy dependency files first (cache optimization)
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-root


# =====================================================
# Stage 2 — Runtime 
# =====================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Only runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY . .

ENTRYPOINT ["python", "scripts/inference.py"]