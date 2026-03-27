
# Stage 1 — Builder
# Purpose:
#   Install and compile all Python dependencies.
#   Heavy tools are kept ONLY in this stage.

# Use lightweight official Python image compatible with MONAI and PyTorch
FROM python:3.11-slim AS builder

# Set environment variables for non-interactive installation and Poetry configuration

    # Disable interactive prompts during apt installs 
ENV DEBIAN_FRONTEND=noninteractive \
    # Show Python logs immediately (better Docker logging)
    PYTHONUNBUFFERED=1 \
    # Avoid pip cache -> smaller image size
    PIP_NO_CACHE_DIR=1 \
    # Install packages globally (no .venv inside container)
    POETRY_VIRTUALENVS_CREATE=false \
    # Prevent Poetry prompts
    POETRY_NO_INTERACTION=1

# Set working directory inside container 
WORKDIR /app

# Install BUILD dependencies (compiler + headers) required to compile Python packages
# (will NOT exist in the final runtime image).

RUN apt-get update && apt-get install -y --no-install-recommends \
        # gcc, make, etc. required to build wheels
        build-essential \
        # PostgreSQL client libraries (dependency of some packages) 
        libpq-dev \
        # # Utility for downloading files
        curl \
    # Clean apt cache to reduce layer size
    && rm -rf /var/lib/apt/lists/*

#Install Poetry dependency manager
RUN pip install --upgrade pip \
    && pip install poetry

# Copy dependency definition FIRST enabling Docker layer caching:
# dependencies reinstall ONLY when these files change.
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
# --no-root avoids installing the project itself
# (only external dependencies are installed)
RUN poetry install --no-root


# Stage 2 — Runtime
# Purpose:
#   Create minimal image containing only what is
#   required to run inference.

# Start from clean lightweight Python image
FROM python:3.11-slim

# Ensure Python outputs logs immediately
ENV PYTHONUNBUFFERED=1

# Set application directory
WORKDIR /app

# Install RUNTIME libraries ONLY (no compilers → smaller + safer image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python environment from builder stage
# This transfers all dependencies WITHOUT build tools.
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source code into container
COPY . .

# Default command executed when container starts
# Turns container into a ready-to-run inference service
ENTRYPOINT ["python", "inference.py"]