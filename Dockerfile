# Multi-stage Dockerfile for Open Memory Suite
# Supports both CPU-only and GPU acceleration for ML models

# Stage 1: Base Python environment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-dev && rm -rf $POETRY_CACHE_DIR

# Stage 3: Production image
FROM base as production

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from dependencies stage
COPY --from=dependencies /app/.venv /app/.venv

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/storage && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import open_memory_suite; print('OK')" || exit 1

# Expose port for API service
EXPOSE 8000

# Default command
CMD ["python", "-m", "open_memory_suite.server"]

# Stage 4: Development image with additional tools
FROM production as development

USER root

# Install development dependencies
RUN poetry install

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Override command for development
CMD ["python", "-m", "open_memory_suite.server", "--reload"]

# Stage 5: GPU-enabled image
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy Poetry configuration files
COPY pyproject.toml poetry.lock ./

# Install dependencies with GPU support
RUN poetry install --no-dev

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/storage && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Expose port for API service
EXPOSE 8000

# Default command
CMD ["python", "-m", "open_memory_suite.server", "--gpu"]