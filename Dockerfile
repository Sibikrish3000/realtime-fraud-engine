# Stage 1: Builder
FROM python:3.12-slim-trixie AS builder

# Install uv by copying the binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy only dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Disable development dependencies
ENV UV_NO_DEV=1

# Sync the project into a new environment
RUN uv sync --frozen --no-install-project

# Stage 2: Final
FROM python:3.12-slim-trixie

# Set working directory
WORKDIR /app

# Install system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Create logs directory
RUN mkdir -p logs

# Create a non-root user for security
# Note: HF Spaces runs as user 1000 by default, so we align with that
RUN useradd -m -u 1000 payshield

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH=/app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Ensure the installed binary is on the `PATH`
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application code
COPY src /app/src
COPY configs /app/configs
COPY models /app/models
COPY entrypoint.sh /app/entrypoint.sh

# Change ownership to the non-root user
RUN chown -R payshield:payshield /app

# Switch to the non-root user
USER 1000

# Expose ports
# 7860 is the default port for Hugging Face Spaces
EXPOSE 7860 8000

# Set environment variables for HF Spaces
ENV API_URL="http://localhost:8000/v1/predict"
ENV PORT=7860

# Use entrypoint script to run both services
CMD ["/bin/bash", "/app/entrypoint.sh"]
