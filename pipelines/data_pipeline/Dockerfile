# ==============================================================================
# STAGE 1: Builder
# ==============================================================================
FROM python:3.12.1-slim-bookworm AS builder

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required for building C-linked Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the pipeline-specific requirements
# (Assumes the build context is the repository root)
COPY pipelines/data_pipeline/requirements.txt /tmp/requirements.txt

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ==============================================================================
# STAGE 2: Runtime
# ==============================================================================
FROM python:3.12.1-slim-bookworm AS runtime

# Set environment variables for runtime execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app"

# Install root CA certificates for boto3 S3 TLS verification
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group for security compliance
RUN groupadd -r pipelineuser && useradd -r -g pipelineuser -d /app -s /sbin/nologin pipelineuser

WORKDIR /app

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy only the shared core and the specific pipeline source code.
# Avoiding COPY . . prevents massive datasets, caches, or other pipelines from bloating the image.
COPY shared_core/ /app/shared_core/
COPY pipelines/data_pipeline/ /app/pipelines/data_pipeline/

# Ensure the non-root user owns the application directory
RUN chown -R pipelineuser:pipelineuser /app

# Explicitly create and permission the DuckDB spill directory defined in pipeline_config.yaml
# Prevents "Permission Denied" crashes during out-of-core materialized joins
RUN mkdir -p /tmp/data_pipeline_spill && chown -R pipelineuser:pipelineuser /tmp/data_pipeline_spill

# Downgrade privileges to the non-root user
USER pipelineuser

# Expose no ports; this is a batch processing container, not an API

# Use module invocation (-m) to automatically resolve the PYTHONPATH to /app
ENTRYPOINT ["python", "-m", "pipelines.data_pipeline.src.runner"]