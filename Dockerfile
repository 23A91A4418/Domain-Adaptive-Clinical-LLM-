FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set non-interactive installation and environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python packages
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r /workspace/requirements.txt

# Copy source code and scripts
COPY src/ /workspace/src/
COPY data/ /workspace/data/
COPY docs/ /workspace/docs/
COPY .env.example /workspace/.env.example

# Create output and results directories with permissions
RUN mkdir -p /workspace/results /workspace/output/final_adapter && \
    chmod +x /workspace/src/main.sh

# Default entrypoint
ENTRYPOINT ["/bin/bash", "/workspace/src/main.sh"]
