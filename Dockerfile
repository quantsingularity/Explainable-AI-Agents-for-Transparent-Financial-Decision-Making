FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    graphviz \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Set environment variables
ENV PYTHONPATH=/workspace/code
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Run tests on build
RUN pytest tests/ -v || echo "Tests will be run manually"

# Default command
CMD ["/bin/bash"]
