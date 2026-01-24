#!/bin/bash
set -e

echo "Building Docker image for XAI Finance Agents..."
docker build -t xai-finance-agents .

echo "Running Docker container..."
docker run -it \
  -v "$(pwd)/data:/workspace/data" \
  -v "$(pwd)/results:/workspace/results" \
  -v "$(pwd)/figures:/workspace/figures" \
  -e PYTHONPATH=/workspace/code \
  -e DETERMINISTIC_SEED=42 \
  xai-finance-agents \
  /bin/bash
