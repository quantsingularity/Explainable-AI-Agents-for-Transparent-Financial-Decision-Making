#!/bin/bash
set -e

echo "=========================================="
echo "XAI Finance Agents - Quick Run (30 min)"
echo "=========================================="

export PYTHONPATH=$(pwd)/code
export DETERMINISTIC_SEED=42
export EXPERIMENT_MODE=quick

echo "Step 1: Setting up data..."
python code/data/fetch_data.py --mode quick

echo "Step 2: Running quick experiment..."
python code/scripts/run_experiment.py --mode quick --seed 42

echo "Step 3: Generating figures..."
python code/scripts/generate_figures.py --results results/quick_run/

echo "Step 4: Running evaluation..."
python code/scripts/evaluate_results.py --results results/quick_run/

echo "=========================================="
echo "Quick run complete!"
echo "Results: results/quick_run/"
echo "Figures: figures/"
echo "=========================================="
