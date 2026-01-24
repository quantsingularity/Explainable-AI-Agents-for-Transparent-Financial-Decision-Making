#!/bin/bash
set -e

echo "=========================================="
echo "XAI Finance Agents - Full Run (~6 hours)"
echo "=========================================="

export PYTHONPATH=$(pwd)/code
export DETERMINISTIC_SEED=42
export EXPERIMENT_MODE=full

echo "Step 1: Setting up data..."
python code/data/fetch_data.py --mode full

echo "Step 2: Running full experiment with all methods..."
python code/scripts/run_experiment.py --mode full --seed 42 --n-folds 5

echo "Step 3: Generating all figures..."
python code/scripts/generate_figures.py --results results/full_run/

echo "Step 4: Running comprehensive evaluation..."
python code/scripts/evaluate_results.py --results results/full_run/ --stats

echo "Step 5: Running human study simulator..."
python code/eval/human_study.py --results results/full_run/

echo "=========================================="
echo "Full run complete!"
echo "Results: results/full_run/"
echo "Figures: figures/"
echo "Human study: results/full_run/human_study.csv"
echo "=========================================="
