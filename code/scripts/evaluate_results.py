"""
Evaluate experiment results.

Reads ``experiment_results.csv`` produced by ``run_experiment.py`` and computes a
consolidated evaluation: per-method explainability quality, per-model predictive
performance, the performance-vs-explainability trade-off, and overall
best-in-class selections. Writes both a machine-readable ``evaluation_summary.json``
and a human-readable ``evaluation_summary.md`` into the results directory, and
prints a short summary to stdout.

This is referenced as the final step of ``scripts/run_quick.sh``.
"""

import argparse
import json
import os
from typing import Any, Dict

import pandas as pd
from loguru import logger

# Predictive-performance columns reported per (model, method) row.
_PERF_COLS = ["auc", "precision", "recall", "accuracy"]
# Explainability-quality columns.
_XAI_COLS = ["mean_faithfulness", "mean_completeness", "mean_explanation_time"]


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if col in df.columns and len(df) else float("nan")


def evaluate(results_dir: str) -> Dict[str, Any]:
    """Compute the evaluation summary from ``experiment_results.csv``."""
    csv_path = os.path.join(results_dir, "experiment_results.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Results file not found: {csv_path}. Run run_experiment.py first."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"Results file is empty: {csv_path}")

    summary: Dict[str, Any] = {
        "n_experiments": int(len(df)),
        "models": sorted(df["model_type"].unique().tolist()),
        "xai_methods": sorted(df["xai_method"].unique().tolist()),
    }

    # Per-XAI-method explainability quality (averaged across models).
    by_method = {}
    for method, group in df.groupby("xai_method"):
        by_method[method] = {col: _safe_mean(group, col) for col in _XAI_COLS}
    summary["by_xai_method"] = by_method

    # Per-model predictive performance (averaged across methods; the predictive
    # metrics are identical across XAI methods for a given model, but averaging is
    # robust to that).
    by_model = {}
    for model, group in df.groupby("model_type"):
        by_model[model] = {col: _safe_mean(group, col) for col in _PERF_COLS}
    summary["by_model"] = by_model

    # Best-in-class selections.
    best_auc_row = df.loc[df["auc"].idxmax()]
    best_faith_row = df.loc[df["mean_faithfulness"].idxmax()]
    fastest_row = df.loc[df["mean_explanation_time"].idxmin()]
    summary["best"] = {
        "highest_auc": {
            "model_type": best_auc_row["model_type"],
            "xai_method": best_auc_row["xai_method"],
            "auc": float(best_auc_row["auc"]),
        },
        "most_faithful_explanations": {
            "model_type": best_faith_row["model_type"],
            "xai_method": best_faith_row["xai_method"],
            "mean_faithfulness": float(best_faith_row["mean_faithfulness"]),
        },
        "fastest_explanations": {
            "model_type": fastest_row["model_type"],
            "xai_method": fastest_row["xai_method"],
            "mean_explanation_time": float(fastest_row["mean_explanation_time"]),
        },
    }

    # Performance vs. explainability trade-off: rank methods by faithfulness and
    # note the average AUC achieved alongside.
    tradeoff = []
    for method, stats in by_method.items():
        method_df = df[df["xai_method"] == method]
        tradeoff.append(
            {
                "xai_method": method,
                "mean_faithfulness": stats["mean_faithfulness"],
                "mean_auc": _safe_mean(method_df, "auc"),
                "mean_explanation_time": stats["mean_explanation_time"],
            }
        )
    tradeoff.sort(key=lambda d: d["mean_faithfulness"], reverse=True)
    summary["tradeoff_ranking"] = tradeoff

    return summary


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines = ["# Evaluation Summary", ""]
    lines.append(f"- Experiments: {summary['n_experiments']}")
    lines.append(f"- Models: {', '.join(summary['models'])}")
    lines.append(f"- XAI methods: {', '.join(summary['xai_methods'])}")
    lines.append("")

    lines.append("## Best in class")
    best = summary["best"]
    auc = best["highest_auc"]
    faith = best["most_faithful_explanations"]
    fast = best["fastest_explanations"]
    lines.append(
        f"- Highest AUC: {auc['model_type']} / {auc['xai_method']} "
        f"(AUC = {auc['auc']:.3f})"
    )
    lines.append(
        f"- Most faithful explanations: {faith['model_type']} / "
        f"{faith['xai_method']} (faithfulness = {faith['mean_faithfulness']:.3f})"
    )
    lines.append(
        f"- Fastest explanations: {fast['model_type']} / {fast['xai_method']} "
        f"({fast['mean_explanation_time']*1000:.1f} ms)"
    )
    lines.append("")

    lines.append("## Explainability quality by method")
    lines.append("")
    lines.append("| Method | Faithfulness | Completeness | Time (ms) |")
    lines.append("| --- | --- | --- | --- |")
    for method, stats in summary["by_xai_method"].items():
        lines.append(
            f"| {method} | {stats['mean_faithfulness']:.3f} | "
            f"{stats['mean_completeness']:.3f} | "
            f"{stats['mean_explanation_time']*1000:.1f} |"
        )
    lines.append("")

    lines.append("## Predictive performance by model")
    lines.append("")
    lines.append("| Model | AUC | Precision | Recall | Accuracy |")
    lines.append("| --- | --- | --- | --- | --- |")
    for model, stats in summary["by_model"].items():
        lines.append(
            f"| {model} | {stats['auc']:.3f} | {stats['precision']:.3f} | "
            f"{stats['recall']:.3f} | {stats['accuracy']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(results_dir: str = "results/quick_run/") -> Dict[str, Any]:
    """Evaluate results and write JSON + Markdown summaries.

    Returns:
        The evaluation summary dictionary.
    """
    summary = evaluate(results_dir)

    json_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    md_path = os.path.join(results_dir, "evaluation_summary.md")
    with open(md_path, "w") as f:
        f.write(_render_markdown(summary))

    logger.info(f"Evaluation summary written to {json_path} and {md_path}")

    best = summary["best"]["highest_auc"]
    top = summary["tradeoff_ranking"][0]
    logger.info(
        f"Best AUC: {best['model_type']}/{best['xai_method']} = {best['auc']:.3f}; "
        f"most faithful method: {top['xai_method']} "
        f"(faithfulness {top['mean_faithfulness']:.3f})"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate experiment results.")
    parser.add_argument("--results", default="results/quick_run/")
    args = parser.parse_args()
    main(args.results)
