"""
Generate additional comprehensive figures for Stage 2.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300

print("Generating additional comprehensive figures...")

figures_dir = "figures"
os.makedirs(figures_dir, exist_ok=True)

# Load results
results_df = pd.read_csv("results/quick_run/experiment_results.csv")

# Figure 6: Model Performance Comparison
print("\n[6/8] Model Performance Comparison...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["auc", "precision", "recall"]
titles = ["ROC-AUC", "Precision", "Recall"]

for ax, metric, title in zip(axes, metrics, titles):
    models = results_df.groupby("model_type")[metric].mean()
    ax.bar(range(len(models)), models.values, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models.index, rotation=0)
    ax.set_ylabel(title, fontweight="bold")
    ax.set_title(f"{title} by Model", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, v in enumerate(models.values):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, "model_performance_comparison.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close()
print(f"  ✓ Saved: {figures_dir}/model_performance_comparison.png")

# Figure 7: Explanation Time Analysis
print("\n[7/8] Explanation Time Analysis...")
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(results_df))
width = 0.35

shap_times = results_df[results_df["xai_method"] == "shap"]["mean_explanation_time"]
lime_times = results_df[results_df["xai_method"] == "lime"]["mean_explanation_time"]

models = results_df[results_df["xai_method"] == "shap"]["model_type"].values
x_pos = np.arange(len(models))

ax.bar(x_pos - width / 2, shap_times, width, label="SHAP", color="#2ca02c", alpha=0.8)
ax.bar(x_pos + width / 2, lime_times, width, label="LIME", color="#ff7f0e", alpha=0.8)

ax.set_xlabel("Model Type", fontweight="bold", fontsize=12)
ax.set_ylabel("Explanation Time (seconds)", fontweight="bold", fontsize=12)
ax.set_title(
    "Explanation Generation Time by Method and Model", fontweight="bold", fontsize=14
)
ax.set_xticks(x_pos)
ax.set_xticklabels(models)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# Add value labels
for i, (s, l) in enumerate(zip(shap_times, lime_times)):
    ax.text(i - width / 2, s + 0.005, f"{s:.3f}s", ha="center", fontsize=8)
    ax.text(i + width / 2, l + 0.005, f"{l:.3f}s", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, "explanation_time_analysis.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close()
print(f"  ✓ Saved: {figures_dir}/explanation_time_analysis.png")

# Figure 8: Comprehensive Metrics Heatmap
print("\n[8/8] Comprehensive Metrics Heatmap...")
fig, ax = plt.subplots(figsize=(10, 6))

# Create pivot table
heatmap_data = results_df.pivot_table(
    values=["auc", "mean_faithfulness", "mean_completeness"],
    index="model_type",
    columns="xai_method",
    aggfunc="mean",
)

# Flatten multi-index columns
heatmap_data.columns = [f"{col[1]}_{col[0]}" for col in heatmap_data.columns]

# Plot heatmap
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu",
    ax=ax,
    cbar_kws={"label": "Score"},
    vmin=0.6,
    vmax=0.9,
)

ax.set_title(
    "Comprehensive Performance Heatmap\n(AUC, Faithfulness, Completeness)",
    fontweight="bold",
    fontsize=13,
    pad=15,
)
ax.set_xlabel("XAI Method & Metric", fontweight="bold", fontsize=11)
ax.set_ylabel("Model Type", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig(
    os.path.join(figures_dir, "comprehensive_metrics_heatmap.png"),
    bbox_inches="tight",
    dpi=300,
)
plt.close()
print(f"  ✓ Saved: {figures_dir}/comprehensive_metrics_heatmap.png")

print("\n" + "=" * 60)
print(f"All additional figures generated!")
print(f"Total figures: 8 (5 from Stage 1 + 3 new)")
print(f"Location: {figures_dir}/")
print("=" * 60)
