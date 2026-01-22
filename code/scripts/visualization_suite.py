"""
Publication-Quality Visualization Suite
Generates 6-8 figures for system architecture, XAI comparisons, and results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withStroke


# Set publication-quality defaults
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 14

# Color scheme
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#06A77D",
    "warning": "#F6AE2D",
    "danger": "#D62828",
    "neutral": "#6C757D",
}


class VisualizationSuite:
    """Comprehensive visualization suite for XAI agents"""

    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = output_dir
        import os

        os.makedirs(output_dir, exist_ok=True)

    def plot_system_architecture(self, save_path: Optional[str] = None):
        """
        Figure 1: System Architecture Diagram
        Shows the multi-agent architecture with data flow
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Define agent positions
        agents = {
            "Orchestrator": (5, 9),
            "Evidence\nCollector": (2, 7),
            "Decision\nAgent": (5, 7),
            "XAI\nAgent": (8, 7),
            "Explanation\nAgent": (3.5, 4.5),
            "Privacy\nGuard": (6.5, 4.5),
        }

        # Draw agents
        for agent, (x, y) in agents.items():
            box = FancyBboxPatch(
                (x - 0.6, y - 0.4),
                1.2,
                0.8,
                boxstyle="round,pad=0.1",
                facecolor=COLORS["primary"],
                edgecolor="black",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(box)
            ax.text(
                x,
                y,
                agent,
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="white",
                path_effects=[withStroke(linewidth=2, foreground="black")],
            )

        # Draw connections
        connections = [
            ("Orchestrator", "Evidence\nCollector"),
            ("Orchestrator", "Decision\nAgent"),
            ("Orchestrator", "XAI\nAgent"),
            ("Decision\nAgent", "Explanation\nAgent"),
            ("XAI\nAgent", "Explanation\nAgent"),
            ("Explanation\nAgent", "Privacy\nGuard"),
        ]

        for src, dst in connections:
            x1, y1 = agents[src]
            x2, y2 = agents[dst]
            arrow = FancyArrowPatch(
                (x1, y1 - 0.4),
                (x2, y2 + 0.4),
                arrowstyle="->,head_width=0.4,head_length=0.8",
                color=COLORS["accent"],
                linewidth=2,
                alpha=0.7,
            )
            ax.add_patch(arrow)

        # Add data flow labels
        ax.text(
            5,
            2,
            "Final Explanation Output",
            ha="center",
            fontsize=12,
            weight="bold",
            bbox=dict(boxstyle="round", facecolor=COLORS["success"], alpha=0.3),
        )

        ax.text(
            5,
            0.5,
            "Input: Financial Application Data",
            ha="center",
            fontsize=10,
            style="italic",
        )

        plt.title(
            "Multi-Agent XAI System Architecture", fontsize=16, weight="bold", pad=20
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(
                f"{self.output_dir}/01_system_architecture.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_xai_method_comparison(
        self, comparison_data: Dict[str, Dict], save_path: Optional[str] = None
    ):
        """
        Figure 2: XAI Method Comparison (Side-by-Side)
        Compares SHAP, LIME, IG, and Counterfactuals
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "XAI Method Comparison: Performance & Characteristics",
            fontsize=16,
            weight="bold",
            y=0.98,
        )

        methods = list(comparison_data.keys())

        # Subplot 1: Computational Cost
        ax = axes[0, 0]
        cost_map = {
            "Low": 1,
            "Medium": 2,
            "Medium-High": 2.5,
            "High": 3,
            "Very High": 4,
        }
        costs = [
            cost_map.get(comparison_data[m]["computational_cost"], 2) for m in methods
        ]
        bars = ax.barh(
            methods,
            costs,
            color=[
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["accent"],
                COLORS["success"],
            ],
        )
        ax.set_xlabel("Computational Cost (Relative)", weight="bold")
        ax.set_title("Computational Cost Comparison", weight="bold")
        ax.set_xlim(0, 4.5)

        # Subplot 2: Time Comparison
        ax = axes[0, 1]
        times = []
        for m in methods:
            time_str = comparison_data[m]["typical_time_ms"]
            avg_time = np.mean([float(x) for x in time_str.split("-")])
            times.append(avg_time)
        bars = ax.barh(
            methods,
            times,
            color=[
                COLORS["primary"],
                COLORS["secondary"],
                COLORS["accent"],
                COLORS["success"],
            ],
        )
        ax.set_xlabel("Typical Execution Time (ms)", weight="bold")
        ax.set_title("Execution Time Comparison", weight="bold")
        ax.set_xscale("log")

        # Subplot 3: Capabilities Radar
        ax = axes[1, 0]
        capabilities = [
            "Model\nAgnostic",
            "Local\nExpl.",
            "Global\nExpl.",
            "Stability",
            "Speed",
        ]

        for i, method in enumerate(methods[:2]):  # Show 2 methods for clarity
            values = [
                1 if comparison_data[method]["model_agnostic"] else 0,
                1 if comparison_data[method]["local_explanation"] else 0,
                1 if comparison_data[method]["global_explanation"] else 0,
                0.8 if method == "SHAP" else 0.5,
                0.3 if method == "SHAP" else 0.7,
            ]
            values += values[:1]  # Complete the circle

            angles = np.linspace(
                0, 2 * np.pi, len(capabilities), endpoint=False
            ).tolist()
            angles += angles[:1]

            ax_polar = plt.subplot(2, 2, 3, projection="polar")
            ax_polar.plot(
                angles,
                values,
                "o-",
                linewidth=2,
                label=method,
                color=list(COLORS.values())[i],
            )
            ax_polar.fill(angles, values, alpha=0.15, color=list(COLORS.values())[i])
            ax_polar.set_xticks(angles[:-1])
            ax_polar.set_xticklabels(capabilities, size=8)
            ax_polar.set_ylim(0, 1)
            ax_polar.set_title("Method Capabilities", weight="bold", pad=20)
            ax_polar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            ax_polar.grid(True)

        # Subplot 4: Use Case Matrix
        ax = axes[1, 1]
        use_cases = ["Any Model", "Large Data", "Neural Nets", "Quick Expl.", "What-if"]
        matrix = np.zeros((len(methods), len(use_cases)))

        for i, method in enumerate(methods):
            cases = comparison_data[method]["use_cases"]
            for j, uc in enumerate(use_cases):
                if any(uc.lower() in case.lower() for case in cases):
                    matrix[i, j] = 1

        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(use_cases)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(use_cases, rotation=45, ha="right")
        ax.set_yticklabels(methods)
        ax.set_title("Recommended Use Cases", weight="bold")

        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(use_cases)):
                text = ax.text(
                    j,
                    i,
                    "✓" if matrix[i, j] else "✗",
                    ha="center",
                    va="center",
                    color="darkgreen" if matrix[i, j] else "darkred",
                    fontsize=14,
                    weight="bold",
                )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(
                f"{self.output_dir}/02_xai_method_comparison.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_dict: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
    ):
        """
        Figure 3: Feature Importance Comparison Across Methods
        """
        fig, axes = plt.subplots(1, len(importance_dict), figsize=(16, 6), sharey=True)
        if len(importance_dict) == 1:
            axes = [axes]

        fig.suptitle(
            "Feature Importance Across Different Methods", fontsize=16, weight="bold"
        )

        for idx, (method_name, importances) in enumerate(importance_dict.items()):
            ax = axes[idx]

            # Sort features by importance
            sorted_idx = np.argsort(importances)[-10:]  # Top 10
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importances = importances[sorted_idx]

            # Create horizontal bar plot
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_importances)))
            ax.barh(range(len(sorted_features)), sorted_importances, color=colors)
            ax.set_yticks(range(len(sorted_features)))
            ax.set_yticklabels(sorted_features)
            ax.set_xlabel("Importance Score", weight="bold")
            ax.set_title(f"{method_name}", weight="bold")
            ax.grid(axis="x", alpha=0.3)

            # Add value labels
            for i, v in enumerate(sorted_importances):
                ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(
                f"{self.output_dir}/03_feature_importance.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_model_performance_comparison(
        self, results_df: pd.DataFrame, save_path: Optional[str] = None
    ):
        """
        Figure 4: Baseline Model Performance Comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Baseline Model Performance Comparison", fontsize=16, weight="bold"
        )

        metrics = ["roc_auc", "accuracy", "f1", "inference_time_ms"]
        titles = ["ROC-AUC Score", "Accuracy", "F1 Score", "Inference Time (ms)"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            if metric in results_df.columns:
                bars = ax.bar(
                    results_df["model"],
                    results_df[metric],
                    color=[
                        COLORS["primary"],
                        COLORS["secondary"],
                        COLORS["accent"],
                        COLORS["success"],
                    ],
                )
                ax.set_ylabel(title, weight="bold")
                ax.set_xlabel("Model", weight="bold")
                ax.set_title(title, weight="bold")
                ax.grid(axis="y", alpha=0.3)

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )

                # Rotate x labels
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(
                f"{self.output_dir}/04_model_performance.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_human_study_results(
        self, trust_scores: Dict[str, float], save_path: Optional[str] = None
    ):
        """
        Figure 5: Human Trust Study Results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Human Trust Study Results", fontsize=16, weight="bold")

        # Trust scores bar chart
        models = list(trust_scores.keys())
        scores = list(trust_scores.values())
        colors_list = [
            (
                COLORS["danger"]
                if s < 3.5
                else COLORS["warning"] if s < 4.0 else COLORS["success"]
            )
            for s in scores
        ]

        bars = ax1.barh(models, scores, color=colors_list)
        ax1.set_xlabel("Trust Score (1-5)", weight="bold")
        ax1.set_title("Trust Score by Model Type", weight="bold")
        ax1.axvline(
            x=3.5, color="red", linestyle="--", alpha=0.5, label="Acceptable Threshold"
        )
        ax1.set_xlim(0, 5)
        ax1.legend()
        ax1.grid(axis="x", alpha=0.3)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax1.text(score, i, f" {score:.2f}", va="center", fontsize=10, weight="bold")

        # Improvement over baseline
        baseline = min(scores)
        improvements = [(s - baseline) / baseline * 100 for s in scores]

        bars = ax2.barh(models, improvements, color=colors_list)
        ax2.set_xlabel("Improvement over Baseline (%)", weight="bold")
        ax2.set_title("Trust Improvement Analysis", weight="bold")
        ax2.grid(axis="x", alpha=0.3)

        for i, (bar, imp) in enumerate(zip(bars, improvements)):
            ax2.text(imp, i, f" +{imp:.1f}%", va="center", fontsize=10, weight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(
                f"{self.output_dir}/05_human_trust_results.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_xai_performance_tradeoffs(
        self, methods_data: Dict[str, Dict], save_path: Optional[str] = None
    ):
        """
        Figure 6: XAI Method Performance Trade-offs (Scatter)
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        methods = list(methods_data.keys())
        times = [methods_data[m]["time"] for m in methods]
        faithfulness = [methods_data[m]["faithfulness"] for m in methods]

        colors_list = [
            COLORS["primary"],
            COLORS["secondary"],
            COLORS["accent"],
            COLORS["success"],
        ]

        for i, method in enumerate(methods):
            ax.scatter(
                times[i],
                faithfulness[i],
                s=500,
                alpha=0.6,
                color=colors_list[i % len(colors_list)],
                edgecolors="black",
                linewidth=2,
                label=method,
            )
            ax.annotate(
                method,
                (times[i], faithfulness[i]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=11,
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.7),
            )

        ax.set_xlabel("Computation Time (ms)", fontsize=12, weight="bold")
        ax.set_ylabel("Faithfulness Score", fontsize=12, weight="bold")
        ax.set_title(
            "XAI Method Performance Trade-offs\n(Speed vs. Quality)",
            fontsize=14,
            weight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

        # Add reference lines
        ax.axhline(
            y=0.8, color="green", linestyle="--", alpha=0.5, label="High Faithfulness"
        )
        ax.axvline(
            x=500, color="orange", linestyle="--", alpha=0.5, label="Acceptable Speed"
        )

        ax.legend(loc="best", framealpha=0.9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(
                f"{self.output_dir}/06_xai_performance_tradeoffs.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_explanation_latency_breakdown(
        self, latency_data: Dict[str, float], save_path: Optional[str] = None
    ):
        """
        Figure 7: Explanation Generation Latency Breakdown
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Explanation Generation Latency Analysis", fontsize=16, weight="bold"
        )

        # Pie chart
        components = list(latency_data.keys())
        times = list(latency_data.values())
        colors_list = list(COLORS.values())[: len(components)]

        wedges, texts, autotexts = ax1.pie(
            times,
            labels=components,
            autopct="%1.1f%%",
            colors=colors_list,
            startangle=90,
            textprops={"weight": "bold"},
        )
        ax1.set_title("Latency Breakdown", weight="bold")

        # Bar chart
        ax2.barh(components, times, color=colors_list)
        ax2.set_xlabel("Time (ms)", weight="bold")
        ax2.set_title("Component Latency", weight="bold")
        ax2.grid(axis="x", alpha=0.3)

        for i, v in enumerate(times):
            ax2.text(v, i, f" {v:.0f}ms", va="center", fontsize=10, weight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(
                f"{self.output_dir}/07_latency_breakdown.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def plot_decision_tree_guide(self, save_path: Optional[str] = None):
        """
        Figure 8: XAI Method Selection Decision Tree
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        # Decision nodes
        nodes = {
            "start": (5, 9, "Start:\nChoose XAI Method"),
            "time": (5, 7.5, "Time Budget\n< 100ms?"),
            "model": (3, 6, "Tree-based\nModel?"),
            "linear": (7, 6, "Linear\nModel?"),
            "large": (2, 4.5, "Dataset\n> 10k?"),
            "features": (4, 4.5, "Features\n< 20?"),
            "counterfactual": (6, 4.5, "Need\nWhat-if?"),
            "neural": (8, 4.5, "Neural\nNetwork?"),
        }

        # Leaf nodes (recommendations)
        leaves = {
            "feat_imp": (5, 3, "Feature\nImportance", COLORS["success"]),
            "shap_tree": (3, 3, "SHAP\nTree", COLORS["success"]),
            "coef": (7, 3, "Coefficients", COLORS["success"]),
            "lime": (2, 1.5, "LIME", COLORS["primary"]),
            "shap": (4, 1.5, "SHAP", COLORS["primary"]),
            "counter": (6, 1.5, "Counterfactual", COLORS["accent"]),
            "ig": (8, 1.5, "Integrated\nGradients", COLORS["secondary"]),
        }

        # Draw decision nodes
        for node, (x, y, label) in nodes.items():
            box = FancyBboxPatch(
                (x - 0.5, y - 0.3),
                1,
                0.6,
                boxstyle="round,pad=0.05",
                facecolor=COLORS["warning"],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(box)
            ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")

        # Draw leaf nodes
        for leaf, (x, y, label, color) in leaves.items():
            box = FancyBboxPatch(
                (x - 0.4, y - 0.25),
                0.8,
                0.5,
                boxstyle="round,pad=0.05",
                facecolor=color,
                edgecolor="black",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(box)
            ax.text(
                x,
                y,
                label,
                ha="center",
                va="center",
                fontsize=10,
                weight="bold",
                color="white",
            )

        # Draw connections with labels
        connections = [
            ("start", "time", "yes/no"),
            ("time", "feat_imp", "yes"),
            ("time", "model", "no"),
            ("model", "shap_tree", "yes"),
            ("model", "linear", "no"),
            ("linear", "coef", "yes"),
            ("linear", "large", "no"),
            ("large", "lime", "yes"),
            ("large", "features", "no"),
            ("features", "shap", "yes"),
            ("features", "counterfactual", "no"),
            ("counterfactual", "counter", "yes"),
            ("counterfactual", "neural", "no"),
            ("neural", "ig", "yes"),
            ("neural", "lime", "no"),
        ]

        for src, dst, label in connections:
            if src in nodes:
                x1, y1, _ = nodes[src]
                if dst in nodes:
                    x2, y2, _ = nodes[dst]
                else:
                    x2, y2, _, _ = leaves[dst]
            else:
                continue

            arrow = FancyArrowPatch(
                (x1, y1 - 0.3),
                (x2, y2 + 0.25 if dst in leaves else y2 + 0.3),
                arrowstyle="->,head_width=0.3,head_length=0.6",
                color="black",
                linewidth=1.5,
                alpha=0.6,
            )
            ax.add_patch(arrow)

            # Add label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mid_x,
                mid_y,
                label,
                ha="center",
                fontsize=7,
                style="italic",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, pad=0.2),
            )

        plt.title(
            "XAI Method Selection Decision Tree", fontsize=16, weight="bold", pad=20
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
        else:
            plt.savefig(
                f"{self.output_dir}/08_xai_decision_tree.png",
                bbox_inches="tight",
                dpi=300,
            )
        plt.close()

    def generate_all_figures(self, data: Dict[str, Any]):
        """Generate all visualization figures"""
        print("Generating all publication-quality figures...")

        self.plot_system_architecture()
        print("✓ Generated: System Architecture")

        if "xai_comparison" in data:
            self.plot_xai_method_comparison(data["xai_comparison"])
            print("✓ Generated: XAI Method Comparison")

        if "feature_importance" in data:
            self.plot_feature_importance(
                data.get("feature_names", [f"Feature_{i}" for i in range(10)]),
                data["feature_importance"],
            )
            print("✓ Generated: Feature Importance")

        if "model_performance" in data:
            self.plot_model_performance_comparison(data["model_performance"])
            print("✓ Generated: Model Performance Comparison")

        if "trust_scores" in data:
            self.plot_human_study_results(data["trust_scores"])
            print("✓ Generated: Human Trust Results")

        if "xai_performance" in data:
            self.plot_xai_performance_tradeoffs(data["xai_performance"])
            print("✓ Generated: XAI Performance Trade-offs")

        if "latency_breakdown" in data:
            self.plot_explanation_latency_breakdown(data["latency_breakdown"])
            print("✓ Generated: Latency Breakdown")

        self.plot_decision_tree_guide()
        print("✓ Generated: XAI Decision Tree Guide")

        print(f"\nAll figures saved to: {self.output_dir}/")
