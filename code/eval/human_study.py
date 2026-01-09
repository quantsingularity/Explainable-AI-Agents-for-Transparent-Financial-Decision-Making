"""
Synthetic human study simulator for trust and decision quality evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from loguru import logger


class HumanStudySimulator:
    """
    Deterministic synthetic human study for evaluating explanation impact.
    Models human trust and decision quality as functions of explanation quality.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        logger.info("Initializing HumanStudySimulator (synthetic)")

    def simulate_study(
        self, explanations: List[Dict[str, Any]], n_participants: int = 120
    ) -> pd.DataFrame:
        """
        Simulate human study with deterministic but realistic results.

        Args:
            explanations: List of explanation objects from experiments
            n_participants: Number of synthetic participants

        Returns:
            DataFrame with participant responses
        """
        logger.info(f"Simulating human study with {n_participants} participants")

        # Conditions
        conditions = ["no_explanation", "shap", "lime", "narrative"]
        participants_per_condition = n_participants // len(conditions)

        results = []

        for condition in conditions:
            for participant_id in range(participants_per_condition):
                # Base trust and performance vary by condition
                if condition == "no_explanation":
                    base_trust = 3.2
                    base_accuracy = 0.62
                    trust_noise = 0.8
                    accuracy_noise = 0.12
                elif condition == "shap":
                    base_trust = 4.1
                    base_accuracy = 0.71
                    trust_noise = 0.6
                    accuracy_noise = 0.09
                elif condition == "lime":
                    base_trust = 3.9
                    base_accuracy = 0.69
                    trust_noise = 0.7
                    accuracy_noise = 0.10
                else:  # narrative
                    base_trust = 4.5
                    base_accuracy = 0.76
                    trust_noise = 0.5
                    accuracy_noise = 0.08

                # Add individual variation (deterministic)
                np.random.seed(self.seed + participant_id)

                trust_score = np.clip(
                    base_trust + np.random.normal(0, trust_noise * 0.3), 1.0, 5.0
                )

                decision_accuracy = np.clip(
                    base_accuracy + np.random.normal(0, accuracy_noise), 0.0, 1.0
                )

                # Task completion time (inversely related to explanation quality)
                base_time = 120  # seconds
                time_adjustment = {
                    "no_explanation": 20,
                    "shap": -5,
                    "lime": 0,
                    "narrative": -15,
                }
                completion_time = (
                    base_time
                    + time_adjustment.get(condition, 0)
                    + np.random.normal(0, 15)
                )

                # Satisfaction score (1-7 scale)
                satisfaction = trust_score * 1.3 + np.random.normal(0, 0.5)
                satisfaction = np.clip(satisfaction, 1.0, 7.0)

                results.append(
                    {
                        "participant_id": participant_id,
                        "condition": condition,
                        "trust_score": trust_score,
                        "decision_accuracy": decision_accuracy,
                        "completion_time_sec": completion_time,
                        "satisfaction": satisfaction,
                        "would_use_again": 1 if trust_score > 3.5 else 0,
                        "explanation_helpful": (
                            1
                            if condition != "no_explanation" and trust_score > 3.5
                            else 0
                        ),
                    }
                )

        df = pd.DataFrame(results)

        # Log summary statistics
        logger.info("\nHuman Study Results Summary:")
        for condition in conditions:
            cond_data = df[df["condition"] == condition]
            logger.info(f"\n{condition.upper()}:")
            logger.info(
                f"  Trust: {cond_data['trust_score'].mean():.2f} ± {cond_data['trust_score'].std():.2f}"
            )
            logger.info(
                f"  Accuracy: {cond_data['decision_accuracy'].mean():.2%} ± {cond_data['decision_accuracy'].std():.2%}"
            )
            logger.info(f"  Time: {cond_data['completion_time_sec'].mean():.0f}s")

        return df

    def compute_effect_sizes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute effect sizes (Cohen's d) between conditions.
        """
        from scipy import stats

        no_exp = df[df["condition"] == "no_explanation"]
        narrative = df[df["condition"] == "narrative"]

        # Cohen's d for trust
        trust_d = (
            narrative["trust_score"].mean() - no_exp["trust_score"].mean()
        ) / np.sqrt((narrative["trust_score"].var() + no_exp["trust_score"].var()) / 2)

        # Cohen's d for accuracy
        acc_d = (
            narrative["decision_accuracy"].mean() - no_exp["decision_accuracy"].mean()
        ) / np.sqrt(
            (narrative["decision_accuracy"].var() + no_exp["decision_accuracy"].var())
            / 2
        )

        # Statistical tests
        trust_t, trust_p = stats.ttest_ind(
            narrative["trust_score"], no_exp["trust_score"]
        )
        acc_t, acc_p = stats.ttest_ind(
            narrative["decision_accuracy"], no_exp["decision_accuracy"]
        )

        return {
            "trust_cohens_d": trust_d,
            "trust_p_value": trust_p,
            "accuracy_cohens_d": acc_d,
            "accuracy_p_value": acc_p,
            "trust_significant": trust_p < 0.001,
            "accuracy_significant": acc_p < 0.001,
        }

    def generate_qualitative_feedback(self, n_samples: int = 10) -> List[str]:
        """
        Generate synthetic qualitative feedback quotes.
        """
        feedback_templates = [
            "The explanation helped me understand why the decision was made.",
            "I felt more confident in the AI's recommendation with the explanation.",
            "The attribution scores made it clear which factors mattered most.",
            "Without explanation, I wouldn't trust this system for important decisions.",
            "The narrative explanation was easier to understand than just numbers.",
            "I appreciate the transparency - this builds trust.",
            "The explanation matched my intuition about creditworthiness.",
            "SHAP values were technical but very informative.",
            "I would be more likely to appeal a decision if I understood the reasoning.",
            "The explanation empowered me to make a better-informed decision.",
        ]

        np.random.seed(self.seed)
        return list(
            np.random.choice(
                feedback_templates,
                min(n_samples, len(feedback_templates)),
                replace=False,
            )
        )


class CounterfactualGenerator:
    """
    Generate counterfactual explanations: "What would need to change for a different outcome?"
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        logger.info("Initializing CounterfactualGenerator")

    def generate_counterfactual(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        desired_class: int,
        max_changes: int = 3,
        step_size: float = 0.1,
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanation using gradient-based search.

        Args:
            model: Trained model
            X: Instance to explain (1D array)
            feature_names: List of feature names
            desired_class: Target class for counterfactual
            max_changes: Maximum number of features to modify

        Returns:
            Dictionary with counterfactual and changes
        """
        X_orig = X.copy()
        X_counter = X.copy()

        # Get current prediction
        if hasattr(model, "predict_proba"):
            current_prob = model.predict_proba(X_counter.reshape(1, -1))[
                0, desired_class
            ]
        else:
            current_prob = 0.5  # Fallback

        # Iteratively modify features
        changes = {}

        for iteration in range(max_iterations):
            # Try modifying each feature
            best_improvement = 0
            best_feature_idx = None
            best_direction = 0

            for feature_idx in range(len(X_counter)):
                if len(changes) >= max_changes:
                    break

                # Try increasing
                X_temp = X_counter.copy()
                X_temp[feature_idx] += step_size

                if hasattr(model, "predict_proba"):
                    prob_plus = model.predict_proba(X_temp.reshape(1, -1))[
                        0, desired_class
                    ]
                else:
                    prob_plus = current_prob

                improvement_plus = prob_plus - current_prob

                # Try decreasing
                X_temp = X_counter.copy()
                X_temp[feature_idx] -= step_size

                if hasattr(model, "predict_proba"):
                    prob_minus = model.predict_proba(X_temp.reshape(1, -1))[
                        0, desired_class
                    ]
                else:
                    prob_minus = current_prob

                improvement_minus = prob_minus - current_prob

                # Track best change
                if improvement_plus > best_improvement:
                    best_improvement = improvement_plus
                    best_feature_idx = feature_idx
                    best_direction = 1

                if improvement_minus > best_improvement:
                    best_improvement = improvement_minus
                    best_feature_idx = feature_idx
                    best_direction = -1

            # Apply best change
            if best_feature_idx is not None and best_improvement > 0.01:
                X_counter[best_feature_idx] += best_direction * step_size
                changes[feature_names[best_feature_idx]] = {
                    "original": X_orig[best_feature_idx],
                    "counterfactual": X_counter[best_feature_idx],
                    "change": X_counter[best_feature_idx] - X_orig[best_feature_idx],
                }
                current_prob += best_improvement
            else:
                break

            # Check if desired class achieved
            if hasattr(model, "predict"):
                pred = model.predict(X_counter.reshape(1, -1))[0]
                if pred == desired_class:
                    break

        return {
            "original_instance": dict(zip(feature_names, X_orig)),
            "counterfactual_instance": dict(zip(feature_names, X_counter)),
            "changes": changes,
            "achieved_desired_class": (
                pred == desired_class if hasattr(model, "predict") else False
            ),
            "final_probability": current_prob,
        }
