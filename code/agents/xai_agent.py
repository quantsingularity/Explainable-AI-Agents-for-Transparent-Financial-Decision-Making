"""
XAI Agent: Provides post-hoc explanations using SHAP, LIME, and Integrated Gradients.
"""

import numpy as np
from typing import Dict, Any
import shap
from lime import lime_tabular
from captum.attr import IntegratedGradients
import torch
from loguru import logger


class XAIAgent:
    """
    Agent responsible for generating post-hoc explanations.
    Supports SHAP, LIME, and Integrated Gradients.
    """

    def __init__(self, method: str = "shap", seed: int = 42):
        """
        Args:
            method: 'shap', 'lime', or 'integrated_gradients'
            seed: Random seed for reproducibility
        """
        self.method = method
        self.seed = seed
        self.explainer = None

        logger.info(f"Initializing XAIAgent with method={method}")

    def initialize(self, model: Any, X_train: np.ndarray, feature_names: list):
        """Initialize the explainer with training data."""
        np.random.seed(self.seed)

        if self.method == "shap":
            # Use KernelExplainer for model-agnostic explanations
            if hasattr(model, "predict_proba"):
                predict_fn = lambda x: model.predict_proba(x)[:, 1]
            else:
                # For neural nets
                def predict_fn(x):
                    model.eval()
                    with torch.no_grad():
                        return model(torch.FloatTensor(x)).numpy().flatten()

            # Sample background data for efficiency
            background = shap.sample(
                X_train, min(100, len(X_train)), random_state=self.seed
            )
            self.explainer = shap.KernelExplainer(predict_fn, background)

        elif self.method == "lime":
            if hasattr(model, "predict_proba"):
                predict_fn = model.predict_proba
            else:

                def predict_fn(x):
                    model.eval()
                    with torch.no_grad():
                        probs = model(torch.FloatTensor(x)).numpy()
                    return np.column_stack([1 - probs, probs])

            self.explainer = lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                mode="classification",
                random_state=self.seed,
            )
            self.predict_fn = predict_fn

        elif self.method == "integrated_gradients":
            # Only works with neural networks
            if not isinstance(model, torch.nn.Module):
                raise ValueError("Integrated Gradients requires a PyTorch model")
            self.explainer = IntegratedGradients(model)
            self.baseline = torch.FloatTensor(X_train.mean(axis=0)).unsqueeze(0)

        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.feature_names = feature_names
        logger.info(f"XAIAgent initialized with {self.method}")

    def explain(self, X: np.ndarray, idx: int) -> Dict[str, Any]:
        """
        Generate explanation for a single instance.

        Returns:
            Dictionary with feature attributions and metadata
        """
        instance = X[idx : idx + 1]

        if self.method == "shap":
            shap_values = self.explainer.shap_values(instance, nsamples=100)

            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification

            attributions = dict(zip(self.feature_names, shap_values[0]))

            return {
                "method": "shap",
                "attributions": attributions,
                "base_value": float(self.explainer.expected_value),
                "instance_values": dict(zip(self.feature_names, instance[0])),
            }

        elif self.method == "lime":
            explanation = self.explainer.explain_instance(
                instance[0], self.predict_fn, num_features=len(self.feature_names)
            )

            # Extract feature importance
            lime_values = dict(explanation.as_list())

            # Map back to feature names (LIME returns feature names with conditions)
            attributions = {}
            for feature_name in self.feature_names:
                # Find matching LIME explanation
                for lime_key, lime_val in lime_values.items():
                    if feature_name in lime_key:
                        attributions[feature_name] = lime_val
                        break
                if feature_name not in attributions:
                    attributions[feature_name] = 0.0

            return {
                "method": "lime",
                "attributions": attributions,
                "score": explanation.score,
                "instance_values": dict(zip(self.feature_names, instance[0])),
            }

        elif self.method == "integrated_gradients":
            instance_tensor = torch.FloatTensor(instance)
            instance_tensor.requires_grad = True

            attributions_tensor = self.explainer.attribute(
                instance_tensor, baselines=self.baseline, n_steps=50
            )

            attributions = dict(
                zip(self.feature_names, attributions_tensor.detach().numpy()[0])
            )

            return {
                "method": "integrated_gradients",
                "attributions": attributions,
                "baseline": self.baseline.numpy()[0].tolist(),
                "instance_values": dict(zip(self.feature_names, instance[0])),
            }

    def get_faithfulness_score(
        self, model: Any, X: np.ndarray, idx: int, top_k: int = 5
    ) -> float:
        """
        Compute faithfulness by masking top-k features and measuring prediction change.

        Returns:
            Faithfulness score (higher = more faithful)
        """
        explanation = self.explain(X, idx)
        attributions = explanation["attributions"]

        # Get original prediction
        if hasattr(model, "predict_proba"):
            original_pred = model.predict_proba(X[idx : idx + 1])[0, 1]
        else:
            model.eval()
            with torch.no_grad():
                original_pred = model(torch.FloatTensor(X[idx : idx + 1])).item()

        # Get top-k most important features
        sorted_features = sorted(
            attributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_features = [f[0] for f in sorted_features[:top_k]]
        top_indices = [self.feature_names.index(f) for f in top_features]

        # Mask top features (set to mean)
        X_masked = X[idx : idx + 1].copy()
        X_masked[0, top_indices] = 0.0  # or use mean

        # Get masked prediction
        if hasattr(model, "predict_proba"):
            masked_pred = model.predict_proba(X_masked)[0, 1]
        else:
            model.eval()
            with torch.no_grad():
                masked_pred = model(torch.FloatTensor(X_masked)).item()

        # Faithfulness = prediction change when important features masked
        faithfulness = abs(original_pred - masked_pred)

        return faithfulness
