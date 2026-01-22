"""
Comprehensive XAI Methods Implementation
SHAP, LIME, Integrated Gradients, and Counterfactuals with performance optimization
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")


class XAIMethod(ABC):
    """Abstract base class for XAI methods"""

    def __init__(self, model, name: str):
        self.model = model
        self.name = name
        self.computation_time = 0
        self.memory_usage = 0

    @abstractmethod
    def explain(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Generate explanation for input samples"""

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics of the explanation method"""
        return {
            "computation_time_ms": self.computation_time,
            "memory_usage_mb": self.memory_usage,
        }


class SHAPExplainer(XAIMethod):
    """SHAP-based explanations with caching and batch processing"""

    def __init__(self, model, background_data: Optional[np.ndarray] = None):
        super().__init__(model, "SHAP")
        self.background_data = background_data
        self.explainer = None
        self.cache = {}

    def _init_explainer(self):
        """Initialize SHAP explainer (lazy initialization)"""
        if self.explainer is None:
            try:
                import shap

                # Use KernelExplainer for model-agnostic explanations
                if self.background_data is not None:
                    bg_sample = shap.sample(
                        self.background_data, min(100, len(self.background_data))
                    )
                    self.explainer = shap.KernelExplainer(
                        self.model.predict_proba, bg_sample
                    )
                else:
                    raise ValueError("Background data required for SHAP")
            except ImportError:
                print("SHAP not installed. Using mock explainer.")
                self.explainer = self._mock_explainer()

    def _mock_explainer(self):
        """Mock SHAP explainer for testing"""

        class MockExplainer:
            def shap_values(self, X, nsamples=100):
                return np.random.randn(*X.shape) * 0.1

        return MockExplainer()

    def explain(self, X: np.ndarray, nsamples: int = 100) -> Dict[str, Any]:
        """Generate SHAP explanations with optional caching"""
        self._init_explainer()

        start_time = time.time()

        # Check cache
        cache_key = hash(X.tobytes())
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X, nsamples=nsamples)

        self.computation_time = (time.time() - start_time) * 1000

        result = {
            "method": "SHAP",
            "attributions": (
                shap_values if hasattr(shap_values, "values") else shap_values
            ),
            "computation_time_ms": self.computation_time,
            "base_value": getattr(self.explainer, "expected_value", 0),
        }

        # Cache result
        self.cache[cache_key] = result

        return result

    def explain_batch(
        self, X: np.ndarray, batch_size: int = 32, nsamples: int = 100
    ) -> List[Dict[str, Any]]:
        """Batch processing for multiple samples"""
        results = []
        for i in range(0, len(X), batch_size):
            batch = X[i : i + batch_size]
            batch_result = self.explain(batch, nsamples=nsamples)
            results.append(batch_result)
        return results


class LIMEExplainer(XAIMethod):
    """LIME-based explanations with optimization"""

    def __init__(self, model, feature_names: Optional[List[str]] = None):
        super().__init__(model, "LIME")
        self.feature_names = feature_names
        self.explainer = None

    def _init_explainer(self):
        """Initialize LIME explainer"""
        if self.explainer is None:
            try:
                from lime.lime_tabular import LimeTabularExplainer

                self.explainer = LimeTabularExplainer(
                    training_data=np.zeros(
                        (1, len(self.feature_names) if self.feature_names else 10)
                    ),
                    feature_names=self.feature_names,
                    mode="classification",
                )
            except ImportError:
                print("LIME not installed. Using mock explainer.")
                self.explainer = self._mock_explainer()

    def _mock_explainer(self):
        """Mock LIME explainer"""

        class MockLIME:
            def explain_instance(
                self, x, predict_fn, num_features=10, num_samples=5000
            ):
                class MockExp:
                    def as_list(self):
                        return [
                            (f"feature_{i}", np.random.randn() * 0.1)
                            for i in range(num_features)
                        ]

                    def as_map(self):
                        return {
                            1: [
                                (i, np.random.randn() * 0.1)
                                for i in range(num_features)
                            ]
                        }

                return MockExp()

        return MockLIME()

    def explain(
        self, X: np.ndarray, num_features: int = 10, num_samples: int = 5000
    ) -> Dict[str, Any]:
        """Generate LIME explanation"""
        self._init_explainer()

        start_time = time.time()

        explanations = []
        for instance in X:
            exp = self.explainer.explain_instance(
                instance,
                self.model.predict_proba,
                num_features=num_features,
                num_samples=num_samples,
            )
            explanations.append(dict(exp.as_list()))

        self.computation_time = (time.time() - start_time) * 1000

        return {
            "method": "LIME",
            "explanations": explanations,
            "computation_time_ms": self.computation_time,
        }


class IntegratedGradientsExplainer(XAIMethod):
    """Integrated Gradients for neural network models"""

    def __init__(self, model, baseline: Optional[np.ndarray] = None):
        super().__init__(model, "IntegratedGradients")
        self.baseline = baseline

    def explain(self, X: np.ndarray, steps: int = 50) -> Dict[str, Any]:
        """Compute integrated gradients"""
        start_time = time.time()

        if self.baseline is None:
            self.baseline = np.zeros_like(X)

        # Interpolate between baseline and input
        alphas = np.linspace(0, 1, steps)
        gradients = []

        for alpha in alphas:
            interpolated = self.baseline + alpha * (X - self.baseline)
            # Approximate gradient using finite differences
            eps = 1e-5
            grad = np.zeros_like(X)
            for i in range(X.shape[1]):
                X_plus = interpolated.copy()
                X_plus[:, i] += eps
                X_minus = interpolated.copy()
                X_minus[:, i] -= eps

                pred_plus = self.model.predict_proba(X_plus)
                pred_minus = self.model.predict_proba(X_minus)

                grad[:, i] = (pred_plus - pred_minus) / (2 * eps)

            gradients.append(grad)

        # Integrate gradients
        integrated_grads = np.mean(gradients, axis=0) * (X - self.baseline)

        self.computation_time = (time.time() - start_time) * 1000

        return {
            "method": "IntegratedGradients",
            "attributions": integrated_grads,
            "computation_time_ms": self.computation_time,
        }


class CounterfactualExplainer(XAIMethod):
    """Counterfactual explanations using simple perturbation search"""

    def __init__(self, model):
        super().__init__(model, "Counterfactual")

    def explain(
        self, X: np.ndarray, target_class: int = None, max_iterations: int = 100
    ) -> Dict[str, Any]:
        """Generate counterfactual explanation"""
        start_time = time.time()

        counterfactuals = []

        for instance in X:
            original_pred = self.model.predict_proba(instance.reshape(1, -1))[0]
            original_class = int(original_pred > 0.5)

            if target_class is None:
                target_class = 1 - original_class

            # Simple random perturbation search
            best_cf = instance.copy()
            best_distance = float("inf")

            for _ in range(max_iterations):
                # Random perturbation
                perturbed = instance + np.random.randn(len(instance)) * 0.1
                pred = self.model.predict_proba(perturbed.reshape(1, -1))[0]
                pred_class = int(pred > 0.5)

                if pred_class == target_class:
                    distance = np.linalg.norm(perturbed - instance)
                    if distance < best_distance:
                        best_distance = distance
                        best_cf = perturbed

            counterfactuals.append(
                {
                    "original": instance,
                    "counterfactual": best_cf,
                    "distance": best_distance,
                }
            )

        self.computation_time = (time.time() - start_time) * 1000

        return {
            "method": "Counterfactual",
            "counterfactuals": counterfactuals,
            "computation_time_ms": self.computation_time,
        }


class XAIMethodSelector:
    """Decision tree for selecting appropriate XAI method"""

    @staticmethod
    def recommend_method(
        model_type: str,
        dataset_size: int,
        feature_count: int,
        time_budget_ms: float = 1000,
        need_local: bool = True,
    ) -> str:
        """
        Recommend XAI method based on context

        Decision tree logic:
        1. If time_budget < 100ms → Use simple feature importance
        2. If model is tree-based → Use SHAP TreeExplainer (fast)
        3. If model is linear → Use coefficients or LIME
        4. If need counterfactual → Use Counterfactual
        5. If dataset is large (>10k) → Use LIME (faster than SHAP Kernel)
        6. If feature_count < 20 → Use SHAP
        7. Default → LIME
        """

        if time_budget_ms < 100:
            return "FeatureImportance"

        if model_type in ["random_forest", "xgboost", "lightgbm"]:
            return "SHAP_Tree"

        if model_type == "logistic":
            return "Coefficients"

        if need_local and dataset_size > 10000:
            return "LIME"

        if feature_count < 20:
            return "SHAP"

        return "LIME"

    @staticmethod
    def get_method_comparison() -> Dict[str, Dict[str, Any]]:
        """Get comprehensive comparison of XAI methods"""
        return {
            "SHAP": {
                "computational_cost": "High",
                "memory_usage": "High",
                "typical_time_ms": "500-2000",
                "model_agnostic": True,
                "local_explanation": True,
                "global_explanation": True,
                "use_cases": [
                    "Any model",
                    "Need faithful explanations",
                    "Feature interactions",
                ],
                "limitations": ["Slow for large datasets", "High memory"],
            },
            "LIME": {
                "computational_cost": "Medium",
                "memory_usage": "Low",
                "typical_time_ms": "100-500",
                "model_agnostic": True,
                "local_explanation": True,
                "global_explanation": False,
                "use_cases": ["Large datasets", "Quick explanations", "Any model"],
                "limitations": ["Less stable", "Sampling dependent"],
            },
            "IntegratedGradients": {
                "computational_cost": "Medium-High",
                "memory_usage": "Medium",
                "typical_time_ms": "200-1000",
                "model_agnostic": False,
                "local_explanation": True,
                "global_explanation": False,
                "use_cases": ["Neural networks", "Gradient-based models"],
                "limitations": ["Requires gradients", "Baseline selection"],
            },
            "Counterfactual": {
                "computational_cost": "Very High",
                "memory_usage": "Low",
                "typical_time_ms": "1000-5000",
                "model_agnostic": True,
                "local_explanation": True,
                "global_explanation": False,
                "use_cases": ["What-if scenarios", "Actionable insights"],
                "limitations": ["Very slow", "May not find valid CF"],
            },
        }
