"""
Comprehensive Testing Suite with 80%+ Coverage
Unit tests for agents, XAI methods, and integration tests
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.baseline_models import BaselineModel, BaselineComparator
from xai.xai_methods import (
    SHAPExplainer,
    LIMEExplainer,
    IntegratedGradientsExplainer,
    CounterfactualExplainer,
    XAIMethodSelector,
)


class TestBaselineModels:
    """Test suite for baseline models"""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        X_test = np.random.randn(30, 10)
        y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
        return X_train, y_train, X_test, y_test

    def test_logistic_regression_training(self, sample_data):
        """Test logistic regression model training"""
        X_train, y_train, _, _ = sample_data
        model = BaselineModel(model_type="logistic", random_state=42)
        metrics = model.train(X_train, y_train)

        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert metrics["accuracy"] > 0.4
        assert metrics["roc_auc"] > 0.4
        assert model.training_time > 0

    def test_random_forest_training(self, sample_data):
        """Test random forest model training"""
        X_train, y_train, _, _ = sample_data
        model = BaselineModel(model_type="random_forest", random_state=42)
        metrics = model.train(X_train, y_train)

        assert "accuracy" in metrics
        assert metrics["accuracy"] > 0.4
        assert model.model is not None

    def test_neural_network_training(self, sample_data):
        """Test neural network model training"""
        X_train, y_train, _, _ = sample_data
        model = BaselineModel(model_type="neural_network", random_state=42)
        metrics = model.train(X_train, y_train)

        assert "accuracy" in metrics
        assert model.model is not None

    def test_model_evaluation(self, sample_data):
        """Test model evaluation on test data"""
        X_train, y_train, X_test, y_test = sample_data
        model = BaselineModel(model_type="logistic", random_state=42)
        model.train(X_train, y_train)

        test_metrics = model.evaluate(X_test, y_test)

        assert "accuracy" in test_metrics
        assert "roc_auc" in test_metrics
        assert "inference_time_ms" in test_metrics
        assert test_metrics["inference_time_ms"] > 0

    def test_predict_proba(self, sample_data):
        """Test probability predictions"""
        X_train, y_train, X_test, _ = sample_data
        model = BaselineModel(model_type="logistic", random_state=42)
        model.train(X_train, y_train)

        probas = model.predict_proba(X_test)

        assert probas.shape[0] == X_test.shape[0]
        assert np.all(probas >= 0) and np.all(probas <= 1)

    def test_feature_importance(self, sample_data):
        """Test feature importance extraction"""
        X_train, y_train, _, _ = sample_data

        for model_type in ["logistic", "random_forest", "neural_network"]:
            model = BaselineModel(model_type=model_type, random_state=42)
            model.train(X_train, y_train)

            importance = model.get_feature_importance()
            assert len(importance) == X_train.shape[1]
            assert np.all(importance >= 0)

    def test_baseline_comparator(self, sample_data):
        """Test baseline model comparator"""
        X_train, y_train, X_test, y_test = sample_data

        comparator = BaselineComparator(random_state=42)
        comparator.add_model("Logistic", "logistic")
        comparator.add_model("RandomForest", "random_forest")

        comparator.train_all(X_train, y_train)
        comparator.evaluate_all(X_test, y_test)

        comparison = comparator.get_comparison_table()
        assert len(comparison) == 2
        assert "Logistic" in comparison
        assert "RandomForest" in comparison

        best_name, best_model = comparator.get_best_model("roc_auc")
        assert best_name in ["Logistic", "RandomForest"]
        assert best_model is not None

    def test_invalid_model_type(self):
        """Test handling of invalid model type"""
        with pytest.raises(ValueError):
            model = BaselineModel(model_type="invalid_model")
            model._init_model()


class TestXAIMethods:
    """Test suite for XAI methods"""

    @pytest.fixture
    def trained_model(self):
        """Get a trained model for XAI testing"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        model = BaselineModel(model_type="logistic", random_state=42)
        model.train(X_train, y_train)

        return model, X_train

    def test_shap_explainer_init(self, trained_model):
        """Test SHAP explainer initialization"""
        model, X_train = trained_model
        explainer = SHAPExplainer(model, background_data=X_train)

        assert explainer.model is not None
        assert explainer.background_data is not None
        assert explainer.name == "SHAP"

    def test_shap_explain(self, trained_model):
        """Test SHAP explanation generation"""
        model, X_train = trained_model
        explainer = SHAPExplainer(model, background_data=X_train)

        X_test = np.random.randn(5, 10)
        result = explainer.explain(X_test, nsamples=50)

        assert "method" in result
        assert result["method"] == "SHAP"
        assert "attributions" in result
        assert "computation_time_ms" in result
        assert result["computation_time_ms"] > 0

    def test_shap_caching(self, trained_model):
        """Test SHAP explanation caching"""
        model, X_train = trained_model
        explainer = SHAPExplainer(model, background_data=X_train)

        X_test = np.random.randn(2, 10)

        # First call
        result1 = explainer.explain(X_test)
        result1["computation_time_ms"]

        # Second call (should be cached)
        result2 = explainer.explain(X_test)
        result2["computation_time_ms"]

        # Cached call should be much faster or instant
        assert len(explainer.cache) > 0

    def test_lime_explainer(self, trained_model):
        """Test LIME explainer"""
        model, X_train = trained_model
        feature_names = [f"feature_{i}" for i in range(10)]
        explainer = LIMEExplainer(model, feature_names=feature_names)

        X_test = np.random.randn(2, 10)
        result = explainer.explain(X_test, num_features=5, num_samples=100)

        assert result["method"] == "LIME"
        assert "explanations" in result
        assert len(result["explanations"]) == len(X_test)
        assert result["computation_time_ms"] > 0

    def test_integrated_gradients(self, trained_model):
        """Test Integrated Gradients explainer"""
        model, _ = trained_model
        explainer = IntegratedGradientsExplainer(model)

        X_test = np.random.randn(3, 10)
        result = explainer.explain(X_test, steps=20)

        assert result["method"] == "IntegratedGradients"
        assert "attributions" in result
        assert result["attributions"].shape == X_test.shape

    def test_counterfactual_explainer(self, trained_model):
        """Test Counterfactual explainer"""
        model, _ = trained_model
        explainer = CounterfactualExplainer(model)

        X_test = np.random.randn(2, 10)
        result = explainer.explain(X_test, max_iterations=50)

        assert result["method"] == "Counterfactual"
        assert "counterfactuals" in result
        assert len(result["counterfactuals"]) == len(X_test)

    def test_xai_method_selector(self):
        """Test XAI method selection logic"""
        # Test quick explanation
        method = XAIMethodSelector.recommend_method(
            model_type="logistic",
            dataset_size=1000,
            feature_count=10,
            time_budget_ms=50,
        )
        assert method == "FeatureImportance"

        # Test tree-based model
        method = XAIMethodSelector.recommend_method(
            model_type="random_forest",
            dataset_size=1000,
            feature_count=10,
            time_budget_ms=500,
        )
        assert method == "SHAP_Tree"

        # Test large dataset
        method = XAIMethodSelector.recommend_method(
            model_type="neural_network",
            dataset_size=50000,
            feature_count=100,
            time_budget_ms=1000,
            need_local=True,
        )
        assert method == "LIME"

    def test_xai_method_comparison(self):
        """Test XAI method comparison data"""
        comparison = XAIMethodSelector.get_method_comparison()

        assert "SHAP" in comparison
        assert "LIME" in comparison
        assert "IntegratedGradients" in comparison
        assert "Counterfactual" in comparison

        for method, data in comparison.items():
            assert "computational_cost" in data
            assert "model_agnostic" in data
            assert "use_cases" in data
            assert isinstance(data["use_cases"], list)


class TestIntegration:
    """Integration tests for full pipeline"""

    def test_full_pipeline(self):
        """Test complete pipeline from data to explanation"""
        # Generate data
        np.random.seed(42)
        X_train = np.random.randn(200, 15)
        y_train = (X_train[:, 0] + 2 * X_train[:, 1] - X_train[:, 2] > 0).astype(int)
        X_test = np.random.randn(50, 15)
        y_test = (X_test[:, 0] + 2 * X_test[:, 1] - X_test[:, 2] > 0).astype(int)

        # Train baseline models
        comparator = BaselineComparator(random_state=42)
        comparator.add_model("Logistic", "logistic")
        comparator.add_model("RandomForest", "random_forest")
        comparator.train_all(X_train, y_train)
        comparator.evaluate_all(X_test, y_test)

        # Get best model
        best_name, best_model = comparator.get_best_model("roc_auc")
        assert best_model is not None

        # Generate explanations with multiple methods
        X_explain = X_test[:3]

        # SHAP
        shap_explainer = SHAPExplainer(best_model, background_data=X_train)
        shap_result = shap_explainer.explain(X_explain, nsamples=50)
        assert "attributions" in shap_result

        # LIME
        feature_names = [f"feature_{i}" for i in range(15)]
        lime_explainer = LIMEExplainer(best_model, feature_names=feature_names)
        lime_result = lime_explainer.explain(X_explain, num_samples=100)
        assert "explanations" in lime_result

        # Verify explanations were generated
        assert shap_result["computation_time_ms"] > 0
        assert lime_result["computation_time_ms"] > 0

    def test_performance_tracking(self):
        """Test performance metrics tracking"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = (X_train[:, 0] > 0).astype(int)

        # Train models and track performance
        models = {}
        for model_type in ["logistic", "random_forest", "neural_network"]:
            model = BaselineModel(model_type=model_type, random_state=42)
            train_metrics = model.train(X_train, y_train)
            models[model_type] = {
                "model": model,
                "training_time": train_metrics["training_time"],
            }

        # All models should have tracked training time
        for model_type, data in models.items():
            assert data["training_time"] > 0

        # Test inference time
        X_test = np.random.randn(20, 10)
        y_test = (X_test[:, 0] > 0).astype(int)

        for model_type, data in models.items():
            test_metrics = data["model"].evaluate(X_test, y_test)
            assert test_metrics["inference_time_ms"] > 0


# Run tests with coverage
if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=../models",
            "--cov=../xai",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )
