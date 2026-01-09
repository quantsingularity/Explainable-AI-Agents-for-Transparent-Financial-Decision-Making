"""
Unit tests for XAI Finance Agents
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.decision_agent import DecisionAgent
from agents.xai_agent import XAIAgent
from agents.privacy import PIIRedactor, ExplanationSanityChecker
from data.fetch_data import DataFetcher


class TestDecisionAgent:
    def test_logistic_training(self):
        """Test logistic regression training."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        agent = DecisionAgent(model_type="logistic", seed=42)
        agent.train(X, y, ["f1", "f2", "f3", "f4", "f5"])

        assert agent.is_trained
        predictions, probas = agent.predict(X[:10])
        assert len(predictions) == 10
        assert len(probas) == 10
        assert all(p in [0, 1] for p in predictions)
        assert all(0 <= p <= 1 for p in probas)

    def test_tree_training(self):
        """Test decision tree training."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        agent = DecisionAgent(model_type="tree", seed=42)
        agent.train(X, y, ["f1", "f2", "f3", "f4", "f5"])

        assert agent.is_trained
        predictions, probas = agent.predict(X[:10])
        assert len(predictions) == 10


class TestXAIAgent:
    def test_shap_initialization(self):
        """Test SHAP explainer initialization."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        decision_agent = DecisionAgent(model_type="logistic", seed=42)
        decision_agent.train(X, y, ["f1", "f2", "f3", "f4", "f5"])

        xai_agent = XAIAgent(method="shap", seed=42)
        xai_agent.initialize(decision_agent.model, X, ["f1", "f2", "f3", "f4", "f5"])

        assert xai_agent.explainer is not None

    def test_lime_initialization(self):
        """Test LIME explainer initialization."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        decision_agent = DecisionAgent(model_type="logistic", seed=42)
        decision_agent.train(X, y, ["f1", "f2", "f3", "f4", "f5"])

        xai_agent = XAIAgent(method="lime", seed=42)
        xai_agent.initialize(decision_agent.model, X, ["f1", "f2", "f3", "f4", "f5"])

        assert xai_agent.explainer is not None


class TestPrivacy:
    def test_pii_redaction(self):
        """Test PII redaction functionality."""
        redactor = PIIRedactor()

        data = {
            "name": "John Doe",
            "ssn": "123-45-6789",
            "email": "john@example.com",
            "income": 50000,
            "credit_score": 720,
        }

        redacted = redactor.redact(data)

        assert redacted["name"] == "[REDACTED]"
        assert redacted["ssn"] in ["[REDACTED]", "[REDACTED-SSN]"]
        assert redacted["email"] in ["[REDACTED]", "[REDACTED-EMAIL]"]
        assert redacted["income"] == 50000  # Not PII
        assert redacted["credit_score"] == 720  # Not PII

    def test_sanity_checker(self):
        """Test explanation sanity checker."""
        checker = ExplanationSanityChecker()

        explanation = {
            "xai_explanation": {"attributions": {"f1": 0.5, "f2": -0.3, "f3": 0.1}},
            "narrative": {
                "narrative": "f1 and f2 are important factors",
                "top_features": ["f1", "f2", "f3"],
            },
            "decision": {"probability": 0.8},
        }

        result = checker.check(explanation)
        assert isinstance(result, dict)
        assert "passed" in result
        assert "warnings" in result


class TestDataFetcher:
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        fetcher = DataFetcher(seed=42)
        df, feature_cols = fetcher.fetch_lending_data(mode="quick")

        assert len(df) == 1000  # Quick mode
        assert len(feature_cols) > 0
        assert "loan_status" in df.columns

        # Check data quality
        assert df["loan_status"].isin([0, 1]).all()
        assert df.isna().sum().sum() == 0  # No missing values

    def test_train_test_split(self):
        """Test train/test split preparation."""
        fetcher = DataFetcher(seed=42)
        df, feature_cols = fetcher.fetch_lending_data(mode="quick")

        X_train, X_test, y_train, y_test, scaler, feature_names = (
            fetcher.prepare_train_test_split(df, feature_cols)
        )

        assert len(X_train) == 800  # 80% of 1000
        assert len(X_test) == 200  # 20% of 1000
        assert len(y_train) == 800
        assert len(y_test) == 200
        assert len(feature_names) == X_train.shape[1]


class TestIntegration:
    def test_end_to_end_quick(self):
        """Integration test: full pipeline."""
        from agents.orchestrator import Orchestrator
        from data.fetch_data import DataFetcher

        # Fetch data
        fetcher = DataFetcher(seed=42)
        df, feature_cols = fetcher.fetch_lending_data(mode="quick")
        X_train, X_test, y_train, y_test, scaler, feature_names = (
            fetcher.prepare_train_test_split(df, feature_cols)
        )

        # Initialize orchestrator
        orchestrator = Orchestrator(
            model_type="logistic",
            xai_method="shap",
            explanation_style="regulatory",
            seed=42,
        )

        # Train
        orchestrator.train(X_train, y_train, feature_names)

        # Generate explanation
        explanation = orchestrator.explain_decision(X_test, 0)

        # Verify explanation structure
        assert "decision" in explanation
        assert "xai_explanation" in explanation
        assert "narrative" in explanation
        assert "metadata" in explanation

        # Verify decision
        assert explanation["decision"]["prediction"] in [0, 1]
        assert 0 <= explanation["decision"]["probability"] <= 1

        # Verify XAI
        assert "attributions" in explanation["xai_explanation"]
        assert len(explanation["xai_explanation"]["attributions"]) > 0

        # Verify narrative
        assert isinstance(explanation["narrative"]["narrative"], str)
        assert len(explanation["narrative"]["narrative"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
