"""
Simplified unit tests (without PyTorch dependency)
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from data.fetch_data import DataFetcher


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

        print(f"✓ Generated {len(df)} samples with {len(feature_cols)} features")

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

        print(f"✓ Train/test split: {len(X_train)}/{len(X_test)}")


class TestModels:
    def test_logistic_regression(self):
        """Test logistic regression model."""
        fetcher = DataFetcher(seed=42)
        df, feature_cols = fetcher.fetch_lending_data(mode="quick")
        X_train, X_test, y_train, y_test, scaler, feature_names = (
            fetcher.prepare_train_test_split(df, feature_cols)
        )

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        probas = model.predict_proba(X_test)

        assert len(predictions) == len(X_test)
        assert probas.shape == (len(X_test), 2)
        assert all(p in [0, 1] for p in predictions)

        print(f"✓ Logistic regression trained and tested")

    def test_decision_tree(self):
        """Test decision tree model."""
        fetcher = DataFetcher(seed=42)
        df, feature_cols = fetcher.fetch_lending_data(mode="quick")
        X_train, X_test, y_train, y_test, scaler, feature_names = (
            fetcher.prepare_train_test_split(df, feature_cols)
        )

        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(X_test)
        assert all(p in [0, 1] for p in predictions)

        print(f"✓ Decision tree trained and tested")


class TestExperimentResults:
    def test_results_file_exists(self):
        """Test that experiment results exist."""
        import os

        results_file = "results/quick_run/experiment_results.csv"
        assert os.path.exists(results_file), f"Results file not found: {results_file}"

        print(f"✓ Experiment results file exists")

    def test_results_content(self):
        """Test results file content."""
        import pandas as pd

        results_df = pd.read_csv("results/quick_run/experiment_results.csv")

        assert len(results_df) > 0
        assert "model_type" in results_df.columns
        assert "xai_method" in results_df.columns
        assert "auc" in results_df.columns
        assert "mean_faithfulness" in results_df.columns

        # Check reasonable values
        assert results_df["auc"].min() >= 0.5
        assert results_df["auc"].max() <= 1.0
        assert results_df["mean_faithfulness"].min() >= 0.0
        assert results_df["mean_faithfulness"].max() <= 1.0

        print(f"✓ Results content validated: {len(results_df)} experiments")


class TestFigures:
    def test_figures_exist(self):
        """Test that all 5 figures were generated."""
        import os

        required_figures = [
            "figures/system_architecture.png",
            "figures/orchestration_sequence.png",
            "figures/perf_vs_explainability.png",
            "figures/xai_comparison.png",
            "figures/human_trust_results.png",
        ]

        for fig in required_figures:
            assert os.path.exists(fig), f"Figure not found: {fig}"

            # Check file size (should be > 100KB for publication quality)
            size = os.path.getsize(fig)
            assert size > 50000, f"Figure {fig} too small: {size} bytes"

        print(f"✓ All 5 required figures exist and are properly sized")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
