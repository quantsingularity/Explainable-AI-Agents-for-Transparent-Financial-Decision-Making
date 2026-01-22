"""
Baseline Models for Comparison
Implements Logistic Regression, Random Forest, and Neural Network baselines
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)
import time
from typing import Dict, Any, Tuple
import joblib


class BaselineModel:
    """Base class for baseline models"""

    def __init__(self, model_type: str = "logistic", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.training_time = 0
        self.inference_time = 0

    def _init_model(self):
        """Initialize the specific model"""
        if self.model_type == "logistic":
            self.model = LogisticRegression(
                random_state=self.random_state, max_iter=1000, solver="lbfgs"
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "neural_network":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation="relu",
                random_state=self.random_state,
                max_iter=500,
                early_stopping=True,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train the model and return training metrics"""
        self._init_model()

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        self.training_time = time.time() - start_time

        # Get training predictions
        y_pred_proba = self.model.predict_proba(X_train_scaled)[:, 1]
        y_pred = self.model.predict(X_train_scaled)

        metrics = {
            "accuracy": accuracy_score(y_train, y_pred),
            "roc_auc": roc_auc_score(y_train, y_pred_proba),
            "precision": precision_score(y_train, y_pred, zero_division=0),
            "recall": recall_score(y_train, y_pred, zero_division=0),
            "f1": f1_score(y_train, y_pred, zero_division=0),
            "log_loss": log_loss(y_train, y_pred_proba),
            "training_time": self.training_time,
        }

        return metrics

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model on test data"""
        X_test_scaled = self.scaler.transform(X_test)

        # Measure inference time
        start_time = time.time()
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = self.model.predict(X_test_scaled)
        self.inference_time = (
            (time.time() - start_time) / len(X_test) * 1000
        )  # ms per sample

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "log_loss": log_loss(y_test, y_pred_proba),
            "inference_time_ms": self.inference_time,
        }

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance (model-specific)"""
        if self.model_type == "logistic":
            return np.abs(self.model.coef_[0])
        elif self.model_type == "random_forest":
            return self.model.feature_importances_
        elif self.model_type == "neural_network":
            # For neural networks, use first layer weights
            return np.abs(self.model.coefs_[0]).mean(axis=1)
        return np.array([])

    def save(self, path: str):
        """Save model to disk"""
        joblib.dump(
            {"model": self.model, "scaler": self.scaler, "model_type": self.model_type},
            path,
        )

    def load(self, path: str):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.model_type = data["model_type"]


class BaselineComparator:
    """Compare multiple baseline models"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def add_model(self, name: str, model_type: str):
        """Add a baseline model to compare"""
        self.models[name] = BaselineModel(model_type, self.random_state)

    def train_all(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train all baseline models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            train_metrics = model.train(X_train, y_train)
            self.results[name] = {"train": train_metrics}

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluate all baseline models"""
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            test_metrics = model.evaluate(X_test, y_test)
            self.results[name]["test"] = test_metrics

    def get_comparison_table(self) -> Dict[str, Dict[str, float]]:
        """Get a comparison table of all models"""
        comparison = {}
        for name, metrics in self.results.items():
            comparison[name] = metrics.get("test", {})
        return comparison

    def get_best_model(self, metric: str = "roc_auc") -> Tuple[str, BaselineModel]:
        """Get the best performing model based on a metric"""
        best_score = -float("inf")
        best_name = None

        for name, metrics in self.results.items():
            score = metrics.get("test", {}).get(metric, -float("inf"))
            if score > best_score:
                best_score = score
                best_name = name

        return best_name, self.models[best_name]
