"""
Decision Agent: Makes financial decisions with optional interpretable constraints.
"""
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import torch
import torch.nn as nn
from loguru import logger


class DecisionAgent:
    """
    Agent responsible for making financial decisions.
    Supports both interpretable models (trees, logistic regression) 
    and black-box models (neural nets, gradient boosting).
    """
    
    def __init__(self, model_type: str = "logistic", seed: int = 42):
        """
        Args:
            model_type: 'logistic', 'tree', 'random_forest', 'gbm', or 'neural_net'
            seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.seed = seed
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        logger.info(f"Initializing DecisionAgent with model_type={model_type}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              feature_names: Optional[list] = None):
        """Train the decision model."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        
        if self.model_type == "logistic":
            self.model = LogisticRegression(random_state=self.seed, max_iter=1000)
            self.model.fit(X_train, y_train)
            
        elif self.model_type == "tree":
            self.model = DecisionTreeClassifier(
                max_depth=5, 
                random_state=self.seed,
                min_samples_split=20
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.seed,
                n_jobs=-1
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == "gbm":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.seed
            )
            self.model.fit(X_train, y_train)
            
        elif self.model_type == "neural_net":
            self.model = SimpleNeuralNet(X_train.shape[1])
            self._train_neural_net(X_train, y_train)
            
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        self.is_trained = True
        logger.info(f"DecisionAgent trained successfully with {self.model_type}")
        
    def _train_neural_net(self, X_train: np.ndarray, y_train: np.ndarray, 
                          epochs: int = 50):
        """Train neural network model."""
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions.
        
        Returns:
            predictions: Binary predictions (0 or 1)
            probabilities: Probability scores
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
            
        if self.model_type == "neural_net":
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                probas = self.model(X_tensor).numpy()
            preds = (probas > 0.5).astype(int).flatten()
            probas = probas.flatten()
        else:
            preds = self.model.predict(X)
            probas = self.model.predict_proba(X)[:, 1]
            
        return preds, probas
    
    def get_intrinsic_explanation(self, X: np.ndarray, idx: int) -> Dict[str, Any]:
        """
        Get intrinsic explanation for interpretable models.
        
        Returns:
            Dictionary with model-specific explanation
        """
        if self.model_type == "logistic":
            coeffs = self.model.coef_[0]
            feature_importance = dict(zip(self.feature_names, coeffs * X[idx]))
            return {
                "type": "linear_coefficients",
                "feature_importance": feature_importance,
                "intercept": float(self.model.intercept_[0])
            }
            
        elif self.model_type == "tree":
            decision_path = self.model.decision_path([X[idx]]).toarray()[0]
            feature_importance = dict(zip(
                self.feature_names, 
                self.model.feature_importances_
            ))
            return {
                "type": "decision_path",
                "feature_importance": feature_importance,
                "path": decision_path
            }
            
        else:
            return {"type": "not_intrinsically_interpretable"}


class SimpleNeuralNet(nn.Module):
    """Simple feed-forward neural network for binary classification."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)
