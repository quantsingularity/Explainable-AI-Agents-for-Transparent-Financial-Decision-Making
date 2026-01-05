"""
Comprehensive evaluation metrics for XAI methods.
"""
import numpy as np
from typing import Dict, Any, List, Tuple
from scipy import stats
from loguru import logger


class XAIEvaluator:
    """
    Comprehensive evaluation of XAI methods:
    - Faithfulness
    - Fidelity
    - Completeness
    - Stability
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def compute_faithfulness(self, model: Any, X: np.ndarray, 
                            attributions: Dict[str, float],
                            feature_names: List[str],
                            top_k: int = 5) -> float:
        """
        Faithfulness: How much does prediction change when important features masked?
        Higher = more faithful
        """
        # Get original prediction
        if hasattr(model, 'predict_proba'):
            original_pred = model.predict_proba(X)[0, 1]
        else:
            import torch
            model.eval()
            with torch.no_grad():
                original_pred = model(torch.FloatTensor(X)).item()
        
        # Get top-k features
        sorted_attrs = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = [f[0] for f in sorted_attrs[:top_k]]
        top_indices = [feature_names.index(f) for f in top_features]
        
        # Mask top features (set to 0)
        X_masked = X.copy()
        X_masked[0, top_indices] = 0.0
        
        # Get masked prediction
        if hasattr(model, 'predict_proba'):
            masked_pred = model.predict_proba(X_masked)[0, 1]
        else:
            model.eval()
            with torch.no_grad():
                masked_pred = model(torch.FloatTensor(X_masked)).item()
        
        # Faithfulness = absolute prediction change
        faithfulness = abs(original_pred - masked_pred)
        
        return faithfulness
    
    def compute_fidelity(self, model: Any, X: np.ndarray,
                        attributions: Dict[str, float],
                        feature_names: List[str],
                        n_steps: int = 10) -> float:
        """
        Fidelity: Correlation between attribution magnitude and prediction change
        when features are progressively removed.
        """
        if hasattr(model, 'predict_proba'):
            original_pred = model.predict_proba(X)[0, 1]
        else:
            import torch
            model.eval()
            with torch.no_grad():
                original_pred = model(torch.FloatTensor(X)).item()
        
        # Sort features by attribution magnitude
        sorted_attrs = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        pred_changes = []
        attr_magnitudes = []
        
        for i in range(1, min(n_steps, len(sorted_attrs)) + 1):
            # Remove top i features
            features_to_mask = [f[0] for f in sorted_attrs[:i]]
            indices = [feature_names.index(f) for f in features_to_mask]
            
            X_masked = X.copy()
            X_masked[0, indices] = 0.0
            
            if hasattr(model, 'predict_proba'):
                masked_pred = model.predict_proba(X_masked)[0, 1]
            else:
                import torch
                model.eval()
                with torch.no_grad():
                    masked_pred = model(torch.FloatTensor(X_masked)).item()
            
            pred_changes.append(abs(original_pred - masked_pred))
            attr_magnitudes.append(sum(abs(sorted_attrs[j][1]) for j in range(i)))
        
        # Fidelity = correlation between cumulative attributions and prediction changes
        if len(pred_changes) > 1:
            correlation, _ = stats.pearsonr(attr_magnitudes, pred_changes)
            return max(0, correlation)  # Clip to [0, 1]
        else:
            return 0.0
    
    def compute_completeness(self, attributions: Dict[str, float],
                            threshold: float = 0.01) -> float:
        """
        Completeness: Fraction of features with non-negligible attribution.
        Higher = more complete explanation
        """
        total_features = len(attributions)
        significant_features = sum(1 for v in attributions.values() 
                                  if abs(v) > threshold)
        
        return significant_features / total_features if total_features > 0 else 0.0
    
    def compute_stability(self, model: Any, X: np.ndarray,
                         explainer_fn: callable,
                         n_perturbations: int = 10,
                         noise_level: float = 0.1) -> float:
        """
        Stability: How consistent are explanations for slightly perturbed inputs?
        Higher = more stable
        """
        # Get original explanation
        original_exp = explainer_fn(X)
        original_attrs = list(original_exp['attributions'].values())
        
        # Generate perturbed versions
        perturbed_attrs = []
        for _ in range(n_perturbations):
            noise = np.random.normal(0, noise_level, X.shape)
            X_perturbed = X + noise
            
            try:
                perturbed_exp = explainer_fn(X_perturbed)
                perturbed_attrs.append(list(perturbed_exp['attributions'].values()))
            except:
                continue
        
        if not perturbed_attrs:
            return 0.0
        
        # Compute average correlation with original
        correlations = []
        for p_attrs in perturbed_attrs:
            if len(p_attrs) == len(original_attrs):
                corr, _ = stats.pearsonr(original_attrs, p_attrs)
                correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def evaluate_comprehensive(self, model: Any, X: np.ndarray,
                              attributions: Dict[str, float],
                              feature_names: List[str],
                              explainer_fn: callable = None) -> Dict[str, float]:
        """
        Run all evaluation metrics.
        
        Returns:
            Dictionary with all metrics
        """
        results = {
            'faithfulness': self.compute_faithfulness(model, X, attributions, feature_names),
            'fidelity': self.compute_fidelity(model, X, attributions, feature_names),
            'completeness': self.compute_completeness(attributions)
        }
        
        # Stability requires explainer function
        if explainer_fn is not None:
            results['stability'] = self.compute_stability(model, X, explainer_fn, n_perturbations=5)
        else:
            results['stability'] = None
        
        return results


class StatisticalTester:
    """
    Statistical significance testing for XAI comparisons.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
    
    def bootstrap_test(self, sample1: np.ndarray, sample2: np.ndarray,
                      n_iterations: int = 1000,
                      confidence: float = 0.95) -> Dict[str, Any]:
        """
        Bootstrap test for difference in means with confidence intervals.
        """
        observed_diff = np.mean(sample1) - np.mean(sample2)
        
        # Bootstrap resampling
        bootstrap_diffs = []
        combined = np.concatenate([sample1, sample2])
        n1, n2 = len(sample1), len(sample2)
        
        for _ in range(n_iterations):
            # Resample with replacement
            indices = np.random.choice(len(combined), len(combined), replace=True)
            resampled = combined[indices]
            
            boot_sample1 = resampled[:n1]
            boot_sample2 = resampled[n1:]
            
            boot_diff = np.mean(boot_sample1) - np.mean(boot_sample2)
            bootstrap_diffs.append(boot_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Compute confidence interval
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_diffs, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_diffs, (1 - alpha/2) * 100)
        
        # P-value: proportion of bootstrap samples with opposite sign
        p_value = np.mean(np.sign(bootstrap_diffs) != np.sign(observed_diff))
        
        return {
            'observed_difference': observed_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'significant': p_value < (1 - confidence)
        }
    
    def permutation_test(self, sample1: np.ndarray, sample2: np.ndarray,
                        n_permutations: int = 1000) -> Dict[str, Any]:
        """
        Permutation test for difference in means.
        """
        observed_diff = np.mean(sample1) - np.mean(sample2)
        
        # Combine samples
        combined = np.concatenate([sample1, sample2])
        n1 = len(sample1)
        
        # Permutation test
        perm_diffs = []
        for _ in range(n_permutations):
            # Randomly shuffle
            np.random.shuffle(combined)
            perm_sample1 = combined[:n1]
            perm_sample2 = combined[n1:]
            
            perm_diff = np.mean(perm_sample1) - np.mean(perm_sample2)
            perm_diffs.append(perm_diff)
        
        perm_diffs = np.array(perm_diffs)
        
        # P-value: proportion of permutations with difference >= observed
        p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_difference': observed_diff,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
