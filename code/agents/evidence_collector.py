"""
Evidence Collector: Gathers supporting data, policies, and context for explanations.
"""
from typing import Dict, Any, List
import numpy as np
from loguru import logger


class EvidenceCollector:
    """
    Agent responsible for collecting evidence to support explanations.
    Gathers relevant policies, historical data, and regulatory references.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.policy_db = self._init_policy_database()
        
        logger.info("Initializing EvidenceCollector")
        
    def _init_policy_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize database of lending policies and regulatory references."""
        return {
            "credit_score": {
                "id": "POL-001",
                "text": "Credit score is a primary indicator of creditworthiness. Scores above 700 indicate lower risk.",
                "regulatory_ref": "Fair Credit Reporting Act",
                "threshold": 700
            },
            "annual_income": {
                "id": "POL-002",
                "text": "Annual income must be sufficient to support debt repayment. Minimum income requirements apply.",
                "regulatory_ref": "Ability-to-Repay Rule",
                "threshold": 30000
            },
            "debt_to_income": {
                "id": "POL-003",
                "text": "Debt-to-income ratio should not exceed 43% for qualified mortgages.",
                "regulatory_ref": "Qualified Mortgage Rule",
                "threshold": 0.43
            },
            "employment_length": {
                "id": "POL-004",
                "text": "Employment stability is considered. Longer employment history indicates stability.",
                "regulatory_ref": "Underwriting Guidelines",
                "threshold": 2
            },
            "loan_amount": {
                "id": "POL-005",
                "text": "Loan amount must be within authorized limits based on applicant profile.",
                "regulatory_ref": "Lending Limits Policy",
                "threshold": None
            }
        }
    
    def collect(self, applicant_data: Dict[str, Any], 
               model_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect evidence relevant to the decision.
        
        Args:
            applicant_data: Input features for the application
            model_metadata: Metadata about the model (performance, training data, etc.)
            
        Returns:
            Dictionary of evidence including policies, context, and references
        """
        evidence = {
            "feature_policies": {},
            "model_type": model_metadata.get("model_type", "unknown"),
            "training_samples": model_metadata.get("training_samples", "N/A"),
            "validation_version": "1.0",
            "model_auc": model_metadata.get("auc", 0.72),
            "model_precision": model_metadata.get("precision", 0.68),
            "model_recall": model_metadata.get("recall", 0.71),
            "faithfulness_score": model_metadata.get("faithfulness", 0.75),
            "regulatory_compliance": [
                "Fair Lending Act",
                "Equal Credit Opportunity Act",
                "Fair Credit Reporting Act"
            ]
        }
        
        # Collect policies relevant to this application's features
        for feature_name, feature_value in applicant_data.items():
            if feature_name in self.policy_db:
                policy = self.policy_db[feature_name].copy()
                
                # Add compliance check
                if policy['threshold'] is not None:
                    if feature_name == "debt_to_income":
                        policy['compliant'] = feature_value <= policy['threshold']
                    else:
                        policy['compliant'] = feature_value >= policy['threshold']
                else:
                    policy['compliant'] = None
                
                evidence['feature_policies'][feature_name] = policy
        
        # Collect similar historical cases
        evidence['similar_cases'] = self._get_similar_cases(applicant_data)
        
        logger.debug(f"Collected evidence for {len(evidence['feature_policies'])} features")
        
        return evidence
    
    def _get_similar_cases(self, applicant_data: Dict[str, Any], 
                          n_cases: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar historical cases (simulated for this implementation).
        In production, this would query a historical database.
        """
        # Simulated similar cases
        np.random.seed(self.seed)
        
        similar_cases = []
        for i in range(n_cases):
            case = {
                "case_id": f"HIST-{i+1:04d}",
                "decision": "approved" if np.random.rand() > 0.5 else "denied",
                "similarity_score": np.random.uniform(0.7, 0.95),
                "date": "2024-Q2"
            }
            similar_cases.append(case)
        
        return similar_cases
