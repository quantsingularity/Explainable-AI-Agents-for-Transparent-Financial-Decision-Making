"""
Orchestrator: Coordinates multi-agent workflow for explainable decisions.
"""
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from agents.decision_agent import DecisionAgent
from agents.xai_agent import XAIAgent
from agents.explanation_agent import ExplanationAgent
from agents.evidence_collector import EvidenceCollector
from agents.privacy import PIIRedactor


class Orchestrator:
    """
    Orchestrates the multi-agent workflow:
    1. Evidence Collector gathers context
    2. Decision Agent makes prediction
    3. XAI Agent generates attribution
    4. Explanation Agent creates narrative
    5. Privacy layer redacts PII
    6. Audit log captures everything
    """
    
    def __init__(self, 
                 model_type: str = "logistic",
                 xai_method: str = "shap",
                 explanation_style: str = "regulatory",
                 seed: int = 42):
        """
        Args:
            model_type: Model for DecisionAgent
            xai_method: XAI method for explanations
            explanation_style: Style for narrative generation
            seed: Random seed for reproducibility
        """
        self.seed = seed
        
        # Initialize agents
        self.decision_agent = DecisionAgent(model_type=model_type, seed=seed)
        self.xai_agent = XAIAgent(method=xai_method, seed=seed)
        self.explanation_agent = ExplanationAgent(style=explanation_style, seed=seed)
        self.evidence_collector = EvidenceCollector(seed=seed)
        self.pii_redactor = PIIRedactor()
        
        # Audit log
        self.audit_log = []
        
        logger.info(f"Orchestrator initialized: model={model_type}, xai={xai_method}, style={explanation_style}")
    
    def train(self, X_train, y_train, feature_names):
        """Train the decision model and initialize XAI explainer."""
        logger.info("Training decision agent...")
        self.decision_agent.train(X_train, y_train, feature_names)
        
        logger.info("Initializing XAI agent...")
        self.xai_agent.initialize(
            self.decision_agent.model,
            X_train,
            feature_names
        )
        
        self.feature_names = feature_names
        logger.info("Orchestrator training complete")
    
    def explain_decision(self, X, idx: int, 
                        applicant_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate complete explanation for a single decision.
        
        Args:
            X: Feature array
            idx: Index of instance to explain
            applicant_data: Optional dict with feature names (for readability)
            
        Returns:
            Complete explanation with decision, XAI output, narrative, and metadata
        """
        start_time = time.time()
        
        # Convert numpy to dict if not provided
        if applicant_data is None:
            applicant_data = dict(zip(self.feature_names, X[idx]))
        
        # Redact PII before processing
        applicant_data_clean = self.pii_redactor.redact(applicant_data)
        
        # Step 1: Make decision
        logger.debug(f"Step 1: Making decision for instance {idx}")
        predictions, probabilities = self.decision_agent.predict(X[idx:idx+1])
        
        decision_output = {
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0]),
            "model_type": self.decision_agent.model_type
        }
        
        # Step 2: Collect evidence
        logger.debug(f"Step 2: Collecting evidence")
        model_metadata = {
            "model_type": self.decision_agent.model_type,
            "training_samples": len(X),
            # These would come from actual validation in production
            "auc": 0.72,
            "precision": 0.68,
            "recall": 0.71
        }
        evidence = self.evidence_collector.collect(applicant_data_clean, model_metadata)
        
        # Step 3: Generate XAI explanation
        logger.debug(f"Step 3: Generating XAI explanation with {self.xai_agent.method}")
        xai_output = self.xai_agent.explain(X, idx)
        
        # Compute faithfulness score
        faithfulness = self.xai_agent.get_faithfulness_score(
            self.decision_agent.model, X, idx, top_k=5
        )
        xai_output['faithfulness'] = faithfulness
        evidence['faithfulness_score'] = faithfulness
        
        # Step 4: Generate narrative explanation
        logger.debug(f"Step 4: Generating narrative explanation")
        explanation_output = self.explanation_agent.generate_explanation(
            decision_output,
            xai_output,
            evidence,
            applicant_data_clean
        )
        
        # Compute timings
        elapsed_time = time.time() - start_time
        
        # Assemble complete output
        complete_explanation = {
            "timestamp": datetime.now().isoformat(),
            "instance_id": idx,
            "decision": decision_output,
            "xai_explanation": xai_output,
            "narrative": explanation_output,
            "evidence": evidence,
            "metadata": {
                "model_type": self.decision_agent.model_type,
                "xai_method": self.xai_agent.method,
                "explanation_style": self.explanation_agent.style,
                "elapsed_time_seconds": elapsed_time,
                "seed": self.seed
            }
        }
        
        # Add to audit log
        self._log_explanation(complete_explanation, applicant_data)
        
        logger.info(f"Explanation generated in {elapsed_time:.2f}s")
        
        return complete_explanation
    
    def _log_explanation(self, explanation: Dict[str, Any], 
                        original_data: Dict[str, Any]):
        """Add explanation to audit log."""
        log_entry = {
            "timestamp": explanation['timestamp'],
            "instance_id": explanation['instance_id'],
            "decision": explanation['decision']['prediction'],
            "probability": explanation['decision']['probability'],
            "xai_method": explanation['xai_explanation']['method'],
            "faithfulness": explanation['xai_explanation']['faithfulness'],
            "top_features": explanation['narrative']['top_features'],
            "elapsed_time": explanation['metadata']['elapsed_time_seconds'],
            # Store hash of original data for audit (not the actual PII)
            "data_hash": hash(str(sorted(original_data.items())))
        }
        
        self.audit_log.append(log_entry)
    
    def save_audit_log(self, filepath: str):
        """Save audit log to JSONL file."""
        with open(filepath, 'w') as f:
            for entry in self.audit_log:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Audit log saved to {filepath} ({len(self.audit_log)} entries)")
    
    def batch_explain(self, X, indices: list) -> list:
        """Generate explanations for multiple instances."""
        explanations = []
        
        for idx in indices:
            try:
                explanation = self.explain_decision(X, idx)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain instance {idx}: {e}")
                explanations.append(None)
        
        return explanations
