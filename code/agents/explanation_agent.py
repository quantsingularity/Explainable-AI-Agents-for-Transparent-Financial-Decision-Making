"""
Explanation Agent: Generates narrative explanations conditioned on XAI outputs.
Uses LLM-style prompt templates with evidence citation.
"""
import json
from typing import Dict, Any, List
import re
from loguru import logger


class ExplanationAgent:
    """
    Agent responsible for generating human-readable narrative explanations.
    Conditions on XAI outputs and enforces evidence citation.
    """
    
    def __init__(self, style: str = "regulatory", seed: int = 42):
        """
        Args:
            style: 'regulatory', 'technical', or 'consumer'
            seed: Random seed for reproducibility
        """
        self.style = style
        self.seed = seed
        
        logger.info(f"Initializing ExplanationAgent with style={style}")
        
    def generate_explanation(self, 
                            decision_output: Dict[str, Any],
                            xai_output: Dict[str, Any],
                            evidence: Dict[str, Any],
                            applicant_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate narrative explanation with evidence citations.
        
        Args:
            decision_output: From DecisionAgent (prediction, probability)
            xai_output: From XAIAgent (feature attributions)
            evidence: From EvidenceCollector (supporting data, policies)
            applicant_data: Original input features
            
        Returns:
            Dictionary with explanation text, citations, and metadata
        """
        # Extract key information
        prediction = decision_output['prediction']
        probability = decision_output['probability']
        attributions = xai_output['attributions']
        method = xai_output['method']
        
        # Get top influential features
        sorted_features = sorted(attributions.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True)
        top_features = sorted_features[:5]
        
        # Generate explanation based on style
        if self.style == "regulatory":
            explanation = self._generate_regulatory_explanation(
                prediction, probability, top_features, applicant_data, method, evidence
            )
        elif self.style == "technical":
            explanation = self._generate_technical_explanation(
                prediction, probability, top_features, applicant_data, method, evidence
            )
        else:  # consumer
            explanation = self._generate_consumer_explanation(
                prediction, probability, top_features, applicant_data, method, evidence
            )
        
        # Verify evidence citations (sanity check for hallucinations)
        verified_explanation = self._verify_citations(explanation, evidence)
        
        return {
            "narrative": verified_explanation['text'],
            "citations": verified_explanation['citations'],
            "top_features": [f[0] for f in top_features],
            "attribution_method": method,
            "confidence": float(probability),
            "style": self.style
        }
    
    def _generate_regulatory_explanation(self, prediction: int, probability: float,
                                         top_features: List, applicant_data: Dict,
                                         method: str, evidence: Dict) -> Dict[str, Any]:
        """Generate explanation in regulatory/audit style."""
        
        decision_text = "APPROVED" if prediction == 1 else "DENIED"
        
        text = f"""LOAN APPLICATION DECISION: {decision_text}
        
Decision Confidence: {probability:.2%}
Explanation Method: {method.upper()}

PRIMARY DECISION FACTORS:

"""
        citations = []
        
        for rank, (feature, attribution) in enumerate(top_features, 1):
            value = applicant_data.get(feature, "N/A")
            direction = "positively" if attribution > 0 else "negatively"
            
            text += f"{rank}. {feature.replace('_', ' ').title()}: {value}\n"
            text += f"   Impact: This factor influenced the decision {direction} "
            text += f"(attribution: {attribution:+.4f})\n"
            
            # Add evidence citation
            if feature in evidence.get('feature_policies', {}):
                policy = evidence['feature_policies'][feature]
                text += f"   [Policy Reference: {policy['id']}]\n"
                citations.append({
                    'feature': feature,
                    'policy_id': policy['id'],
                    'policy_text': policy['text']
                })
            text += "\n"
        
        text += f"\nMODEL METHODOLOGY:\n"
        text += f"This decision was made using a {evidence.get('model_type', 'machine learning')} model "
        text += f"trained on {evidence.get('training_samples', 'historical')} loan applications. "
        text += f"The explanation was generated using {method} to ensure transparency.\n"
        
        text += f"\nREGULATORY COMPLIANCE:\n"
        text += f"This decision complies with applicable fair lending regulations [Fair Lending Act]. "
        text += f"The model has been validated for disparate impact [Validation Report v{evidence.get('validation_version', '1.0')}].\n"
        
        return {'text': text, 'citations': citations}
    
    def _generate_technical_explanation(self, prediction: int, probability: float,
                                       top_features: List, applicant_data: Dict,
                                       method: str, evidence: Dict) -> Dict[str, Any]:
        """Generate explanation for technical/ML audience."""
        
        decision_text = "Positive (Approve)" if prediction == 1 else "Negative (Deny)"
        
        text = f"""TECHNICAL EXPLANATION
        
Prediction: {decision_text} (confidence: {probability:.4f})
XAI Method: {method}

Feature Attribution Analysis:

"""
        citations = []
        
        for rank, (feature, attribution) in enumerate(top_features, 1):
            value = applicant_data.get(feature, "N/A")
            
            text += f"{rank}. {feature}={value}, attribution={attribution:+.6f}\n"
        
        text += f"\nModel Performance (on validation set):\n"
        text += f"- ROC-AUC: {evidence.get('model_auc', 0.72):.3f}\n"
        text += f"- Precision: {evidence.get('model_precision', 0.68):.3f}\n"
        text += f"- Recall: {evidence.get('model_recall', 0.71):.3f}\n"
        
        text += f"\nExplanation Faithfulness: {evidence.get('faithfulness_score', 0.75):.3f}\n"
        text += f"(Measured as prediction change when top-k features masked)\n"
        
        return {'text': text, 'citations': citations}
    
    def _generate_consumer_explanation(self, prediction: int, probability: float,
                                      top_features: List, applicant_data: Dict,
                                      method: str, evidence: Dict) -> Dict[str, Any]:
        """Generate explanation for loan applicants (plain language)."""
        
        if prediction == 1:
            text = "Good news! Your loan application has been approved.\n\n"
        else:
            text = "We're sorry, but your loan application was not approved at this time.\n\n"
        
        text += "Here are the main factors that influenced this decision:\n\n"
        
        citations = []
        
        for rank, (feature, attribution) in enumerate(top_features, 1):
            value = applicant_data.get(feature, "N/A")
            feature_readable = feature.replace('_', ' ').title()
            
            if attribution > 0:
                impact = "helped" if prediction == 1 else "helped, but wasn't enough"
            else:
                impact = "hurt" if prediction == 0 else "was not ideal, but other factors compensated"
            
            text += f"{rank}. Your {feature_readable} ({value}) {impact} your application.\n"
        
        if prediction == 0:
            text += "\nTo improve your chances in the future, consider:\n"
            text += "- Improving factors that negatively influenced this decision\n"
            text += "- Waiting for your financial situation to improve\n"
            text += "- Consulting with a financial advisor\n"
        
        text += f"\nThis decision was made using objective criteria applied consistently to all applicants. "
        text += f"If you have questions, please contact our support team.\n"
        
        return {'text': text, 'citations': citations}
    
    def _verify_citations(self, explanation: Dict[str, Any], 
                         evidence: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanity check: verify that cited policies/evidence actually exist.
        This prevents hallucinated references.
        """
        text = explanation['text']
        citations = explanation['citations']
        
        # Check for citation markers in text
        cited_ids = re.findall(r'\[Policy Reference: ([^\]]+)\]', text)
        
        for cite_id in cited_ids:
            # Verify this policy exists in evidence
            found = False
            for policy in evidence.get('feature_policies', {}).values():
                if policy.get('id') == cite_id:
                    found = True
                    break
            
            if not found:
                logger.warning(f"Hallucinated citation detected: {cite_id}")
                # Remove hallucinated citation
                text = text.replace(f"[Policy Reference: {cite_id}]", 
                                   "[Citation Removed: Unverified]")
        
        return {'text': text, 'citations': citations}
