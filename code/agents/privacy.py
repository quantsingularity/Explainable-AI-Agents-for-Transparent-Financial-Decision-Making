"""
Privacy: PII redaction and safeguards.
"""

import re
from typing import Dict, Any
from loguru import logger


class PIIRedactor:
    """
    Redacts personally identifiable information (PII) from data
    before generating explanations.
    """

    def __init__(self):
        # Patterns for common PII
        self.patterns = {
            "ssn": re.compile(r"\d{3}-\d{2}-\d{4}"),
            "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "phone": re.compile(r"\d{3}[-.]?\d{3}[-.]?\d{4}"),
            "address": re.compile(r"\d+\s+[\w\s]+,\s*\w+,\s*[A-Z]{2}\s+\d{5}"),
        }

        # PII fields that should be redacted
        self.pii_fields = {
            "name",
            "ssn",
            "social_security",
            "email",
            "phone",
            "address",
            "zip_code",
            "account_number",
        }

        logger.info("PIIRedactor initialized")

    def redact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact PII from data dictionary.

        Args:
            data: Dictionary potentially containing PII

        Returns:
            Dictionary with PII redacted
        """
        redacted = data.copy()

        for key, value in redacted.items():
            # Check if field name suggests PII
            if any(pii_term in key.lower() for pii_term in self.pii_fields):
                redacted[key] = "[REDACTED]"
                logger.debug(f"Redacted PII field: {key}")
                continue

            # Check if value matches PII patterns
            if isinstance(value, str):
                for pii_type, pattern in self.patterns.items():
                    if pattern.search(value):
                        redacted[key] = f"[REDACTED-{pii_type.upper()}]"
                        logger.debug(f"Redacted {pii_type} in field: {key}")
                        break

        return redacted

    def redact_text(self, text: str) -> str:
        """Redact PII from free-form text."""
        redacted_text = text

        for pii_type, pattern in self.patterns.items():
            redacted_text = pattern.sub(f"[REDACTED-{pii_type.upper()}]", redacted_text)

        return redacted_text


class ExplanationSanityChecker:
    """
    Checks explanations for potential issues:
    - Hallucinated claims
    - Inconsistent attributions
    - Unbacked citations
    """

    def __init__(self):
        logger.info("ExplanationSanityChecker initialized")

    def check(self, explanation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run sanity checks on explanation.

        Returns:
            Dictionary with check results and any warnings
        """
        warnings = []

        # Check 1: Attribution magnitudes are reasonable
        attributions = explanation["xai_explanation"]["attributions"]
        max_attr = max(abs(v) for v in attributions.values())

        if max_attr > 10.0:  # Arbitrary threshold
            warnings.append(
                {
                    "type": "high_attribution",
                    "message": f"Unusually high attribution magnitude: {max_attr:.2f}",
                    "severity": "warning",
                }
            )

        # Check 2: Narrative mentions top features
        narrative = explanation["narrative"]["narrative"].lower()
        top_features = explanation["narrative"]["top_features"]

        for feature in top_features[:3]:  # Check top 3
            feature_readable = feature.replace("_", " ").lower()
            if feature_readable not in narrative:
                warnings.append(
                    {
                        "type": "missing_feature",
                        "message": f"Top feature '{feature}' not mentioned in narrative",
                        "severity": "warning",
                    }
                )

        # Check 3: Citations are valid
        citations = explanation["narrative"].get("citations", [])
        for citation in citations:
            if "policy_id" not in citation:
                warnings.append(
                    {
                        "type": "invalid_citation",
                        "message": "Citation missing policy_id",
                        "severity": "error",
                    }
                )

        # Check 4: Confidence is reasonable
        confidence = explanation["decision"]["probability"]
        if confidence > 0.99 or confidence < 0.01:
            warnings.append(
                {
                    "type": "extreme_confidence",
                    "message": f"Extreme confidence level: {confidence:.4f}",
                    "severity": "warning",
                }
            )

        return {
            "passed": len([w for w in warnings if w["severity"] == "error"]) == 0,
            "warnings": warnings,
        }


class RateLimiter:
    """
    Rate limiter for explanation generation (prevents abuse).
    """

    def __init__(self, max_requests_per_minute: int = 100):
        self.max_requests = max_requests_per_minute
        self.request_log = []

        logger.info(f"RateLimiter initialized: {max_requests_per_minute} req/min")

    def check_rate_limit(self, user_id: str) -> bool:
        """
        Check if user has exceeded rate limit.

        Returns:
            True if within limit, False if exceeded
        """
        import time

        current_time = time.time()
        minute_ago = current_time - 60

        # Clean old requests
        self.request_log = [
            (uid, ts) for uid, ts in self.request_log if ts > minute_ago
        ]

        # Count requests from this user
        user_requests = sum(1 for uid, ts in self.request_log if uid == user_id)

        if user_requests >= self.max_requests:
            logger.warning(f"Rate limit exceeded for user {user_id}")
            return False

        # Log this request
        self.request_log.append((user_id, current_time))
        return True
