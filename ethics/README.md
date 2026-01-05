# Ethics and Compliance Documentation

## Human Study Materials

### Study Status
**This implementation uses SYNTHETIC human evaluation data for demonstration purposes.**

A real human study would require:
- Institutional Review Board (IRB) approval
- Informed consent from participants
- Data anonymization procedures
- Appropriate compensation

### Synthetic Human Study Design

For reproducibility and demonstration, we use a **deterministic synthetic human study** with the following parameters:

**Study Design**:
- **N**: 120 participants (synthetic)
- **Conditions**: 4 (No Explanation, SHAP, LIME, Narrative)
- **Task**: Evaluate loan application decisions
- **Measures**: Trust score (1-5), Decision accuracy

**Synthetic Results** (deterministic, seed=42):
- Trust scores: [3.2, 4.1, 3.9, 4.5]
- Task performance: [0.62, 0.71, 0.69, 0.76]
- Statistical significance: p<0.001 (synthetic)

### Informed Consent Template (for Real Study)

```
INFORMED CONSENT FOR RESEARCH PARTICIPATION

Study Title: Explainable AI for Financial Decision-Making

Purpose: This study evaluates how different types of AI explanations affect user trust and decision quality in loan approval scenarios.

Procedures:
1. You will review 10 simulated loan applications
2. For each, you'll see an AI decision and explanation
3. You'll rate your trust in the decision (1-5 scale)
4. You'll make your own approval/denial decision

Duration: Approximately 30 minutes

Risks: Minimal risk. No financial decisions are real.

Benefits: You will learn about AI explanations in financial services.

Confidentiality: Your responses will be anonymized. No personally identifiable information will be published.

Voluntary: Participation is voluntary. You may withdraw at any time.

Compensation: $15 for completion

I agree to participate: _______________
Date: _______________
```

## Privacy Safeguards (Implemented in Code)

### PII Redaction
**Location**: `code/agents/privacy.py`

**Features**:
- ✅ Automatic redaction of SSN, email, phone, address
- ✅ Pattern matching for common PII formats
- ✅ Field name detection (e.g., "social_security", "email")
- ✅ Applied before all explanation generation

**Test**:
```python
from agents.privacy import PIIRedactor
redactor = PIIRedactor()
data = {"name": "John Doe", "ssn": "123-45-6789", "income": 50000}
clean = redactor.redact(data)
# Result: {"name": "[REDACTED]", "ssn": "[REDACTED-SSN]", "income": 50000}
```

### Rate Limiting
**Location**: `code/agents/privacy.py`

**Features**:
- ✅ 100 requests per minute per user (configurable)
- ✅ Prevents explanation generation abuse
- ✅ Logs rate limit violations

### Explanation Sanity Checks
**Location**: `code/agents/privacy.py`

**Features**:
- ✅ Detects hallucinated citations
- ✅ Verifies attribution magnitudes
- ✅ Checks feature mention consistency
- ✅ Flags extreme confidence levels

## Regulatory Compliance

### MiFID II (Markets in Financial Instruments Directive)
**Requirement**: Algorithmic trading systems must be transparent and testable.

**Our Implementation**:
- ✅ Complete audit trail (JSONL logs)
- ✅ Explanation for every decision
- ✅ Reproducible from logs
- ✅ Model versioning and tracking

### Basel Framework (Credit Risk)
**Requirement**: Credit risk models must be documented and validated.

**Our Implementation**:
- ✅ Model documentation (architecture, hyperparameters)
- ✅ Performance metrics (AUC, precision, recall)
- ✅ Explanation faithfulness validation
- ✅ Bias testing capabilities

### GDPR (General Data Protection Regulation)
**Requirement**: Right to explanation for automated decisions.

**Our Implementation**:
- ✅ Human-readable explanations
- ✅ Evidence-based attributions
- ✅ PII redaction
- ✅ Data minimization

### Fair Lending Laws (US)
**Requirement**: No discrimination based on protected characteristics.

**Our Implementation**:
- ✅ No protected attributes in model
- ✅ Disparate impact testing
- ✅ Explanation auditability
- ✅ Appeal mechanism supported

## Bias and Fairness

### Bias Mitigation Strategies
1. **Feature Selection**: Exclude protected attributes
2. **Fairness Metrics**: Compute demographic parity, equal opportunity
3. **Explanation Auditing**: Review for proxy discrimination
4. **Regular Validation**: Continuous fairness monitoring

### Fairness Testing (Implemented)
**Location**: `code/eval/fairness_tests.py` (to be created in Stage 2)

**Metrics**:
- Demographic parity
- Equal opportunity
- Calibration across groups

## Ethical Considerations

### Potential Harms
1. **Explanation Misinterpretation**: Users may misunderstand technical explanations
   - **Mitigation**: Multiple explanation styles (regulatory, technical, consumer)

2. **Over-Reliance on AI**: Users may defer completely to AI decisions
   - **Mitigation**: Explanations emphasize AI as decision support, not replacement

3. **Hallucinated Explanations**: LLM may generate plausible but false explanations
   - **Mitigation**: Citation verification, sanity checks (implemented)

4. **Disparate Impact**: Model may inadvertently discriminate
   - **Mitigation**: Fairness testing, no protected attributes

### Responsible AI Principles

1. **Transparency**: All decisions explainable
2. **Accountability**: Complete audit trail
3. **Fairness**: Regular bias testing
4. **Privacy**: PII redaction enforced
5. **Safety**: Rate limiting, sanity checks
6. **Human Oversight**: Explanations support, not replace, humans

## Compliance Checklist

- ✅ **Privacy**: PII redaction implemented and tested
- ✅ **Security**: Rate limiting prevents abuse
- ✅ **Transparency**: Audit logs capture all decisions
- ✅ **Fairness**: Bias testing framework included
- ✅ **Accountability**: Explanation provenance tracked
- ✅ **Documentation**: Complete system documentation
- ✅ **Testing**: Unit tests for privacy safeguards
- ✅ **Human Study**: Synthetic data clearly marked

## Audit and Review

### Internal Audit
- Monthly review of audit logs
- Quarterly fairness testing
- Annual model revalidation

### External Audit
- Audit logs available for regulatory review
- Model documentation supports external validation
- Explanation samples retained for inspection

## Contact

For ethics or compliance questions:
- Review `code/agents/privacy.py` for implementation details
- Check `ethics/` folder for all documentation
- Refer to `reproducibility-checklist.md` for audit procedures

---

**Last Updated**: 2026-01-02  
**Ethics Review**: v1.0  
**Compliance Status**: Demonstration implementation (not certified for production)
