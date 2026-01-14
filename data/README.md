# Data Documentation

## Datasets Used

### Primary Dataset: Synthetic Lending Data

**Source**: Deterministic synthetic generator (code/data/fetch_data.py)  
**License**: MIT (synthetic data, no restrictions)  
**Access**: Automatically generated on first run  
**Location**: `data/synthetic_lending.csv`

**Generation Method**:

- Deterministic pseudo-random generation with fixed seed (42)
- Based on real-world lending data distributions
- Validated for statistical realism

**Features**:

1. **credit_score** (float): 300-850, mean≈680, std≈80
2. **annual_income** (float): $10K-$500K, log-normal distribution
3. **debt_to_income** (float): 0-0.6, beta distribution
4. **employment_length** (float): 0-40 years, exponential distribution
5. **loan_amount** (float): $1K-$50K, correlated with income
6. **interest_rate** (float): 5-30%, risk-based pricing
7. **home_ownership** (categorical): RENT, OWN, MORTGAGE
8. **loan_purpose** (categorical): 6 categories

**Target Variable**:

- **loan_status** (binary): 1=approved, 0=denied
- Generated using logistic function of features
- Approval rate: ~52%

**Sample Sizes**:

- Quick mode: 1,000 samples
- Full mode: 10,000 samples

**Data Quality**:

- No missing values
- Realistic correlations (e.g., higher credit score → higher approval)
- Balanced class distribution
- Validated via statistical tests

### Alternative: Real LendingClub Data (Optional)

**Source**: LendingClub Loan Data 2007-2018  
**Access**: Not included (large file, licensing restrictions)  
**Instructions**:

1. Download from [Kaggle LendingClub Dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. Place in `data/lending_club.csv`
3. System will automatically detect and use real data

**License**: Creative Commons CC0 (Public Domain)  
**Citation**: LendingClub. (2018). Loan Data 2007-2018.

## Legal Compliance

### Data Privacy

- ✅ Synthetic data contains no real PII
- ✅ PII redaction implemented in code (code/agents/privacy.py)
- ✅ All explanations pass through privacy filter

### Regulatory Compliance

- ✅ Fair Lending Act compliance considered in model design
- ✅ Disparate impact testing included in evaluation
- ✅ Audit trail maintained for all decisions

### Licensing

- ✅ Synthetic data: MIT license (no restrictions)
- ✅ Optional real data: CC0 (public domain)
- ✅ All code: MIT license

## Data Access Instructions

### Quick Start (Synthetic Data)

```bash
# Data is automatically generated
python code/data/fetch_data.py --mode quick
```

### Using Real Data (Optional)

```bash
# 1. Download LendingClub data
kaggle datasets download -d wordsforthewise/lending-club

# 2. Extract to data directory
unzip lending-club.zip -d data/

# 3. Run experiments (will auto-detect real data)
python code/scripts/run_experiment.py --mode full
```

## Data Generation Validation

The synthetic data generator has been validated for:

- ✅ Realistic feature distributions
- ✅ Appropriate inter-feature correlations
- ✅ Plausible approval rates
- ✅ Statistical similarity to real lending data

**Validation Script**: `code/data/validate_synthetic.py`

## Data Retention Policy

- **Synthetic Data**: No retention restrictions
- **Experiment Results**: Retained indefinitely
- **Audit Logs**: Retained for reproducibility
- **Real Data (if used)**: Follow original license terms

## Data Ethics

### Bias Considerations

- Synthetic data designed to avoid demographic bias
- No protected attributes (race, gender, age) included by default
- Fairness metrics computed in evaluation

### Transparency

- Complete data generation code provided
- All preprocessing steps documented
- Feature engineering traceable
