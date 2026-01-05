"""
Data fetcher and processor for lending datasets.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from loguru import logger
from typing import Tuple


class DataFetcher:
    """
    Fetches and processes financial datasets.
    Falls back to synthetic data if real data unavailable.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("DataFetcher initialized")
    
    def fetch_lending_data(self, mode: str = "quick") -> Tuple[pd.DataFrame, list]:
        """
        Fetch lending dataset (LendingClub-style data).
        Falls back to synthetic if real data unavailable.
        
        Args:
            mode: 'quick' (1000 samples) or 'full' (10000+ samples)
            
        Returns:
            DataFrame and list of feature names
        """
        # Try to load real data first
        real_data_path = os.path.join(self.data_dir, "lending_club.csv")
        
        if os.path.exists(real_data_path):
            logger.info(f"Loading real lending data from {real_data_path}")
            df = pd.read_csv(real_data_path)
        else:
            logger.warning("Real lending data not found, generating synthetic data")
            df = self._generate_synthetic_lending_data(mode)
            
            # Save synthetic data
            synthetic_path = os.path.join(self.data_dir, "synthetic_lending.csv")
            df.to_csv(synthetic_path, index=False)
            logger.info(f"Synthetic data saved to {synthetic_path}")
        
        # Process data
        df_processed, feature_names = self._process_lending_data(df, mode)
        
        return df_processed, feature_names
    
    def _generate_synthetic_lending_data(self, mode: str) -> pd.DataFrame:
        """
        Generate deterministic synthetic lending data.
        Designed to be realistic with known correlations.
        """
        np.random.seed(self.seed)
        
        n_samples = 1000 if mode == "quick" else 10000
        
        logger.info(f"Generating {n_samples} synthetic loan applications...")
        
        # Generate correlated features
        credit_score = np.random.normal(680, 80, n_samples).clip(300, 850)
        annual_income = np.random.lognormal(10.5, 0.8, n_samples).clip(10000, 500000)
        
        # Debt-to-income ratio (correlated with income)
        debt_to_income = np.random.beta(2, 5, n_samples) * 0.6
        
        # Employment length (years)
        employment_length = np.random.exponential(5, n_samples).clip(0, 40)
        
        # Loan amount (correlated with income)
        loan_amount = (annual_income * np.random.uniform(0.2, 0.5, n_samples)).clip(1000, 50000)
        
        # Interest rate (risk-based)
        base_rate = 10.0
        rate_adjustment = (700 - credit_score) / 100 + debt_to_income * 5
        interest_rate = (base_rate + rate_adjustment).clip(5, 30)
        
        # Home ownership
        home_ownership = np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples, 
                                         p=[0.4, 0.2, 0.4])
        
        # Loan purpose
        loan_purpose = np.random.choice([
            'debt_consolidation', 'credit_card', 'home_improvement',
            'major_purchase', 'small_business', 'medical'
        ], n_samples, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
        
        # Generate target (loan approved) based on realistic rules
        # Higher credit score, lower DTI, higher income -> more likely approval
        approval_score = (
            (credit_score - 600) / 250 +  # 0 to 1 for score 600-850
            (1 - debt_to_income / 0.6) +   # 0 to 1 for DTI 0-0.6
            np.log(annual_income) / 12 +   # 0 to 1 for income 10k-500k
            employment_length / 40          # 0 to 1 for 0-40 years
        )
        
        # Add noise and threshold
        approval_prob = 1 / (1 + np.exp(-approval_score + 2))  # Sigmoid
        loan_status = (approval_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'credit_score': credit_score,
            'annual_income': annual_income,
            'debt_to_income': debt_to_income,
            'employment_length': employment_length,
            'loan_amount': loan_amount,
            'interest_rate': interest_rate,
            'home_ownership': home_ownership,
            'loan_purpose': loan_purpose,
            'loan_status': loan_status  # 1 = approved, 0 = denied
        })
        
        logger.info(f"Synthetic data generated: {len(df)} samples, "
                   f"{df['loan_status'].mean():.1%} approval rate")
        
        return df
    
    def _process_lending_data(self, df: pd.DataFrame, 
                             mode: str) -> Tuple[pd.DataFrame, list]:
        """Process and clean lending data."""
        
        # Select subset for quick mode
        if mode == "quick":
            df = df.head(1000)
        
        # Handle categorical variables
        df_encoded = pd.get_dummies(df, columns=['home_ownership', 'loan_purpose'], 
                                     drop_first=True)
        
        # Select features
        feature_cols = [col for col in df_encoded.columns if col != 'loan_status']
        
        logger.info(f"Processed data: {len(df_encoded)} samples, {len(feature_cols)} features")
        
        return df_encoded, feature_cols
    
    def prepare_train_test_split(self, df: pd.DataFrame, 
                                feature_cols: list) -> Tuple:
        """
        Prepare train/test split with scaling.
        
        Returns:
            X_train, X_test, y_train, y_test, scaler, feature_names
        """
        X = df[feature_cols].values
        y = df['loan_status'].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=y
        )
        
        # Scale continuous features (first 6 are continuous in our synthetic data)
        scaler = StandardScaler()
        X_train[:, :6] = scaler.fit_transform(X_train[:, :6])
        X_test[:, :6] = scaler.transform(X_test[:, :6])
        
        logger.info(f"Train/test split: {len(X_train)} train, {len(X_test)} test")
        
        return X_train, X_test, y_train, y_test, scaler, feature_cols
