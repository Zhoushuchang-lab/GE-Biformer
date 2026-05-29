# -*- coding: utf-8 -*-
"""
GBLUP (Genomic Best Linear Unbiased Prediction) implementation
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, RegressorMixin


class GBLUP(BaseEstimator, RegressorMixin):
    """
    Genomic Best Linear Unbiased Prediction using Ridge Regression
    
    Note: SNP data is assumed to be already preprocessed/standardized
    """
    
    def __init__(self, use_env=False, alpha=1.0):
        """
        Initialize GBLUP model
        
        Args:
            use_env: Whether to use environment features
            alpha: Regularization strength (higher = more regularization)
        """
        self.use_env = use_env
        self.alpha = alpha
        self.model = None
        self.intercept = None
    
    def fit(self, X_snp, y, X_env=None):
        """
        Fit GBLUP model
        
        Args:
            X_snp: SNP features (n_samples, n_snps) - already standardized
            y: Target values (n_samples,)
            X_env: Environment features (n_samples, n_env) [optional]
        """
        # Combine features if needed
        if self.use_env and X_env is not None:
            X = np.hstack([X_snp, X_env])
        else:
            X = X_snp
        
        # Use Ridge regression directly
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(X, y)
        
        return self
    
    def predict(self, X_snp_test, X_env_test=None):
        """
        Make predictions
        
        Args:
            X_snp_test: Test SNP features (n_test, n_snps)
            X_env_test: Test environment features (n_test, n_env) [optional]
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.use_env and X_env_test is not None:
            X_test = np.hstack([X_snp_test, X_env_test])
        else:
            X_test = X_snp_test
        
        return self.model.predict(X_test)
