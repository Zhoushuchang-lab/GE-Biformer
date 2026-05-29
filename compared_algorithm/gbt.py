# -*- coding: utf-8 -*-
"""
Gradient Boosting Trees (GBT) implementation
"""
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class GBT(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting Trees for genomic prediction
    """
    
    def __init__(self, use_env=False, n_estimators=100, learning_rate=0.1,
                 max_depth=3, random_state=42):
        """
        Initialize GBT
        
        Args:
            use_env: Whether to use environment features
            n_estimators: Number of boosting stages
            learning_rate: Learning rate
            max_depth: Maximum depth of individual trees
            random_state: Random state
        """
        self.use_env = use_env
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None
        
    def fit(self, X_snp, y, X_env=None):
        """
        Fit GBT model
        
        Args:
            X_snp: SNP features (n_samples, n_snps)
            y: Target values (n_samples,)
            X_env: Environment features (n_samples, n_env) [optional]
        """
        if self.use_env and X_env is not None:
            X = np.hstack([X_snp, X_env])
        else:
            X = X_snp
        
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        
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
