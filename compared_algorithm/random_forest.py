# -*- coding: utf-8 -*-
"""
Random Forest implementation
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class RandomForest(BaseEstimator, RegressorMixin):
    """
    Random Forest for genomic prediction
    """
    
    def __init__(self, use_env=False, n_estimators=100, max_depth=None, 
                 min_samples_split=2, random_state=42):
        """
        Initialize Random Forest
        
        Args:
            use_env: Whether to use environment features
            n_estimators: Number of trees
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            random_state: Random state
        """
        self.use_env = use_env
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = None
        
    def fit(self, X_snp, y, X_env=None):
        """
        Fit Random Forest model
        
        Args:
            X_snp: SNP features (n_samples, n_snps)
            y: Target values (n_samples,)
            X_env: Environment features (n_samples, n_env) [optional]
        """
        if self.use_env and X_env is not None:
            X = np.hstack([X_snp, X_env])
        else:
            X = X_snp
        
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1
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
