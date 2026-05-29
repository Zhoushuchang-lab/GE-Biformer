# -*- coding: utf-8 -*-
"""
K-Nearest Neighbors implementation
"""
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class KNN(BaseEstimator, RegressorMixin):
    """
    K-Nearest Neighbors for genomic prediction
    """
    
    def __init__(self, use_env=False, n_neighbors=5, weights='uniform', 
                 metric='euclidean'):
        """
        Initialize KNN
        
        Args:
            use_env: Whether to use environment features
            n_neighbors: Number of neighbors
            weights: Weighting scheme ('uniform' or 'distance')
            metric: Distance metric
        """
        self.use_env = use_env
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.model = None
        
    def fit(self, X_snp, y, X_env=None):
        """
        Fit KNN model
        
        Args:
            X_snp: SNP features (n_samples, n_snps)
            y: Target values (n_samples,)
            X_env: Environment features (n_samples, n_env) [optional]
        """
        if self.use_env and X_env is not None:
            X = np.hstack([X_snp, X_env])
        else:
            X = X_snp
        
        self.model = KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
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
