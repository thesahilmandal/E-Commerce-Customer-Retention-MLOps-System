from typing import List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalSchemaEnforcer(BaseEstimator, TransformerMixin):
    """
    A custom Scikit-Learn transformer that acts as an immovable contract between 
    the data and the model. It ensures that categorical columns are strictly cast 
    to Pandas Categorical dtypes with the exact categories observed during training,
    which is a strict requirement for XGBoost when enable_categorical=True.
    """
    def __init__(self, categorical_features: List[str], numerical_features: List[str]) -> None:
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.categories_: Dict[str, List[Any]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CategoricalSchemaEnforcer":
        """Learns the categories present in the training data."""
        for col in self.categorical_features:
            if col in X.columns:
                # Extract unique, non-null categories and cast to string for consistency
                unique_vals = X[col].dropna().astype(str).unique().tolist()
                self.categories_[col] = unique_vals
            else:
                self.categories_[col] = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Enforces numerical casting, categorical casting, and column ordering."""
        X_transformed = X.copy()
        
        # 1. Enforce numerical types
        for col in self.numerical_features:
            if col in X_transformed.columns:
                X_transformed[col] = pd.to_numeric(X_transformed[col], errors="coerce")
            else:
                X_transformed[col] = np.nan

        # 2. Enforce categorical types
        for col in self.categorical_features:
            if col in X_transformed.columns:
                # Convert to string first to match fit behavior
                X_transformed[col] = X_transformed[col].astype(str)
                # Cast to strict categorical dtype based on fitted dictionary
                X_transformed[col] = pd.Categorical(
                    X_transformed[col], 
                    categories=self.categories_.get(col, []),
                    ordered=False
                )
            else:
                X_transformed[col] = pd.Categorical(
                    [np.nan] * len(X_transformed), 
                    categories=self.categories_.get(col, [])
                )

        # 3. Ensure strict column ordering
        expected_columns = self.numerical_features + self.categorical_features
        return X_transformed[expected_columns]