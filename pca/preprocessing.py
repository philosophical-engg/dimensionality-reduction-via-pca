"""Simple preprocessing utilities for tabular PCA."""

import numpy as np


def prepare_numeric_data(df, standardize=True):
    clean = df.copy()

    # 1. Replace any +inf / -inf with NaN first
    clean = clean.replace([np.inf, -np.inf], np.nan)

    # 2. Drop columns that are 100% NaN (completely empty numeric columns)
    clean = clean.dropna(axis=1, how="all")

    # 3. Median imputation (with 0.0 fallback just in case)
    for column in clean.columns:
        if clean[column].isna().any():
            med = clean[column].median()
            fill_val = 0.0 if np.isnan(med) else med
            clean[column] = clean[column].fillna(fill_val)

    X = clean.to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[(std == 0) | np.isnan(std)] = 1.0
        X = (X - mean) / std

    return X, list(clean.columns)
