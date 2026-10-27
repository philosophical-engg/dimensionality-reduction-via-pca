"""Simple preprocessing utilities for tabular PCA."""


import numpy as np


def prepare_numeric_data(df, standardize=True):
    clean = df.copy()

    # Median imputation keeps the example simple and is less sensitive to outliers than the mean.
    for column in clean.columns:
        if clean[column].isna().any():
            clean[column] = clean[column].fillna(clean[column].median())

    X = clean.to_numpy(dtype=float)

    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std[std == 0] = 1.0
        X = (X - mean) / std

    return X, list(clean.columns)
