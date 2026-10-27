"""Quantitative measures used to evaluate PCA reconstruction."""

import numpy as np


def reconstruction_metrics(X, X_hat):
    X = np.asarray(X, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)

    mse = float(np.mean((X - X_hat) ** 2))
    rmse = float(np.sqrt(mse))

    # Approximate storage: original matrix versus PCA representation.
    m, n = X.shape
    k = min(m, n)
    compression_ratio = (m * n) / (m * k + n * k + k) if k else 1.0

    return {
        "mse": mse,
        "rmse": rmse,
        "compression_ratio": compression_ratio,
    }
