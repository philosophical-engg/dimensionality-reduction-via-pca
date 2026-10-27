"""PCA engine implemented from first principles."""

import numpy as np


class PCAFromScratch:
    """A small, readable implementation of PCA using NumPy linear algebra."""

    def __init__(self, method="eigen"):
        if method not in {"eigen", "svd"}:
            raise ValueError("method must be 'eigen' or 'svd'")
        self.method = method

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        if X.shape[0] < 2:
            raise ValueError("PCA needs at least two observations.")

        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        n = Xc.shape[0]

        if self.method == "eigen":
            covariance = (Xc.T @ Xc) / (n - 1)
            values, vectors = np.linalg.eigh(covariance)
            order = np.argsort(values)[::-1]
            values = values[order]
            vectors = vectors[:, order]
            values = np.maximum(values, 0.0)
            self.explained_variance_ = values
            self.components_ = vectors.T
        else:
            U, singular_values, Vt = np.linalg.svd(Xc, full_matrices=False)
            self.components_ = Vt
            self.explained_variance_ = (singular_values ** 2) / (n - 1)

        total = np.sum(self.explained_variance_)
        self.explained_variance_ratio_ = (
            self.explained_variance_ / total if total > 0 else np.zeros_like(self.explained_variance_)
        )
        self.cumulative_explained_variance_ = np.cumsum(self.explained_variance_ratio_)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, k=None):
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has incompatible dimensions.")
        k = self._validate_k(k)
        return (X - self.mean_) @ self.components_[:k].T

    def inverse_transform(self, Z):
        self._check_fitted()
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2:
            raise ValueError("Z must be a 2D matrix.")
        k = Z.shape[1]
        self._validate_k(k)
        return Z @ self.components_[:k] + self.mean_

    def components_for_variance(self, threshold=0.95):
        self._check_fitted()
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        return int(np.searchsorted(self.cumulative_explained_variance_, threshold) + 1)

    def _validate_k(self, k):
        if k is None:
            return self.n_features_in_
        if not 1 <= k <= min(self.components_.shape):
            raise ValueError("k must be between 1 and the number of available components.")
        return int(k)

    def _check_fitted(self):
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit(X) before using the PCA model.")
