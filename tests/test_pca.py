import numpy as np
from pca.core import PCAFromScratch


def test_components_are_orthonormal():
    X = np.array([[1, 2], [2, 4], [3, 5], [4, 7]], dtype=float)
    pca = PCAFromScratch().fit(X)
    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(2))


def test_variance_ratios_sum_to_one():
    X = np.random.default_rng(1).normal(size=(50, 5))
    pca = PCAFromScratch().fit(X)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)


def test_svd_and_eigen_have_same_variance():
    X = np.random.default_rng(2).normal(size=(40, 4))
    a = PCAFromScratch("eigen").fit(X)
    b = PCAFromScratch("svd").fit(X)
    assert np.allclose(a.explained_variance_, b.explained_variance_, atol=1e-9)
