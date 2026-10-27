# PCA Theory — Complete Concept Guide

## 1. The intuition

Imagine a cloud of points in two dimensions.

If the points form a long diagonal shape, most of the variation is along that diagonal. PCA searches for that direction.

The first principal component is the direction along which the projected data has maximum variance.

The second principal component is perpendicular to the first and captures as much of the remaining variance as possible.

---

## 2. Variance

Variance measures how spread out values are around their mean.

For observations \(x_1,\ldots,x_n\):

\[
Var(X)=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2
\]

PCA uses variance as its definition of the amount of information represented by a direction.

---

## 3. Covariance

Covariance describes whether two variables tend to increase or decrease together.

\[
Cov(X,Y)=\frac{1}{n-1}\sum_i(x_i-\bar{x})(y_i-\bar{y})
\]

Positive covariance means they tend to move together.

Negative covariance means they tend to move in opposite directions.

Near-zero covariance means there is little linear relationship.

---

## 4. Covariance matrix

For several variables, pairwise covariances are collected into:

\[
C=\frac{1}{n-1}X_c^TX_c
\]

The diagonal contains variances.

The off-diagonal entries contain covariances.

The covariance matrix is symmetric, which is why its eigenvectors can be chosen orthogonal.

---

## 5. Eigenvectors and eigenvalues

An eigenvector satisfies:

\[
Cv=\lambda v
\]

The direction \(v\) is special because multiplying the covariance matrix by it changes its magnitude but not its direction.

The eigenvalue \(\lambda\) tells us how much variance exists along that direction.

Therefore:

**largest eigenvalue → most important principal direction**

---

## 6. Why sorting eigenvalues matters

Numerical eigendecomposition does not necessarily return eigenvalues in descending order.

The implementation explicitly sorts them:

\[
\lambda_1\geq\lambda_2\geq\cdots
\]

and rearranges the corresponding eigenvectors in the same order.

This ensures that the first component is actually the most informative one.

---

## 7. Projection

Once the principal directions have been found, the centered data is projected:

\[
Z=X_cV_k
\]

If \(X\) has \(p\) features and \(k<p\), the new representation has only \(k\) dimensions.

This is the dimensionality-reduction step.

---

## 8. Reconstruction

The reduced data can be mapped back:

\[
\hat X=ZV_k^T+\mu
\]

Because only \(k\) directions were retained, the reconstruction is generally an approximation.

---

## 9. Why increasing k improves reconstruction

If more principal directions are retained, the reconstruction has more information available.

At \(k=p\), assuming the complete PCA basis is retained, the centered data can be reconstructed up to numerical precision.

At smaller \(k\), some directions are deliberately discarded.

---

## 10. SVD connection

SVD decomposes:

\[
X_c=U\Sigma V^T
\]

The principal directions are contained in \(V^T\).

The covariance matrix is:

\[
C=\frac{X_c^TX_c}{n-1}
\]

Substituting the SVD:

\[
C=\frac{V\Sigma^TU^TU\Sigma V^T}{n-1}
\]

Since \(U^TU=I\):

\[
C=V\frac{\Sigma^2}{n-1}V^T
\]

Therefore the eigenvalues of the covariance matrix are:

\[
\lambda_i=\frac{\sigma_i^2}{n-1}
\]

This explains why eigenvalue PCA and SVD PCA produce the same principal variances.

---

## 11. Scaling

If one feature ranges from 0 to 1 and another ranges from 0 to 1,000, the second feature can dominate covariance.

Standardization removes this unit-scale problem:

\[
z=\frac{x-\mu}{\sigma}
\]

The project therefore allows the user to choose whether tabular features should be standardized.

---

## 12. PCA is not feature selection

PCA does not normally choose some original columns and throw away the rest.

Instead it creates new variables:

\[
PC_1=a_1X_1+a_2X_2+\cdots+a_pX_p
\]

Thus each component can contain contributions from many original features.

This distinction is important:

**feature selection:** keep original variables.

**PCA:** create new variables from combinations of original variables.

---

## 13. Practical component selection

There is no universal correct value of \(k\).

Useful approaches include:

- explained-variance threshold;
- reconstruction error;
- scree plot;
- downstream model performance;
- computational/storage constraints.

This project exposes explained variance and reconstruction error so the user can study the trade-off rather than blindly choosing \(k\).

---

## 14. Complexity considerations

For the covariance approach, constructing the covariance matrix involves matrix multiplication and creates a \(p\times p\) matrix.

For very high-dimensional problems this matrix can become expensive.

SVD can be preferable in many situations because it works directly with the centered data matrix.

This is one reason production PCA implementations often provide multiple solver strategies.

---

## 15. What the code is doing

`PCAFromScratch.fit(X)`

- converts input to floating-point data;
- calculates feature means;
- centers the matrix;
- computes eigenvalue decomposition or SVD;
- orders the principal directions;
- calculates explained variance;
- calculates explained variance ratios;
- calculates cumulative explained variance.

`transform(X, k)`

- centers new data using the training mean;
- projects it onto the first \(k\) principal directions.

`inverse_transform(Z)`

- projects the reduced representation back into the original feature space;
- adds the mean vector.

`components_for_variance(threshold)`

- finds the smallest number of components whose cumulative explained variance reaches the requested threshold.

`reconstruction_metrics(X, X_hat)`

- calculates MSE;
- calculates RMSE;
- estimates the storage compression ratio.

---

## 16. The central idea

The entire project can be remembered as:

\[
\boxed{
X
\rightarrow
X-\mu
\rightarrow
C
\rightarrow
(\lambda,V)
\rightarrow
V_k
\rightarrow
Z
\rightarrow
\hat X
}
\]

That sequence is the mathematical heart of PCA Lab.
