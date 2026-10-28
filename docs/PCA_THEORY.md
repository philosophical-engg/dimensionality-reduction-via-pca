# PCA Theory — Comprehensive Mathematical and Practical Guide

This guide explains Principal Component Analysis (PCA) from geometric intuition through its mathematical derivation, implementation choices, practical applications, limitations, and the differences between covariance-eigenvalue PCA and Singular Value Decomposition (SVD)-based PCA.

The equations use standard Markdown math delimiters (`$...$` for inline mathematics and `$$...$$` for display equations). If your application uses a renderer that supports only a subset of LaTeX, these delimiters and commands are generally more portable than custom macros such as `\operatorname{}`.

---

## 1. What PCA does

**Principal Component Analysis (PCA)** is an unsupervised linear transformation that represents a dataset using new, mutually orthogonal directions. The first direction captures as much variance as possible; each subsequent direction captures as much of the remaining variance as possible while remaining orthogonal to the earlier directions.

PCA is commonly used for:

- reducing the number of numerical features;
- visualizing high-dimensional data in two or three dimensions;
- compressing data while controlling reconstruction loss;
- removing some low-variance noise;
- reducing redundancy among correlated measurements;
- creating compact inputs for downstream machine-learning models.

PCA does **not** automatically identify the variables that matter most for a prediction target. It is unsupervised: it uses the feature values, not target labels, to determine its directions.

### A geometric intuition

Imagine a cloud of points shaped like a long, narrow ellipse in two dimensions. Although each point has two coordinates, most of the cloud's spread lies along the ellipse's long axis. PCA chooses that axis as the first principal component. The perpendicular short axis becomes the second component.

If the points are tightly concentrated around the long axis, keeping only the first component can preserve much of their variation using one number per point instead of two.

In three or more dimensions, the same idea applies: PCA finds orthogonal directions that successively capture the largest remaining variance.

---

## 2. Data-matrix notation

Suppose the dataset contains $n$ observations and $p$ numerical features. Arrange it into a matrix:

$$
X \in \mathbb{R}^{n \times p}.
$$

- $n$ is the number of rows or observations.
- $p$ is the number of columns or features.
- $X_{ij}$ is the value of feature $j$ for observation $i$.

For example, rows might represent houses and columns might represent floor area, number of rooms, age, and distance to a city centre.

Let the column vector of feature means be:

$$
\boldsymbol{\mu} =
\begin{bmatrix}
\mu_1 & \mu_2 & \cdots & \mu_p
\end{bmatrix}^{\mathsf T},
\qquad
\mu_j = \frac{1}{n}\sum_{i=1}^{n} X_{ij}.
$$

Let $\mathbf{1}_n$ denote an $n$-dimensional column vector of ones.

### Why the data is centred

PCA usually begins by subtracting each feature's mean:

$$
X_c = X - \mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}.
$$

After centring, each column of $X_c$ has mean zero, apart from small floating-point error.

This matters because PCA is intended to describe variation **around the mean**. Without centring, the directions can be influenced by the dataset's offset from the origin rather than just by its spread.

For new data, always subtract the means calculated on the training data. Recomputing means independently for a test set changes the coordinate system and can cause data leakage in a machine-learning workflow.

---

## 3. Variance: measuring spread

Variance measures how far values tend to lie from their mean. For observations $x_1,\ldots,x_n$, the sample variance is:

$$
s_x^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2,
\qquad
\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i.
$$

The divisor $n-1$ is used for the conventional unbiased sample-variance estimate. Some algorithms instead use $n$; that changes the scale of the eigenvalues but not the principal directions or explained-variance ratios when used consistently.

PCA uses variance as a measure of how much the observations spread along a direction. A direction with high projected variance is considered a useful direction for representing the dataset under PCA's objective.

**Important qualification:** high variance does not always mean high practical or predictive importance. A feature can vary substantially because of measurement noise, while a low-variance feature may be highly informative for a particular prediction task.

---

## 4. Covariance: how two features vary together

Covariance describes whether two features tend to be above or below their means together. For paired observations $(x_i,y_i)$:

$$
\operatorname{Cov}(X,Y)
=
\frac{1}{n-1}
\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y}).
$$

- **Positive covariance:** the features tend to move in the same direction relative to their means.
- **Negative covariance:** when one is above its mean, the other tends to be below its mean.
- **Covariance near zero:** there is little linear co-variation, though a nonlinear relationship may still exist.

Covariance depends on measurement units. For example, changing a feature from metres to millimetres multiplies its variance by one million. This is why feature scaling can be important before PCA.

---

## 5. The covariance matrix

The sample covariance matrix of the centred data is:

$$
C = \frac{1}{n-1}X_c^{\mathsf T}X_c.
$$

It has shape $p \times p$. Its entries are:

$$
C_{ab} = \operatorname{Cov}(X_a,X_b).
$$

The diagonal entries contain feature variances; the off-diagonal entries contain pairwise covariances.

The covariance matrix is symmetric:

$$
C^{\mathsf T}=C.
$$

It is also positive semidefinite, so its eigenvalues are non-negative apart from tiny negative values that may appear due to numerical rounding. Symmetry is important because a real symmetric matrix has real eigenvalues and can be decomposed using an orthonormal set of eigenvectors.

---

## 6. Eigenvectors and eigenvalues

An eigenvector of $C$ satisfies:

$$
C\mathbf{v}_i = \lambda_i\mathbf{v}_i.
$$

Here:

- $\mathbf{v}_i$ is an eigenvector, a direction in feature space;
- $\lambda_i$ is the eigenvalue associated with that direction.

Multiplying $C$ by an eigenvector scales it by $\lambda_i$ without changing its direction. In PCA, the eigenvectors give the principal directions, and the corresponding eigenvalues give the variance along those directions.

For a symmetric covariance matrix, the eigenvectors can be chosen to be orthonormal:

$$
\mathbf{v}_i^{\mathsf T}\mathbf{v}_j =
\begin{cases}
1, & i=j,\\
0, & i\ne j.
\end{cases}
$$

Collect the eigenvectors into a matrix:

$$
V = \begin{bmatrix}
\mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_p
\end{bmatrix}.
$$

Then the eigendecomposition is:

$$
C = V\Lambda V^{\mathsf T},
$$

where $\Lambda$ is the diagonal matrix of eigenvalues.

Sort the eigenvalues in descending order and reorder the eigenvectors in exactly the same way:

$$
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0.
$$

The first principal component is associated with $\lambda_1$, the second with $\lambda_2$, and so on. Eigenvector signs are arbitrary: $\mathbf{v}$ and $-\mathbf{v}$ describe the same principal axis.

---

## 7. Projection into principal-component space

Keep the first $k$ principal directions:

$$
V_k = \begin{bmatrix}
\mathbf{v}_1 & \mathbf{v}_2 & \cdots & \mathbf{v}_k
\end{bmatrix}.
$$

The matrix $V_k$ has shape $p \times k$. Project the centred observations into the reduced space:

$$
Z = X_c V_k.
$$

The reduced matrix $Z$ has shape $n \times k$. Each row is an observation represented by its scores on the first $k$ principal components.

If $k<p$, each observation is represented using fewer numbers. The new columns are linear combinations of the original features, not selected original columns.

### What does a principal-component score mean?

For observation $i$, the score on component $j$ is:

$$
z_{ij} = \sum_{\ell=1}^{p}(X_{i\ell}-\mu_\ell)V_{\ell j}.
$$

A score indicates where an observation lies along that principal direction relative to the centred dataset.

---

## 8. Reconstruction and information loss

To approximate the original data from the reduced representation, project back into the original feature space and add the means:

$$
\widehat{X} = ZV_k^{\mathsf T}
+ \mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}.
$$

Equivalently:

$$
\widehat{X} = X_cV_kV_k^{\mathsf T}
+ \mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}.
$$

When $k<p$, the discarded directions cannot be recovered exactly. The reconstruction is therefore generally an approximation.

Keeping more components usually reduces reconstruction error. If all nonzero principal directions are retained, the data can be reconstructed up to numerical precision. If $p$ exceeds the number of independent centred observations, some directions have zero variance, so retaining every column is not necessary to reconstruct the observed data.

PCA's truncated reconstruction has a useful optimality property: among rank-$k$ linear reconstructions under the usual squared-error criterion, PCA minimises the total squared reconstruction error.

---

## 9. Explained variance and choosing the number of components

The explained-variance ratio of component $i$ is:

$$
\mathrm{EVR}_i =
\frac{\lambda_i}{\sum_{j=1}^{p}\lambda_j}.
$$

The cumulative explained-variance ratio for the first $k$ components is:

$$
\mathrm{CEV}_k =
\sum_{i=1}^{k}\mathrm{EVR}_i.
$$

For example, if $\mathrm{CEV}_5=0.96$, the first five components account for 96% of the total variance represented by the covariance matrix. This does **not** mean that 96% of every type of information or 96% of predictive performance is preserved.

Common component-selection strategies include:

1. **Explained-variance threshold:** choose the smallest $k$ that reaches a target such as 90%, 95%, or 99%.
2. **Scree plot:** inspect where the sorted eigenvalues begin to flatten, often called the elbow.
3. **Reconstruction error:** choose a $k$ that provides an acceptable error for the application.
4. **Downstream validation:** choose $k$ based on performance on held-out data when PCA is used before a predictive model.
5. **Operational constraints:** balance accuracy against memory, storage, and latency.

There is no universal correct value of $k$. The best choice depends on what the representation will be used for.

---

## 10. Reconstruction error

Given original values $x_i$ and reconstructed values $\widehat{x}_i$, the mean squared error is:

$$
\mathrm{MSE} =
\frac{1}{N}\sum_{i=1}^{N}(x_i-\widehat{x}_i)^2.
$$

The root mean squared error is:

$$
\mathrm{RMSE} = \sqrt{\mathrm{MSE}}.
$$

Here $N$ is the total number of scalar values being compared, such as all entries in a data matrix. If the features use different units or scales, an aggregate error can be dominated by large-scale features. Evaluate error in the appropriate scaled or original space for the intended use.

A lower reconstruction error means the reconstruction is numerically closer under squared error. It does not by itself establish that the representation is more useful for classification, forecasting, or another task.

---

## 11. Two ways to compute PCA

There are two closely related approaches in this project:

1. **Covariance/eigenvalue PCA:** explicitly form the covariance matrix and calculate its eigenvectors and eigenvalues.
2. **SVD-based PCA:** apply Singular Value Decomposition directly to the centred data matrix.

They implement the same underlying PCA geometry when applied to the same centred matrix with consistent conventions. Their principal directions and variance estimates should agree up to numerical precision, sign choices, and differences caused by truncation or numerical tolerances.

The important difference is **how the calculation is carried out**, including its computational cost and numerical behaviour.

---

## 12. Method A — Covariance matrix and eigendecomposition

### Step-by-step procedure

**Step 1: Centre the data.**

$$
X_c = X-\mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}.
$$

**Step 2: Form the covariance matrix.**

$$
C = \frac{1}{n-1}X_c^{\mathsf T}X_c.
$$

**Step 3: Eigendecompose the covariance matrix.**

$$
C = V\Lambda V^{\mathsf T}.
$$

**Step 4: Sort the eigenvalues and their corresponding eigenvectors.**

$$
\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p.
$$

**Step 5: Keep the first $k$ eigenvectors.**

$$
V_k = [\mathbf{v}_1,\ldots,\mathbf{v}_k].
$$

**Step 6: Project the data.**

$$
Z = X_cV_k.
$$

### Why this method works

The covariance matrix encodes the variance and covariance of all features. Its eigenvectors are the directions that diagonalise that covariance structure. Sorting the eigenvalues ranks those directions by variance.

In the principal-component coordinate system, the covariance matrix becomes diagonal:

$$
V^{\mathsf T}CV = \Lambda.
$$

The diagonal entries are the variances of the principal-component scores.

### Strengths

- The intermediate covariance matrix is explicit and easy to inspect.
- The derivation closely matches many introductory statistics and linear-algebra explanations of PCA.
- It is a useful teaching implementation because it makes the relationship between covariance, eigenvectors, eigenvalues, and explained variance visible.
- It can be effective when the number of features $p$ is moderate and forming a $p\times p$ matrix is affordable.

### Weaknesses

- It must form a $p\times p$ covariance matrix, which requires memory proportional to $p^2$.
- Forming $X_c^{\mathsf T}X_c$ costs roughly $O(np^2)$ operations for a dense matrix.
- The eigendecomposition of a dense $p\times p$ matrix costs roughly $O(p^3)$ operations.
- Forming the Gram/covariance matrix can worsen numerical conditioning: in the nonzero singular spectrum, the condition number of $X_c^{\mathsf T}X_c$ is the square of the condition number of $X_c$.

The last point matters when singular values differ greatly in magnitude. Small components can become less accurately estimated when the covariance matrix is formed explicitly.

### When it is a sensible choice

- The dataset has a manageable number of features.
- You want to teach or inspect covariance structure directly.
- You need the full covariance matrix for other analysis.
- You want a straightforward from-scratch implementation and can afford the intermediate matrix.

---

## 13. Method B — Singular Value Decomposition (SVD)

SVD factorises a matrix into three matrices. Applied to the centred data:

$$
X_c = U\Sigma V^{\mathsf T}.
$$

The columns of $U$ describe directions in observation space, the columns of $V$ describe directions in feature space, and $\Sigma$ contains the non-negative singular values. The principal directions are the columns of $V$.

For a thin SVD, only the singular vectors needed to represent the matrix are returned. The exact shapes depend on the SVD variant and on whether $n$ or $p$ is smaller.

### Step-by-step procedure

**Step 1: Centre the data.**

$$
X_c = X-\mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}.
$$

**Step 2: Compute its SVD.**

$$
X_c = U\Sigma V^{\mathsf T}.
$$

**Step 3: Read the principal directions from $V$.**

Keep the first $k$ columns:

$$
V_k = [\mathbf{v}_1,\ldots,\mathbf{v}_k].
$$

**Step 4: Calculate component scores.**

$$
Z = X_cV_k.
$$

Equivalently, using the SVD factors, the scores are the corresponding columns of $U$ multiplied by their singular values.

**Step 5: Calculate the component variances.**

If $\sigma_i$ is the $i$th singular value, then:

$$
\lambda_i = \frac{\sigma_i^2}{n-1}.
$$

The explained-variance ratios can then be calculated from these variances.

### Why SVD gives PCA

Start with the covariance matrix:

$$
C = \frac{X_c^{\mathsf T}X_c}{n-1}.
$$

Substitute $X_c=U\Sigma V^{\mathsf T}$:

$$
C =
\frac{V\Sigma^{\mathsf T}U^{\mathsf T}U\Sigma V^{\mathsf T}}{n-1}.
$$

The columns of $U$ are orthonormal, so $U^{\mathsf T}U=I$ for the corresponding thin/full decomposition. Thus:

$$
C = V\frac{\Sigma^{\mathsf T}\Sigma}{n-1}V^{\mathsf T}.
$$

For the usual diagonal singular-value matrix, this gives:

$$
\lambda_i = \frac{\sigma_i^2}{n-1}.
$$

So the right singular vectors are covariance eigenvectors, and the squared singular values divided by $n-1$ are the corresponding covariance eigenvalues.

### Strengths

- It works directly on the centred data matrix and does not need to explicitly construct the covariance matrix.
- It is often numerically more stable than forming $X_c^{\mathsf T}X_c$ first.
- It can be attractive when the data has many features, especially when a suitable thin, truncated, or iterative SVD implementation is available.
- It provides singular values that directly reveal the relative scale of the principal directions.

### Weaknesses

- A full SVD can still be expensive for a very large dense matrix.
- SVD implementations and solver choices may be more complex to explain.
- If only a small number of components is needed, a full exact SVD may do more work than necessary; truncated or randomized methods may be better.
- The best algorithm depends on matrix shape, sparsity, required precision, and hardware.

### When it is a sensible choice

- Numerical stability is important.
- You want to avoid explicitly forming a covariance matrix.
- You have a large or high-dimensional dataset and can use an appropriate SVD solver.
- You want a strong general-purpose foundation for practical PCA implementations.

---

## 14. Covariance PCA vs SVD PCA — direct comparison

| Property | Covariance/eigenvalue PCA | SVD-based PCA |
|---|---|---|
| Matrix decomposed | Covariance matrix $C$ | Centred data matrix $X_c$ |
| Main factorisation | $C=V\Lambda V^{\mathsf T}$ | $X_c=U\Sigma V^{\mathsf T}$ |
| Principal directions | Eigenvectors of $C$ | Right singular vectors of $X_c$ |
| Variances | Eigenvalues $\lambda_i$ | $\sigma_i^2/(n-1)$ |
| Explicit covariance matrix | Yes | No |
| Extra matrix memory | Often $O(p^2)$ for $C$ | Depends on SVD algorithm and returned factors |
| Numerical conditioning | Forming $X_c^{\mathsf T}X_c$ squares the nonzero-spectrum condition number | Usually avoids that squaring step |
| Best teaching value | Makes covariance/eigenvalue interpretation explicit | Shows a general matrix factorisation and its connection to PCA |
| Often suitable for | Moderate feature counts; when covariance is useful | General-purpose numerical PCA; high-dimensional cases with suitable solvers |
| Are results mathematically equivalent? | Yes, under matching assumptions and conventions | Yes, under matching assumptions and conventions |

### Which one is faster?

There is no universal winner. Performance depends on $n$, $p$, whether the matrix is dense or sparse, whether all components or only a few are needed, the numerical library, and the hardware.

For a dense $n\times p$ matrix:

- covariance formation costs approximately $O(np^2)$ and its full eigendecomposition approximately $O(p^3)$;
- a conventional full SVD has a cost that depends on the matrix shape, often described in broad terms as cubic in the smaller dimension times the larger dimension for common dense algorithms.

If $p$ is small and $n$ is very large, covariance PCA can be practical. If $p$ is very large, the $p\times p$ covariance matrix can become the bottleneck. When $k$ is small, truncated or randomized SVD may avoid computing every component.

A useful rule of thumb is: **use covariance eigendecomposition to make the covariance-based derivation explicit; use a stable SVD solver as a strong default for numerical work, unless the data shape or performance requirements suggest another solver.** Benchmark on the actual workload when speed matters.

---

## 15. Scaling and standardisation

PCA on the covariance matrix is sensitive to feature units. Suppose one feature is measured between 0 and 1 and another between 0 and 1,000. The second feature may dominate the covariance structure simply because of its scale.

Standardisation transforms a feature value using:

$$
z = \frac{x-\mu}{s},
$$

where $\mu$ is the feature mean and $s$ is its standard deviation, estimated on the training data.

After standardisation, each non-constant feature has approximately zero mean and unit variance. PCA on standardised data is equivalent to performing PCA on a correlation-scaled representation rather than allowing raw feature variances to set the scale.

### When should you standardise?

Standardisation is often appropriate when:

- features use different units, such as metres, kilograms, and currency;
- the relative scale of each feature should not determine its influence;
- all features should begin with comparable variance.

It may be inappropriate when absolute variance is meaningful and should determine the directions, or when scaling would erase a meaningful distinction in feature reliability. Constant features require special handling because their standard deviation is zero.

Any imputation, scaling, or PCA fit used in a predictive workflow should be learned from the training split only and then applied to validation/test data.

---

## 16. PCA is not feature selection

**Feature selection** retains a subset of the original columns. PCA usually creates new features by combining the original columns.

For example, the first principal component may be:

$$
\mathrm{PC}_1 =
a_1X_1+a_2X_2+\cdots+a_pX_p,
$$

where the coefficients $a_j$ are entries of the first principal direction.

This means PC1 can combine information from every original feature. The coefficients are often called **loadings** (terminology varies by convention and scaling).

| Feature selection | PCA |
|---|---|
| Keeps selected original variables | Creates linear combinations of variables |
| Often preserves direct interpretability of chosen columns | Components may be harder to interpret |
| Can use labels or domain knowledge, depending on the method | Standard PCA does not use target labels |
| Does not necessarily decorrelate retained variables | Principal-component scores are uncorrelated in the training sample |

PCA is useful when compactness and redundancy reduction matter. Feature selection may be preferable when domain interpretability, measurement cost, or a requirement to retain original variables is central.

---

## 17. Real-world applications

### 17.1 Image compression

An image can be represented as a matrix of pixel values. PCA can approximate that matrix using a limited number of principal directions:

$$
\widehat{X} = X_cV_kV_k^{\mathsf T}
+ \mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}.
$$

The smaller $k$ is, the fewer component scores need to be stored, but the reconstruction generally loses more detail. Larger $k$ tends to improve fidelity while reducing the compression benefit.

For colour images, each channel may be processed separately or the image may be represented using another matrix arrangement. The quality/cost trade-off depends on the representation.

**Practical caveat:** PCA compression is a teaching and analytical technique, not always the best production image codec. Modern image codecs exploit spatial structure and perceptual properties in ways that plain PCA does not.

### 17.2 Visualising high-dimensional datasets

A dataset with dozens or hundreds of features cannot be plotted directly in ordinary 2D space. PCA can map it to two or three components and make broad patterns visible.

Examples include exploratory analysis of sensor readings, customer measurements, or biological samples.

**Caveat:** separation in a PC1-versus-PC2 plot is not proof of class separability, and patterns in discarded dimensions will not be visible.

### 17.3 Genomics and biological measurements

Gene-expression and other biological datasets can have thousands of measured variables but comparatively few samples. PCA is used to explore dominant patterns, identify possible batch effects, inspect sample similarity, and visualise broad variation.

**Caveat:** a component can reflect technical variation, sample preparation, or other confounders rather than the biological process of interest. Interpretation requires domain knowledge and careful experimental design.

### 17.4 Finance and market data

PCA can summarise correlated movements across asset returns, interest-rate curves, or other financial variables. A small number of components may represent broad common movement and additional patterns.

For example, multiple interest-rate maturities often move together; principal components can provide a compact description of level-like, slope-like, and curvature-like changes.

**Caveat:** components are estimated from a particular dataset and time period. Their meaning can change over time, and PCA alone is not a trading strategy or a guarantee of risk reduction.

### 17.5 Industrial monitoring and sensor systems

Factories and buildings may collect readings from many correlated sensors. PCA can compress those readings, reveal dominant operating patterns, and support monitoring for unusual deviations from a baseline.

A common approach is to inspect component scores and reconstruction residuals. Observations that differ strongly from the patterns learned during normal operation may merit investigation.

**Caveat:** a large residual is an alerting signal, not a diagnosis of the cause. Sensor drift, missing values, and changes in operating conditions need separate treatment.

### 17.6 Telecommunications and signal processing

Measurements from multiple channels can be correlated. PCA can transform them into orthogonal coordinates and reduce dimensionality before other processing steps. Related eigenvector methods are used in areas such as array processing and signal analysis.

**Caveat:** the exact method used in a signal-processing system may be a specialised variant rather than ordinary textbook PCA.

### 17.7 Machine-learning preprocessing

PCA can reduce the input dimension before a downstream model. This may lower storage requirements, speed up training, and reduce redundant directions.

It can be useful for models that benefit from compact, less-correlated inputs, but the result is model- and dataset-dependent.

**Important:** fit PCA inside the training pipeline. Choosing components based on the entire dataset before evaluating a model can leak information from the test set. Always compare against a baseline without PCA using suitable validation.

### 17.8 Text and document representations

PCA can be applied to numerical document-feature matrices or dense embeddings to find directions of high variance and create compact representations. In practice, sparse text matrices often call for specialised methods such as truncated SVD rather than explicitly constructing a dense covariance matrix.

**Caveat:** PCA is not the same as Latent Semantic Analysis (LSA), though truncated SVD of a term-document matrix is a central technique in LSA. Centring sparse matrices can also destroy sparsity if done explicitly, so the choice of algorithm matters.

### 17.9 Quality control and anomaly exploration

PCA can model the dominant variation of a process. An observation with an unusual score pattern or a large reconstruction residual may be worth inspecting.

This is useful as an exploratory or monitoring tool, but the thresholds must be calibrated to the process and the cost of false alarms.

---

## 18. Limitations and common misunderstandings

1. **PCA is linear.** It cannot generally represent curved or nonlinear structure with a small number of linear components.
2. **High variance is not the same as usefulness.** PCA does not know which features predict a target.
3. **Scaling matters.** Raw units can dominate the result.
4. **Outliers can strongly influence the mean and covariance.** Robust preprocessing or robust PCA variants may be needed.
5. **Components may be difficult to interpret.** Each component can combine many original variables.
6. **Results depend on the data used to fit PCA.** A new population or time period may have a different covariance structure.
7. **Near-equal eigenvalues can make individual directions unstable.** The subspace can be stable even when the ordering or orientation of individual components changes.
8. **PCA does not automatically improve model accuracy.** It should be validated against the task.
9. **Missing values need explicit handling.** Ordinary PCA implementations generally expect a complete numerical matrix.
10. **PCA is not inherently a causal method.** A component describes a pattern of variation, not a cause.

---

## 19. Computational complexity in practice

Let $X_c$ be a dense $n\times p$ matrix.

### Covariance/eigenvalue route

- Centre the data: $O(np)$.
- Form $X_c^{\mathsf T}X_c$: approximately $O(np^2)$.
- Store covariance matrix: $O(p^2)$ memory.
- Full eigendecomposition: approximately $O(p^3)$.

These broad estimates assume dense linear algebra and do not include implementation-specific optimisations.

### SVD route

The cost depends on the matrix shape and solver:

- A full dense SVD can be expensive for large $n$ and $p$.
- A thin SVD avoids storing unnecessary full orthogonal matrices.
- A truncated SVD computes only a requested number of dominant directions.
- Randomised or iterative methods can be useful when only a small $k$ is required and approximate solutions are acceptable.

### Shape matters

- If $p$ is moderate, covariance PCA may be efficient and easy to interpret.
- If $p$ is extremely large, constructing a $p\times p$ matrix can be impractical.
- If $n$ is much smaller than $p$, methods that exploit the smaller dimension or use truncated decompositions can be advantageous.
- Sparse datasets require methods that preserve or exploit sparsity; naïvely centring a sparse matrix can make it dense.

Choose the solver based on matrix shape, memory, precision requirements, and the number of components needed—not solely on a blanket claim that one method is always faster.

---

## 20. What the implementation does

The PCA Lab implementation exposes the main stages of the algorithm.

### `PCAFromScratch.fit(X)`

The fitting stage typically:

1. converts input into floating-point data;
2. calculates and stores feature means;
3. centres the matrix;
4. calculates the covariance eigendecomposition or SVD;
5. orders principal directions by decreasing variance;
6. stores the component variances;
7. calculates explained-variance ratios and their cumulative sums.

The means and principal directions learned here must be reused when transforming new observations.

### `transform(X, k)`

The transformation stage centres new data with the stored training means and projects it onto the first $k$ components:

$$
Z = (X-\mathbf{1}_n\boldsymbol{\mu}^{\mathsf T})V_k.
$$

### `inverse_transform(Z)`

The inverse transformation maps scores back to the original feature coordinates and restores the mean:

$$
\widehat{X} = ZV_k^{\mathsf T}
+ \mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}.
$$

The result is exact only if the retained directions span the relevant data space; otherwise it is an approximation.

### `components_for_variance(threshold)`

This operation finds the smallest $k$ whose cumulative explained-variance ratio reaches the requested threshold. A threshold of 0.95, for example, means selecting enough leading components to account for at least 95% of the fitted data's variance.

### `reconstruction_metrics(X, X_hat)`

This operation may calculate metrics such as MSE and RMSE and estimate a compression ratio. A compression ratio is meaningful only when the storage assumptions are stated—for example, whether component vectors, means, metadata, and numeric precision are included.

---

## 21. Practical workflow for applying PCA responsibly

A robust workflow for a tabular machine-learning problem is:

1. **Define the purpose.** Is the goal visualisation, compression, noise reduction, or prediction?
2. **Separate training and test data first** when evaluating predictive performance.
3. **Inspect the data.** Check feature types, units, missingness, outliers, and constant columns.
4. **Impute missing values** using a method suitable for the dataset.
5. **Decide whether to standardise.** Base this on feature units and the intended meaning of variance.
6. **Fit PCA on the training data only.**
7. **Inspect explained variance and component loadings.**
8. **Choose $k$** using variance, reconstruction error, validation performance, or operational constraints.
9. **Transform validation and test data** using the fitted means, scales, and components.
10. **Compare against a baseline.** Verify that PCA actually helps the task.
11. **Monitor changes over time** if PCA is used in a deployed system.

For exploratory visualisation, it can be reasonable to fit PCA to the dataset being explored. For an unbiased predictive evaluation, however, fitting preprocessing on held-out data can leak information.

---

## 22. Worked conceptual example

Suppose a dataset contains two highly correlated features: a measurement in centimetres and the same measurement in millimetres. Much of their variation is redundant because one is nearly a constant multiple of the other.

After centring, PCA will find a direction along which the two features vary together and a perpendicular direction that contains little variance. Keeping the first component can represent most of the variation with one number instead of two.

This example illustrates dimensionality reduction, but it also highlights an important practical issue: units and scaling affect PCA. Standardising the two columns changes their relative scale, while the underlying redundancy remains.

In a more realistic dataset, components often combine several correlated measurements rather than duplicating a single quantity.

---

## 23. Frequently asked questions

### Does PCA require normally distributed data?

No. PCA can be calculated without assuming that the features follow a normal distribution. Normality can matter for some statistical interpretations or probabilistic models built around PCA, but it is not required for the basic linear-algebra calculation.

### Must the components be orthogonal?

Standard PCA produces mutually orthogonal component directions. The component scores are uncorrelated in the fitted sample under the usual covariance-based formulation. Uncorrelated does not necessarily mean statistically independent.

### Can eigenvalue PCA and SVD PCA produce different answers?

They should agree on the principal subspace and principal variances when applied consistently. Differences may arise from floating-point error, eigenvector sign conventions, nearly repeated eigenvalues, truncated solvers, or inconsistent centring and scaling.

### Why can an eigenvector's sign change between runs?

If $\mathbf{v}$ is an eigenvector, then $-\mathbf{v}$ is also an eigenvector for the same eigenvalue. A sign change flips the corresponding component scores but does not change the represented axis or explained variance.

### Does retaining 95% variance mean retaining 95% of the information?

Not in every sense. It means retaining 95% of the variance measured by the fitted PCA representation. Information important for a particular label, rare event, or downstream objective may lie in low-variance directions.

### Is SVD always better?

No method is best for every workload. SVD is often a strong numerical default, but covariance eigendecomposition can be practical and educational for moderate feature counts. For very large problems, truncated, iterative, or randomised methods may be more suitable than either full decomposition.

### Does PCA make features independent?

No. It makes component scores uncorrelated in the fitted sample under the standard formulation. Statistical independence requires stronger conditions and does not follow from PCA in general.

---

## 24. The entire PCA pipeline

The main mathematical sequence is:

$$
X
\longrightarrow
X_c
\longrightarrow
\begin{cases}
C = \dfrac{X_c^{\mathsf T}X_c}{n-1}
& \text{(covariance route)},\\[8pt]
X_c = U\Sigma V^{\mathsf T}
& \text{(SVD route)}
\end{cases}
$$

Both routes identify the leading principal directions $V_k$. The data is then projected:

$$
Z = X_cV_k,
$$

and optionally reconstructed:

$$
\widehat{X} = ZV_k^{\mathsf T}
+ \mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}.
$$

The goal is not simply to produce fewer columns. It is to choose a compact representation that preserves the variation relevant to the intended use, while understanding the approximation error and computational trade-offs.

---

## 25. Final takeaway

PCA is a bridge between geometry, statistics, and linear algebra:

- **Variance** measures spread.
- **Covariance** describes linear co-variation.
- **Eigenvectors** give principal directions of the covariance matrix.
- **Eigenvalues** quantify the variance along those directions.
- **SVD** finds the same principal directions directly from the centred data matrix.
- **Projection** creates a lower-dimensional representation.
- **Reconstruction** shows what was lost by discarding components.

The covariance/eigenvalue method makes the covariance structure explicit and is valuable for learning and moderate-sized feature spaces. SVD avoids explicitly forming the covariance matrix and is often more numerically stable. Both are ways to compute the same underlying PCA model; the practical choice depends on data dimensions, numerical requirements, memory, and whether a full or truncated solution is needed.
