import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pca.core import PCAFromScratch
from pca.metrics import reconstruction_metrics
from pca.preprocessing import prepare_numeric_data
from visualization.plots import variance_plot, reconstruction_plot, scatter_plot
from utils.data_loader import image_to_matrix

st.set_page_config(page_title="PCA Lab", page_icon="📐", layout="wide")

st.title("📐 PCA Lab")
st.caption("Interactive Principal Component Analysis, dimensionality reduction and image compression — implemented from first principles.")

with st.sidebar:
    st.header("Experiment")
    mode = st.radio("Choose a laboratory", ["Image Lab", "CSV / Dataset Lab", "PCA Theory"])
    method = st.selectbox("PCA implementation", ["Eigenvalue PCA", "SVD PCA"])
    st.divider()
    st.info("This project deliberately implements the PCA mathematics with NumPy instead of using sklearn's PCA as the main engine.")

if mode == "PCA Theory":
    st.header("Principal Component Analysis")
    st.markdown(r"""PCA Theory — Comprehensive Mathematical and Practical Guide
This guide explains Principal Component Analysis (PCA) from geometric intuition through its mathematical derivation, implementation choices, practical applications, limitations, and the differences between covariance-eigenvalue PCA and Singular Value Decomposition (SVD)-based PCA.

The equations use standard Markdown math delimiters ($...$ for inline mathematics and $$...$$ for display equations). If your application uses a renderer that supports only a subset of LaTeX, these delimiters and commands are generally more portable than custom macros such as \operatorname{}.

1. What PCA does
Principal Component Analysis (PCA) is an unsupervised linear transformation that represents a dataset using new, mutually orthogonal directions. The first direction captures as much variance as possible; each subsequent direction captures as much of the remaining variance as possible while remaining orthogonal to the earlier directions.

PCA is commonly used for:

reducing the number of numerical features;
visualizing high-dimensional data in two or three dimensions;
compressing data while controlling reconstruction loss;
removing some low-variance noise;
reducing redundancy among correlated measurements;
creating compact inputs for downstream machine-learning models.
PCA does not automatically identify the variables that matter most for a prediction target. It is unsupervised: it uses the feature values, not target labels, to determine its directions.

A geometric intuition
Imagine a cloud of points shaped like a long, narrow ellipse in two dimensions. Although each point has two coordinates, most of the cloud's spread lies along the ellipse's long axis. PCA chooses that axis as the first principal component. The perpendicular short axis becomes the second component.

If the points are tightly concentrated around the long axis, keeping only the first component can preserve much of their variation using one number per point instead of two.

In three or more dimensions, the same idea applies: PCA finds orthogonal directions that successively capture the largest remaining variance.

2. Data-matrix notation
Suppose the dataset contains 
n
 observations and 
p
 numerical features. Arrange it into a matrix:

X
∈
R
n
×
p
.

n
 is the number of rows or observations.
p
 is the number of columns or features.
X
i
j
 is the value of feature 
j
 for observation 
i
.
For example, rows might represent houses and columns might represent floor area, number of rooms, age, and distance to a city centre.

Let the column vector of feature means be:

μ
=
[
μ
1
μ
2
⋯
μ
p
]
T
,
μ
j
=
1
n
∑
i
=
1
n
X
i
j
.

Let 
1
n
 denote an 
n
-dimensional column vector of ones.

Why the data is centred
PCA usually begins by subtracting each feature's mean:

X
c
=
X
−
1
n
μ
T
.

After centring, each column of 
X
c
 has mean zero, apart from small floating-point error.

This matters because PCA is intended to describe variation around the mean. Without centring, the directions can be influenced by the dataset's offset from the origin rather than just by its spread.

For new data, always subtract the means calculated on the training data. Recomputing means independently for a test set changes the coordinate system and can cause data leakage in a machine-learning workflow.

3. Variance: measuring spread
Variance measures how far values tend to lie from their mean. For observations 
x
1
,
…
,
x
n
, the sample variance is:

s
x
2
=
1
n
−
1
∑
i
=
1
n
(
x
i
−
x
¯
)
2
,
x
¯
=
1
n
∑
i
=
1
n
x
i
.

The divisor 
n
−
1
 is used for the conventional unbiased sample-variance estimate. Some algorithms instead use 
n
; that changes the scale of the eigenvalues but not the principal directions or explained-variance ratios when used consistently.

PCA uses variance as a measure of how much the observations spread along a direction. A direction with high projected variance is considered a useful direction for representing the dataset under PCA's objective.

Important qualification: high variance does not always mean high practical or predictive importance. A feature can vary substantially because of measurement noise, while a low-variance feature may be highly informative for a particular prediction task.

4. Covariance: how two features vary together
Covariance describes whether two features tend to be above or below their means together. For paired observations 
(
x
i
,
y
i
)
:

$$ \operatorname{Cov}(X,Y)
\frac{1}{n-1} \sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y}). $$

Positive covariance: the features tend to move in the same direction relative to their means.
Negative covariance: when one is above its mean, the other tends to be below its mean.
Covariance near zero: there is little linear co-variation, though a nonlinear relationship may still exist.
Covariance depends on measurement units. For example, changing a feature from metres to millimetres multiplies its variance by one million. This is why feature scaling can be important before PCA.

5. The covariance matrix
The sample covariance matrix of the centred data is:

C
=
1
n
−
1
X
c
T
X
c
.

It has shape 
p
×
p
. Its entries are:

The following macros are not allowed: operatorname

$$
C_{ab} = \operatorname{Cov}(X_a,X_b).
$$

The diagonal entries contain feature variances; the off-diagonal entries contain pairwise covariances.

The covariance matrix is symmetric:

C
T
=
C
.

It is also positive semidefinite, so its eigenvalues are non-negative apart from tiny negative values that may appear due to numerical rounding. Symmetry is important because a real symmetric matrix has real eigenvalues and can be decomposed using an orthonormal set of eigenvectors.

6. Eigenvectors and eigenvalues
An eigenvector of 
C
 satisfies:

C
v
i
=
λ
i
v
i
.

Here:

v
i
 is an eigenvector, a direction in feature space;
λ
i
 is the eigenvalue associated with that direction.
Multiplying 
C
 by an eigenvector scales it by 
λ
i
 without changing its direction. In PCA, the eigenvectors give the principal directions, and the corresponding eigenvalues give the variance along those directions.

For a symmetric covariance matrix, the eigenvectors can be chosen to be orthonormal:

v
i
T
v
j
=
{
1
,
i
=
j
,
0
,
i
≠
j
.

Collect the eigenvectors into a matrix:

V
=
[
v
1
v
2
⋯
v
p
]
.

Then the eigendecomposition is:

C
=
V
Λ
V
T
,

where 
Λ
 is the diagonal matrix of eigenvalues.

Sort the eigenvalues in descending order and reorder the eigenvectors in exactly the same way:

λ
1
≥
λ
2
≥
⋯
≥
λ
p
≥
0.

The first principal component is associated with 
λ
1
, the second with 
λ
2
, and so on. Eigenvector signs are arbitrary: 
v
 and 
−
v
 describe the same principal axis.

7. Projection into principal-component space
Keep the first 
k
 principal directions:

V
k
=
[
v
1
v
2
⋯
v
k
]
.

The matrix 
V
k
 has shape 
p
×
k
. Project the centred observations into the reduced space:

Z
=
X
c
V
k
.

The reduced matrix 
Z
 has shape 
n
×
k
. Each row is an observation represented by its scores on the first 
k
 principal components.

If 
k
<
p
, each observation is represented using fewer numbers. The new columns are linear combinations of the original features, not selected original columns.

What does a principal-component score mean?
For observation 
i
, the score on component 
j
 is:

z
i
j
=
∑
ℓ
=
1
p
(
X
i
ℓ
−
μ
ℓ
)
V
ℓ
j
.

A score indicates where an observation lies along that principal direction relative to the centred dataset.

8. Reconstruction and information loss
To approximate the original data from the reduced representation, project back into the original feature space and add the means:

$$ \widehat{X} = ZV_k^{\mathsf T}

\mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}. $$
Equivalently:

$$ \widehat{X} = X_cV_kV_k^{\mathsf T}

\mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}. $$
When 
k
<
p
, the discarded directions cannot be recovered exactly. The reconstruction is therefore generally an approximation.

Keeping more components usually reduces reconstruction error. If all nonzero principal directions are retained, the data can be reconstructed up to numerical precision. If 
p
 exceeds the number of independent centred observations, some directions have zero variance, so retaining every column is not necessary to reconstruct the observed data.

PCA's truncated reconstruction has a useful optimality property: among rank-$k$ linear reconstructions under the usual squared-error criterion, PCA minimises the total squared reconstruction error.

9. Explained variance and choosing the number of components
The explained-variance ratio of component 
i
 is:

EVR
i
=
λ
i
∑
j
=
1
p
λ
j
.

The cumulative explained-variance ratio for the first 
k
 components is:

CEV
k
=
∑
i
=
1
k
EVR
i
.

For example, if 
CEV
5
=
0.96
, the first five components account for 96% of the total variance represented by the covariance matrix. This does not mean that 96% of every type of information or 96% of predictive performance is preserved.

Common component-selection strategies include:

Explained-variance threshold: choose the smallest 
k
 that reaches a target such as 90%, 95%, or 99%.
Scree plot: inspect where the sorted eigenvalues begin to flatten, often called the elbow.
Reconstruction error: choose a 
k
 that provides an acceptable error for the application.
Downstream validation: choose 
k
 based on performance on held-out data when PCA is used before a predictive model.
Operational constraints: balance accuracy against memory, storage, and latency.
There is no universal correct value of 
k
. The best choice depends on what the representation will be used for.

10. Reconstruction error
Given original values 
x
i
 and reconstructed values 
x
^
i
, the mean squared error is:

MSE
=
1
N
∑
i
=
1
N
(
x
i
−
x
^
i
)
2
.

The root mean squared error is:

RMSE
=
MSE
.

Here 
N
 is the total number of scalar values being compared, such as all entries in a data matrix. If the features use different units or scales, an aggregate error can be dominated by large-scale features. Evaluate error in the appropriate scaled or original space for the intended use.

A lower reconstruction error means the reconstruction is numerically closer under squared error. It does not by itself establish that the representation is more useful for classification, forecasting, or another task.

11. Two ways to compute PCA
There are two closely related approaches in this project:

Covariance/eigenvalue PCA: explicitly form the covariance matrix and calculate its eigenvectors and eigenvalues.
SVD-based PCA: apply Singular Value Decomposition directly to the centred data matrix.
They implement the same underlying PCA geometry when applied to the same centred matrix with consistent conventions. Their principal directions and variance estimates should agree up to numerical precision, sign choices, and differences caused by truncation or numerical tolerances.

The important difference is how the calculation is carried out, including its computational cost and numerical behaviour.

12. Method A — Covariance matrix and eigendecomposition
Step-by-step procedure
Step 1: Centre the data.

X
c
=
X
−
1
n
μ
T
.

Step 2: Form the covariance matrix.

C
=
1
n
−
1
X
c
T
X
c
.

Step 3: Eigendecompose the covariance matrix.

C
=
V
Λ
V
T
.

Step 4: Sort the eigenvalues and their corresponding eigenvectors.

λ
1
≥
λ
2
≥
⋯
≥
λ
p
.

Step 5: Keep the first 
k
 eigenvectors.

V
k
=
[
v
1
,
…
,
v
k
]
.

Step 6: Project the data.

Z
=
X
c
V
k
.

Why this method works
The covariance matrix encodes the variance and covariance of all features. Its eigenvectors are the directions that diagonalise that covariance structure. Sorting the eigenvalues ranks those directions by variance.

In the principal-component coordinate system, the covariance matrix becomes diagonal:

V
T
C
V
=
Λ
.

The diagonal entries are the variances of the principal-component scores.

Strengths
The intermediate covariance matrix is explicit and easy to inspect.
The derivation closely matches many introductory statistics and linear-algebra explanations of PCA.
It is a useful teaching implementation because it makes the relationship between covariance, eigenvectors, eigenvalues, and explained variance visible.
It can be effective when the number of features 
p
 is moderate and forming a 
p
×
p
 matrix is affordable.
Weaknesses
It must form a 
p
×
p
 covariance matrix, which requires memory proportional to 
p
2
.
Forming 
X
c
T
X
c
 costs roughly 
O
(
n
p
2
)
 operations for a dense matrix.
The eigendecomposition of a dense 
p
×
p
 matrix costs roughly 
O
(
p
3
)
 operations.
Forming the Gram/covariance matrix can worsen numerical conditioning: in the nonzero singular spectrum, the condition number of 
X
c
T
X
c
 is the square of the condition number of 
X
c
.
The last point matters when singular values differ greatly in magnitude. Small components can become less accurately estimated when the covariance matrix is formed explicitly.

When it is a sensible choice
The dataset has a manageable number of features.
You want to teach or inspect covariance structure directly.
You need the full covariance matrix for other analysis.
You want a straightforward from-scratch implementation and can afford the intermediate matrix.
13. Method B — Singular Value Decomposition (SVD)
SVD factorises a matrix into three matrices. Applied to the centred data:

X
c
=
U
Σ
V
T
.

The columns of 
U
 describe directions in observation space, the columns of 
V
 describe directions in feature space, and 
Σ
 contains the non-negative singular values. The principal directions are the columns of 
V
.

For a thin SVD, only the singular vectors needed to represent the matrix are returned. The exact shapes depend on the SVD variant and on whether 
n
 or 
p
 is smaller.

Step-by-step procedure
Step 1: Centre the data.

X
c
=
X
−
1
n
μ
T
.

Step 2: Compute its SVD.

X
c
=
U
Σ
V
T
.

Step 3: Read the principal directions from 
V
.

Keep the first 
k
 columns:

V
k
=
[
v
1
,
…
,
v
k
]
.

Step 4: Calculate component scores.

Z
=
X
c
V
k
.

Equivalently, using the SVD factors, the scores are the corresponding columns of 
U
 multiplied by their singular values.

Step 5: Calculate the component variances.

If 
σ
i
 is the $i$th singular value, then:

λ
i
=
σ
i
2
n
−
1
.

The explained-variance ratios can then be calculated from these variances.

Why SVD gives PCA
Start with the covariance matrix:

C
=
X
c
T
X
c
n
−
1
.

Substitute 
X
c
=
U
Σ
V
T
:

C
=
V
Σ
T
U
T
U
Σ
V
T
n
−
1
.

The columns of 
U
 are orthonormal, so 
U
T
U
=
I
 for the corresponding thin/full decomposition. Thus:

C
=
V
Σ
T
Σ
n
−
1
V
T
.

For the usual diagonal singular-value matrix, this gives:

λ
i
=
σ
i
2
n
−
1
.

So the right singular vectors are covariance eigenvectors, and the squared singular values divided by 
n
−
1
 are the corresponding covariance eigenvalues.

Strengths
It works directly on the centred data matrix and does not need to explicitly construct the covariance matrix.
It is often numerically more stable than forming 
X
c
T
X
c
 first.
It can be attractive when the data has many features, especially when a suitable thin, truncated, or iterative SVD implementation is available.
It provides singular values that directly reveal the relative scale of the principal directions.
Weaknesses
A full SVD can still be expensive for a very large dense matrix.
SVD implementations and solver choices may be more complex to explain.
If only a small number of components is needed, a full exact SVD may do more work than necessary; truncated or randomized methods may be better.
The best algorithm depends on matrix shape, sparsity, required precision, and hardware.
When it is a sensible choice
Numerical stability is important.
You want to avoid explicitly forming a covariance matrix.
You have a large or high-dimensional dataset and can use an appropriate SVD solver.
You want a strong general-purpose foundation for practical PCA implementations.
14. Covariance PCA vs SVD PCA — direct comparison
Property	Covariance/eigenvalue PCA	SVD-based PCA
Matrix decomposed	Covariance matrix 
C
Centred data matrix 
X
c
Main factorisation	
C
=
V
Λ
V
T
X
c
=
U
Σ
V
T
Principal directions	Eigenvectors of 
C
Right singular vectors of 
X
c
Variances	Eigenvalues 
λ
i
σ
i
2
/
(
n
−
1
)
Explicit covariance matrix	Yes	No
Extra matrix memory	Often 
O
(
p
2
)
 for 
C
Depends on SVD algorithm and returned factors
Numerical conditioning	Forming 
X
c
T
X
c
 squares the nonzero-spectrum condition number	Usually avoids that squaring step
Best teaching value	Makes covariance/eigenvalue interpretation explicit	Shows a general matrix factorisation and its connection to PCA
Often suitable for	Moderate feature counts; when covariance is useful	General-purpose numerical PCA; high-dimensional cases with suitable solvers
Are results mathematically equivalent?	Yes, under matching assumptions and conventions	Yes, under matching assumptions and conventions
Which one is faster?
There is no universal winner. Performance depends on 
n
, 
p
, whether the matrix is dense or sparse, whether all components or only a few are needed, the numerical library, and the hardware.

For a dense 
n
×
p
 matrix:

covariance formation costs approximately 
O
(
n
p
2
)
 and its full eigendecomposition approximately 
O
(
p
3
)
;
a conventional full SVD has a cost that depends on the matrix shape, often described in broad terms as cubic in the smaller dimension times the larger dimension for common dense algorithms.
If 
p
 is small and 
n
 is very large, covariance PCA can be practical. If 
p
 is very large, the 
p
×
p
 covariance matrix can become the bottleneck. When 
k
 is small, truncated or randomized SVD may avoid computing every component.

A useful rule of thumb is: use covariance eigendecomposition to make the covariance-based derivation explicit; use a stable SVD solver as a strong default for numerical work, unless the data shape or performance requirements suggest another solver. Benchmark on the actual workload when speed matters.

15. Scaling and standardisation
PCA on the covariance matrix is sensitive to feature units. Suppose one feature is measured between 0 and 1 and another between 0 and 1,000. The second feature may dominate the covariance structure simply because of its scale.

Standardisation transforms a feature value using:

z
=
x
−
μ
s
,

where 
μ
 is the feature mean and 
s
 is its standard deviation, estimated on the training data.

After standardisation, each non-constant feature has approximately zero mean and unit variance. PCA on standardised data is equivalent to performing PCA on a correlation-scaled representation rather than allowing raw feature variances to set the scale.

When should you standardise?
Standardisation is often appropriate when:

features use different units, such as metres, kilograms, and currency;
the relative scale of each feature should not determine its influence;
all features should begin with comparable variance.
It may be inappropriate when absolute variance is meaningful and should determine the directions, or when scaling would erase a meaningful distinction in feature reliability. Constant features require special handling because their standard deviation is zero.

Any imputation, scaling, or PCA fit used in a predictive workflow should be learned from the training split only and then applied to validation/test data.

16. PCA is not feature selection
Feature selection retains a subset of the original columns. PCA usually creates new features by combining the original columns.

For example, the first principal component may be:

PC
1
=
a
1
X
1
+
a
2
X
2
+
⋯
+
a
p
X
p
,

where the coefficients 
a
j
 are entries of the first principal direction.

This means PC1 can combine information from every original feature. The coefficients are often called loadings (terminology varies by convention and scaling).

Feature selection	PCA
Keeps selected original variables	Creates linear combinations of variables
Often preserves direct interpretability of chosen columns	Components may be harder to interpret
Can use labels or domain knowledge, depending on the method	Standard PCA does not use target labels
Does not necessarily decorrelate retained variables	Principal-component scores are uncorrelated in the training sample
PCA is useful when compactness and redundancy reduction matter. Feature selection may be preferable when domain interpretability, measurement cost, or a requirement to retain original variables is central.

17. Real-world applications
17.1 Image compression
An image can be represented as a matrix of pixel values. PCA can approximate that matrix using a limited number of principal directions:

$$ \widehat{X} = X_cV_kV_k^{\mathsf T}

\mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}. $$
The smaller 
k
 is, the fewer component scores need to be stored, but the reconstruction generally loses more detail. Larger 
k
 tends to improve fidelity while reducing the compression benefit.

For colour images, each channel may be processed separately or the image may be represented using another matrix arrangement. The quality/cost trade-off depends on the representation.

Practical caveat: PCA compression is a teaching and analytical technique, not always the best production image codec. Modern image codecs exploit spatial structure and perceptual properties in ways that plain PCA does not.

17.2 Visualising high-dimensional datasets
A dataset with dozens or hundreds of features cannot be plotted directly in ordinary 2D space. PCA can map it to two or three components and make broad patterns visible.

Examples include exploratory analysis of sensor readings, customer measurements, or biological samples.

Caveat: separation in a PC1-versus-PC2 plot is not proof of class separability, and patterns in discarded dimensions will not be visible.

17.3 Genomics and biological measurements
Gene-expression and other biological datasets can have thousands of measured variables but comparatively few samples. PCA is used to explore dominant patterns, identify possible batch effects, inspect sample similarity, and visualise broad variation.

Caveat: a component can reflect technical variation, sample preparation, or other confounders rather than the biological process of interest. Interpretation requires domain knowledge and careful experimental design.

17.4 Finance and market data
PCA can summarise correlated movements across asset returns, interest-rate curves, or other financial variables. A small number of components may represent broad common movement and additional patterns.

For example, multiple interest-rate maturities often move together; principal components can provide a compact description of level-like, slope-like, and curvature-like changes.

Caveat: components are estimated from a particular dataset and time period. Their meaning can change over time, and PCA alone is not a trading strategy or a guarantee of risk reduction.

17.5 Industrial monitoring and sensor systems
Factories and buildings may collect readings from many correlated sensors. PCA can compress those readings, reveal dominant operating patterns, and support monitoring for unusual deviations from a baseline.

A common approach is to inspect component scores and reconstruction residuals. Observations that differ strongly from the patterns learned during normal operation may merit investigation.

Caveat: a large residual is an alerting signal, not a diagnosis of the cause. Sensor drift, missing values, and changes in operating conditions need separate treatment.

17.6 Telecommunications and signal processing
Measurements from multiple channels can be correlated. PCA can transform them into orthogonal coordinates and reduce dimensionality before other processing steps. Related eigenvector methods are used in areas such as array processing and signal analysis.

Caveat: the exact method used in a signal-processing system may be a specialised variant rather than ordinary textbook PCA.

17.7 Machine-learning preprocessing
PCA can reduce the input dimension before a downstream model. This may lower storage requirements, speed up training, and reduce redundant directions.

It can be useful for models that benefit from compact, less-correlated inputs, but the result is model- and dataset-dependent.

Important: fit PCA inside the training pipeline. Choosing components based on the entire dataset before evaluating a model can leak information from the test set. Always compare against a baseline without PCA using suitable validation.

17.8 Text and document representations
PCA can be applied to numerical document-feature matrices or dense embeddings to find directions of high variance and create compact representations. In practice, sparse text matrices often call for specialised methods such as truncated SVD rather than explicitly constructing a dense covariance matrix.

Caveat: PCA is not the same as Latent Semantic Analysis (LSA), though truncated SVD of a term-document matrix is a central technique in LSA. Centring sparse matrices can also destroy sparsity if done explicitly, so the choice of algorithm matters.

17.9 Quality control and anomaly exploration
PCA can model the dominant variation of a process. An observation with an unusual score pattern or a large reconstruction residual may be worth inspecting.

This is useful as an exploratory or monitoring tool, but the thresholds must be calibrated to the process and the cost of false alarms.

18. Limitations and common misunderstandings
PCA is linear. It cannot generally represent curved or nonlinear structure with a small number of linear components.
High variance is not the same as usefulness. PCA does not know which features predict a target.
Scaling matters. Raw units can dominate the result.
Outliers can strongly influence the mean and covariance. Robust preprocessing or robust PCA variants may be needed.
Components may be difficult to interpret. Each component can combine many original variables.
Results depend on the data used to fit PCA. A new population or time period may have a different covariance structure.
Near-equal eigenvalues can make individual directions unstable. The subspace can be stable even when the ordering or orientation of individual components changes.
PCA does not automatically improve model accuracy. It should be validated against the task.
Missing values need explicit handling. Ordinary PCA implementations generally expect a complete numerical matrix.
PCA is not inherently a causal method. A component describes a pattern of variation, not a cause.
19. Computational complexity in practice
Let 
X
c
 be a dense 
n
×
p
 matrix.

Covariance/eigenvalue route
Centre the data: 
O
(
n
p
)
.
Form 
X
c
T
X
c
: approximately 
O
(
n
p
2
)
.
Store covariance matrix: 
O
(
p
2
)
 memory.
Full eigendecomposition: approximately 
O
(
p
3
)
.
These broad estimates assume dense linear algebra and do not include implementation-specific optimisations.

SVD route
The cost depends on the matrix shape and solver:

A full dense SVD can be expensive for large 
n
 and 
p
.
A thin SVD avoids storing unnecessary full orthogonal matrices.
A truncated SVD computes only a requested number of dominant directions.
Randomised or iterative methods can be useful when only a small 
k
 is required and approximate solutions are acceptable.
Shape matters
If 
p
 is moderate, covariance PCA may be efficient and easy to interpret.
If 
p
 is extremely large, constructing a 
p
×
p
 matrix can be impractical.
If 
n
 is much smaller than 
p
, methods that exploit the smaller dimension or use truncated decompositions can be advantageous.
Sparse datasets require methods that preserve or exploit sparsity; naïvely centring a sparse matrix can make it dense.
Choose the solver based on matrix shape, memory, precision requirements, and the number of components needed—not solely on a blanket claim that one method is always faster.

20. What the implementation does
The PCA Lab implementation exposes the main stages of the algorithm.

PCAFromScratch.fit(X)
The fitting stage typically:

converts input into floating-point data;
calculates and stores feature means;
centres the matrix;
calculates the covariance eigendecomposition or SVD;
orders principal directions by decreasing variance;
stores the component variances;
calculates explained-variance ratios and their cumulative sums.
The means and principal directions learned here must be reused when transforming new observations.

transform(X, k)
The transformation stage centres new data with the stored training means and projects it onto the first 
k
 components:

Z
=
(
X
−
1
n
μ
T
)
V
k
.

inverse_transform(Z)
The inverse transformation maps scores back to the original feature coordinates and restores the mean:

$$ \widehat{X} = ZV_k^{\mathsf T}

\mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}. $$
The result is exact only if the retained directions span the relevant data space; otherwise it is an approximation.

components_for_variance(threshold)
This operation finds the smallest 
k
 whose cumulative explained-variance ratio reaches the requested threshold. A threshold of 0.95, for example, means selecting enough leading components to account for at least 95% of the fitted data's variance.

reconstruction_metrics(X, X_hat)
This operation may calculate metrics such as MSE and RMSE and estimate a compression ratio. A compression ratio is meaningful only when the storage assumptions are stated—for example, whether component vectors, means, metadata, and numeric precision are included.

21. Practical workflow for applying PCA responsibly
A robust workflow for a tabular machine-learning problem is:

Define the purpose. Is the goal visualisation, compression, noise reduction, or prediction?
Separate training and test data first when evaluating predictive performance.
Inspect the data. Check feature types, units, missingness, outliers, and constant columns.
Impute missing values using a method suitable for the dataset.
Decide whether to standardise. Base this on feature units and the intended meaning of variance.
Fit PCA on the training data only.
Inspect explained variance and component loadings.
Choose 
k
 using variance, reconstruction error, validation performance, or operational constraints.
Transform validation and test data using the fitted means, scales, and components.
Compare against a baseline. Verify that PCA actually helps the task.
Monitor changes over time if PCA is used in a deployed system.
For exploratory visualisation, it can be reasonable to fit PCA to the dataset being explored. For an unbiased predictive evaluation, however, fitting preprocessing on held-out data can leak information.

22. Worked conceptual example
Suppose a dataset contains two highly correlated features: a measurement in centimetres and the same measurement in millimetres. Much of their variation is redundant because one is nearly a constant multiple of the other.

After centring, PCA will find a direction along which the two features vary together and a perpendicular direction that contains little variance. Keeping the first component can represent most of the variation with one number instead of two.

This example illustrates dimensionality reduction, but it also highlights an important practical issue: units and scaling affect PCA. Standardising the two columns changes their relative scale, while the underlying redundancy remains.

In a more realistic dataset, components often combine several correlated measurements rather than duplicating a single quantity.

23. Frequently asked questions
Does PCA require normally distributed data?
No. PCA can be calculated without assuming that the features follow a normal distribution. Normality can matter for some statistical interpretations or probabilistic models built around PCA, but it is not required for the basic linear-algebra calculation.

Must the components be orthogonal?
Standard PCA produces mutually orthogonal component directions. The component scores are uncorrelated in the fitted sample under the usual covariance-based formulation. Uncorrelated does not necessarily mean statistically independent.

Can eigenvalue PCA and SVD PCA produce different answers?
They should agree on the principal subspace and principal variances when applied consistently. Differences may arise from floating-point error, eigenvector sign conventions, nearly repeated eigenvalues, truncated solvers, or inconsistent centring and scaling.

Why can an eigenvector's sign change between runs?
If 
v
 is an eigenvector, then 
−
v
 is also an eigenvector for the same eigenvalue. A sign change flips the corresponding component scores but does not change the represented axis or explained variance.

Does retaining 95% variance mean retaining 95% of the information?
Not in every sense. It means retaining 95% of the variance measured by the fitted PCA representation. Information important for a particular label, rare event, or downstream objective may lie in low-variance directions.

Is SVD always better?
No method is best for every workload. SVD is often a strong numerical default, but covariance eigendecomposition can be practical and educational for moderate feature counts. For very large problems, truncated, iterative, or randomised methods may be more suitable than either full decomposition.

Does PCA make features independent?
No. It makes component scores uncorrelated in the fitted sample under the standard formulation. Statistical independence requires stronger conditions and does not follow from PCA in general.

24. The entire PCA pipeline
The main mathematical sequence is:

X
⟶
X
c
⟶
{
C
=
X
c
T
X
c
n
−
1
(covariance route)
,
\[
8
p
t
]
X
c
=
U
Σ
V
T
(SVD route)

Both routes identify the leading principal directions 
V
k
. The data is then projected:

Z
=
X
c
V
k
,

and optionally reconstructed:

$$ \widehat{X} = ZV_k^{\mathsf T}

\mathbf{1}_n\boldsymbol{\mu}^{\mathsf T}. $$
The goal is not simply to produce fewer columns. It is to choose a compact representation that preserves the variation relevant to the intended use, while understanding the approximation error and computational trade-offs.

25. Final takeaway
PCA is a bridge between geometry, statistics, and linear algebra:

Variance measures spread.
Covariance describes linear co-variation.
Eigenvectors give principal directions of the covariance matrix.
Eigenvalues quantify the variance along those directions.
SVD finds the same principal directions directly from the centred data matrix.
Projection creates a lower-dimensional representation.
Reconstruction shows what was lost by discarding components.
The covariance/eigenvalue method makes the covariance structure explicit and is valuable for learning and moderate-sized feature spaces. SVD avoids explicitly forming the covariance matrix and is often more numerically stable. Both are ways to compute the same underlying PCA model; the practical choice depends on data dimensions, numerical requirements, memory, and whether a full or truncated solution is needed.
""")

elif mode == "Image Lab":
    st.header("🖼️ Image PCA Laboratory")
    uploaded = st.file_uploader("Upload a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        X = image_to_matrix(image)
        n_samples, n_features = X.shape

        st.write(f"**Image matrix:** {n_samples} samples × {n_features} features")

        with st.spinner("Computing PCA spectrum..."):
            start = time.perf_counter()
            pca = PCAFromScratch(method="eigen" if method == "Eigenvalue PCA" else "svd")
            pca.fit(X)
            elapsed = time.perf_counter() - start

        default_k = min(50, n_features)
        threshold = st.slider("Target variance", 0.50, 0.999, 0.95, 0.001)
        auto_k = pca.components_for_variance(threshold)

        k = st.slider(
            "Number of principal components (k)",
            1, n_features, min(default_k, n_features), 1
        )

        st.info(f"At least **{auto_k} components** are required to retain {threshold:.1%} of variance.")

        X_hat = pca.inverse_transform(pca.transform(X, k=k))
        metrics = reconstruction_metrics(X, X_hat)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variance retained", f"{pca.cumulative_explained_variance_[k-1]:.2%}")
        c2.metric("MSE", f"{metrics['mse']:.3f}")
        c3.metric("RMSE", f"{metrics['rmse']:.3f}")
        c4.metric("Approx. compression", f"{metrics['compression_ratio']:.2f}×")

        original = image.convert("L")
        reconstructed = Image.fromarray(np.clip(X_hat, 0, 255).astype(np.uint8))

        left, right = st.columns(2)
        with left:
            st.subheader("Original")
            st.image(original, use_container_width=True)
        with right:
            st.subheader(f"Reconstructed — k={k}")
            st.image(reconstructed, use_container_width=True)

        st.subheader("Variance analysis")
        st.pyplot(variance_plot(pca.explained_variance_ratio_, pca.cumulative_explained_variance_))

        st.subheader("Reconstruction error")
        ks = sorted(set([1, min(5, n_features), min(10, n_features), min(25, n_features),
                         min(50, n_features), min(100, n_features), k]))
        errors = []
        for kk in ks:
            pred = pca.inverse_transform(pca.transform(X, k=kk))
            errors.append(reconstruction_metrics(X, pred)["mse"])
        st.pyplot(reconstruction_plot(ks, errors))

        with st.expander("🔬 PCA Microscope — inspect the mathematics"):
            st.write("Mean vector (first 10 values)")
            st.code(np.array2string(pca.mean_[:10], precision=3))
            st.write("Largest eigenvalues / singular-value-derived variances")
            st.dataframe(pd.DataFrame({
                "Component": np.arange(1, min(10, len(pca.explained_variance_)) + 1),
                "Explained variance": pca.explained_variance_[:10],
                "Explained variance ratio": pca.explained_variance_ratio_[:10],
                "Cumulative ratio": pca.cumulative_explained_variance_[:10],
            }), use_container_width=True)
            st.write(f"Computation time: **{elapsed:.4f} seconds**")

else:
    st.header("📊 CSV / Dataset Laboratory")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Dataset overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", len(df))
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing values", int(df.isna().sum().sum()))

        st.dataframe(df.head(10), use_container_width=True)

        numeric = df.select_dtypes(include=np.number)
        if numeric.shape[1] < 2:
            st.error("The CSV needs at least two numeric columns for PCA.")
            st.stop()

        scale = st.checkbox("Standardize features (zero mean, unit variance)", value=True)
        X, feature_names = prepare_numeric_data(numeric, standardize=scale)

        pca = PCAFromScratch(method="eigen" if method == "Eigenvalue PCA" else "svd")
        pca.fit(X)

        max_k = min(X.shape)
        threshold = st.slider("Target variance", 0.50, 0.999, 0.95, 0.001)
        auto_k = pca.components_for_variance(threshold)
        k = st.slider("Components", 1, max_k, min(auto_k, max_k))

        Z = pca.transform(X, k=k)
        X_hat = pca.inverse_transform(Z)
        metrics = reconstruction_metrics(X, X_hat)

        c1, c2, c3 = st.columns(3)
        c1.metric("Components", k)
        c2.metric("Variance retained", f"{pca.cumulative_explained_variance_[k-1]:.2%}")
        c3.metric("Reconstruction RMSE", f"{metrics['rmse']:.4f}")

        st.pyplot(variance_plot(pca.explained_variance_ratio_, pca.cumulative_explained_variance_))

        if k >= 2:
            st.subheader("Data in principal-component space")
            st.pyplot(scatter_plot(Z[:, 0], Z[:, 1], "PC1", "PC2"))

        st.subheader("Principal component loadings")
        loadings = pd.DataFrame(
            pca.components_[:k].T,
            index=feature_names,
            columns=[f"PC{i+1}" for i in range(k)]
        )
        st.dataframe(loadings, use_container_width=True)

        st.download_button(
            "Download reduced dataset",
            pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(k)]).to_csv(index=False),
            "pca_reduced_dataset.csv",
            "text/csv"
        )
