# PCA Lab

## Interactive Principal Component Analysis, Dimensionality Reduction & Image Compression

PCA Lab is an educational and experimental implementation of **Principal Component Analysis (PCA)** built with Python, NumPy and Streamlit.

The central goal is not merely to call a machine-learning library and obtain an answer. The project implements the mathematical PCA pipeline explicitly:

**data → centering → covariance / SVD → eigenvectors → component selection → projection → reconstruction → evaluation**

The application provides two laboratories:

- **Image Lab:** visual PCA-based image reconstruction and compression.
- **CSV / Dataset Lab:** PCA on numerical tabular data, with scaling, missing-value handling, component analysis and reduced-data export.

A third section, **PCA Theory**, explains the mathematics used by the implementation.

---

## 1. What is PCA?

Principal Component Analysis is an unsupervised linear dimensionality-reduction method.

Suppose a dataset contains many variables. Some of them may be strongly correlated, meaning that the dataset contains redundant information.

PCA creates a new coordinate system in which:

1. The first principal component captures the maximum possible variance.
2. The second captures the maximum remaining variance while being orthogonal to the first.
3. The process continues for subsequent components.

The original variables are therefore replaced by a smaller number of mathematically constructed variables.

---

## 2. Mathematical derivation

Let the data matrix be

\[
X \in \mathbb{R}^{n\times p}
\]

where \(n\) is the number of observations and \(p\) is the number of features.

### Mean centering

For each feature \(j\):

\[
\mu_j = \frac{1}{n}\sum_{i=1}^{n}X_{ij}
\]

The centered matrix is:

\[
X_c = X-\mu
\]

### Covariance matrix

\[
C = \frac{1}{n-1}X_c^T X_c
\]

The covariance matrix tells us how features vary together.

### Eigenvalue problem

We solve:

\[
Cv_i=\lambda_i v_i
\]

where:

- \(v_i\) is an eigenvector and becomes a principal direction.
- \(\lambda_i\) is its eigenvalue and measures the variance represented by that direction.

The eigenvalues are ordered:

\[
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p
\]

### Projection

Keep the first \(k\) principal directions:

\[
V_k=[v_1,v_2,\ldots,v_k]
\]

Then project:

\[
Z=X_cV_k
\]

The matrix \(Z\) is the lower-dimensional representation.

### Reconstruction

An approximation of the original data is:

\[
\hat X=ZV_k^T+\mu
\]

As \(k\) increases, more information is retained and reconstruction usually improves.

---

## 3. Explained variance

For component \(i\):

\[
EVR_i=\frac{\lambda_i}{\sum_j\lambda_j}
\]

Cumulative explained variance:

\[
CEV_k=\sum_{i=1}^{k}EVR_i
\]

For example, if the first five components explain 96% of the variance, the data can be represented using five dimensions while retaining 96% of the variance captured by the original representation.

The application automatically finds the smallest \(k\) that reaches a selected variance threshold such as 90%, 95% or 99%.

---

## 4. Reconstruction error

The project measures information lost during reconstruction.

\[
MSE=\frac{1}{N}\sum_i(x_i-\hat{x}_i)^2
\]

and

\[
RMSE=\sqrt{MSE}
\]

Lower reconstruction error generally means that the reduced representation preserves the original data more accurately.

---

## 5. Eigenvalue PCA vs SVD PCA

The project contains two implementations.

### Covariance/eigenvalue approach

1. Center the matrix.
2. Compute \(C=X_c^TX_c/(n-1)\).
3. Calculate eigenvalues and eigenvectors.
4. Sort them.
5. Select the leading components.

### SVD approach

The centered matrix is decomposed as:

\[
X_c=U\Sigma V^T
\]

The rows of \(V^T\) provide the principal directions.

The explained variances are obtained from the singular values:

\[
\lambda_i=\frac{\sigma_i^2}{n-1}
\]

The two implementations should produce the same principal variances up to numerical precision.

---

## 6. Image compression

For an image, each row can be treated as an observation and each column as a feature.

PCA finds directions describing major spatial patterns in the image.

Keeping only \(k\) components gives:

\[
\hat X = X_cV_kV_k^T+\mu
\]

Small \(k\) produces stronger compression but more visible reconstruction loss.

Large \(k\) produces a reconstruction closer to the original.

This creates a useful experimental trade-off:

**number of components ↔ information retained ↔ reconstruction quality ↔ storage**

---

## 7. Tabular-data laboratory

The CSV laboratory:

- accepts numerical CSV datasets;
- reports dataset dimensions and missing values;
- performs median imputation where required;
- optionally standardizes features;
- computes PCA;
- displays explained variance;
- visualizes PC1 vs PC2;
- displays component loadings;
- exports the reduced dataset.

Standardization transforms a feature using:

\[
z=\frac{x-\mu}{\sigma}
\]

This is important because PCA is sensitive to feature scale.

---

## 8. Why PCA is useful

PCA can be used for:

- dimensionality reduction;
- visualization;
- noise reduction;
- data compression;
- feature engineering;
- exploratory data analysis;
- preprocessing before some machine-learning algorithms.

It is not, however, automatically a method for improving predictive accuracy.

---

## 9. Limitations

PCA has important limitations:

- It is fundamentally linear.
- It can be sensitive to feature scaling.
- Outliers can strongly influence the covariance structure.
- Principal components can be difficult to interpret.
- Variance is not always equivalent to predictive usefulness.
- PCA does not automatically improve every machine-learning model.

---

## 10. Project architecture

```text
pca-project/
├── app.py
├── pca/
│   ├── core.py
│   ├── metrics.py
│   └── preprocessing.py
├── visualization/
│   └── plots.py
├── utils/
│   └── data_loader.py
├── tests/
│   └── test_pca.py
├── docs/
│   └── PCA_THEORY.md
├── examples/
├── requirements.txt
└── README.md
```

The separation keeps the mathematics, user interface, plotting and utilities independent.

---

## 11. Running the project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest
```

---

## 12. Educational objective

The project is designed to make PCA observable rather than treating it as a black box.

A user can move from:

**original data**

to

**centered data**

to

**covariance structure**

to

**principal directions**

to

**reduced representation**

to

**reconstruction**

and finally quantify how much information was retained.

This makes the project both a software application and a practical demonstration of linear algebra, statistics and unsupervised machine learning.
