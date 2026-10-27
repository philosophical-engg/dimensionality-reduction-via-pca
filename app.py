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
    st.markdown("""
### What is PCA?

Principal Component Analysis (PCA) is an **unsupervised linear dimensionality-reduction technique**.
It transforms correlated variables into a smaller set of new, orthogonal variables called **principal components**.

The first component captures the largest possible amount of variance, the second captures the largest remaining amount, and so on.

### Mathematical pipeline

For a data matrix \(X\):

**1. Mean centering**
\[
X_c = X - \\mu
\]

**2. Covariance matrix**
\[
C = \\frac{1}{n-1}X_c^T X_c
\]

**3. Eigenvalue problem**
\[
Cv_i = \\lambda_i v_i
\]

**4. Keep the first \(k\) components**
\[
V_k = [v_1,v_2,\\ldots,v_k]
\]

**5. Project**
\[
Z = X_cV_k
\]

**6. Reconstruct**
\[
\\hat X = ZV_k^T + \\mu
\]

### Explained variance

Each eigenvalue tells us how much variance its component represents:

\[
EVR_i = \\frac{\\lambda_i}{\\sum_j \\lambda_j}
\]

The cumulative explained variance is:

\[
CEV_k = \\sum_{i=1}^{k} EVR_i
\]

A common practical question is therefore:

> **How many components can we keep while retaining 95% of the information represented by variance?**

### Important limitations

PCA is linear, sensitive to feature scaling and can be influenced by outliers. Principal components can also be harder to interpret than the original features.
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
