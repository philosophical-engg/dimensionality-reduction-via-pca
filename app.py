import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from pca.core import PCAFromScratch
from pca.metrics import reconstruction_metrics
from pca.preprocessing import prepare_numeric_data
from visualization.plots import variance_plot, reconstruction_plot, scatter_plot
from utils.data_loader import image_to_matrix

THEORY_FILE = Path(__file__).parent / "PCA_THEORY.md"


def load_theory():
    """Load the PCA theory from the Markdown file."""
    if not THEORY_FILE.is_file():
        return None
    return THEORY_FILE.read_text(encoding="utf-8")


# --- Pure-NumPy Helper Functions for Advanced Analytics Tabs (Zero Extra Dependencies) ---

def kmeans_numpy(X, n_clusters=3, max_iter=100, seed=42):
    """Lightweight K-Means in pure NumPy so no external ML library is required."""
    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]
    if n_samples < n_clusters:
        return np.zeros(n_samples, dtype=int)
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X[indices].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X[mask].mean(axis=0)
    return labels


def colored_latent_scatter(pc1, pc2, groups, group_label="Group"):
    """Plot 2D PCA projection color-coded by cluster or target category."""
    fig, ax = plt.subplots(figsize=(7, 4.8))
    unique_groups = pd.unique(groups)

    # If continuous/many unique values, use a continuous colormap
    if pd.api.types.is_numeric_dtype(groups) and len(unique_groups) > 12:
        sc = ax.scatter(pc1, pc2, c=groups, cmap="viridis", alpha=0.8, edgecolors="k", linewidth=0.3)
        fig.colorbar(sc, ax=ax, label=group_label)
    else:
        cmap = plt.get_cmap("tab10")
        for idx, g in enumerate(unique_groups[:10]):
            mask = (groups == g)
            ax.scatter(
                pc1[mask], pc2[mask],
                label=str(g),
                color=cmap(idx % 10),
                alpha=0.8,
                edgecolors="white",
                linewidth=0.4,
                s=45
            )
        ax.legend(title=group_label, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"2D Principal Subspace Projection (Colored by {group_label})")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def evaluate_ridge_benchmark(X_raw, X_pca, y_series):
    """
    Pure-NumPy 5-Fold Cross-Validation comparing downstream model performance
    on Raw High-Dimensional Features vs. PCA-Compressed Features.
    """
    # Clean target variable
    valid_mask = ~y_series.isna()
    X_r = X_raw[valid_mask]
    X_p = X_pca[valid_mask]
    y_clean = y_series[valid_mask]

    is_classification = (not pd.api.types.is_numeric_dtype(y_clean)) or (y_clean.nunique() <= 10)

    if is_classification:
        classes, y_encoded = np.unique(y_clean.astype(str), return_inverse=True)
        n_classes = len(classes)
        Y_mat = np.eye(n_classes)[y_encoded]
        metric_name = "5-Fold CV Accuracy"
    else:
        y_vals = y_clean.to_numpy(dtype=float)
        Y_mat = y_vals[:, None]
        metric_name = "5-Fold CV R² Score"

    n = len(X_r)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    n_folds = min(5, n)
    folds = np.array_split(indices, n_folds)

    def run_cv(Features):
        t0 = time.perf_counter()
        scores = []
        # Run multiple passes to get stable timing on smaller datasets
        for _ in range(10):
            fold_scores = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.hstack([folds[j] for j in range(n_folds) if j != i])

                X_tr, Y_tr = Features[train_idx], Y_mat[train_idx]
                X_va, Y_va = Features[val_idx], Y_mat[val_idx]

                # Add bias term
                X_tr_b = np.column_stack([np.ones(len(X_tr)), X_tr])
                X_va_b = np.column_stack([np.ones(len(X_va)), X_va])

                # Closed-form Ridge solver: (X^T X + alpha*I)^(-1) X^T Y
                reg = 1.0 * np.eye(X_tr_b.shape[1])
                reg[0, 0] = 0.0
                W = np.linalg.solve(X_tr_b.T @ X_tr_b + reg, X_tr_b.T @ Y_tr)
                preds = X_va_b @ W

                if is_classification:
                    pred_cls = np.argmax(preds, axis=1)
                    true_cls = np.argmax(Y_va, axis=1)
                    fold_scores.append(np.mean(pred_cls == true_cls))
                else:
                    ss_res = np.sum((Y_va - preds) ** 2)
                    ss_tot = np.sum((Y_va - np.mean(Y_va)) ** 2)
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
                    fold_scores.append(r2)
            scores = fold_scores
        elapsed_ms = ((time.perf_counter() - t0) / 10.0) * 1000.0
        return float(np.mean(scores)), elapsed_ms

    raw_score, raw_time = run_cv(X_r)
    pca_score, pca_time = run_cv(X_p)
    return metric_name, is_classification, raw_score, raw_time, pca_score, pca_time


# --- Streamlit Page Configuration ---

st.set_page_config(page_title="PCA Lab", page_icon="📐", layout="wide")

st.title("📐 PCA Lab & Subspace Analytics Workbench")
st.caption(
    "Interactive Principal Component Analysis, dimensionality reduction, anomaly detection, "
    "and image compression — implemented from first principles."
)

with st.sidebar:
    st.header("Experiment")
    mode = st.radio("Choose a laboratory", ["Image Lab", "CSV / Dataset Lab", "PCA Theory"])
    method = st.selectbox("PCA implementation", ["Eigenvalue PCA", "SVD PCA"])
    st.divider()
    st.info(
        "This project deliberately implements the PCA mathematics with NumPy "
        "instead of using sklearn's PCA as the main engine."
    )

# ==========================================
# MODE 1: PCA THEORY
# ==========================================
if mode == "PCA Theory":
    st.header("📚 Principal Component Analysis — First Principles")
    theory_content = load_theory()
    if theory_content:
        st.markdown(theory_content)
    else:
        st.markdown(
            """
            ### 1. Mathematical Formulation
            Given a mean-centered data matrix $X \\in \\mathbb{R}^{n \\times p}$, Principal Component Analysis seeks an
            orthonormal basis $V_k = [v_1, v_2, \\dots, v_k] \\in \\mathbb{R}^{p \\times k}$ that maximizes the projected variance
            or equivalently minimizes the Frobenius reconstruction error:
            $$\\min_{V_k^T V_k = I_k} \\| X - X V_k V_k^T \\|_F^2$$

            ### 2. Eigenvalue Decomposition vs. Singular Value Decomposition (SVD)
            * **Spectral Covariance Method:** Computes the sample covariance matrix $C = \\frac{1}{n-1}X^T X$ and solves
              $C v_i = \\lambda_i v_i$.
            * **Direct SVD Method:** Factorizes $X = U \\Sigma V^T$, where the right singular vectors $V$ form the principal directions
              and the eigenvalues relate to singular values via $\\lambda_i = \\frac{\\sigma_i^2}{n - 1}$.

            ### 3. Why Subspace Analytics Matters in Practice
            1. **Latent Compression:** Strips collinearity while preserving global geometric structure.
            2. **Unsupervised Anomaly Detection:** Normal samples reside close to the top-$k$ principal subspace; anomalies exhibit
               large residual error $\\|x_i - \\hat{x}_i\\|_2^2$ in the orthogonal complement.
            """
        )

# ==========================================
# MODE 2: IMAGE LAB
# ==========================================
elif mode == "Image Lab":
    st.header("🖼️ Image PCA Laboratory")
    uploaded = st.file_uploader("Upload a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        X = image_to_matrix(image)
        n_samples, n_features = X.shape
        max_k = min(n_samples, n_features)

        st.write(f"**Image matrix:** {n_samples} samples × {n_features} features (Max rank: {max_k})")

        with st.spinner("Computing PCA spectrum..."):
            start = time.perf_counter()
            pca = PCAFromScratch(method="eigen" if method == "Eigenvalue PCA" else "svd")
            pca.fit(X)
            elapsed = time.perf_counter() - start

        default_k = min(50, max_k)
        threshold = st.slider("Target variance", 0.50, 0.999, 0.95, 0.001)
        auto_k = min(pca.components_for_variance(threshold), max_k)

        k = st.slider(
            "Number of principal components (k)",
            1, max_k, min(default_k, max_k), 1
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
        ks = sorted(set([1, min(5, max_k), min(10, max_k), min(25, max_k),
                         min(50, max_k), min(100, max_k), k]))
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

# ==========================================
# MODE 3: CSV / DATASET LAB (UPGRADED WORKBENCH)
# ==========================================
else:
    st.header("📊 CSV / Subspace Analytics Workbench")
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
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            threshold = st.slider("Target variance", 0.50, 0.999, 0.95, 0.001)
            auto_k = min(pca.components_for_variance(threshold), max_k)
            st.caption(f"Recommended components for {threshold:.1%} variance: **{auto_k}**")
        with col_s2:
            k = st.slider("Selected components (k)", 1, max_k, min(auto_k, max_k))

        Z = pca.transform(X, k=k)
        X_hat = pca.inverse_transform(Z)
        metrics = reconstruction_metrics(X, X_hat)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Features", X.shape[1])
        c2.metric("Compressed Components (k)", k, delta=f"-{X.shape[1] - k} dims")
        c3.metric("Variance Retained", f"{pca.cumulative_explained_variance_[k-1]:.2%}")
        c4.metric("Reconstruction RMSE", f"{metrics['rmse']:.4f}")

        st.divider()

        # --- Multi-Tab Diagnostic Workbench ---
        tab_overview, tab_clusters, tab_anomaly, tab_ml = st.tabs([
            "📈 Spectrum & Loadings",
            "🧭 2D Cluster & Pattern Explorer",
            "🚨 Subspace Anomaly Radar",
            "⚡ Downstream ML Benchmark"
        ])

        # TAB 1: Original Spectrum, Loadings & Export
        with tab_overview:
            st.subheader("Explained Variance Spectrum")
            st.pyplot(variance_plot(pca.explained_variance_ratio_, pca.cumulative_explained_variance_))

            st.subheader("Principal Component Feature Loadings")
            st.caption("Inspect how much each original feature contributes to each principal axis.")
            loadings = pd.DataFrame(
                pca.components_[:k].T,
                index=feature_names,
                columns=[f"PC{i+1}" for i in range(k)]
            )
            st.dataframe(loadings.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)

            st.download_button(
                "📥 Download PCA-Reduced Dataset (CSV)",
                pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(k)]).to_csv(index=False),
                "pca_reduced_dataset.csv",
                "text/csv"
            )

        # TAB 2: 2D Cluster & Pattern Discovery
        with tab_clusters:
            st.subheader("2D Latent Space Pattern & Cluster Discovery")
            if max_k < 2:
                st.warning("Need at least 2 numeric features to visualize 2D projections.")
            else:
                # Always grab at least 2 PCs for 2D visualization even if k=1 is selected above
                Z_2d = pca.transform(X, k=2)
                color_mode = st.radio(
                    "Color points by:",
                    ["Auto-Cluster (K-Means in Latent Space)", "Existing Dataset Column", "None (Uniform)"],
                    horizontal=True
                )

                if color_mode == "Auto-Cluster (K-Means in Latent Space)":
                    n_clusters = st.slider("Number of K-Means clusters", 2, 8, 3)
                    cluster_labels = kmeans_numpy(Z_2d, n_clusters=n_clusters)
                    fig_cl = colored_latent_scatter(
                        Z_2d[:, 0], Z_2d[:, 1],
                        pd.Series([f"Cluster {c+1}" for c in cluster_labels]),
                        group_label="Latent Cluster"
                    )
                    st.pyplot(fig_cl)
                    plt.close(fig_cl)

                    # Cluster summary table
                    cluster_summary = numeric.copy()
                    cluster_summary["Cluster"] = [f"Cluster {c+1}" for c in cluster_labels]
                    st.write("**Cluster Centroid Profiles (Mean of Original Features per Cluster):**")
                    st.dataframe(cluster_summary.groupby("Cluster").mean(), use_container_width=True)

                elif color_mode == "Existing Dataset Column":
                    selected_col = st.selectbox("Select column for color encoding", df.columns)
                    fig_col = colored_latent_scatter(
                        Z_2d[:, 0], Z_2d[:, 1],
                        df[selected_col].fillna("Missing").to_numpy(),
                        group_label=selected_col
                    )
                    st.pyplot(fig_col)
                    plt.close(fig_col)
                else:
                    st.pyplot(scatter_plot(Z_2d[:, 0], Z_2d[:, 1], "PC1", "PC2"))

        # TAB 3: Subspace Anomaly Radar
       # TAB 3: Subspace Anomaly Radar
        with tab_anomaly:
            st.subheader("🚨 Unsupervised Subspace Anomaly Detection")
            st.markdown(
                "Normal records lie close to the top-$k$ principal subspace and reconstruct accurately. "
                "**Anomalies and corrupted records** violate the dominant correlation structure and exhibit "
                "high reconstruction error $\\|x_i - \\hat{x}_i\\|^2$."
            )

            # Take real part (in case Eigen PCA produced 0j complex numbers) and sanitize NaNs/Infs
            X_hat_clean = np.nan_to_num(np.real(X_hat), nan=0.0, posinf=0.0, neginf=0.0)
            X_clean = np.nan_to_num(np.real(X), nan=0.0, posinf=0.0, neginf=0.0)

            sq_residuals = (X_clean - X_hat_clean) ** 2
            row_mse = np.mean(sq_residuals, axis=1)
            row_mse = np.nan_to_num(row_mse, nan=0.0, posinf=0.0, neginf=0.0)

            worst_feature_idx = np.argmax(sq_residuals, axis=1)
            worst_feature_names = [feature_names[idx] for idx in worst_feature_idx]

            pct_cutoff = st.slider("Anomaly Alert Percentile Threshold (%)", 90.0, 99.5, 95.0, 0.5)
            cutoff_val = float(np.percentile(row_mse, pct_cutoff))
            is_anomaly = row_mse >= cutoff_val
            n_anomalies = int(np.sum(is_anomaly))

            a1, a2, a3 = st.columns(3)
            a1.metric("Flagged Anomalies", f"{n_anomalies} rows", f"Top {100 - pct_cutoff:.1f}%")
            a2.metric("Error Threshold (MSE)", f"{cutoff_val:.4f}")
            a3.metric("Max Row Error", f"{np.max(row_mse):.4f}")

            # Plot anomaly score distribution safely
            fig_anom, ax_anom = plt.subplots(figsize=(7, 3.2))
            finite_mse = row_mse[np.isfinite(row_mse)]
            if len(finite_mse) > 0 and np.max(finite_mse) > np.min(finite_mse):
                ax_anom.hist(finite_mse, bins=35, color="#4C78A8", edgecolor="white", alpha=0.85)
            else:
                ax_anom.hist(finite_mse, bins=1, color="#4C78A8", edgecolor="white", alpha=0.85)
            ax_anom.axvline(cutoff_val, color="#E45756", linestyle="--", linewidth=2, label=f"{pct_cutoff}% Cutoff")
            ax_anom.set_xlabel("Row Reconstruction Error (MSE)")
            ax_anom.set_ylabel("Record Count")
            ax_anom.set_title("Subspace Reconstruction Error Distribution")
            ax_anom.legend()
            fig_anom.tight_layout()
            st.pyplot(fig_anom)
            plt.close(fig_anom)

            # Build ranked anomaly inspection table
            anom_df = df.copy()
            anom_df.insert(0, "Primary_Deviating_Feature", worst_feature_names)
            anom_df.insert(0, "Reconstruction_MSE", np.round(row_mse, 4))
            flagged_df = anom_df[is_anomaly].sort_values(by="Reconstruction_MSE", ascending=False)

            st.write(f"**Top Flagged Anomalous Records (Ranked by Subspace Deviation at k={k}):**")
            st.dataframe(flagged_df.head(25), use_container_width=True)

        # TAB 4: Downstream ML Benchmark (Before vs After PCA)
        with tab_ml:
            st.subheader("⚡ Downstream Machine Learning Impact Benchmark")
            st.markdown(
                "Compare predictive accuracy and training speed when training a model on **all raw numeric features** "
                "versus your **PCA-compressed features ($k$ components)**."
            )

            target_col = st.selectbox("Select Target Column (y) to Predict", df.columns)

            # Exclude target column from input features X if target is numeric so there is no data leakage
            other_numeric_cols = [c for c in feature_names if c != target_col]

            if len(other_numeric_cols) < 2:
                st.warning("Need at least 2 numeric predictor columns (excluding the target) to run the ML benchmark.")
            else:
                X_bench_raw, _ = prepare_numeric_data(df[other_numeric_cols], standardize=scale)
                pca_bench = PCAFromScratch(method="eigen" if method == "Eigenvalue PCA" else "svd")
                pca_bench.fit(X_bench_raw)

                k_bench = min(k, min(X_bench_raw.shape))
                Z_bench = pca_bench.transform(X_bench_raw, k=k_bench)

                metric_name, is_clf, raw_score, raw_ms, pca_score, pca_ms = evaluate_ridge_benchmark(
                    X_bench_raw, Z_bench, df[target_col]
                )

                task_type = "Classification (Ridge Classifier)" if is_clf else "Regression (Ridge Regressor)"
                st.info(f"**Detected Task:** {task_type} | **Evaluation:** {metric_name}")

                m1, m2, m3 = st.columns(3)
                dim_reduction_pct = (1.0 - (k_bench / X_bench_raw.shape[1])) * 100.0
                speedup = raw_ms / max(pca_ms, 1e-6)

                m1.metric(
                    "Feature Space Reduction",
                    f"{X_bench_raw.shape[1]} → {k_bench} cols",
                    delta=f"-{dim_reduction_pct:.1f}% dimensions"
                )
                m2.metric(
                    f"{metric_name} (PCA vs Raw)",
                    f"{pca_score:.4f}",
                    delta=f"{pca_score - raw_score:+.4f} vs Raw ({raw_score:.4f})"
                )
                m3.metric(
                    "CV Training Time",
                    f"{pca_ms:.2f} ms",
                    delta=f"{speedup:.2f}× speedup (Raw: {raw_ms:.2f} ms)"
                )

                bench_table = pd.DataFrame({
                    "Pipeline": ["1. Raw High-Dimensional Features", f"2. PCA-Compressed Features (k={k_bench})"],
                    "Input Dimensions": [X_bench_raw.shape[1], k_bench],
                    "Variance Retained": ["100.00%", f"{pca_bench.cumulative_explained_variance_[k_bench-1]:.2%}"],
                    metric_name: [round(raw_score, 4), round(pca_score, 4)],
                    "Fit Time (ms)": [round(raw_ms, 2), round(pca_ms, 2)]
                })
                st.table(bench_table)
