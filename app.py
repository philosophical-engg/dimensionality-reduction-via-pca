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
    """Load the PCA theory from the Markdown file or return comprehensive fallback theory."""
    if THEORY_FILE.is_file():
        text = THEORY_FILE.read_text(encoding="utf-8")
        if text.strip():
            return text
    return None


# ==========================================
# PURE-NUMPY ADVANCED ANALYTICS HELPERS
# ==========================================

def kmeans_numpy(X, n_clusters=3, max_iter=100, seed=42):
    """Lightweight K-Means in pure NumPy so no external ML library is required."""
    X_clean = np.nan_to_num(np.real(X), nan=0.0, posinf=0.0, neginf=0.0)
    rng = np.random.default_rng(seed)
    n_samples = X_clean.shape[0]
    if n_samples < n_clusters:
        return np.zeros(n_samples, dtype=int)
    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X_clean[indices].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        dists = np.linalg.norm(X_clean[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for c in range(n_clusters):
            mask = labels == c
            if np.any(mask):
                centroids[c] = X_clean[mask].mean(axis=0)
    return labels


def colored_latent_scatter(
    pc1, pc2, groups, group_label="Group",
    components_2d=None, feature_names=None, top_n_vectors=5, highlight_point=None
):
    """Plot 2D PCA projection with optional Biplot feature vectors and counterfactual marker."""
    pc1 = np.nan_to_num(np.real(pc1), nan=0.0)
    pc2 = np.nan_to_num(np.real(pc2), nan=0.0)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    unique_groups = pd.unique(groups)

    if pd.api.types.is_numeric_dtype(groups) and len(unique_groups) > 12:
        sc = ax.scatter(pc1, pc2, c=groups, cmap="viridis", alpha=0.75, edgecolors="k", linewidth=0.3, s=40)
        fig.colorbar(sc, ax=ax, label=group_label)
    else:
        cmap = plt.get_cmap("tab10")
        for idx, g in enumerate(unique_groups[:10]):
            mask = (groups == g)
            ax.scatter(
                pc1[mask], pc2[mask],
                label=str(g),
                color=cmap(idx % 10),
                alpha=0.78,
                edgecolors="white",
                linewidth=0.4,
                s=45
            )
        ax.legend(title=group_label, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    if components_2d is not None and feature_names is not None:
        comp = np.nan_to_num(np.real(components_2d), nan=0.0)
        scale_x = np.percentile(np.abs(pc1), 90) if len(pc1) > 0 else 1.0
        scale_y = np.percentile(np.abs(pc2), 90) if len(pc2) > 0 else 1.0
        arrow_scale = max(min(scale_x, scale_y) * 1.15, 1e-3)

        mags = np.hypot(comp[0], comp[1])
        top_idx = np.argsort(mags)[::-1][:min(top_n_vectors, len(feature_names))]

        for i in top_idx:
            vx = comp[0, i] * arrow_scale / (np.max(mags) + 1e-12)
            vy = comp[1, i] * arrow_scale / (np.max(mags) + 1e-12)
            ax.annotate(
                "", xy=(vx, vy), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="#D62728", lw=1.8, alpha=0.9)
            )
            ax.text(
                vx * 1.08, vy * 1.08, feature_names[i],
                color="#B2182B", fontsize=8.5, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#D62728", alpha=0.75, lw=0.5)
            )

    if highlight_point is not None:
        orig_pt, mod_pt = highlight_point
        ax.scatter([orig_pt[0]], [orig_pt[1]], color="#111111", s=130, marker="o", zorder=9, label="Selected Row (Original)")
        ax.scatter([mod_pt[0]], [mod_pt[1]], color="#FFD700", edgecolors="black", s=220, marker="*", zorder=10, label="What-If Modified (★)")
        ax.annotate(
            "", xy=(mod_pt[0], mod_pt[1]), xytext=(orig_pt[0], orig_pt[1]),
            arrowprops=dict(arrowstyle="->", color="black", lw=2.0, ls="--")
        )
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)

    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.axvline(0, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("Principal Component 1 (PC1)")
    ax.set_ylabel("Principal Component 2 (PC2)")
    ax.set_title(f"2D Latent Subspace Projection ({group_label})")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def evaluate_ridge_benchmark(X_raw, X_pca, y_series):
    """Pure-NumPy 5-Fold Cross-Validation comparing Raw vs. PCA-Compressed Features."""
    X_raw = np.nan_to_num(np.real(X_raw), nan=0.0, posinf=0.0, neginf=0.0)
    X_pca = np.nan_to_num(np.real(X_pca), nan=0.0, posinf=0.0, neginf=0.0)

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
        Y_mat = np.nan_to_num(y_vals[:, None], nan=0.0)
        metric_name = "5-Fold CV R² Score"

    n = len(X_r)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)
    n_folds = min(5, n)
    folds = np.array_split(indices, n_folds)

    def run_cv(Features):
        t0 = time.perf_counter()
        scores = []
        for _ in range(10):
            fold_scores = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.hstack([folds[j] for j in range(n_folds) if j != i])

                X_tr, Y_tr = Features[train_idx], Y_mat[train_idx]
                X_va, Y_va = Features[val_idx], Y_mat[val_idx]

                X_tr_b = np.column_stack([np.ones(len(X_tr)), X_tr])
                X_va_b = np.column_stack([np.ones(len(X_va)), X_va])

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


def marchenko_pastur_analysis(X, eigenvalues):
    """Random Matrix Theory noise filter."""
    n, p = X.shape
    gamma = p / max(n, 1)
    sigma2 = float(np.mean(eigenvalues)) if len(eigenvalues) > 0 else 1.0
    lambda_plus = sigma2 * ((1.0 + np.sqrt(gamma)) ** 2)
    signal_mask = eigenvalues > lambda_plus
    n_signal = int(np.sum(signal_mask))
    return lambda_plus, max(n_signal, 1)


# ==========================================
# DATA SANITIZER & HYGIENE LABORATORY
# ==========================================

def render_data_sanitizer_lab():
    st.header("🧹 Data Hygiene & ML-Ready Sanitizer Lab")
    st.markdown(
        "Upload a raw, 'infected' dataset containing common real-world data flaws "
        "(missing values, infinite values, mixed types, extreme outliers, or unscaled ranges). "
        "Diagnose defects and export a mathematically sanitized, ML-ready dataset."
    )

    uploaded_raw = st.file_uploader("Upload Raw / Infected CSV Dataset", type=["csv"], key="sanitizer_uploader")

    if uploaded_raw:
        raw_df = pd.read_csv(uploaded_raw)
        
        st.subheader("🔍 Automated Data Defect Audit")
        
        total_cells = raw_df.size
        missing_count = int(raw_df.isna().sum().sum())
        inf_count = int(np.isinf(raw_df.select_dtypes(include=np.number)).sum().sum())
        duplicate_count = int(raw_df.duplicated().sum())
        
        col_types = raw_df.dtypes.astype(str)
        non_numeric_cols = [col for col, dtype in col_types.items() if dtype not in ['int64', 'float64', 'int32', 'float32']]

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Missing Cells (NaN)", f"{missing_count}", delta=f"{missing_count/max(total_cells,1):.1%} of data", delta_color="inverse")
        d2.metric("Infinite Values (Inf)", f"{inf_count}", delta="Critical Error" if inf_count > 0 else "Clean", delta_color="inverse")
        d3.metric("Duplicate Rows", f"{duplicate_count}", delta="Redundant" if duplicate_count > 0 else "Unique", delta_color="inverse")
        d4.metric("Non-Numeric Columns", f"{len(non_numeric_cols)}", delta="Requires Encoding" if len(non_numeric_cols) > 0 else "Pure Numeric", delta_color="off")

        with st.expander("📋 View Raw Data Head & Column Types"):
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.write(pd.DataFrame({"Data Type": col_types, "Missing Values": raw_df.isna().sum()}))

        st.divider()

        st.subheader("⚙️ Sanitization Pipeline Configuration")
        
        c_opt1, c_opt2 = st.columns(2)
        with c_opt1:
            impute_strategy = st.selectbox(
                "Missing Value Strategy",
                ["Median (Robust to Outliers)", "Mean (Parametric)", "Drop Rows with Missing Values", "Constant Fill (0.0)"]
            )
            handle_inf = st.checkbox("Sanitize Infinite Values (`±inf` → `NaN` → Imputed)", value=True)
            drop_dups = st.checkbox("Remove Duplicate Rows", value=True)

        with c_opt2:
            outlier_handling = st.selectbox(
                "Extreme Outlier Handling",
                ["None (Keep All)", "Clip Outliers (Winsorize at 1st / 99th Percentile)", "Drop Rows with Z-score > 3.5"]
            )
            scale_method = st.selectbox(
                "Feature Scaling Method for ML",
                ["Standard Scaler (Zero Mean, Unit Variance)", "MinMax Scaler (Range [0, 1])", "None (Raw Values)"]
            )

        if st.button("🚀 Run Sanitization & Generate ML-Ready Dataset", type="primary"):
            cleaned = raw_df.copy()

            if drop_dups:
                cleaned = cleaned.drop_duplicates()

            if handle_inf:
                num_cols_sel = cleaned.select_dtypes(include=np.number).columns
                cleaned[num_cols_sel] = cleaned[num_cols_sel].replace([np.inf, -np.inf], np.nan)

            for col in cleaned.columns:
                if cleaned[col].dtype == object or cleaned[col].dtype.name == 'category':
                    cleaned[col] = cleaned[col].astype(str).factorize()[0]

            num_cols = cleaned.select_dtypes(include=np.number).columns
            for col in num_cols:
                if cleaned[col].isna().any():
                    if "Median" in impute_strategy:
                        fill_val = cleaned[col].median()
                    elif "Mean" in impute_strategy:
                        fill_val = cleaned[col].mean()
                    elif "Constant" in impute_strategy:
                        fill_val = 0.0
                    cleaned[col] = cleaned[col].fillna(0.0 if np.isnan(fill_val) else fill_val)

            if "Drop Rows" in impute_strategy:
                cleaned = cleaned.dropna()

            if "Clip" in outlier_handling:
                for col in num_cols:
                    p1, p99 = np.percentile(cleaned[col].dropna(), [1, 99])
                    cleaned[col] = np.clip(cleaned[col], p1, p99)
            elif "Drop" in outlier_handling:
                for col in num_cols:
                    mean_val = cleaned[col].mean()
                    std_val = cleaned[col].std()
                    if std_val > 0:
                        z_scores = np.abs((cleaned[col] - mean_val) / std_val)
                        cleaned = cleaned[cleaned[col].isna() | (z_scores <= 3.5)]

            X_clean_mat = cleaned.select_dtypes(include=np.number).to_numpy(dtype=float)
            X_clean_mat = np.nan_to_num(X_clean_mat, nan=0.0, posinf=0.0, neginf=0.0)

            if "Standard" in scale_method:
                mean_v = X_clean_mat.mean(axis=0)
                std_v = X_clean_mat.std(axis=0, ddof=0)
                std_v[(std_v == 0) | np.isnan(std_v)] = 1.0
                X_clean_mat = (X_clean_mat - mean_v) / std_v
            elif "MinMax" in scale_method:
                min_v = X_clean_mat.min(axis=0)
                max_v = X_clean_mat.max(axis=0)
                ptp_v = max_v - min_v
                ptp_v[ptp_v == 0] = 1.0
                X_clean_mat = (X_clean_mat - min_v) / ptp_v

            final_clean_df = pd.DataFrame(X_clean_mat, columns=cleaned.select_dtypes(include=np.number).columns)

            st.success("✨ Data Sanitization Complete! Your dataset is 100% finite, imputed, scaled, and ready for ML ingestion.")

            rc1, rc2 = st.columns(2)
            rc1.metric("Original Dataset Shape", f"{raw_df.shape[0]} rows × {raw_df.shape[1]} cols")
            rc2.metric("Sanitized ML-Ready Shape", f"{final_clean_df.shape[0]} rows × {final_clean_df.shape[1]} cols", delta=f"-{raw_df.shape[0]-final_clean_df.shape[0]} rows dropped")

            st.subheader("✨ Sanitized Data Preview (First 10 Rows)")
            st.dataframe(final_clean_df.head(10), use_container_width=True)

            st.download_button(
                "📥 Download ML-Ready Sanitized Dataset (.csv)",
                final_clean_df.to_csv(index=False),
                "ml_ready_sanitized_dataset.csv",
                "text/csv"
            )


# ==========================================
# STREAMLIT PAGE SETUP
# ==========================================

st.set_page_config(page_title="LatentLabs | Advanced PCA Workbench", page_icon="🧪", layout="wide")

st.title("🧪 LatentLabs: Advanced PCA & Subspace Analytics Workbench")
st.caption(
    "First-principles Principal Component Analysis, Random Matrix Noise Filtering, Biplot Cluster Discovery, "
    "Subspace Anomaly Detection, Composite Index Scoring, Data Hygiene Sanitization, and Multi-Channel Image Compression."
)

with st.sidebar:
    st.header("Control Panel")
    mode = st.radio("Choose a laboratory", [
        "CSV / Subspace Analytics", 
        "Data Sanitizer & Hygiene", 
        "Image Compression Lab", 
        "PCA Mathematical Theory"
    ])
    method = st.selectbox("PCA Engine Implementation", ["Eigenvalue PCA", "SVD PCA"])
    st.divider()
    st.info(
        "Engine Architecture: Built purely on NumPy linear algebra operations "
        "without relying on scikit-learn's internal decomposition wrappers."
    )

# ==========================================
# MAIN ROUTING FLOW
# ==========================================
if mode == "PCA Mathematical Theory":
    st.header("📚 Principal Component Analysis: Complete Theoretical Foundation")
    file_theory = load_theory()
    if file_theory:
        st.markdown(file_theory)
    else:
        st.markdown(
            """
            ### 1. Variational & Spectral Foundations
            Given a mean-centered data matrix $X \\in \\mathbb{R}^{n \\times p}$ (where $n$ is samples and $p$ is features), 
            Principal Component Analysis (PCA) seeks an orthonormal projection basis $V_k = [v_1, v_2, \\dots, v_k] \\in \\mathbb{R}^{p \\times k}$ 
            that maximizes the variance of the projected coordinates:
            $$\\max_{V_k^T V_k = I_k} \\text{Tr}(V_k^T X^T X V_k)$$
            
            By the **Eckart–Young–Mirsky Theorem**, this is mathematically equivalent to finding the rank-$k$ linear subspace 
            that minimizes the Frobenius reconstruction error norm:
            $$\\min_{V_k} \\| X - X V_k V_k^T \\|_F^2 = \\sum_{i=k+1}^{\\min(n,p)} \\sigma_i^2$$

            ---

            ### 2. Algorithmic Dualism: Eigendecomposition vs. Singular Value Decomposition
            * **Covariance Eigendecomposition ($X^T X$):** 
              Constructs the sample covariance matrix $C = \\frac{1}{n-1} X^T X \\in \\mathbb{R}^{p \\times p}$ and solves the characteristic equation:
              $$C v_i = \\lambda_i v_i$$
              *Limitation:* Squaring $X$ squares its condition number $\\kappa(X^T X) = \\kappa(X)^2$, which can introduce severe numerical round-off errors and instability when features are highly collinear.
            * **Direct Singular Value Decomposition (SVD):** 
              Factorizes $X$ directly without forming the covariance matrix:
              $$X = U \\Sigma V^T$$
              where $U \\in \\mathbb{R}^{n \\times n}$ contains left singular vectors, $\\Sigma$ holds singular values $\\sigma_i$, and $V$ contains right singular vectors (principal axes). The eigenvalues relate directly via $\\lambda_i = \\frac{\\sigma_i^2}{n - 1}$.

            ---

            ### 3. Random Matrix Theory (Marchenko–Pastur Law)
            When analyzing high-dimensional datasets where feature count $p$ approaches or exceeds sample size $n$ (aspect ratio $\\gamma = p/n$), 
            standard scree plots often fail to distinguish true signal from sampling noise. 
            According to the **Marchenko–Pastur Law**, eigenvalues of pure random Gaussian noise matrices are bounded above by:
            $$\\lambda_+ = \\sigma^2 \\left(1 + \\sqrt{\\gamma}\\right)^2$$
            Any principal component whose eigenvalue exceeds $\\lambda_+$ represents genuine structural correlation, 
            while components below $\\lambda_+$ are statistically indistinguishable from random noise.

            ---

            ### 4. Downstream Applications in Data Science
            1. **Multi-Channel Image Compression:** Applying spatial subspace factorization across pixel matrices or independent RGB channels 
               to achieve high storage compression ratios while preserving perceptual structures.
            2. **Unsupervised Subspace Anomaly Detection:** Normal records conform to the dominant correlation subspace, yielding minimal reconstruction error. 
               Anomalies and outliers deviate sharply, registering large residual error in the orthogonal complement space ($\\|x_i - \\hat{x}_i\\|^2$).
            3. **Objective Composite Index Scoring:** Projecting high-dimensional economic, academic, or financial indicators onto `PC1` 
               to eliminate subjective weighting bias and generate objective ranking leaderboards.
            """
        )

elif mode == "Data Sanitizer & Hygiene":
    render_data_sanitizer_lab()

elif mode == "Image Compression Lab":
    st.header("🖼️ Multi-Channel Image Compression & Subspace Reconstruction Lab")
    uploaded = st.file_uploader("Upload a JPG, JPEG or PNG image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded)
        color_mode = st.radio(
            "Image Compression Architecture",
            ["Grayscale (Single Matrix)", "Full RGB Color (3-Channel Independent PCA)"],
            horizontal=True
        )

        X_gray = image_to_matrix(image)
        n_samples, n_features = X_gray.shape
        max_k = min(n_samples, n_features)

        st.write(f"**Image Dimensions:** {n_samples} rows × {n_features} columns (Maximum Achievable Rank: {max_k})")

        impl_key = "eigen" if method == "Eigenvalue PCA" else "svd"

        with st.spinner("Computing spectral decomposition..."):
            start = time.perf_counter()
            pca = PCAFromScratch(method=impl_key)
            pca.fit(X_gray)
            elapsed = time.perf_counter() - start

        default_k = min(50, max_k)
        c_sl1, c_sl2 = st.columns(2)
        with c_sl1:
            threshold = st.slider("Target Cumulative Variance Threshold", 0.50, 0.999, 0.95, 0.001)
            auto_k = min(pca.components_for_variance(threshold), max_k)
        with c_sl2:
            k = st.slider("Number of Principal Components (k)", 1, max_k, min(default_k, max_k), 1)

        st.info(f"Target variance requirement ({threshold:.1%}) is satisfied by at least **{auto_k} principal components**.")

        if color_mode == "Grayscale (Single Matrix)":
            X_hat = np.real(pca.inverse_transform(pca.transform(X_gray, k=k)))
            metrics = reconstruction_metrics(X_gray, X_hat)
            disp_orig = image.convert("L")
            disp_recon = Image.fromarray(np.clip(X_hat, 0, 255).astype(np.uint8))
            residual_map = np.abs(X_gray - X_hat)
        else:
            rgb_arr = np.asarray(image.convert("RGB"), dtype=float)
            recon_channels = []
            for ch in range(3):
                pca_ch = PCAFromScratch(method=impl_key)
                pca_ch.fit(rgb_arr[:, :, ch])
                ch_hat = np.real(pca_ch.inverse_transform(pca_ch.transform(rgb_arr[:, :, ch], k=k)))
                recon_channels.append(ch_hat)
            rgb_hat = np.stack(recon_channels, axis=2)
            metrics = reconstruction_metrics(rgb_arr.reshape(n_samples, -1), rgb_hat.reshape(n_samples, -1))
            disp_orig = image.convert("RGB")
            disp_recon = Image.fromarray(np.clip(rgb_hat, 0, 255).astype(np.uint8))
            residual_map = np.mean(np.abs(rgb_arr - rgb_hat), axis=2)

        true_storage_ratio = (n_samples * n_features) / max(k * (n_samples + n_features) + n_features, 1)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variance Retained", f"{pca.cumulative_explained_variance_[k-1]:.2%}")
        c2.metric("Pixel RMSE", f"{metrics['rmse']:.3f}")
        c3.metric("Rank Ratio (p/k)", f"{metrics['compression_ratio']:.2f}×")
        c4.metric("True Storage Ratio", f"{true_storage_ratio:.2f}×")

        st.divider()
        col_img1, col_img2, col_img3 = st.columns(3)
        with col_img1:
            st.subheader("Original Image")
            st.image(disp_orig, use_container_width=True)
        with col_img2:
            st.subheader(f"Reconstructed (k={k})")
            st.image(disp_recon, use_container_width=True)
        with col_img3:
            st.subheader("Residual Error Heatmap")
            fig_res, ax_res = plt.subplots(figsize=(4.5, 4.5))
            im_res = ax_res.imshow(residual_map, cmap="inferno")
            ax_res.axis("off")
            fig_res.colorbar(im_res, ax=ax_res, fraction=0.046, pad=0.04, label="Absolute Pixel Error")
            fig_res.tight_layout()
            st.pyplot(fig_res)
            plt.close(fig_res)

        st.subheader("Explained Variance Spectrum")
        st.pyplot(variance_plot(pca.explained_variance_ratio_, pca.cumulative_explained_variance_))

        st.subheader("Reconstruction Error Decay Curve")
        ks = sorted(set([1, min(5, max_k), min(10, max_k), min(25, max_k),
                         min(50, max_k), min(100, max_k), k]))
        errors = []
        for kk in ks:
            pred = np.real(pca.inverse_transform(pca.transform(X_gray, k=kk)))
            errors.append(reconstruction_metrics(X_gray, pred)["mse"])
        st.pyplot(reconstruction_plot(ks, errors))

        with st.expander("🔬 Mathematical Inspection Microscope"):
            st.write("Mean Vector Preview (First 10 Features):")
            st.code(np.array2string(pca.mean_[:10], precision=3))
            st.write("Top Eigenvalues & Explained Variance Ratios:")
            st.dataframe(pd.DataFrame({
                "Component": np.arange(1, min(10, len(pca.explained_variance_)) + 1),
                "Explained Variance": pca.explained_variance_[:10],
                "Explained Variance Ratio": pca.explained_variance_ratio_[:10],
                "Cumulative Ratio": pca.cumulative_explained_variance_[:10],
            }), use_container_width=True)
            st.write(f"Computation Execution Time: **{elapsed:.4f} seconds**")

else:
    st.header("📊 CSV / Subspace Analytics Workbench")
    uploaded = st.file_uploader("Upload a tabular CSV dataset", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", len(df))
        c2.metric("Total Columns", len(df.columns))
        c3.metric("Missing Values", int(df.isna().sum().sum()))

        st.dataframe(df.head(10), use_container_width=True)

        numeric = df.select_dtypes(include=np.number)
        if numeric.shape[1] < 2:
            st.error("The CSV needs at least two numeric columns to perform PCA.")
            st.stop()

        scale = st.checkbox("Standardize Features (Zero mean, unit variance $\\sigma^2=1$)", value=True)
        X, feature_names = prepare_numeric_data(numeric, standardize=scale)
        X = np.nan_to_num(np.real(X), nan=0.0, posinf=0.0, neginf=0.0)

        pca = PCAFromScratch(method="eigen" if method == "Eigenvalue PCA" else "svd")
        pca.fit(X)

        max_k = min(X.shape)
        eig_vals = np.nan_to_num(np.real(pca.explained_variance_), nan=0.0)
        mp_bound, mp_k = marchenko_pastur_analysis(X, eig_vals)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            threshold = st.slider("Target Variance Threshold", 0.50, 0.999, 0.95, 0.001)
            auto_k = min(pca.components_for_variance(threshold), max_k)
            st.caption(
                f"**Variance threshold ($95\\%$):** {auto_k} components | "
                f"**Marchenko–Pastur Signal Floor:** {min(mp_k, max_k)} components"
            )
        with col_s2:
            k = st.slider("Selected Subspace Dimensions (k)", 1, max_k, min(auto_k, max_k))

        Z = np.nan_to_num(np.real(pca.transform(X, k=k)), nan=0.0)
        X_hat = np.nan_to_num(np.real(pca.inverse_transform(Z)), nan=0.0)
        metrics = reconstruction_metrics(X, X_hat)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Numeric Features", X.shape[1])
        c2.metric("Selected Components (k)", k, delta=f"-{X.shape[1] - k} dims")
        c3.metric("Variance Retained", f"{pca.cumulative_explained_variance_[k-1]:.2%}")
        c4.metric("Reconstruction RMSE", f"{metrics['rmse']:.4f}")

        st.divider()

        # --- 5 Research-Grade Analytical Tabs ---
        tab_overview, tab_clusters, tab_anomaly, tab_ml, tab_index = st.tabs([
            "📈 Spectrum, RMT & Stability",
            "🧭 2D Biplot & Cluster Explorer",
            "🚨 Anomaly Radar & What-If Simulator",
            "⚡ Downstream ML Benchmark",
            "🏆 PC1 Composite Index & Ranking"
        ])

        # TAB 1: Spectrum & RMT
        with tab_overview:
            st.subheader("Explained Variance Spectrum & Marchenko–Pastur Noise Filter")
            st.pyplot(variance_plot(pca.explained_variance_ratio_, pca.cumulative_explained_variance_))

            st.info(
                f"🔬 **Random Matrix Theory Analysis:** For aspect ratio $\\gamma = p/n = {X.shape[1]/max(X.shape[0], 1):.3f}$, "
                f"the theoretical random noise bound is $\\lambda_+ = {mp_bound:.3f}$. "
                f"Exactly **{min(mp_k, max_k)} components** surpass this noise floor as true signal."
            )

            with st.expander("⚖️ Numerical Stability Stress-Test: Eigenvalue PCA vs. Direct SVD"):
                t_eig0 = time.perf_counter()
                pca_eig = PCAFromScratch(method="eigen").fit(X)
                t_eig = (time.perf_counter() - t_eig0) * 1000.0

                t_svd0 = time.perf_counter()
                pca_svd = PCAFromScratch(method="svd").fit(X)
                t_svd = (time.perf_counter() - t_svd0) * 1000.0

                s_vals = np.linalg.svd(X, compute_uv=False)
                cond_x = float(s_vals[0] / max(s_vals[-1], 1e-15))
                spectrum_diff = float(np.max(np.abs(
                    np.real(pca_eig.explained_variance_[:max_k]) - np.real(pca_svd.explained_variance_[:max_k])
                )))

                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric("Eigen PCA Time", f"{t_eig:.2f} ms")
                ec2.metric("SVD PCA Time", f"{t_svd:.2f} ms")
                ec3.metric("Condition Number κ(X)", f"{cond_x:.2e}")
                ec4.metric("Max Spectrum Discrepancy", f"{spectrum_diff:.2e}")
                st.caption("SVD avoids forming $X^T X$, preserving precision against condition number squaring ($\\kappa^2$).")

            st.subheader("Principal Component Feature Loadings Matrix")
            loadings = pd.DataFrame(
                np.real(pca.components_[:k].T),
                index=feature_names,
                columns=[f"PC{i+1}" for i in range(k)]
            )
            try:
                st.dataframe(loadings.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)
            except Exception:
                st.dataframe(loadings, use_container_width=True)

            st.download_button(
                "📥 Download PCA-Reduced Dataset (.csv)",
                pd.DataFrame(Z, columns=[f"PC{i+1}" for i in range(k)]).to_csv(index=False),
                "pca_reduced_dataset.csv",
                "text/csv"
            )

        # TAB 2: Biplot & Clusters
        with tab_clusters:
            st.subheader("2D Latent Space Biplot & Cluster Discovery")
            if max_k < 2:
                st.warning("Need at least 2 numeric features for 2D visualization.")
            else:
                Z_2d = np.nan_to_num(np.real(pca.transform(X, k=2)), nan=0.0)
                c_opt1, c_opt2 = st.columns([2, 1])
                with c_opt1:
                    color_mode = st.radio(
                        "Color Points By:",
                        ["Auto-Cluster (K-Means)", "Existing Dataset Column", "None (Uniform)"],
                        horizontal=True
                    )
                with c_opt2:
                    show_biplot = st.checkbox("🏹 Overlay Biplot Vectors", value=True)
                    top_n_vec = st.slider("Max Vectors", 2, min(10, len(feature_names)), min(5, len(feature_names))) if show_biplot else 0

                comp_2d = pca.components_[:2] if show_biplot else None

                if color_mode == "Auto-Cluster (K-Means)":
                    n_clusters = st.slider("K-Means Clusters", 2, 8, 3)
                    cluster_labels = kmeans_numpy(Z_2d, n_clusters=n_clusters)
                    fig_cl = colored_latent_scatter(
                        Z_2d[:, 0], Z_2d[:, 1],
                        pd.Series([f"Cluster {c+1}" for c in cluster_labels]),
                        group_label="Latent Cluster",
                        components_2d=comp_2d,
                        feature_names=feature_names,
                        top_n_vectors=top_n_vec
                    )
                    st.pyplot(fig_cl)
                    plt.close(fig_cl)

                    cluster_summary = df[feature_names].copy()
                    cluster_summary["Cluster"] = [f"Cluster {c+1}" for c in cluster_labels]
                    st.write("**Cluster Centroid Profiles:**")
                    st.dataframe(cluster_summary.groupby("Cluster").mean(), use_container_width=True)

                elif color_mode == "Existing Dataset Column":
                    selected_col = st.selectbox("Select Column", df.columns)
                    fig_col = colored_latent_scatter(
                        Z_2d[:, 0], Z_2d[:, 1],
                        df[selected_col].fillna("Missing").to_numpy(),
                        group_label=selected_col,
                        components_2d=comp_2d,
                        feature_names=feature_names,
                        top_n_vectors=top_n_vec
                    )
                    st.pyplot(fig_col)
                    plt.close(fig_col)
                else:
                    fig_uni = colored_latent_scatter(
                        Z_2d[:, 0], Z_2d[:, 1],
                        pd.Series(["Sample"] * len(Z_2d)),
                        group_label="All Records",
                        components_2d=comp_2d,
                        feature_names=feature_names,
                        top_n_vectors=top_n_vec
                    )
                    st.pyplot(fig_uni)
                    plt.close(fig_uni)

        # TAB 3: Anomaly Radar & Simulator
        with tab_anomaly:
            st.subheader("🚨 Unsupervised Subspace Anomaly Detection")
            st.markdown(
                "Normal records conform to the dominant principal subspace and reconstruct with low residual error $\\|x_i - \\hat{x}_i\\|^2$. "
                "Anomalous and corrupted rows register large orthogonal errors."
            )

            sq_residuals = (X - X_hat) ** 2
            row_mse = np.mean(sq_residuals, axis=1)
            row_mse = np.nan_to_num(row_mse, nan=0.0, posinf=0.0, neginf=0.0)

            worst_feature_idx = np.argmax(sq_residuals, axis=1)
            worst_feature_names = [feature_names[idx] for idx in worst_feature_idx]

            pct_cutoff = st.slider("Anomaly Percentile Threshold (%)", 90.0, 99.5, 95.0, 0.5)
            cutoff_val = float(np.percentile(row_mse, pct_cutoff))
            is_anomaly = row_mse >= cutoff_val
            n_anomalies = int(np.sum(is_anomaly))

            a1, a2, a3 = st.columns(3)
            a1.metric("Flagged Anomalies", f"{n_anomalies} rows", f"Top {100 - pct_cutoff:.1f}%")
            a2.metric("Error Threshold (MSE)", f"{cutoff_val:.4f}")
            a3.metric("Max Row Error", f"{np.max(row_mse):.4f}")

            fig_anom, ax_anom = plt.subplots(figsize=(7, 3.2))
            finite_mse = row_mse[np.isfinite(row_mse)]
            if len(finite_mse) > 0 and np.max(finite_mse) > np.min(finite_mse):
                ax_anom.hist(finite_mse, bins=35, color="#4C78A8", edgecolor="white", alpha=0.85)
            else:
                ax_anom.hist(finite_mse, bins=1, color="#4C78A8", edgecolor="white", alpha=0.85)
            ax_anom.axvline(cutoff_val, color="#E45756", linestyle="--", linewidth=2, label=f"{pct_cutoff}% Cutoff")
            ax_anom.set_xlabel("Reconstruction MSE")
            ax_anom.set_ylabel("Record Count")
            ax_anom.set_title("Reconstruction Error Distribution")
            ax_anom.legend()
            fig_anom.tight_layout()
            st.pyplot(fig_anom)
            plt.close(fig_anom)

            anom_df = df.copy()
            anom_df.insert(0, "Primary_Deviating_Feature", worst_feature_names)
            anom_df.insert(0, "Reconstruction_MSE", np.round(row_mse, 4))
            flagged_df = anom_df[is_anomaly].sort_values(by="Reconstruction_MSE", ascending=False)

            st.write(f"**Top Flagged Anomalous Records (at k={k}):**")
            st.dataframe(flagged_df.head(25), use_container_width=True)

            st.divider()
            st.subheader("🎛️ Counterfactual 'What-If' Subspace Perturbation Simulator")
            default_row = int(np.argmax(row_mse))
            sim_row_idx = st.number_input("Row Index to Inspect & Perturb", 0, len(X) - 1, default_row, 1)

            x_orig_row = X[sim_row_idx].copy()
            x_mod_row = x_orig_row.copy()

            top3_feat_idx = np.argsort(sq_residuals[sim_row_idx])[::-1][:min(3, len(feature_names))]
            sim_cols = st.columns(len(top3_feat_idx))
            for idx_pos, f_idx in enumerate(top3_feat_idx):
                fname = feature_names[f_idx]
                cur_val = float(x_orig_row[f_idx])
                with sim_cols[idx_pos]:
                    x_mod_row[f_idx] = st.slider(
                        f"Perturb `{fname}`",
                        float(min(-4.0, cur_val - 2.0)),
                        float(max(4.0, cur_val + 2.0)),
                        cur_val,
                        0.1
                    )

            z_mod = np.real(pca.transform(x_mod_row[None, :], k=k))
            x_hat_mod = np.real(pca.inverse_transform(z_mod))
            new_mse = float(np.mean((x_mod_row - x_hat_mod[0]) ** 2))
            old_mse = float(row_mse[sim_row_idx])

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Original Row MSE", f"{old_mse:.4f}")
            sc2.metric("Modified MSE", f"{new_mse:.4f}", delta=f"{new_mse - old_mse:+.4f}", delta_color="inverse")
            sc3.metric("Status After Tweak", "🚨 Still Flagged" if new_mse >= cutoff_val else "✅ Normal")

        # TAB 4: ML Benchmark
        with tab_ml:
            st.subheader("⚡ Downstream Machine Learning Impact Benchmark")
            st.markdown("Compare predictive accuracy and training latency between Raw Features vs. PCA-Compressed Features.")

            target_col = st.selectbox("Select Target Column (y)", df.columns)
            other_numeric_cols = [c for c in feature_names if c != target_col]

            if len(other_numeric_cols) < 2:
                st.warning("Need at least 2 numeric predictor columns.")
            else:
                X_bench_raw, _ = prepare_numeric_data(df[other_numeric_cols], standardize=scale)
                X_bench_raw = np.nan_to_num(np.real(X_bench_raw), nan=0.0, posinf=0.0, neginf=0.0)
                pca_bench = PCAFromScratch(method="eigen" if method == "Eigenvalue PCA" else "svd")
                pca_bench.fit(X_bench_raw)

                k_bench = min(k, min(X_bench_raw.shape))
                Z_bench = np.nan_to_num(np.real(pca_bench.transform(X_bench_raw, k=k_bench)), nan=0.0)

                metric_name, is_clf, raw_score, raw_ms, pca_score, pca_ms = evaluate_ridge_benchmark(
                    X_bench_raw, Z_bench, df[target_col]
                )

                task_type = "Classification" if is_clf else "Regression"
                st.info(f"**Task:** {task_type} | **Metric:** {metric_name}")

                m1, m2, m3 = st.columns(3)
                dim_reduction_pct = (1.0 - (k_bench / X_bench_raw.shape[1])) * 100.0
                speedup = raw_ms / max(pca_ms, 1e-6)

                m1.metric("Dimensionality Reduction", f"{X_bench_raw.shape[1]} → {k_bench} cols", delta=f"-{dim_reduction_pct:.1f}%")
                m2.metric(f"{metric_name}", f"{pca_score:.4f}", delta=f"{pca_score - raw_score:+.4f} vs Raw")
                m3.metric("Fit Training Time", f"{pca_ms:.2f} ms", delta=f"{speedup:.2f}× speedup")

                bench_table = pd.DataFrame({
                    "Pipeline": ["1. Raw Features", f"2. PCA Compressed (k={k_bench})"],
                    "Dimensions": [X_bench_raw.shape[1], k_bench],
                    metric_name: [round(raw_score, 4), round(pca_score, 4)],
                    "Fit Time (ms)": [round(raw_ms, 2), round(pca_ms, 2)]
                })
                st.table(bench_table)

        # TAB 5: Composite Index Leaderboard
        with tab_index:
            st.subheader("🏆 Explainable PC1 Composite Index & Ranking Engine")
            st.markdown("Projects indicators onto `PC1` and scales them to a **0–100 Composite Score** for objective ranking.")

            id_col = st.selectbox("Select Entity Identifier Column", df.columns)
            pc1_raw = np.nan_to_num(np.real(pca.transform(X, k=1)[:, 0]), nan=0.0)

            pc1_weights = np.nan_to_num(np.real(pca.components_[0]), nan=0.0)
            if np.sum(pc1_weights) < 0:
                pc1_raw = -pc1_raw
                pc1_weights = -pc1_weights

            ptp = np.ptp(pc1_raw)
            composite_score = ((pc1_raw - np.min(pc1_raw)) / ptp) * 100.0 if ptp > 1e-12 else np.full_like(pc1_raw, 50.0)

            top_w_idx = np.argsort(np.abs(pc1_weights))[::-1][:min(6, len(feature_names))]
            terms = [f"({pc1_weights[i]:+.3f} × {feature_names[i]})" for i in top_w_idx]
            st.code("Composite_Index ∝ " + " + ".join(terms), language="text")

            rank_df = df.copy()
            rank_df.insert(0, "Composite_Index_0_100", np.round(composite_score, 2))
            rank_df = rank_df.sort_values(by="Composite_Index_0_100", ascending=False).reset_index(drop=True)
            rank_df.insert(0, "Rank", np.arange(1, len(rank_df) + 1))

            r_col1, r_col2 = st.columns(2)
            with r_col1:
                st.write("**🥇 Top 15 Ranked Entities:**")
                st.dataframe(rank_df[["Rank", id_col, "Composite_Index_0_100"] + feature_names[:min(4, len(feature_names))]].head(15), use_container_width=True)
            with r_col2:
                st.write("**📉 Bottom 15 Ranked Entities:**")
                st.dataframe(rank_df[["Rank", id_col, "Composite_Index_0_100"] + feature_names[:min(4, len(feature_names))]].tail(15), use_container_width=True)
