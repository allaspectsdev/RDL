"""
Correlation & Multivariate Analysis Module - Correlation matrices, PCA, t-SNE, Factor Analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import (
    section_header, empty_state, validation_panel,
    interpretation_card, alternative_suggestion,
)
from modules.validation import (
    check_sample_size, check_kmo_bartlett, interpret_correlation,
)


@st.cache_data
def _compute_corr_and_pvals(df_subset, method):
    """Compute correlation matrix and p-values (cached)."""
    selected = df_subset.columns.tolist()
    corr = df_subset.corr(method=method)
    p_matrix = pd.DataFrame(np.ones((len(selected), len(selected))),
                            index=selected, columns=selected)
    for i, c1 in enumerate(selected):
        for j, c2 in enumerate(selected):
            if i != j:
                pair = df_subset[[c1, c2]].dropna()
                if method == "pearson":
                    _, p = stats.pearsonr(pair[c1], pair[c2])
                elif method == "spearman":
                    _, p = stats.spearmanr(pair[c1], pair[c2])
                else:
                    _, p = stats.kendalltau(pair[c1], pair[c2])
                p_matrix.iloc[i, j] = p
    return corr, p_matrix


def render_correlation(df: pd.DataFrame):
    """Render correlation and multivariate analysis interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    tabs = st.tabs([
        "Correlation Matrix", "Scatter Matrix", "Pairwise Scatter",
        "PCA", "t-SNE", "Factor Analysis", "Partial Correlation",
        "Discriminant Analysis", "MDS",
    ])

    with tabs[0]:
        _render_corr_matrix(df)
    with tabs[1]:
        _render_scatter_matrix(df)
    with tabs[2]:
        _render_pairwise(df)
    with tabs[3]:
        _render_pca(df)
    with tabs[4]:
        _render_tsne(df)
    with tabs[5]:
        _render_factor_analysis(df)
    with tabs[6]:
        _render_partial_corr(df)
    with tabs[7]:
        _render_discriminant(df)
    with tabs[8]:
        _render_mds(df)


def _render_corr_matrix(df: pd.DataFrame):
    """Correlation matrix with heatmap."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    selected = st.multiselect("Columns:", num_cols, default=num_cols[:8], key="corr_cols")
    if len(selected) < 2:
        return

    method = st.selectbox("Method:", ["pearson", "spearman", "kendall"], key="corr_method")
    show_tri = st.selectbox("Show:", ["Full", "Upper Triangle", "Lower Triangle"], key="corr_tri")

    corr, p_matrix = _compute_corr_and_pvals(df[selected], method)

    # Validation checks
    try:
        n_obs = len(df[selected].dropna())
        checks = [check_sample_size(n_obs, "correlation")]
        validation_panel(checks, title="Data Quality")
        if method == "pearson" and n_obs < 30:
            alternative_suggestion(
                f"Small sample (n={n_obs}) with Pearson correlation",
                ["Spearman rank correlation"],
            )
    except Exception:
        pass

    # Mask
    mask = np.zeros_like(corr, dtype=bool)
    if show_tri == "Upper Triangle":
        mask[np.tril_indices_from(mask, k=-1)] = True
    elif show_tri == "Lower Triangle":
        mask[np.triu_indices_from(mask, k=1)] = True

    corr_display = corr.copy()
    corr_display = corr_display.where(~mask, other=np.nan)

    # Heatmap
    fig = px.imshow(corr_display, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title=f"{method.title()} Correlation Matrix",
                    aspect="auto")
    fig.update_layout(height=max(400, len(selected) * 40))
    st.plotly_chart(fig, use_container_width=True)

    # P-value matrix
    with st.expander("P-Values"):
        p_display = p_matrix.where(~mask, other=np.nan).round(6)
        st.dataframe(p_display, use_container_width=True)

    # Significant correlations
    with st.expander("Significant Correlations (p < 0.05)"):
        sig_pairs = []
        for i, c1 in enumerate(selected):
            for j, c2 in enumerate(selected):
                if i < j and p_matrix.iloc[i, j] < 0.05:
                    sig_pairs.append({
                        "Variable 1": c1, "Variable 2": c2,
                        "Correlation": round(corr.iloc[i, j], 4),
                        "p-value": round(p_matrix.iloc[i, j], 6),
                    })
        if sig_pairs:
            st.dataframe(pd.DataFrame(sig_pairs), use_container_width=True, hide_index=True)
        else:
            st.info("No significant correlations found.")


def _render_scatter_matrix(df: pd.DataFrame):
    """Scatter plot matrix (SPLOM)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    selected = st.multiselect("Columns:", num_cols, default=num_cols[:4], key="splom_cols")
    if len(selected) < 2:
        return

    color_col = st.selectbox("Color by:", [None] + cat_cols, key="splom_color")

    fig = px.scatter_matrix(df, dimensions=selected, color=color_col,
                            title="Scatter Plot Matrix",
                            opacity=0.6)
    fig.update_traces(diagonal_visible=True, showupperhalf=True)
    fig.update_layout(height=max(600, len(selected) * 200))
    st.plotly_chart(fig, use_container_width=True)


def _render_pairwise(df: pd.DataFrame):
    """Single pairwise scatter plot with detailed analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X:", num_cols, key="pair_x")
    y_col = c2.selectbox("Y:", [c for c in num_cols if c != x_col], key="pair_y")

    color_col = st.selectbox("Color by:", [None] + cat_cols + num_cols, key="pair_color")
    size_col = st.selectbox("Size by:", [None] + num_cols, key="pair_size")
    show_trend = st.checkbox("Show trendline", value=True, key="pair_trend")
    show_marginal = st.selectbox("Marginal plots:", [None, "histogram", "box", "violin", "rug"], key="pair_marg")

    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                     trendline="ols" if show_trend else None,
                     marginal_x=show_marginal, marginal_y=show_marginal,
                     title=f"{y_col} vs {x_col}", opacity=0.7)
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation stats
    pair_data = df[[x_col, y_col]].dropna()
    pearson_r, pearson_p = stats.pearsonr(pair_data[x_col], pair_data[y_col])
    spearman_r, spearman_p = stats.spearmanr(pair_data[x_col], pair_data[y_col])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pearson r", f"{pearson_r:.4f}")
    c2.metric("Pearson p", f"{pearson_p:.6f}")
    c3.metric("Spearman ρ", f"{spearman_r:.4f}")
    c4.metric("Spearman p", f"{spearman_p:.6f}")

    # Confidence interval for Pearson r (Fisher z-transform)
    n = len(pair_data)
    if abs(pearson_r) >= 1.0 or n <= 3:
        st.write("**95% CI for Pearson r:** not computable (|r| = 1 or n ≤ 3)")
    else:
        z = np.arctanh(pearson_r)
        se = 1 / np.sqrt(n - 3)
        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)
        st.write(f"**95% CI for Pearson r:** [{r_lower:.4f}, {r_upper:.4f}]")


def _render_pca(df: pd.DataFrame):
    """Principal Component Analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    selected = st.multiselect("Variables:", num_cols, default=num_cols, key="pca_cols")
    if len(selected) < 2:
        return

    standardize = st.checkbox("Standardize (recommended)", value=True, key="pca_std")
    color_col = st.selectbox("Color by:", [None] + cat_cols, key="pca_color")

    if st.button("Run PCA", key="run_pca"):
        data = df[selected].dropna()
        X = data.values

        # Validation: KMO & Bartlett's test
        try:
            checks = [
                check_sample_size(len(data), "pca"),
                check_kmo_bartlett(data[selected] if isinstance(data, pd.DataFrame) else pd.DataFrame(X, columns=selected)),
            ]
            validation_panel(checks, title="PCA Suitability")
            failed = [c for c in checks if c.status in ("warn", "fail")]
            if any("KMO" in c.name for c in failed):
                alternative_suggestion(
                    "Data may not be suitable for dimension reduction",
                    ["Review variable selection", "Check for low-variance variables"],
                )
        except Exception:
            pass

        if standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        n_components = min(len(selected), len(data))
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(X)

        # Explained variance
        exp_var = pca.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)

        section_header("Explained Variance")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(n_components)],
                             y=exp_var * 100, name="Individual %"), secondary_y=False)
        fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(n_components)],
                                 y=cum_var * 100, name="Cumulative %",
                                 line=dict(color="red", width=2),
                                 mode="lines+markers"), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color="gray", secondary_y=True)
        fig.update_layout(title="Scree Plot", height=400)
        fig.update_yaxes(title_text="Individual %", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", range=[0, 105], secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

        # Kaiser criterion
        n_kaiser = np.sum(pca.explained_variance_ > 1)
        n_80 = np.searchsorted(cum_var, 0.80) + 1
        st.write(f"**Kaiser criterion (eigenvalue > 1):** {n_kaiser} components")
        st.write(f"**Components for 80% variance:** {n_80}")

        # Variance table
        var_df = pd.DataFrame({
            "Component": [f"PC{i+1}" for i in range(n_components)],
            "Eigenvalue": pca.explained_variance_.round(4),
            "Variance %": (exp_var * 100).round(2),
            "Cumulative %": (cum_var * 100).round(2),
        })
        st.dataframe(var_df, use_container_width=True, hide_index=True)

        # Loadings
        section_header("Loadings")
        loadings = pd.DataFrame(pca.components_.T,
                                columns=[f"PC{i+1}" for i in range(n_components)],
                                index=selected).round(4)
        st.dataframe(loadings, use_container_width=True)

        # Biplot (PC1 vs PC2)
        section_header("Biplot")
        scores_df = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"])
        if color_col and color_col in df.columns:
            scores_df["color"] = df[color_col].iloc[data.index].values

        fig = go.Figure()
        if color_col and "color" in scores_df:
            for cat in scores_df["color"].unique():
                mask = scores_df["color"] == cat
                fig.add_trace(go.Scatter(x=scores_df.loc[mask, "PC1"],
                                         y=scores_df.loc[mask, "PC2"],
                                         mode="markers", name=str(cat),
                                         marker=dict(size=6, opacity=0.7)))
        else:
            fig.add_trace(go.Scatter(x=scores_df["PC1"], y=scores_df["PC2"],
                                     mode="markers", name="Scores",
                                     marker=dict(size=6, opacity=0.7)))

        # Add loading vectors
        scale = max(scores_df["PC1"].abs().max(), scores_df["PC2"].abs().max()) * 0.8
        load_scale = max(abs(pca.components_[0]).max(), abs(pca.components_[1]).max())
        for i, var in enumerate(selected):
            x_load = pca.components_[0, i] * scale / load_scale
            y_load = pca.components_[1, i] * scale / load_scale
            fig.add_annotation(x=x_load, y=y_load, ax=0, ay=0, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowsize=1.5, arrowcolor="red",
                               text=var, font=dict(color="red", size=10))

        fig.update_layout(
            title=f"Biplot (PC1: {exp_var[0]*100:.1f}% vs PC2: {exp_var[1]*100:.1f}%)",
            xaxis_title=f"PC1 ({exp_var[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({exp_var[1]*100:.1f}%)",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_tsne(df: pd.DataFrame):
    """t-SNE visualization."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    selected = st.multiselect("Variables:", num_cols, default=num_cols, key="tsne_cols")
    if len(selected) < 2:
        return

    c1, c2, c3 = st.columns(3)
    perplexity = c1.slider("Perplexity:", 5, 50, 30, key="tsne_perp")
    n_dims = c2.selectbox("Dimensions:", [2, 3], key="tsne_dims")
    color_col = c3.selectbox("Color by:", [None] + cat_cols, key="tsne_color")

    max_samples = st.slider("Max samples:", 100, min(5000, len(df)), min(1000, len(df)), key="tsne_max")

    if st.button("Run t-SNE", key="run_tsne"):
        data = df[selected].dropna()
        if len(data) > max_samples:
            data = data.sample(max_samples, random_state=42)

        X = StandardScaler().fit_transform(data.values)
        perp = min(perplexity, len(data) - 1)

        with st.spinner("Computing t-SNE..."):
            tsne = TSNE(n_components=n_dims, perplexity=perp, random_state=42, n_iter=1000)
            embedding = tsne.fit_transform(X)

        emb_df = pd.DataFrame(embedding, columns=[f"Dim{i+1}" for i in range(n_dims)])
        if color_col and color_col in df.columns:
            emb_df["color"] = df[color_col].iloc[data.index].values

        if n_dims == 2:
            fig = px.scatter(emb_df, x="Dim1", y="Dim2",
                             color="color" if "color" in emb_df else None,
                             title="t-SNE 2D", opacity=0.7)
        else:
            fig = px.scatter_3d(emb_df, x="Dim1", y="Dim2", z="Dim3",
                                color="color" if "color" in emb_df else None,
                                title="t-SNE 3D", opacity=0.7)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


def _render_factor_analysis(df: pd.DataFrame):
    """Factor Analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 3:
        empty_state("Need at least 3 numeric columns.")
        return

    selected = st.multiselect("Variables:", num_cols, default=num_cols[:6], key="fa_cols")
    if len(selected) < 3:
        return

    max_factors = min(len(selected) - 1, 10)
    if max_factors < 1:
        st.warning("Need at least 2 variables selected for factor analysis.")
        return
    n_factors = st.slider("Number of factors:", 1, max_factors, min(2, max_factors), key="fa_n")
    rotation = st.selectbox("Rotation:", ["varimax", "none"], key="fa_rot",
                            help="Varimax: orthogonal rotation for simpler structure")

    if st.button("Run Factor Analysis", key="run_fa"):
        data = df[selected].dropna()

        # Validation: KMO & Bartlett's
        try:
            checks = [
                check_sample_size(len(data), "pca"),
                check_kmo_bartlett(data),
            ]
            validation_panel(checks, title="Factor Analysis Suitability")
        except Exception:
            pass

        X = StandardScaler().fit_transform(data.values)

        rot = rotation if rotation != "none" else None
        fa = FactorAnalysis(n_components=n_factors, rotation=rot, random_state=42)
        fa.fit(X)

        # Loadings
        loadings = pd.DataFrame(fa.components_.T,
                                columns=[f"Factor{i+1}" for i in range(n_factors)],
                                index=selected).round(4)
        section_header("Factor Loadings")
        st.dataframe(loadings, use_container_width=True)

        # Communalities
        communalities = np.sum(fa.components_ ** 2, axis=0)
        # Actually communalities are sum of squared loadings for each variable
        comm = np.sum(fa.components_.T ** 2, axis=1)
        comm_df = pd.DataFrame({"Variable": selected, "Communality": comm.round(4)})
        section_header("Communalities")
        st.dataframe(comm_df, use_container_width=True, hide_index=True)

        # Loadings heatmap
        fig = px.imshow(loadings.values, x=loadings.columns, y=loadings.index,
                        text_auto=".2f", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Factor Loadings Heatmap")
        fig.update_layout(height=max(300, len(selected) * 30))
        st.plotly_chart(fig, use_container_width=True)


def _render_partial_corr(df: pd.DataFrame):
    """Partial correlation."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 3:
        empty_state("Need at least 3 numeric columns.")
        return

    c1, c2 = st.columns(2)
    var1 = c1.selectbox("Variable 1:", num_cols, key="pc_v1")
    var2 = c2.selectbox("Variable 2:", [c for c in num_cols if c != var1], key="pc_v2")
    control_vars = st.multiselect("Controlling for:",
                                   [c for c in num_cols if c not in (var1, var2)],
                                   key="pc_ctrl")

    if not control_vars:
        st.info("Select at least one control variable.")
        return

    if st.button("Compute Partial Correlation", key="run_pc"):
        data = df[[var1, var2] + control_vars].dropna()

        # Partial correlation via regression
        from sklearn.linear_model import LinearRegression

        # Regress var1 on controls
        lr1 = LinearRegression()
        lr1.fit(data[control_vars].values, data[var1].values)
        resid1 = data[var1].values - lr1.predict(data[control_vars].values)

        # Regress var2 on controls
        lr2 = LinearRegression()
        lr2.fit(data[control_vars].values, data[var2].values)
        resid2 = data[var2].values - lr2.predict(data[control_vars].values)

        # Correlation of residuals
        partial_r, partial_p = stats.pearsonr(resid1, resid2)

        # Zero-order correlation
        zero_r, zero_p = stats.pearsonr(data[var1], data[var2])

        c1, c2 = st.columns(2)
        c1.metric("Zero-order r", f"{zero_r:.4f} (p={zero_p:.6f})")
        c2.metric("Partial r", f"{partial_r:.4f} (p={partial_p:.6f})")

        st.write(f"Controlling for: **{', '.join(control_vars)}**")
        diff = abs(zero_r) - abs(partial_r)
        if diff > 0.05:
            st.info(f"Control variables account for a portion of the relationship (Δr = {diff:.4f}).")

        # Scatter of residuals
        fig = px.scatter(x=resid1, y=resid2, trendline="ols",
                         labels={"x": f"{var1} (residualized)", "y": f"{var2} (residualized)"},
                         title=f"Partial Correlation: {var1} vs {var2}")
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Tab 8 -- Discriminant Analysis (LDA / QDA)
# ===================================================================

def _render_discriminant(df: pd.DataFrame):
    """Linear and Quadratic Discriminant Analysis."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import confusion_matrix, classification_report

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    target_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 10]
    if not target_options:
        empty_state("Need a categorical target variable.")
        return

    section_header("Discriminant Analysis")

    target = st.selectbox("Target (group) variable:", target_options, key="da_target")
    features = st.multiselect("Feature variables:", [c for c in num_cols if c != target],
                               key="da_features")
    if len(features) < 2:
        st.info("Select at least 2 feature variables.")
        return

    method = st.selectbox("Method:", ["LDA (Linear)", "QDA (Quadratic)"], key="da_method")

    if st.button("Run Analysis", key="da_run"):
        data = df[features + [target]].dropna()
        X = data[features].values
        y = data[target].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == "LDA (Linear)":
            model = LinearDiscriminantAnalysis()
        else:
            model = QuadraticDiscriminantAnalysis()

        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)
        accuracy = (y_pred == y).mean()
        cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")

        # Metrics
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("Training Accuracy", f"{accuracy:.4f}")
        c2m.metric("CV Accuracy (5-fold)", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        classes = np.unique(y)
        c3m.metric("Classes", str(len(classes)))

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            x=[str(c) for c in classes], y=[str(c) for c in classes],
                            labels=dict(x="Predicted", y="Actual"),
                            title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        # LDA-specific: discriminant functions and projections
        if method == "LDA (Linear)" and hasattr(model, 'scalings_'):
            section_header("Discriminant Functions")
            n_components = min(model.scalings_.shape[1], 2)

            if n_components >= 1:
                coef_df = pd.DataFrame(
                    model.scalings_[:, :n_components],
                    index=features,
                    columns=[f"LD{i+1}" for i in range(n_components)],
                ).round(4)
                st.dataframe(coef_df, use_container_width=True)

                # Explained variance ratio
                if hasattr(model, 'explained_variance_ratio_'):
                    st.write("**Explained variance ratio:** " +
                             ", ".join([f"LD{i+1}: {v:.1%}" for i, v in
                                        enumerate(model.explained_variance_ratio_[:n_components])]))

            # LD scatter plot
            if n_components >= 2:
                section_header("LD1 vs LD2 Plot")
                X_lda = model.transform(X_scaled)
                lda_df = pd.DataFrame({
                    "LD1": X_lda[:, 0],
                    "LD2": X_lda[:, 1],
                    "Group": y,
                })
                fig = px.scatter(lda_df, x="LD1", y="LD2", color="Group",
                                 title="Linear Discriminant Projection", opacity=0.7)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            elif n_components == 1:
                section_header("LD1 Distribution by Group")
                X_lda = model.transform(X_scaled)
                lda_df = pd.DataFrame({"LD1": X_lda[:, 0], "Group": y})
                fig = px.histogram(lda_df, x="LD1", color="Group", barmode="overlay",
                                   opacity=0.6, title="LD1 by Group")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Wilks' Lambda (approximate)
        if method == "LDA (Linear)":
            section_header("Wilks' Lambda")
            # Compute Wilks' Lambda as product of 1/(1+eigenvalue)
            if hasattr(model, 'explained_variance_ratio_'):
                # Approximate from eigenvalues
                eigenvalues = model.explained_variance_ratio_ / (1 - model.explained_variance_ratio_ + 1e-10)
                wilks = np.prod(1 / (1 + eigenvalues))
                n = len(y)
                p = len(features)
                g = len(classes)

                # F-approximation for Wilks' Lambda
                df1 = p * (g - 1)
                df2 = n - (p + g) / 2
                if wilks > 0 and wilks < 1:
                    f_stat = ((1 - wilks) / wilks) * (df2 / df1)
                    p_val = 1 - stats.f.cdf(f_stat, df1, max(df2, 1))
                    st.write(f"**Wilks' Λ:** {wilks:.4f}")
                    st.write(f"**Approximate F:** {f_stat:.4f}, p = {p_val:.6f}")


# ===================================================================
# Tab 9 -- Multidimensional Scaling (MDS)
# ===================================================================

def _render_mds(df: pd.DataFrame):
    """Multidimensional Scaling visualization."""
    from sklearn.manifold import MDS
    from sklearn.metrics import pairwise_distances

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    section_header("Multidimensional Scaling (MDS)")

    selected = st.multiselect("Variables:", num_cols, default=num_cols, key="mds_cols")
    if len(selected) < 2:
        return

    c1, c2, c3 = st.columns(3)
    metric = c1.selectbox("Type:", ["Metric", "Non-metric"], key="mds_metric")
    n_dims = c2.selectbox("Dimensions:", [2, 3], key="mds_dims")
    color_col = c3.selectbox("Color by:", [None] + cat_cols, key="mds_color")

    max_samples = st.slider("Max samples:", 50, min(2000, len(df)), min(500, len(df)),
                             key="mds_max")

    if st.button("Run MDS", key="run_mds"):
        data = df[selected].dropna()
        if len(data) > max_samples:
            data = data.sample(max_samples, random_state=42)

        X = StandardScaler().fit_transform(data.values)

        with st.spinner("Computing MDS embedding..."):
            mds = MDS(n_components=n_dims,
                      metric=(metric == "Metric"),
                      random_state=42, n_init=4, max_iter=300,
                      normalized_stress="auto")
            embedding = mds.fit_transform(X)

        stress = mds.stress_
        st.metric("Stress", f"{stress:.4f}")

        emb_df = pd.DataFrame(embedding, columns=[f"Dim{i+1}" for i in range(n_dims)])
        if color_col and color_col in df.columns:
            emb_df["color"] = df[color_col].iloc[data.index].values

        if n_dims == 2:
            fig = px.scatter(emb_df, x="Dim1", y="Dim2",
                             color="color" if "color" in emb_df else None,
                             title=f"MDS 2D ({metric})", opacity=0.7)
        else:
            fig = px.scatter_3d(emb_df, x="Dim1", y="Dim2", z="Dim3",
                                color="color" if "color" in emb_df else None,
                                title=f"MDS 3D ({metric})", opacity=0.7)
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Shepard diagram
        section_header("Shepard Diagram")
        original_dist = pairwise_distances(X)
        embedded_dist = pairwise_distances(embedding)

        # Sample pairs for large datasets
        n = len(X)
        if n > 200:
            idx_pairs = np.random.choice(n * n, min(5000, n * n), replace=False)
            orig_flat = original_dist.flatten()[idx_pairs]
            emb_flat = embedded_dist.flatten()[idx_pairs]
        else:
            triu_idx = np.triu_indices(n, k=1)
            orig_flat = original_dist[triu_idx]
            emb_flat = embedded_dist[triu_idx]

        fig_shep = go.Figure()
        fig_shep.add_trace(go.Scatter(x=orig_flat, y=emb_flat, mode="markers",
                                       marker=dict(size=3, opacity=0.3), name="Pairs"))
        # Perfect fit line
        max_val = max(orig_flat.max(), emb_flat.max())
        fig_shep.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                       mode="lines", line=dict(dash="dash", color="red"),
                                       name="Perfect Fit"))
        fig_shep.update_layout(title="Shepard Diagram",
                               xaxis_title="Original Distance",
                               yaxis_title="Embedded Distance",
                               height=450)
        st.plotly_chart(fig_shep, use_container_width=True)

        # Interpretation
        if stress < 0.05:
            st.success(f"Stress = {stress:.4f}: Excellent fit")
        elif stress < 0.10:
            st.success(f"Stress = {stress:.4f}: Good fit")
        elif stress < 0.20:
            st.info(f"Stress = {stress:.4f}: Fair fit")
        else:
            st.warning(f"Stress = {stress:.4f}: Poor fit — consider more dimensions")
