"""
Machine Learning Module - Clustering, Classification, Regression, Dimensionality Reduction.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import (
    section_header, empty_state, validation_panel, interpretation_card,
    alternative_suggestion, rdl_plotly_chart, help_tip, log_analysis,
)
from modules.validation import (
    check_sample_size, check_class_balance, check_outlier_proportion,
    interpret_effect_size, interpret_silhouette,
)

from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, r2_score, mean_absolute_error, silhouette_score,
    silhouette_samples, precision_recall_curve, average_precision_score,
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


@st.cache_data(show_spinner="Training classifiers...")
def _compare_classifiers(X, y):
    """Compare all classifiers with 5-fold CV (cached). Uses Pipeline to scale inside each fold."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss",
                                           use_label_encoder=False, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1)
    except ImportError:
        pass
    try:
        from sklearn.neural_network import MLPClassifier
        models["Neural Network"] = MLPClassifier(max_iter=1000, random_state=42)
    except ImportError:
        pass
    results = []
    for name, model in models.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
            results.append({
                "Model": name,
                "Mean Accuracy": scores.mean(),
                "Std": scores.std(),
                "Min": scores.min(),
                "Max": scores.max(),
            })
        except Exception:
            pass
    return results


@st.cache_data(show_spinner="Training regressors...")
def _compare_regressors(X, y):
    """Compare all regressors with 5-fold CV (cached). Uses Pipeline to scale inside each fold."""
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(random_state=42, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1)
    except ImportError:
        pass
    try:
        from sklearn.neural_network import MLPRegressor
        models["Neural Network"] = MLPRegressor(max_iter=1000, random_state=42)
    except ImportError:
        pass
    results = []
    for name, model in models.items():
        try:
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            r2_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2")
            neg_rmse = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")
            results.append({
                "Model": name,
                "Mean R²": r2_scores.mean(),
                "Std R²": r2_scores.std(),
                "Mean RMSE": -neg_rmse.mean(),
            })
        except Exception:
            pass
    return results


def render_machine_learning(df: pd.DataFrame):
    """Render machine learning interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    tabs = st.tabs([
        "Clustering", "Classification", "Regression (ML)",
        "Dimensionality Reduction", "Model Comparison",
        "AutoTune", "SHAP Explorer", "Profiler",
    ])

    with tabs[0]:
        _render_clustering(df)
    with tabs[1]:
        _render_classification(df)
    with tabs[2]:
        _render_ml_regression(df)
    with tabs[3]:
        _render_dim_reduction(df)
    with tabs[4]:
        _render_model_comparison(df)
    with tabs[5]:
        _render_auto_tune(df)
    with tabs[6]:
        _render_shap_explorer(df)
    with tabs[7]:
        _render_ml_profiler(df)


def _render_clustering(df: pd.DataFrame):
    """Clustering analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    features = st.multiselect("Features:", num_cols, default=num_cols[:4], key="cl_features")
    if len(features) < 2:
        return

    algorithm = st.selectbox("Algorithm:", ["K-Means", "DBSCAN", "Agglomerative", "Gaussian Mixture"], key="cl_algo")
    standardize = st.checkbox("Standardize features", value=True, key="cl_std")

    data = df[features].dropna()
    X = data.values
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if algorithm == "K-Means":
        c1, c2 = st.columns(2)
        k = c1.slider("Number of clusters (k):", 2, 15, 3, key="cl_k")
        show_elbow = c2.checkbox("Show elbow plot", value=True, key="cl_elbow")

        if show_elbow:
            inertias = []
            sil_scores = []
            K_range = range(2, min(11, len(X)))
            with st.spinner("Computing elbow plot..."):
                for ki in K_range:
                    km = KMeans(n_clusters=ki, random_state=42, n_init=10)
                    km.fit(X)
                    inertias.append(km.inertia_)
                    sil_scores.append(silhouette_score(X, km.labels_))

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Plot", "Silhouette Score"))
            fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers",
                                     name="Inertia"), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(K_range), y=sil_scores, mode="lines+markers",
                                     name="Silhouette"), row=1, col=2)
            fig.update_xaxes(title_text="k", row=1, col=1)
            fig.update_xaxes(title_text="k", row=1, col=2)
            fig.update_layout(height=350)
            rdl_plotly_chart(fig)

        if st.button("Run K-Means", key="run_kmeans"):
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
            _show_cluster_results(data, X, labels, features, cat_cols, df)

    elif algorithm == "DBSCAN":
        c1, c2 = st.columns(2)
        eps = c1.slider("eps:", 0.1, 5.0, 0.5, 0.1, key="cl_eps")
        min_samples = c2.slider("min_samples:", 2, 20, 5, key="cl_min_samples")

        if st.button("Run DBSCAN", key="run_dbscan"):
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            st.write(f"**Clusters found:** {n_clusters} | **Noise points:** {n_noise}")
            if n_clusters >= 2:
                _show_cluster_results(data, X, labels, features, cat_cols, df)
            else:
                st.warning("DBSCAN found fewer than 2 clusters. Try adjusting eps or min_samples.")

    elif algorithm == "Agglomerative":
        c1, c2 = st.columns(2)
        n_clusters = c1.slider("Clusters:", 2, 15, 3, key="cl_agg_k")
        linkage = c2.selectbox("Linkage:", ["ward", "complete", "average", "single"], key="cl_linkage")

        if st.button("Run Agglomerative", key="run_agg"):
            model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
            labels = model.fit_predict(X)
            _show_cluster_results(data, X, labels, features, cat_cols, df)

    elif algorithm == "Gaussian Mixture":
        from sklearn.mixture import GaussianMixture

        c1, c2 = st.columns(2)
        n_components = c1.slider("Number of components:", 2, 15, 3, key="cl_gmm_k")
        cov_type = c2.selectbox("Covariance type:", ["full", "tied", "diag", "spherical"], key="cl_gmm_cov")
        show_bic_aic = st.checkbox("Show BIC/AIC plot", value=True, key="cl_gmm_bic")

        if show_bic_aic:
            bics = []
            aics = []
            K_range = range(2, min(16, len(X)))
            with st.spinner("Computing BIC/AIC..."):
                for ki in K_range:
                    gm = GaussianMixture(n_components=ki, covariance_type=cov_type, random_state=42)
                    gm.fit(X)
                    bics.append(gm.bic(X))
                    aics.append(gm.aic(X))

            fig = make_subplots(rows=1, cols=2, subplot_titles=("BIC vs k", "AIC vs k"))
            fig.add_trace(go.Scatter(x=list(K_range), y=bics, mode="lines+markers",
                                     name="BIC"), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(K_range), y=aics, mode="lines+markers",
                                     name="AIC"), row=1, col=2)
            fig.update_xaxes(title_text="k", row=1, col=1)
            fig.update_xaxes(title_text="k", row=1, col=2)
            fig.update_layout(height=350)
            rdl_plotly_chart(fig)

        if st.button("Run Gaussian Mixture", key="run_gmm"):
            model = GaussianMixture(n_components=n_components, covariance_type=cov_type, random_state=42)
            model.fit(X)
            labels = model.predict(X)
            probs = model.predict_proba(X)

            _show_cluster_results(data, X, labels, features, cat_cols, df)

            # Show cluster probabilities (soft assignments)
            section_header("Cluster Probabilities (Soft Assignments)")
            prob_df = pd.DataFrame(probs, columns=[f"Cluster {i}" for i in range(n_components)])
            prob_df.index = data.index
            st.dataframe(prob_df.head(50).round(4), use_container_width=True)
            if len(prob_df) > 50:
                st.caption(f"Showing first 50 of {len(prob_df):,} rows.")

            # Cluster ellipses on 2D scatter if applicable
            if X.shape[1] == 2:
                section_header("Cluster Ellipses (2D)")
                fig_ell = px.scatter(
                    data.assign(Cluster=labels.astype(str)),
                    x=features[0], y=features[1], color="Cluster",
                    title="Gaussian Mixture Clusters with Covariance Ellipses",
                    opacity=0.6,
                )
                # Draw covariance ellipses
                for i in range(n_components):
                    if cov_type == "full":
                        cov = model.covariances_[i]
                    elif cov_type == "tied":
                        cov = model.covariances_
                    elif cov_type == "diag":
                        cov = np.diag(model.covariances_[i])
                    else:  # spherical
                        cov = np.eye(2) * model.covariances_[i]

                    mean = model.means_[i]
                    # Compute ellipse from covariance
                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                        theta = np.linspace(0, 2 * np.pi, 100)
                        # 95% confidence ellipse (chi-squared with 2 df at 0.95 = 5.991)
                        scale = np.sqrt(5.991)
                        ellipse_x = (scale * np.sqrt(eigenvalues[0]) * np.cos(theta) * np.cos(angle)
                                     - scale * np.sqrt(eigenvalues[1]) * np.sin(theta) * np.sin(angle) + mean[0])
                        ellipse_y = (scale * np.sqrt(eigenvalues[0]) * np.cos(theta) * np.sin(angle)
                                     + scale * np.sqrt(eigenvalues[1]) * np.sin(theta) * np.cos(angle) + mean[1])
                        from modules.ui_helpers import _RDL_COLORWAY
                        fig_ell.add_trace(go.Scatter(
                            x=ellipse_x, y=ellipse_y, mode="lines",
                            line=dict(color=_RDL_COLORWAY[i % len(_RDL_COLORWAY)], width=2, dash="dash"),
                            name=f"Cluster {i} (95% CI)", showlegend=True,
                        ))
                    except Exception:
                        continue

                fig_ell.update_layout(height=500)
                rdl_plotly_chart(fig_ell)
            elif X.shape[1] > 2:
                # Use PCA to project to 2D and draw ellipses
                section_header("Cluster Ellipses (PCA Projection)")
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)

                plot_df = pd.DataFrame(X_2d, columns=["PC1", "PC2"])
                plot_df["Cluster"] = labels.astype(str)
                fig_ell = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                                     title="Gaussian Mixture Clusters (PCA Projection)",
                                     opacity=0.6)

                # Transform means and covariances to PCA space for ellipses
                for i in range(n_components):
                    if cov_type == "full":
                        cov = model.covariances_[i]
                    elif cov_type == "tied":
                        cov = model.covariances_
                    elif cov_type == "diag":
                        cov = np.diag(model.covariances_[i])
                    else:  # spherical
                        cov = np.eye(X.shape[1]) * model.covariances_[i]

                    mean_pca = pca.transform(model.means_[i].reshape(1, -1))[0]
                    cov_pca = pca.components_ @ cov @ pca.components_.T

                    try:
                        eigenvalues, eigenvectors = np.linalg.eigh(cov_pca)
                        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                        theta = np.linspace(0, 2 * np.pi, 100)
                        scale = np.sqrt(5.991)
                        ellipse_x = (scale * np.sqrt(eigenvalues[0]) * np.cos(theta) * np.cos(angle)
                                     - scale * np.sqrt(eigenvalues[1]) * np.sin(theta) * np.sin(angle) + mean_pca[0])
                        ellipse_y = (scale * np.sqrt(eigenvalues[0]) * np.cos(theta) * np.sin(angle)
                                     + scale * np.sqrt(eigenvalues[1]) * np.sin(theta) * np.cos(angle) + mean_pca[1])
                        from modules.ui_helpers import _RDL_COLORWAY
                        fig_ell.add_trace(go.Scatter(
                            x=ellipse_x, y=ellipse_y, mode="lines",
                            line=dict(color=_RDL_COLORWAY[i % len(_RDL_COLORWAY)], width=2, dash="dash"),
                            name=f"Cluster {i} (95% CI)", showlegend=True,
                        ))
                    except Exception:
                        continue

                fig_ell.update_layout(height=500)
                rdl_plotly_chart(fig_ell)


def _show_cluster_results(data, X, labels, features, cat_cols, df):
    """Display clustering results."""
    data = data.copy()
    data["Cluster"] = labels.astype(str)

    # Metrics
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.discard(-1)
    if len(unique_labels) >= 2:
        mask = labels != -1
        if mask.sum() > len(unique_labels):
            sil = silhouette_score(X[mask], labels[mask])
            st.metric("Silhouette Score", f"{sil:.4f}")
            try:
                interpretation_card(interpret_silhouette(sil))
            except Exception:
                pass

    # 2D visualization (PCA if >2 features)
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)
        plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        plot_df["Cluster"] = labels.astype(str)
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="Cluster",
                         title="Clusters (PCA projection)", opacity=0.7)
    else:
        fig = px.scatter(data, x=features[0], y=features[1], color="Cluster",
                         title="Clusters", opacity=0.7)
    fig.update_layout(height=500)
    rdl_plotly_chart(fig)

    # Cluster profiles
    section_header("Cluster Profiles")
    profile = data.groupby("Cluster")[features].mean().round(4)
    st.dataframe(profile, use_container_width=True)

    # Cluster sizes
    sizes = data["Cluster"].value_counts().sort_index()
    fig = px.bar(x=sizes.index, y=sizes.values, labels={"x": "Cluster", "y": "Count"},
                 title="Cluster Sizes", color=sizes.index)
    fig.update_layout(height=300)
    rdl_plotly_chart(fig)


def _render_classification(df: pd.DataFrame):
    """Classification analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Target: categorical or low-cardinality numeric
    target_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 15]
    if not target_options:
        empty_state("No suitable target variable found.")
        return

    target = st.selectbox("Target variable:", target_options, key="clf_target")
    features = st.multiselect("Features:", [c for c in num_cols if c != target], key="clf_features")

    if not features:
        st.info("Select feature variables.")
        return

    _clf_algos = [
        "Logistic Regression", "Random Forest", "Gradient Boosting",
        "SVM", "KNN", "Naive Bayes", "Decision Tree",
    ]
    try:
        import xgboost  # noqa: F401
        _clf_algos.append("XGBoost")
    except ImportError:
        pass
    try:
        import lightgbm  # noqa: F401
        _clf_algos.append("LightGBM")
    except ImportError:
        pass
    _clf_algos.append("Neural Network (MLP)")
    algorithm = st.selectbox("Algorithm:", _clf_algos, key="clf_algo")

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test size:", 0.1, 0.5, 0.2, 0.05, key="clf_test")
    cv_folds = c2.number_input("CV folds:", 2, 20, 5, key="clf_cv")
    random_state = c3.number_input("Random state:", 0, 999, 42, key="clf_rs")

    # Algorithm-specific hyperparameters
    params = {}
    with st.expander("Hyperparameters"):
        if algorithm == "Random Forest":
            params["n_estimators"] = st.slider("n_estimators:", 10, 500, 100, key="clf_rf_n")
            params["max_depth"] = st.slider("max_depth:", 1, 30, 10, key="clf_rf_d")
        elif algorithm == "Gradient Boosting":
            params["n_estimators"] = st.slider("n_estimators:", 10, 500, 100, key="clf_gb_n")
            params["learning_rate"] = st.slider("learning_rate:", 0.01, 1.0, 0.1, key="clf_gb_lr")
            params["max_depth"] = st.slider("max_depth:", 1, 15, 3, key="clf_gb_d")
        elif algorithm == "SVM":
            params["C"] = st.slider("C:", 0.01, 100.0, 1.0, key="clf_svm_c")
            params["kernel"] = st.selectbox("Kernel:", ["rbf", "linear", "poly"], key="clf_svm_k")
        elif algorithm == "KNN":
            params["n_neighbors"] = st.slider("n_neighbors:", 1, 30, 5, key="clf_knn_k")
        elif algorithm == "Decision Tree":
            params["max_depth"] = st.slider("max_depth:", 1, 30, 10, key="clf_dt_d")
        elif algorithm == "XGBoost":
            params["n_estimators"] = st.slider("n_estimators:", 10, 500, 100, key="clf_xgb_n")
            params["learning_rate"] = st.slider("learning_rate:", 0.01, 1.0, 0.1, key="clf_xgb_lr")
            params["max_depth"] = st.slider("max_depth:", 1, 15, 6, key="clf_xgb_d")
        elif algorithm == "LightGBM":
            params["n_estimators"] = st.slider("n_estimators:", 10, 500, 100, key="clf_lgb_n")
            params["learning_rate"] = st.slider("learning_rate:", 0.01, 1.0, 0.1, key="clf_lgb_lr")
            params["max_depth"] = st.slider("max_depth:", -1, 15, -1, key="clf_lgb_d")
        elif algorithm == "Neural Network (MLP)":
            hidden_str = st.text_input("Hidden layers (comma-sep):", "100,50",
                                        key="clf_mlp_hidden")
            try:
                params["hidden_layer_sizes"] = tuple(int(x.strip()) for x in hidden_str.split(","))
            except ValueError:
                params["hidden_layer_sizes"] = (100,)
            params["activation"] = st.selectbox("Activation:", ["relu", "tanh", "logistic"],
                                                 key="clf_mlp_act")

    if st.button("Train Model", key="train_clf"):
        data = df[features + [target]].dropna()
        X = data[features].values
        y_raw = data[target]

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = le.classes_

        # ── Data-quality validation ──────────────────────────────────
        try:
            checks = [
                check_sample_size(len(X), "ml-classification"),
                check_class_balance(y),
            ]
            validation_panel(checks, title="Data Quality")
            if any(c.status in ("warn", "fail") for c in checks if "Class" in c.name):
                alternative_suggestion(
                    "Class imbalance detected",
                    ["Use class_weight='balanced'", "Consider SMOTE oversampling"],
                )
        except Exception:
            pass
        # ────────────────────────────────────────────────────────────

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y)
        except ValueError:
            st.warning("Stratified split failed (too few samples in a class). Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state)

        # Scale after split to prevent data leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build model
        model = _build_classifier(algorithm, params, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics — use Pipeline for CV so scaling is done inside each fold
        accuracy = (y_pred == y_test).mean()
        cv_pipe = Pipeline([("scaler", StandardScaler()), ("model", _build_classifier(algorithm, params, random_state))])
        cv_scores = cross_val_score(cv_pipe, X, y, cv=cv_folds, scoring="accuracy")

        c1, c2, c3 = st.columns(3)
        c1.metric("Test Accuracy", f"{accuracy:.4f}")
        c2.metric(f"CV Accuracy ({cv_folds}-fold)", f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        c3.metric("Training Accuracy", f"{model.score(X_train, y_train):.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=[str(c) for c in classes], y=[str(c) for c in classes],
                        labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
        rdl_plotly_chart(fig)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        # ROC curve (binary)
        if len(classes) == 2:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(X_test)
            else:
                y_prob = y_pred

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc:.4f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="gray"), name="Random"))
            fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=400)
            rdl_plotly_chart(fig)

            # Precision-Recall curve
            try:
                if hasattr(model, "predict_proba"):
                    pr_prob = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    pr_prob = model.decision_function(X_test)
                else:
                    pr_prob = y_pred

                precision_vals, recall_vals, _ = precision_recall_curve(y_test, pr_prob)
                avg_precision = average_precision_score(y_test, pr_prob)
                baseline = y_test.sum() / len(y_test)

                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=recall_vals, y=precision_vals,
                    name=f"PR Curve (AUPRC={avg_precision:.4f})",
                    fill="tozeroy", fillcolor="rgba(99, 102, 241, 0.1)",
                ))
                fig_pr.add_trace(go.Scatter(
                    x=[0, 1], y=[baseline, baseline],
                    line=dict(dash="dash", color="gray"),
                    name=f"Baseline ({baseline:.3f})",
                ))
                fig_pr.update_layout(
                    title="Precision-Recall Curve",
                    xaxis_title="Recall", yaxis_title="Precision",
                    height=400,
                )
                rdl_plotly_chart(fig_pr)
            except Exception:
                pass

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            imp_df = pd.DataFrame({"Feature": features, "Importance": imp}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance")
            fig.update_layout(height=max(300, len(features) * 25))
            rdl_plotly_chart(fig)
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            imp_df = pd.DataFrame({"Feature": features, "Importance": coefs}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance (|coefficients|)")
            fig.update_layout(height=max(300, len(features) * 25))
            rdl_plotly_chart(fig)

        # SHAP analysis
        try:
            import shap
            with st.expander("SHAP Explainability"):
                try:
                    if algorithm in ("Random Forest", "Gradient Boosting", "Decision Tree", "XGBoost", "LightGBM"):
                        explainer = shap.TreeExplainer(model)
                    else:
                        background = shap.sample(pd.DataFrame(X_train, columns=features), min(100, len(X_train)))
                        explainer = shap.Explainer(model.predict, background, feature_names=features)
                    shap_values = explainer(pd.DataFrame(X_test, columns=features))

                    # Mean absolute SHAP values
                    if shap_values.values.ndim == 3:
                        vals = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
                    else:
                        vals = np.abs(shap_values.values).mean(axis=0)
                    shap_imp = pd.DataFrame({"Feature": features, "Mean |SHAP|": vals}).sort_values("Mean |SHAP|", ascending=True)
                    fig = px.bar(shap_imp, x="Mean |SHAP|", y="Feature", orientation="h",
                                 title="SHAP Feature Importance")
                    fig.update_layout(height=max(300, len(features) * 30))
                    rdl_plotly_chart(fig)
                except Exception as e:
                    st.warning(f"SHAP analysis failed: {e}")
        except ImportError:
            pass

        # Loss curve for neural network
        if algorithm == "Neural Network (MLP)" and hasattr(model, 'loss_curve_'):
            with st.expander("Training Loss Curve"):
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=model.loss_curve_, mode="lines",
                                              name="Loss"))
                fig_loss.update_layout(title="Training Loss", xaxis_title="Iteration",
                                       yaxis_title="Loss", height=350)
                rdl_plotly_chart(fig_loss)


def _build_classifier(algorithm, params, random_state):
    """Build classifier from algorithm name and params."""
    if algorithm == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    elif algorithm == "Random Forest":
        return RandomForestClassifier(random_state=random_state, **params)
    elif algorithm == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=random_state, **params)
    elif algorithm == "SVM":
        return SVC(random_state=random_state, probability=True, **params)
    elif algorithm == "KNN":
        return KNeighborsClassifier(**params)
    elif algorithm == "Naive Bayes":
        return GaussianNB()
    elif algorithm == "Decision Tree":
        return DecisionTreeClassifier(random_state=random_state, **params)
    elif algorithm == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=random_state, eval_metric="logloss",
                             use_label_encoder=False, verbosity=0, **params)
    elif algorithm == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=random_state, verbose=-1, **params)
    elif algorithm == "Neural Network (MLP)":
        from sklearn.neural_network import MLPClassifier
        hidden = params.get("hidden_layer_sizes", (100,))
        activation = params.get("activation", "relu")
        return MLPClassifier(hidden_layer_sizes=hidden, activation=activation,
                             max_iter=1000, random_state=random_state)


def _render_ml_regression(df: pd.DataFrame):
    """ML Regression analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    target = st.selectbox("Target:", num_cols, key="reg_target")
    features = st.multiselect("Features:", [c for c in num_cols if c != target], key="reg_features")

    if not features:
        st.info("Select feature variables.")
        return

    _reg_algos = [
        "Linear Regression", "Ridge", "Lasso", "ElasticNet",
        "Random Forest", "Gradient Boosting", "SVR", "KNN",
    ]
    try:
        import xgboost  # noqa: F401
        _reg_algos.append("XGBoost")
    except ImportError:
        pass
    try:
        import lightgbm  # noqa: F401
        _reg_algos.append("LightGBM")
    except ImportError:
        pass
    _reg_algos.append("Neural Network (MLP)")
    algorithm = st.selectbox("Algorithm:", _reg_algos, key="reg_algo")

    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test size:", 0.1, 0.5, 0.2, 0.05, key="reg_test")
    cv_folds = c2.number_input("CV folds:", 2, 20, 5, key="reg_cv")
    random_state = c3.number_input("Random state:", 0, 999, 42, key="reg_rs")

    # Hyperparameters
    params = {}
    with st.expander("Hyperparameters"):
        if algorithm == "Ridge":
            params["alpha"] = st.slider("alpha:", 0.01, 100.0, 1.0, key="reg_ridge_a")
        elif algorithm == "Lasso":
            params["alpha"] = st.slider("alpha:", 0.001, 10.0, 1.0, key="reg_lasso_a")
        elif algorithm == "ElasticNet":
            params["alpha"] = st.slider("alpha:", 0.001, 10.0, 1.0, key="reg_en_a")
            params["l1_ratio"] = st.slider("l1_ratio:", 0.0, 1.0, 0.5, key="reg_en_l1")
        elif algorithm == "Random Forest":
            params["n_estimators"] = st.slider("n_estimators:", 10, 500, 100, key="reg_rf_n")
            params["max_depth"] = st.slider("max_depth:", 1, 30, 10, key="reg_rf_d")
        elif algorithm == "Gradient Boosting":
            params["n_estimators"] = st.slider("n_estimators:", 10, 500, 100, key="reg_gb_n")
            params["learning_rate"] = st.slider("learning_rate:", 0.01, 1.0, 0.1, key="reg_gb_lr")
        elif algorithm == "SVR":
            params["C"] = st.slider("C:", 0.01, 100.0, 1.0, key="reg_svr_c")
            params["kernel"] = st.selectbox("Kernel:", ["rbf", "linear", "poly"], key="reg_svr_k")
        elif algorithm == "KNN":
            params["n_neighbors"] = st.slider("n_neighbors:", 1, 30, 5, key="reg_knn_k")
        elif algorithm == "XGBoost":
            params["n_estimators"] = st.slider("n_estimators:", 10, 500, 100, key="reg_xgb_n")
            params["learning_rate"] = st.slider("learning_rate:", 0.01, 1.0, 0.1, key="reg_xgb_lr")
            params["max_depth"] = st.slider("max_depth:", 1, 15, 6, key="reg_xgb_d")
        elif algorithm == "LightGBM":
            params["n_estimators"] = st.slider("n_estimators:", 10, 500, 100, key="reg_lgb_n")
            params["learning_rate"] = st.slider("learning_rate:", 0.01, 1.0, 0.1, key="reg_lgb_lr")
            params["max_depth"] = st.slider("max_depth:", -1, 15, -1, key="reg_lgb_d")
        elif algorithm == "Neural Network (MLP)":
            hidden_str = st.text_input("Hidden layers (comma-sep):", "100,50",
                                        key="reg_mlp_hidden")
            try:
                params["hidden_layer_sizes"] = tuple(int(x.strip()) for x in hidden_str.split(","))
            except ValueError:
                params["hidden_layer_sizes"] = (100,)
            params["activation"] = st.selectbox("Activation:", ["relu", "tanh", "logistic"],
                                                 key="reg_mlp_act")

    if st.button("Train Model", key="train_reg"):
        data = df[features + [target]].dropna()
        X = data[features].values
        y = data[target].values

        # ── Data-quality validation ──────────────────────────────────
        try:
            checks = [
                check_sample_size(len(X), "ml-regression"),
                check_outlier_proportion(y),
            ]
            validation_panel(checks, title="Data Quality")
        except Exception:
            pass
        # ────────────────────────────────────────────────────────────

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        # Scale after split to prevent data leakage
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Build model
        model = _build_regressor(algorithm, params, random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_train_pred = model.predict(X_train)

        # Metrics — use Pipeline for CV so scaling is done inside each fold
        r2 = r2_score(y_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(features) - 1)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if (y_test != 0).all() else np.nan
        cv_pipe = Pipeline([("scaler", StandardScaler()), ("model", _build_regressor(algorithm, params, random_state))])
        cv_scores = cross_val_score(cv_pipe, X, y, cv=cv_folds, scoring="r2")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{r2:.4f}")
        c2.metric("Adj R²", f"{adj_r2:.4f}")
        c3.metric("RMSE", f"{rmse:.4f}")
        c4.metric("MAE", f"{mae:.4f}")

        st.write(f"**MAPE:** {mape:.2f}%  |  **CV R² ({cv_folds}-fold):** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        st.write(f"**Training R²:** {r2_score(y_train, y_train_pred):.4f}")

        # Predicted vs Actual
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Predicted vs Actual", "Residuals"))
        fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                 marker=dict(size=5, opacity=0.7),
                                 name="Predictions"), row=1, col=1)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode="lines", line=dict(color="red", dash="dash"),
                                 name="Perfect"), row=1, col=1)

        residuals = y_test - y_pred
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                 marker=dict(size=5, opacity=0.7),
                                 name="Residuals"), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_xaxes(title_text="Actual", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_xaxes(title_text="Predicted", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_layout(height=400)
        rdl_plotly_chart(fig)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            imp_df = pd.DataFrame({"Feature": features, "Importance": imp}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance")
            fig.update_layout(height=max(300, len(features) * 25))
            rdl_plotly_chart(fig)
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs}).sort_values("Coefficient")
            fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                         title="Coefficients")
            fig.update_layout(height=max(300, len(features) * 25))
            rdl_plotly_chart(fig)

        # SHAP analysis for regression
        try:
            import shap
            with st.expander("SHAP Explainability"):
                try:
                    if algorithm in ("Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"):
                        explainer = shap.TreeExplainer(model)
                    else:
                        background = shap.sample(pd.DataFrame(X_train, columns=features), min(100, len(X_train)))
                        explainer = shap.Explainer(model.predict, background, feature_names=features)
                    shap_values = explainer(pd.DataFrame(X_test, columns=features))
                    vals = np.abs(shap_values.values).mean(axis=0)
                    shap_imp = pd.DataFrame({"Feature": features, "Mean |SHAP|": vals}).sort_values("Mean |SHAP|", ascending=True)
                    fig = px.bar(shap_imp, x="Mean |SHAP|", y="Feature", orientation="h",
                                 title="SHAP Feature Importance")
                    fig.update_layout(height=max(300, len(features) * 30))
                    rdl_plotly_chart(fig)
                except Exception as e:
                    st.warning(f"SHAP analysis failed: {e}")
        except ImportError:
            pass

        # Loss curve for neural network
        if algorithm == "Neural Network (MLP)" and hasattr(model, 'loss_curve_'):
            with st.expander("Training Loss Curve"):
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(y=model.loss_curve_, mode="lines",
                                              name="Loss"))
                fig_loss.update_layout(title="Training Loss", xaxis_title="Iteration",
                                       yaxis_title="Loss", height=350)
                rdl_plotly_chart(fig_loss)


def _build_regressor(algorithm, params, random_state):
    """Build regressor from algorithm name and params."""
    if algorithm == "Linear Regression":
        return LinearRegression()
    elif algorithm == "Ridge":
        return Ridge(**params)
    elif algorithm == "Lasso":
        return Lasso(**params)
    elif algorithm == "ElasticNet":
        return ElasticNet(**params)
    elif algorithm == "Random Forest":
        return RandomForestRegressor(random_state=random_state, **params)
    elif algorithm == "Gradient Boosting":
        return GradientBoostingRegressor(random_state=random_state, **params)
    elif algorithm == "SVR":
        return SVR(**params)
    elif algorithm == "KNN":
        return KNeighborsRegressor(**params)
    elif algorithm == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=random_state, verbosity=0, **params)
    elif algorithm == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(random_state=random_state, verbose=-1, **params)
    elif algorithm == "Neural Network (MLP)":
        from sklearn.neural_network import MLPRegressor
        hidden = params.get("hidden_layer_sizes", (100,))
        activation = params.get("activation", "relu")
        return MLPRegressor(hidden_layer_sizes=hidden, activation=activation,
                            max_iter=1000, random_state=random_state)


def _render_dim_reduction(df: pd.DataFrame):
    """Dimensionality reduction visualization."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 3:
        empty_state("Need at least 3 numeric columns.")
        return

    features = st.multiselect("Features:", num_cols, default=num_cols, key="dr_features")
    if len(features) < 3:
        return

    method = st.selectbox("Method:", ["PCA", "t-SNE", "UMAP"], key="dr_method")
    color_col = st.selectbox("Color by:", [None] + cat_cols, key="dr_color")
    n_dims = st.selectbox("Dimensions:", [2, 3], key="dr_dims")

    # UMAP-specific parameters
    if method == "UMAP":
        with st.expander("UMAP Parameters"):
            umap_n_neighbors = st.slider("n_neighbors:", 2, 200, 15, key="dr_umap_nn")
            umap_min_dist = st.slider("min_dist:", 0.0, 1.0, 0.1, 0.05, key="dr_umap_md")
            umap_metric = st.selectbox("Metric:", ["euclidean", "manhattan", "cosine", "correlation"], key="dr_umap_metric")

    if st.button("Run", key="run_dr"):
        data = df[features].dropna()
        X = StandardScaler().fit_transform(data.values)

        if method == "PCA":
            reducer = PCA(n_components=n_dims)
            embedding = reducer.fit_transform(X)
            labels = [f"PC{i+1} ({reducer.explained_variance_ratio_[i]*100:.1f}%)" for i in range(n_dims)]
        elif method == "t-SNE":
            max_n = min(len(data), 5000)
            if len(data) > max_n:
                idx = np.random.choice(len(data), max_n, replace=False)
                X = X[idx]
                data = data.iloc[idx]
            perplexity = min(30, len(X) - 1)
            reducer = TSNE(n_components=n_dims, perplexity=perplexity, random_state=42)
            embedding = reducer.fit_transform(X)
            labels = [f"Dim{i+1}" for i in range(n_dims)]
        elif method == "UMAP":
            try:
                import umap
            except ImportError:
                st.error("umap-learn not installed. Run: `pip install umap-learn`")
                return
            max_n = min(len(data), 10000)
            if len(data) > max_n:
                idx = np.random.choice(len(data), max_n, replace=False)
                X = X[idx]
                data = data.iloc[idx]
            with st.spinner("Computing UMAP embedding..."):
                reducer = umap.UMAP(n_components=n_dims, n_neighbors=umap_n_neighbors,
                                    min_dist=umap_min_dist, metric=umap_metric, random_state=42)
                embedding = reducer.fit_transform(X)
            labels = [f"UMAP{i+1}" for i in range(n_dims)]

        emb_df = pd.DataFrame(embedding, columns=labels)
        if color_col and color_col in df.columns:
            emb_df["color"] = df[color_col].iloc[data.index].values

        if n_dims == 2:
            fig = px.scatter(emb_df, x=labels[0], y=labels[1],
                             color="color" if "color" in emb_df else None,
                             title=f"{method} 2D", opacity=0.7)
        else:
            fig = px.scatter_3d(emb_df, x=labels[0], y=labels[1], z=labels[2],
                                color="color" if "color" in emb_df else None,
                                title=f"{method} 3D", opacity=0.7)
        fig.update_layout(height=600)
        rdl_plotly_chart(fig)


def _render_model_comparison(df: pd.DataFrame):
    """Compare multiple models on the same dataset."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    task = st.selectbox("Task:", ["Classification", "Regression"], key="mc_task")

    if task == "Classification":
        target_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 15]
        if not target_options:
            empty_state("No suitable target variable found.")
            return

        target = st.selectbox("Target:", target_options, key="mc_clf_target")
        features = st.multiselect("Features:", [c for c in num_cols if c != target], key="mc_clf_features")
        if not features:
            return

        if st.button("Compare All Classifiers", key="run_mc_clf"):
            data = df[features + [target]].dropna()
            X = data[features].values
            y = LabelEncoder().fit_transform(data[target])

            results = _compare_classifiers(X, y)

            results_df = pd.DataFrame(results).sort_values("Mean Accuracy", ascending=False)
            st.dataframe(results_df.round(4), use_container_width=True, hide_index=True)

            fig = px.bar(results_df, x="Model", y="Mean Accuracy",
                         error_y="Std", title="Model Comparison (5-Fold CV)",
                         color="Mean Accuracy", color_continuous_scale="Blues")
            fig.update_layout(height=400)
            rdl_plotly_chart(fig)

            # ── Best model interpretation ────────────────────────────
            try:
                if not results_df.empty:
                    best = results_df.iloc[0]
                    interpretation_card({
                        "title": "Best Model",
                        "body": (
                            f"{best['Model']} achieved a mean 5-fold CV accuracy of "
                            f"{best['Mean Accuracy']:.4f} (std {best['Std']:.4f})."
                        ),
                        "detail": (
                            f"Accuracy ranged from {best['Min']:.4f} to {best['Max']:.4f} "
                            f"across folds."
                        ),
                    })
            except Exception:
                pass

    else:  # Regression
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns.")
            return

        target = st.selectbox("Target:", num_cols, key="mc_reg_target")
        features = st.multiselect("Features:", [c for c in num_cols if c != target], key="mc_reg_features")
        if not features:
            return

        if st.button("Compare All Regressors", key="run_mc_reg"):
            data = df[features + [target]].dropna()
            X = data[features].values
            y = data[target].values

            results = _compare_regressors(X, y)

            results_df = pd.DataFrame(results).sort_values("Mean R²", ascending=False)
            st.dataframe(results_df.round(4), use_container_width=True, hide_index=True)

            fig = px.bar(results_df, x="Model", y="Mean R²",
                         error_y="Std R²", title="Model Comparison (5-Fold CV)",
                         color="Mean R²", color_continuous_scale="Blues")
            fig.update_layout(height=400)
            rdl_plotly_chart(fig)

            # ── Best model interpretation ────────────────────────────
            try:
                if not results_df.empty:
                    best = results_df.iloc[0]
                    interpretation_card(interpret_effect_size(
                        best["Mean R²"], "r-squared",
                    ))
                    interpretation_card({
                        "title": "Best Model",
                        "body": (
                            f"{best['Model']} achieved a mean 5-fold CV R² of "
                            f"{best['Mean R²']:.4f} (std {best['Std R²']:.4f}) "
                            f"with mean RMSE of {best['Mean RMSE']:.4f}."
                        ),
                    })
            except Exception:
                pass


# ─── AutoTune Tab ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_autotune_search(
    _X_bytes, _y_bytes, n_features, task_type, algorithms, search_strategy,
    n_iter, scoring, cv_folds, random_state, param_configs,
):
    """Run hyperparameter search (cached).

    _X_bytes / _y_bytes are bytes representations so Streamlit can hash them.
    """
    X = np.frombuffer(_X_bytes, dtype=np.float64).reshape(-1, n_features)
    y = np.frombuffer(_y_bytes, dtype=np.float64)

    if task_type == "Classification":
        y = y.astype(int)

    all_results = []

    for algo_name in algorithms:
        model, param_grid = _autotune_model_and_grid(
            algo_name, task_type, random_state, param_configs.get(algo_name, {}),
        )
        if model is None:
            continue

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        # Prefix param names for pipeline
        pipe_grid = {f"model__{k}": v for k, v in param_grid.items()}

        try:
            if search_strategy == "Grid Search":
                searcher = GridSearchCV(
                    pipe, pipe_grid, cv=cv_folds, scoring=scoring,
                    n_jobs=-1, error_score="raise",
                )
            else:
                searcher = RandomizedSearchCV(
                    pipe, pipe_grid, n_iter=min(n_iter, _grid_size(pipe_grid)),
                    cv=cv_folds, scoring=scoring, n_jobs=-1,
                    random_state=random_state, error_score="raise",
                )
            searcher.fit(X, y)

            cv_results = searcher.cv_results_
            for i in range(len(cv_results["mean_test_score"])):
                row = {
                    "Algorithm": algo_name,
                    "Mean Score": cv_results["mean_test_score"][i],
                    "Std Score": cv_results["std_test_score"][i],
                    "Rank": cv_results["rank_test_score"][i],
                }
                raw_params = cv_results["params"][i]
                clean_params = {
                    k.replace("model__", ""): v for k, v in raw_params.items()
                }
                row["Params"] = str(clean_params)
                all_results.append(row)
        except Exception as exc:
            all_results.append({
                "Algorithm": algo_name,
                "Mean Score": None,
                "Std Score": None,
                "Rank": None,
                "Params": f"Error: {exc}",
            })

    return all_results


def _grid_size(grid):
    """Compute total number of combinations in a parameter grid."""
    size = 1
    for v in grid.values():
        size *= len(v)
    return size


def _autotune_model_and_grid(algo_name, task_type, random_state, cfg):
    """Return (estimator, param_grid) for the given algorithm and user config."""
    if task_type == "Classification":
        builders = {
            "Random Forest": (
                RandomForestClassifier(random_state=random_state),
                {
                    "n_estimators": list(range(
                        cfg.get("n_estimators_min", 50),
                        cfg.get("n_estimators_max", 500) + 1,
                        50,
                    )),
                    "max_depth": list(range(
                        cfg.get("max_depth_min", 2),
                        cfg.get("max_depth_max", 30) + 1,
                        4,
                    )),
                    "min_samples_split": list(range(
                        cfg.get("min_samples_split_min", 2),
                        cfg.get("min_samples_split_max", 20) + 1,
                        3,
                    )),
                },
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(random_state=random_state),
                {
                    "n_estimators": list(range(
                        cfg.get("n_estimators_min", 50),
                        cfg.get("n_estimators_max", 500) + 1,
                        50,
                    )),
                    "learning_rate": [
                        round(v, 2)
                        for v in np.arange(
                            cfg.get("learning_rate_min", 0.01),
                            cfg.get("learning_rate_max", 0.5) + 0.01,
                            0.05,
                        ).tolist()
                    ],
                    "max_depth": list(range(
                        cfg.get("max_depth_min", 2),
                        cfg.get("max_depth_max", 15) + 1,
                        2,
                    )),
                },
            ),
            "SVM": (
                SVC(random_state=random_state, probability=True),
                {
                    "C": [
                        round(v, 2)
                        for v in np.logspace(
                            np.log10(cfg.get("C_min", 0.01)),
                            np.log10(cfg.get("C_max", 100)),
                            10,
                        ).tolist()
                    ],
                    "kernel": cfg.get("kernels", ["rbf", "linear", "poly"]),
                },
            ),
            "KNN": (
                KNeighborsClassifier(),
                {
                    "n_neighbors": list(range(
                        cfg.get("n_neighbors_min", 1),
                        cfg.get("n_neighbors_max", 30) + 1,
                        2,
                    )),
                },
            ),
        }
        # XGBoost
        if algo_name == "XGBoost":
            try:
                from xgboost import XGBClassifier
                return (
                    XGBClassifier(
                        random_state=random_state, eval_metric="logloss",
                        use_label_encoder=False, verbosity=0,
                    ),
                    {
                        "n_estimators": list(range(
                            cfg.get("n_estimators_min", 50),
                            cfg.get("n_estimators_max", 500) + 1,
                            50,
                        )),
                        "learning_rate": [
                            round(v, 2)
                            for v in np.arange(
                                cfg.get("learning_rate_min", 0.01),
                                cfg.get("learning_rate_max", 0.5) + 0.01,
                                0.05,
                            ).tolist()
                        ],
                        "max_depth": list(range(
                            cfg.get("max_depth_min", 2),
                            cfg.get("max_depth_max", 15) + 1,
                            2,
                        )),
                    },
                )
            except ImportError:
                return None, None
    else:
        builders = {
            "Random Forest": (
                RandomForestRegressor(random_state=random_state),
                {
                    "n_estimators": list(range(
                        cfg.get("n_estimators_min", 50),
                        cfg.get("n_estimators_max", 500) + 1,
                        50,
                    )),
                    "max_depth": list(range(
                        cfg.get("max_depth_min", 2),
                        cfg.get("max_depth_max", 30) + 1,
                        4,
                    )),
                    "min_samples_split": list(range(
                        cfg.get("min_samples_split_min", 2),
                        cfg.get("min_samples_split_max", 20) + 1,
                        3,
                    )),
                },
            ),
            "Gradient Boosting": (
                GradientBoostingRegressor(random_state=random_state),
                {
                    "n_estimators": list(range(
                        cfg.get("n_estimators_min", 50),
                        cfg.get("n_estimators_max", 500) + 1,
                        50,
                    )),
                    "learning_rate": [
                        round(v, 2)
                        for v in np.arange(
                            cfg.get("learning_rate_min", 0.01),
                            cfg.get("learning_rate_max", 0.5) + 0.01,
                            0.05,
                        ).tolist()
                    ],
                    "max_depth": list(range(
                        cfg.get("max_depth_min", 2),
                        cfg.get("max_depth_max", 15) + 1,
                        2,
                    )),
                },
            ),
            "SVM": (
                SVR(),
                {
                    "C": [
                        round(v, 2)
                        for v in np.logspace(
                            np.log10(cfg.get("C_min", 0.01)),
                            np.log10(cfg.get("C_max", 100)),
                            10,
                        ).tolist()
                    ],
                    "kernel": cfg.get("kernels", ["rbf", "linear", "poly"]),
                },
            ),
            "KNN": (
                KNeighborsRegressor(),
                {
                    "n_neighbors": list(range(
                        cfg.get("n_neighbors_min", 1),
                        cfg.get("n_neighbors_max", 30) + 1,
                        2,
                    )),
                },
            ),
        }
        # XGBoost
        if algo_name == "XGBoost":
            try:
                from xgboost import XGBRegressor
                return (
                    XGBRegressor(random_state=random_state, verbosity=0),
                    {
                        "n_estimators": list(range(
                            cfg.get("n_estimators_min", 50),
                            cfg.get("n_estimators_max", 500) + 1,
                            50,
                        )),
                        "learning_rate": [
                            round(v, 2)
                            for v in np.arange(
                                cfg.get("learning_rate_min", 0.01),
                                cfg.get("learning_rate_max", 0.5) + 0.01,
                                0.05,
                            ).tolist()
                        ],
                        "max_depth": list(range(
                            cfg.get("max_depth_min", 2),
                            cfg.get("max_depth_max", 15) + 1,
                            2,
                        )),
                    },
                )
            except ImportError:
                return None, None

    if algo_name in builders:
        return builders[algo_name]
    return None, None


def _render_auto_tune(df: pd.DataFrame):
    """AutoTune -- automated hyperparameter search across algorithms."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    section_header(
        "AutoTune",
        "Automated hyperparameter search using Grid Search or Random Search across multiple algorithms.",
    )

    help_tip(
        "How AutoTune works",
        "Select one or more algorithms, adjust their hyperparameter ranges, "
        "pick a search strategy, and AutoTune will evaluate all combinations "
        "with cross-validation. The best configuration is highlighted and can "
        "be stored in session state for use elsewhere.",
    )

    # Task type
    task_type = st.radio(
        "Task type:", ["Classification", "Regression"],
        horizontal=True, key="at_task",
    )

    # Target & features
    if task_type == "Classification":
        target_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 15]
        if not target_options:
            empty_state("No suitable classification target found.")
            return
        target = st.selectbox("Target variable:", target_options, key="at_target")
        feature_options = [c for c in num_cols if c != target]
    else:
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns for regression.")
            return
        target = st.selectbox("Target variable:", num_cols, key="at_target")
        feature_options = [c for c in num_cols if c != target]

    features = st.multiselect(
        "Feature columns:", feature_options,
        default=feature_options[:min(5, len(feature_options))],
        key="at_features",
    )
    if not features:
        st.info("Select at least one feature column.")
        return

    # Algorithm selection
    algo_options = ["Random Forest", "Gradient Boosting", "SVM", "KNN"]
    try:
        import xgboost  # noqa: F401
        algo_options.append("XGBoost")
    except ImportError:
        pass
    algorithms = st.multiselect(
        "Algorithms to tune:", algo_options,
        default=algo_options[:2], key="at_algos",
    )
    if not algorithms:
        st.info("Select at least one algorithm.")
        return

    # Per-algorithm hyperparameter sliders
    param_configs = {}
    with st.expander("Hyperparameter Ranges", expanded=True):
        for algo in algorithms:
            st.markdown(f"**{algo}**")
            cfg = {}
            if algo in ("Random Forest", "Gradient Boosting", "XGBoost"):
                c1, c2 = st.columns(2)
                ne_range = c1.slider(
                    f"n_estimators ({algo}):", 50, 500, (50, 300),
                    step=50, key=f"at_{algo}_ne",
                )
                cfg["n_estimators_min"] = ne_range[0]
                cfg["n_estimators_max"] = ne_range[1]
                if algo in ("Gradient Boosting", "XGBoost"):
                    lr_range = c2.slider(
                        f"learning_rate ({algo}):", 0.01, 0.50, (0.01, 0.30),
                        step=0.01, key=f"at_{algo}_lr",
                    )
                    cfg["learning_rate_min"] = lr_range[0]
                    cfg["learning_rate_max"] = lr_range[1]
                if algo == "Random Forest":
                    md_range = c2.slider(
                        f"max_depth ({algo}):", 2, 30, (2, 20),
                        key=f"at_{algo}_md",
                    )
                    mss_range = st.slider(
                        f"min_samples_split ({algo}):", 2, 20, (2, 10),
                        key=f"at_{algo}_mss",
                    )
                    cfg["min_samples_split_min"] = mss_range[0]
                    cfg["min_samples_split_max"] = mss_range[1]
                else:
                    md_max = 15 if algo != "Random Forest" else 30
                    md_range = c2.slider(
                        f"max_depth ({algo}):", 2, md_max, (2, 10),
                        key=f"at_{algo}_md",
                    )
                cfg["max_depth_min"] = md_range[0]
                cfg["max_depth_max"] = md_range[1]

            elif algo == "SVM":
                c1, c2 = st.columns(2)
                c_range = c1.slider(
                    "C (SVM):", 0.01, 100.0, (0.01, 10.0),
                    key="at_svm_c",
                )
                cfg["C_min"] = c_range[0]
                cfg["C_max"] = c_range[1]
                kernels = c2.multiselect(
                    "Kernels (SVM):", ["rbf", "linear", "poly"],
                    default=["rbf", "linear"], key="at_svm_kernels",
                )
                cfg["kernels"] = kernels if kernels else ["rbf"]

            elif algo == "KNN":
                nn_range = st.slider(
                    "n_neighbors (KNN):", 1, 30, (1, 15),
                    key="at_knn_nn",
                )
                cfg["n_neighbors_min"] = nn_range[0]
                cfg["n_neighbors_max"] = nn_range[1]

            param_configs[algo] = cfg
            st.divider()

    # Search strategy
    c1, c2, c3 = st.columns(3)
    search_strategy = c1.radio(
        "Search strategy:", ["Grid Search", "Random Search"],
        key="at_strategy",
    )
    n_iter = 20
    if search_strategy == "Random Search":
        n_iter = c2.slider("Max iterations per algo:", 5, 200, 20, key="at_niter")

    cv_folds = c3.slider("CV folds:", 2, 10, 5, key="at_cv")

    # Scoring metric
    if task_type == "Classification":
        scoring_options = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc_ovr"]
    else:
        scoring_options = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
    scoring = st.selectbox("Scoring metric:", scoring_options, key="at_scoring")

    random_state = 42

    if st.button("Run AutoTune", key="run_autotune", type="primary"):
        data = df[features + [target]].dropna()
        if len(data) < 10:
            st.warning("Not enough data after dropping NaNs (need at least 10 rows).")
            return

        X = data[features].values
        if task_type == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(data[target]).astype(np.float64)
        else:
            y = data[target].values.astype(np.float64)

        with st.spinner("Running hyperparameter search -- this may take a while..."):
            results = _run_autotune_search(
                X.astype(np.float64).tobytes(),
                y.tobytes(),
                len(features),
                task_type,
                algorithms,
                search_strategy,
                n_iter,
                scoring,
                cv_folds,
                random_state,
                param_configs,
            )

        if not results:
            st.warning("No results returned. Check your configuration.")
            return

        results_df = pd.DataFrame(results)
        valid = results_df.dropna(subset=["Mean Score"])
        if valid.empty:
            st.warning("All algorithm evaluations failed. Check data/settings.")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            return

        valid = valid.sort_values("Mean Score", ascending=False).reset_index(drop=True)
        valid["Rank"] = range(1, len(valid) + 1)

        section_header("Results")
        st.dataframe(
            valid[["Rank", "Algorithm", "Mean Score", "Std Score", "Params"]].round(4),
            use_container_width=True, hide_index=True,
        )

        # Bar chart of top results
        top_n = min(20, len(valid))
        top = valid.head(top_n).copy()
        top["Label"] = top["Algorithm"] + " #" + top["Rank"].astype(str)
        fig = px.bar(
            top, x="Mean Score", y="Label", orientation="h",
            error_x="Std Score", color="Algorithm",
            title=f"Top {top_n} Configurations by {scoring}",
        )
        fig.update_layout(
            height=max(350, top_n * 30),
            yaxis=dict(autorange="reversed"),
        )
        rdl_plotly_chart(fig, key="autotune_results_chart")

        # Best result interpretation
        best = valid.iloc[0]
        interpretation_card({
            "title": "Best Configuration",
            "body": (
                f"{best['Algorithm']} achieved a mean CV {scoring} of "
                f"{best['Mean Score']:.4f} (+/- {best['Std Score']:.4f})."
            ),
            "detail": f"Parameters: {best['Params']}",
        })

        # "Use Best" button
        if st.button("Use Best Parameters", key="at_use_best"):
            st.session_state["autotune_best"] = {
                "algorithm": best["Algorithm"],
                "task_type": task_type,
                "scoring": scoring,
                "mean_score": best["Mean Score"],
                "params": best["Params"],
            }
            st.success(
                f"Stored best config in session state: {best['Algorithm']} -- "
                f"{best['Params']}"
            )

        log_analysis(
            "Machine Learning", "AutoTune",
            params={"algorithms": algorithms, "strategy": search_strategy, "scoring": scoring},
            summary=f"Best: {best['Algorithm']} ({best['Mean Score']:.4f})",
        )


# ─── SHAP Explorer Tab ───────────────────────────────────────────────────

def _render_shap_explorer(df: pd.DataFrame):
    """Enhanced SHAP Explorer with beeswarm, dependence, and waterfall plots."""
    try:
        import shap  # noqa: F401
    except ImportError:
        empty_state(
            "SHAP library not installed.",
            "Install it with: pip install shap",
        )
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    section_header(
        "SHAP Explorer",
        "Explore model predictions with SHAP (SHapley Additive exPlanations) values.",
    )

    help_tip(
        "What are SHAP values?",
        "SHAP values decompose a prediction into contributions from each feature. "
        "Positive SHAP values push the prediction higher; negative values push it lower. "
        "The beeswarm plot shows the distribution of SHAP values across all observations.",
    )

    # Task type
    task_type = st.radio(
        "Task type:", ["Classification", "Regression"],
        horizontal=True, key="shap_task",
    )

    # Target & features
    if task_type == "Classification":
        target_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 15]
        if not target_options:
            empty_state("No suitable classification target found.")
            return
        target = st.selectbox("Target:", target_options, key="shap_target")
        feature_options = [c for c in num_cols if c != target]
    else:
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns.")
            return
        target = st.selectbox("Target:", num_cols, key="shap_target")
        feature_options = [c for c in num_cols if c != target]

    features = st.multiselect(
        "Features:", feature_options,
        default=feature_options[:min(8, len(feature_options))],
        key="shap_features",
    )
    if not features:
        st.info("Select at least one feature.")
        return

    model_options = ["Random Forest", "Gradient Boosting"]
    try:
        import xgboost  # noqa: F401
        model_options.append("XGBoost")
    except ImportError:
        pass
    model_type = st.selectbox("Model type:", model_options, key="shap_model")

    if st.button("Compute SHAP Values", key="run_shap", type="primary"):
        import shap

        data = df[features + [target]].dropna()
        if len(data) < 10:
            st.warning("Not enough data (need at least 10 rows after dropping NaNs).")
            return

        X = data[features].values
        feature_vals = data[features].copy()  # keep raw values for coloring

        if task_type == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(data[target])
        else:
            y = data[target].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
        )

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Build model
        with st.spinner("Training model..."):
            if model_type == "Random Forest":
                if task_type == "Classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "Gradient Boosting":
                if task_type == "Classification":
                    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                else:
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == "XGBoost":
                from xgboost import XGBClassifier, XGBRegressor
                if task_type == "Classification":
                    model = XGBClassifier(
                        n_estimators=100, random_state=42,
                        eval_metric="logloss", use_label_encoder=False, verbosity=0,
                    )
                else:
                    model = XGBRegressor(
                        n_estimators=100, random_state=42, verbosity=0,
                    )
            model.fit(X_train_sc, y_train)

        # Compute SHAP
        with st.spinner("Computing SHAP values..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values_obj = explainer(
                    pd.DataFrame(X_test_sc, columns=features),
                )
            except Exception:
                background = shap.sample(
                    pd.DataFrame(X_train_sc, columns=features),
                    min(100, len(X_train_sc)),
                )
                explainer = shap.Explainer(
                    model.predict, background, feature_names=features,
                )
                shap_values_obj = explainer(
                    pd.DataFrame(X_test_sc, columns=features),
                )

        # Extract SHAP values array
        sv = shap_values_obj.values
        if sv.ndim == 3:
            # Multi-class: use class 1 for binary, or mean absolute across classes
            if sv.shape[2] == 2:
                sv = sv[:, :, 1]
            else:
                sv = np.abs(sv).mean(axis=2)

        # Corresponding raw feature values for color mapping
        # Use original (unscaled) test data
        raw_test = feature_vals.iloc[-len(X_test):].values

        # ── Beeswarm Plot ──────────────────────────────────────
        section_header("SHAP Beeswarm Plot")

        mean_abs_shap = np.abs(sv).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)  # ascending for horizontal
        sorted_features = [features[i] for i in sorted_idx]

        fig_bee = go.Figure()
        for rank, feat_idx in enumerate(sorted_idx):
            shap_col = sv[:, feat_idx]
            feat_raw = raw_test[:, feat_idx]
            # Normalize feature values to [0, 1] for color mapping
            fmin, fmax = feat_raw.min(), feat_raw.max()
            if fmax - fmin > 0:
                feat_norm = (feat_raw - fmin) / (fmax - fmin)
            else:
                feat_norm = np.full_like(feat_raw, 0.5)

            # Map to blue (low) -> red (high)
            colors = [
                f"rgb({int(255 * v)}, {int(50 * (1 - abs(2 * v - 1)))}, {int(255 * (1 - v))})"
                for v in feat_norm
            ]

            # Add jitter for beeswarm effect
            jitter = np.random.default_rng(42).normal(0, 0.12, size=len(shap_col))

            fig_bee.add_trace(go.Scatter(
                x=shap_col,
                y=np.full(len(shap_col), rank) + jitter,
                mode="markers",
                marker=dict(size=4, color=colors, opacity=0.7),
                name=sorted_features[rank],
                showlegend=False,
                hovertemplate=(
                    f"<b>{sorted_features[rank]}</b><br>"
                    "SHAP value: %{x:.4f}<br>"
                    "<extra></extra>"
                ),
            ))

        fig_bee.update_layout(
            title="SHAP Beeswarm (feature impact on prediction)",
            xaxis_title="SHAP Value",
            yaxis=dict(
                tickvals=list(range(len(sorted_features))),
                ticktext=sorted_features,
            ),
            height=max(400, len(features) * 40),
        )
        # Add a color bar annotation
        fig_bee.add_annotation(
            text="Feature value: <span style='color:rgb(0,0,255)'>Low</span> to "
                 "<span style='color:rgb(255,0,0)'>High</span>",
            xref="paper", yref="paper", x=1.0, y=1.05,
            showarrow=False, font=dict(size=11),
        )
        rdl_plotly_chart(fig_bee, key="shap_beeswarm")

        # ── Dependence Plots (top 3 features) ─────────────────
        section_header("SHAP Dependence Plots")
        top_n = min(3, len(features))
        top_feat_indices = sorted_idx[-top_n:][::-1]  # descending by importance

        fig_dep = make_subplots(
            rows=1, cols=top_n,
            subplot_titles=[features[i] for i in top_feat_indices],
            horizontal_spacing=0.08,
        )
        for col_i, feat_idx in enumerate(top_feat_indices):
            feat_raw = raw_test[:, feat_idx]
            shap_col = sv[:, feat_idx]
            fig_dep.add_trace(
                go.Scatter(
                    x=feat_raw, y=shap_col, mode="markers",
                    marker=dict(size=4, opacity=0.6, color=shap_col,
                                colorscale="RdBu_r", showscale=(col_i == top_n - 1)),
                    name=features[feat_idx],
                    showlegend=False,
                ),
                row=1, col=col_i + 1,
            )
            fig_dep.update_xaxes(title_text=features[feat_idx], row=1, col=col_i + 1)
            fig_dep.update_yaxes(title_text="SHAP Value" if col_i == 0 else "", row=1, col=col_i + 1)

        fig_dep.update_layout(
            title=f"SHAP Dependence (Top {top_n} Features)",
            height=400,
        )
        rdl_plotly_chart(fig_dep, key="shap_dependence")

        # ── Waterfall for Individual Prediction ───────────────
        section_header("SHAP Waterfall (Individual Prediction)")
        max_row = len(X_test) - 1
        row_idx = st.number_input(
            "Select test observation index:", 0, max_row, 0,
            key="shap_waterfall_row",
        )

        row_shap = sv[row_idx]
        base_value = (
            shap_values_obj.base_values[row_idx]
            if hasattr(shap_values_obj.base_values, '__len__')
            else shap_values_obj.base_values
        )
        if hasattr(base_value, '__len__'):
            # Multi-class: pick class 1 for binary
            base_value = base_value[1] if len(base_value) == 2 else float(np.mean(base_value))

        # Sort by absolute SHAP value
        wf_order = np.argsort(np.abs(row_shap))
        wf_features = [features[i] for i in wf_order]
        wf_values = row_shap[wf_order]
        wf_raw = raw_test[row_idx, wf_order]

        colors = ["#ef4444" if v > 0 else "#3b82f6" for v in wf_values]
        labels = [f"{f} = {r:.2f}" for f, r in zip(wf_features, wf_raw)]

        fig_wf = go.Figure(go.Bar(
            x=wf_values, y=labels, orientation="h",
            marker_color=colors,
            hovertemplate="SHAP: %{x:.4f}<extra></extra>",
        ))
        fig_wf.update_layout(
            title=f"Feature Contributions (Obs #{row_idx})",
            xaxis_title="SHAP Value",
            height=max(350, len(features) * 30),
        )
        fig_wf.add_annotation(
            text=f"Base value: {base_value:.4f}",
            xref="paper", yref="paper", x=0.5, y=-0.12,
            showarrow=False, font=dict(size=12),
        )
        rdl_plotly_chart(fig_wf, key="shap_waterfall")

        log_analysis(
            "Machine Learning", "SHAP Explorer",
            params={"model": model_type, "task": task_type, "n_features": len(features)},
            summary=f"Top feature: {features[sorted_idx[-1]]}",
        )


# ─── ML Prediction Profiler Tab ──────────────────────────────────────────

def _render_ml_profiler(df: pd.DataFrame):
    """Interactive prediction profiler with live sliders and partial dependence."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    section_header(
        "ML Prediction Profiler",
        "Interactively explore how each feature affects the model prediction.",
    )

    help_tip(
        "How the Profiler works",
        "Train a model, then use the sliders to set feature values. The live "
        "prediction updates instantly. Partial dependence plots show how the "
        "prediction changes as each feature varies while all others are held "
        "at their current slider positions.",
    )

    # Task type
    task_type = st.radio(
        "Task type:", ["Classification", "Regression"],
        horizontal=True, key="prof_task",
    )

    # Target & features
    if task_type == "Classification":
        target_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 15]
        if not target_options:
            empty_state("No suitable classification target found.")
            return
        target = st.selectbox("Target:", target_options, key="prof_target")
        feature_options = [c for c in num_cols if c != target]
    else:
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns.")
            return
        target = st.selectbox("Target:", num_cols, key="prof_target")
        feature_options = [c for c in num_cols if c != target]

    features = st.multiselect(
        "Features:", feature_options,
        default=feature_options[:min(6, len(feature_options))],
        key="prof_features",
    )
    if not features:
        st.info("Select at least one feature.")
        return

    # Model type
    model_options = [
        "Random Forest", "Gradient Boosting", "Linear / Logistic",
        "KNN", "SVM",
    ]
    try:
        import xgboost  # noqa: F401
        model_options.append("XGBoost")
    except ImportError:
        pass
    model_type = st.selectbox("Model:", model_options, key="prof_model")

    # Train
    train_key = f"prof_trained_{task_type}_{target}_{model_type}_{'_'.join(sorted(features))}"
    if st.button("Train Model", key="prof_train", type="primary"):
        data = df[features + [target]].dropna()
        if len(data) < 10:
            st.warning("Not enough data (need at least 10 rows).")
            return

        X = data[features].values
        if task_type == "Classification":
            le = LabelEncoder()
            y = le.fit_transform(data[target])
            classes = le.classes_
        else:
            y = data[target].values
            classes = None

        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        with st.spinner("Training model..."):
            model = _build_profiler_model(model_type, task_type)
            model.fit(X_sc, y)

        # Store everything needed for profiling
        st.session_state[train_key] = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "classes": classes,
            "task_type": task_type,
            "feature_mins": data[features].min().to_dict(),
            "feature_maxs": data[features].max().to_dict(),
            "feature_means": data[features].mean().to_dict(),
        }
        st.success("Model trained successfully.")
        log_analysis(
            "Machine Learning", "Profiler Train",
            params={"model": model_type, "task": task_type},
        )

    # Profiling interface
    if train_key not in st.session_state:
        st.info("Train a model first to enable the profiler.")
        return

    trained = st.session_state[train_key]
    model = trained["model"]
    scaler = trained["scaler"]
    feat_names = trained["features"]
    classes = trained["classes"]
    t_type = trained["task_type"]
    feat_mins = trained["feature_mins"]
    feat_maxs = trained["feature_maxs"]
    feat_means = trained["feature_means"]

    st.divider()
    section_header("Feature Sliders")

    # Create sliders for each feature
    slider_values = {}
    n_cols = min(3, len(feat_names))
    cols = st.columns(n_cols)
    for i, feat in enumerate(feat_names):
        col = cols[i % n_cols]
        fmin = float(feat_mins[feat])
        fmax = float(feat_maxs[feat])
        fmean = float(feat_means[feat])
        # Handle edge case where min == max
        if fmin == fmax:
            fmin -= 1.0
            fmax += 1.0
        step = (fmax - fmin) / 100.0
        slider_values[feat] = col.slider(
            feat, min_value=fmin, max_value=fmax, value=fmean,
            step=step, key=f"prof_slider_{feat}",
        )

    # Live prediction
    input_arr = np.array([[slider_values[f] for f in feat_names]])
    input_sc = scaler.transform(input_arr)

    st.divider()
    section_header("Prediction")

    if t_type == "Classification":
        pred_class_idx = model.predict(input_sc)[0]
        pred_class = classes[pred_class_idx] if classes is not None else pred_class_idx

        st.metric("Predicted Class", str(pred_class))

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_sc)[0]
            prob_labels = [str(c) for c in classes] if classes is not None else [str(i) for i in range(len(probs))]
            fig_prob = go.Figure(go.Bar(
                x=probs, y=prob_labels, orientation="h",
                marker_color=["#6366f1" if i == pred_class_idx else "#cbd5e1"
                               for i in range(len(probs))],
                text=[f"{p:.3f}" for p in probs],
                textposition="auto",
            ))
            fig_prob.update_layout(
                title="Class Probabilities",
                xaxis_title="Probability", xaxis_range=[0, 1],
                height=max(200, len(prob_labels) * 40),
            )
            rdl_plotly_chart(fig_prob, key="prof_probabilities")
    else:
        pred_val = model.predict(input_sc)[0]
        st.metric("Predicted Value", f"{pred_val:.4f}")

    # ── Partial Dependence Plots ──────────────────────────
    section_header("Partial Dependence Plots")

    n_feats = len(feat_names)
    n_pdp_cols = min(3, n_feats)
    n_pdp_rows = (n_feats + n_pdp_cols - 1) // n_pdp_cols

    fig_pdp = make_subplots(
        rows=n_pdp_rows, cols=n_pdp_cols,
        subplot_titles=feat_names,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    n_grid = 50
    for i, feat in enumerate(feat_names):
        row = i // n_pdp_cols + 1
        col = i % n_pdp_cols + 1

        fmin = float(feat_mins[feat])
        fmax = float(feat_maxs[feat])
        if fmin == fmax:
            fmin -= 1.0
            fmax += 1.0
        grid = np.linspace(fmin, fmax, n_grid)

        # Build input grid: all features at slider value except this one
        grid_input = np.tile(input_arr, (n_grid, 1))
        feat_idx = feat_names.index(feat)
        grid_input[:, feat_idx] = grid

        grid_sc = scaler.transform(grid_input)

        if t_type == "Classification" and hasattr(model, "predict_proba"):
            preds = model.predict_proba(grid_sc)[:, -1]  # probability of last class
            y_label = "P(positive)" if len(classes) == 2 else "P(last class)"
        else:
            preds = model.predict(grid_sc)
            y_label = "Prediction"

        fig_pdp.add_trace(
            go.Scatter(
                x=grid, y=preds, mode="lines",
                line=dict(width=2),
                showlegend=False,
                hovertemplate=f"{feat}: " + "%{x:.3f}<br>" + f"{y_label}: " + "%{y:.4f}<extra></extra>",
            ),
            row=row, col=col,
        )

        # Mark current slider position
        current_val = slider_values[feat]
        current_pred_idx = np.argmin(np.abs(grid - current_val))
        fig_pdp.add_trace(
            go.Scatter(
                x=[current_val], y=[preds[current_pred_idx]],
                mode="markers",
                marker=dict(size=10, color="#ef4444", symbol="diamond"),
                showlegend=False,
                hovertemplate=f"Current: {current_val:.3f}<extra></extra>",
            ),
            row=row, col=col,
        )

        fig_pdp.update_xaxes(title_text=feat, row=row, col=col)
        if col == 1:
            fig_pdp.update_yaxes(title_text=y_label, row=row, col=col)

    fig_pdp.update_layout(
        title="Partial Dependence (other features held at slider values)",
        height=max(350, n_pdp_rows * 300),
    )
    rdl_plotly_chart(fig_pdp, key="prof_pdp")


def _build_profiler_model(model_type, task_type):
    """Build a model for the profiler."""
    if model_type == "Random Forest":
        if task_type == "Classification":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == "Gradient Boosting":
        if task_type == "Classification":
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == "Linear / Logistic":
        if task_type == "Classification":
            return LogisticRegression(max_iter=1000, random_state=42)
        return LinearRegression()
    elif model_type == "KNN":
        if task_type == "Classification":
            return KNeighborsClassifier()
        return KNeighborsRegressor()
    elif model_type == "SVM":
        if task_type == "Classification":
            return SVC(probability=True, random_state=42)
        return SVR()
    elif model_type == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        if task_type == "Classification":
            return XGBClassifier(
                n_estimators=100, random_state=42,
                eval_metric="logloss", use_label_encoder=False, verbosity=0,
            )
        return XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    # Fallback
    if task_type == "Classification":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    return RandomForestRegressor(n_estimators=100, random_state=42)
