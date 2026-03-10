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

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, r2_score, mean_absolute_error, silhouette_score,
    silhouette_samples,
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
        st.warning("No data loaded.")
        return

    tabs = st.tabs([
        "Clustering", "Classification", "Regression (ML)",
        "Dimensionality Reduction", "Model Comparison",
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


def _render_clustering(df: pd.DataFrame):
    """Clustering analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
        return

    features = st.multiselect("Features:", num_cols, default=num_cols[:4], key="cl_features")
    if len(features) < 2:
        return

    algorithm = st.selectbox("Algorithm:", ["K-Means", "DBSCAN", "Agglomerative"], key="cl_algo")
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
            for ki in K_range:
                km = KMeans(n_clusters=ki, random_state=42, n_init=10)
                km.fit(X)
                inertias.append(km.inertia_)
                sil_scores.append(silhouette_score(X, km.labels_))

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Plot", "Silhouette Score"))
            fig.add_trace(go.Scatter(x=list(K_range), y=inertias, mode="lines+markers",
                                     name="Inertia", marker=dict(color="steelblue")), row=1, col=1)
            fig.add_trace(go.Scatter(x=list(K_range), y=sil_scores, mode="lines+markers",
                                     name="Silhouette", marker=dict(color="steelblue")), row=1, col=2)
            fig.update_xaxes(title_text="k", row=1, col=1)
            fig.update_xaxes(title_text="k", row=1, col=2)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)

    # Cluster profiles
    st.markdown("#### Cluster Profiles")
    profile = data.groupby("Cluster")[features].mean().round(4)
    st.dataframe(profile, use_container_width=True)

    # Cluster sizes
    sizes = data["Cluster"].value_counts().sort_index()
    fig = px.bar(x=sizes.index, y=sizes.values, labels={"x": "Cluster", "y": "Count"},
                 title="Cluster Sizes", color=sizes.index)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def _render_classification(df: pd.DataFrame):
    """Classification analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Target: categorical or low-cardinality numeric
    target_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 15]
    if not target_options:
        st.warning("No suitable target variable found.")
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

    if st.button("Train Model", key="train_clf"):
        data = df[features + [target]].dropna()
        X = data[features].values
        y_raw = data[target]

        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = le.classes_

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
        st.plotly_chart(fig, use_container_width=True)

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
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            imp_df = pd.DataFrame({"Feature": features, "Importance": imp}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance")
            fig.update_layout(height=max(300, len(features) * 25))
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, "coef_"):
            coefs = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
            imp_df = pd.DataFrame({"Feature": features, "Importance": coefs}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance (|coefficients|)")
            fig.update_layout(height=max(300, len(features) * 25))
            st.plotly_chart(fig, use_container_width=True)

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
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP analysis failed: {e}")
        except ImportError:
            pass


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


def _render_ml_regression(df: pd.DataFrame):
    """ML Regression analysis."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
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

    if st.button("Train Model", key="train_reg"):
        data = df[features + [target]].dropna()
        X = data[features].values
        y = data[target].values

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
                                 marker=dict(color="steelblue", size=5, opacity=0.7),
                                 name="Predictions"), row=1, col=1)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode="lines", line=dict(color="red", dash="dash"),
                                 name="Perfect"), row=1, col=1)

        residuals = y_test - y_pred
        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                 marker=dict(color="steelblue", size=5, opacity=0.7),
                                 name="Residuals"), row=1, col=2)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_xaxes(title_text="Actual", row=1, col=1)
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_xaxes(title_text="Predicted", row=1, col=2)
        fig.update_yaxes(title_text="Residuals", row=1, col=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            imp_df = pd.DataFrame({"Feature": features, "Importance": imp}).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance")
            fig.update_layout(height=max(300, len(features) * 25))
            st.plotly_chart(fig, use_container_width=True)
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            coef_df = pd.DataFrame({"Feature": features, "Coefficient": coefs}).sort_values("Coefficient")
            fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                         title="Coefficients")
            fig.update_layout(height=max(300, len(features) * 25))
            st.plotly_chart(fig, use_container_width=True)

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
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"SHAP analysis failed: {e}")
        except ImportError:
            pass


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


def _render_dim_reduction(df: pd.DataFrame):
    """Dimensionality reduction visualization."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(num_cols) < 3:
        st.warning("Need at least 3 numeric columns.")
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
        st.plotly_chart(fig, use_container_width=True)


def _render_model_comparison(df: pd.DataFrame):
    """Compare multiple models on the same dataset."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    task = st.selectbox("Task:", ["Classification", "Regression"], key="mc_task")

    if task == "Classification":
        target_options = cat_cols + [c for c in num_cols if df[c].nunique() <= 15]
        if not target_options:
            st.warning("No suitable target variable found.")
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
            st.plotly_chart(fig, use_container_width=True)

    else:  # Regression
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns.")
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
            st.plotly_chart(fig, use_container_width=True)
