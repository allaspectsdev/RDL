"""
Regression Analysis Module - Linear, multiple, polynomial, logistic, curve fitting.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats, optimize
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import section_header, empty_state, help_tip, validation_panel, interpretation_card, confidence_badge
from modules.validation import (
    check_normality, check_sample_size, check_multicollinearity,
    check_independence, check_homoscedasticity, check_residual_normality,
    check_outlier_proportion, interpret_r_squared, interpret_p_value,
)

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
    HAS_SM = True
except ImportError:
    HAS_SM = False


def render_regression(df: pd.DataFrame):
    """Render regression analysis interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    tabs = st.tabs([
        "Simple Linear", "Multiple Linear", "Polynomial",
        "Logistic", "Curve Fitting", "Diagnostics",
        "GLM", "Robust & Quantile", "Mixed Models",
        "Regularized", "Nonlinear", "Profiler",
        "Variable Selection",
    ])

    with tabs[0]:
        _render_simple_linear(df)
    with tabs[1]:
        _render_multiple_linear(df)
    with tabs[2]:
        _render_polynomial(df)
    with tabs[3]:
        _render_logistic(df)
    with tabs[4]:
        _render_curve_fitting(df)
    with tabs[5]:
        _render_diagnostics(df)
    with tabs[6]:
        _render_glm(df)
    with tabs[7]:
        _render_robust_quantile(df)
    with tabs[8]:
        _render_mixed_models(df)
    with tabs[9]:
        _render_regularized(df)
    with tabs[10]:
        _render_nonlinear(df)
    with tabs[11]:
        _render_profiler(df)
    with tabs[12]:
        _render_variable_selection(df)


def _render_simple_linear(df: pd.DataFrame):
    """Simple linear regression."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X (predictor):", num_cols, key="slr_x")
    y_col = c2.selectbox("Y (response):", [c for c in num_cols if c != x_col], key="slr_y")
    show_ci = st.checkbox("Show confidence band", value=True, key="slr_ci")
    show_pi = st.checkbox("Show prediction band", value=True, key="slr_pi")

    if st.button("Fit Model", key="fit_slr"):
        data = df[[x_col, y_col]].dropna()
        x, y = data[x_col].values, data[y_col].values
        n = len(x)

        if HAS_SM:
            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()

            # --- Validation checks ---
            try:
                residuals = model.resid
                checks = [check_sample_size(n, "regression")]
                checks.extend([
                    check_residual_normality(residuals),
                    check_independence(residuals),
                    check_homoscedasticity(residuals, x),
                ])
                validation_panel(checks)
            except Exception:
                pass

            section_header("Regression Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{model.rsquared:.4f}")
            c2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
            c3.metric("F-statistic", f"{model.fvalue:.4f}")
            c4.metric("p (F-test)", f"{model.f_pvalue:.6f}")

            # --- Interpretation card for R² ---
            try:
                interpretation_card(interpret_r_squared(model.rsquared, model.rsquared_adj))
            except Exception:
                pass

            coef_df = pd.DataFrame({
                "Coefficient": ["Intercept", x_col],
                "Estimate": model.params,
                "Std Error": model.bse,
                "t-value": model.tvalues,
                "p-value": model.pvalues,
                "CI Lower (95%)": model.conf_int()[0],
                "CI Upper (95%)": model.conf_int()[1],
            }).round(6)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

            b0, b1 = model.params
            st.latex(f"y = {b0:.4f} + {b1:.4f} \\cdot x")
            st.markdown(f"**AIC:** {model.aic:.2f}  |  **BIC:** {model.bic:.2f}  |  **RMSE:** {np.sqrt(model.mse_resid):.4f}")
            help_tip("AIC vs BIC", "Both measure model fit penalized for complexity. **AIC** favors predictive accuracy; **BIC** penalizes complexity more heavily. Lower is better for both.")

            # Plot
            x_line = np.linspace(x.min(), x.max(), 200)
            X_line = sm.add_constant(x_line)
            y_pred = model.predict(X_line)
            predictions = model.get_prediction(X_line)
            ci_frame = predictions.summary_frame(alpha=0.05)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                                     marker=dict(size=6, opacity=0.7)))
            fig.add_trace(go.Scatter(x=x_line, y=y_pred, mode="lines", name="Fit",
                                     line=dict(color="red", width=2)))
            if show_ci:
                fig.add_trace(go.Scatter(x=np.concatenate([x_line, x_line[::-1]]),
                                         y=np.concatenate([ci_frame["mean_ci_lower"].values,
                                                           ci_frame["mean_ci_upper"].values[::-1]]),
                                         fill="toself", fillcolor="rgba(255,0,0,0.1)",
                                         line=dict(color="rgba(255,0,0,0)"), name="95% CI"))
            if show_pi:
                fig.add_trace(go.Scatter(x=np.concatenate([x_line, x_line[::-1]]),
                                         y=np.concatenate([ci_frame["obs_ci_lower"].values,
                                                           ci_frame["obs_ci_upper"].values[::-1]]),
                                         fill="toself", fillcolor="rgba(0,0,255,0.05)",
                                         line=dict(color="rgba(0,0,255,0)"), name="95% PI"))
            fig.update_layout(title=f"{y_col} vs {x_col}", xaxis_title=x_col, yaxis_title=y_col, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Residual plots
            _plot_residuals(model, x_col, y_col)
        else:
            # Fallback without statsmodels
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            # --- Validation checks (fallback) ---
            try:
                checks = [check_sample_size(n, "regression")]
                validation_panel(checks)
            except Exception:
                pass
            st.metric("R²", f"{r_value**2:.4f}")
            # --- Interpretation card for R² (fallback) ---
            try:
                interpretation_card(interpret_r_squared(r_value**2))
            except Exception:
                pass
            st.write(f"**y = {intercept:.4f} + {slope:.4f} · x**  (p = {p_value:.6f})")
            fig = px.scatter(data, x=x_col, y=y_col, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)


def _render_multiple_linear(df: pd.DataFrame):
    """Multiple linear regression with optional WLS."""
    if not HAS_SM:
        empty_state("statsmodels required for multiple regression.", "Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 3:
        empty_state("Need at least 3 numeric columns.")
        return

    y_col = st.selectbox("Y (response):", num_cols, key="mlr_y")
    x_cols = st.multiselect("X (predictors):", [c for c in num_cols if c != y_col], key="mlr_x")

    if not x_cols:
        st.info("Select predictor variables.")
        return

    # WLS option
    use_wls = st.checkbox("Use Weighted Least Squares (WLS)", value=False, key="mlr_wls")
    wls_weight_col = None
    wls_method = None
    if use_wls:
        wls_method = st.selectbox(
            "Weight source:",
            ["Inverse Variance (automatic)"] + [c for c in num_cols if c != y_col and c not in x_cols],
            key="mlr_wls_method",
        )
        if wls_method != "Inverse Variance (automatic)":
            wls_weight_col = wls_method
        help_tip("Weighted Least Squares", """
WLS is used when residuals have non-constant variance (heteroscedasticity).
- **Inverse Variance:** Weights are estimated from a preliminary OLS fit (1/fitted_values^2 of |residuals| regressed on predictors).
- **Custom column:** Use a column of known weights (e.g., sample sizes, inverse measurement variance).
Observations with higher weight have more influence on the fitted model.
""")

    if st.button("Fit Model", key="fit_mlr"):
        data = df[[y_col] + x_cols].dropna()
        if wls_weight_col:
            data = df[[y_col] + x_cols + [wls_weight_col]].dropna()

        y = data[y_col].values
        X = sm.add_constant(data[x_cols].values)
        n = len(y)

        # Fit OLS first (always needed for comparison or as base)
        ols_model = sm.OLS(y, X).fit()

        if use_wls:
            # Determine weights
            if wls_method == "Inverse Variance (automatic)":
                # Estimate weights from OLS residuals
                abs_resid = np.abs(ols_model.resid)
                # Regress |residuals| on X to get fitted variance proxy
                resid_model = sm.OLS(abs_resid, X).fit()
                fitted_var = resid_model.fittedvalues ** 2
                # Avoid zero/negative weights
                fitted_var = np.maximum(fitted_var, 1e-10)
                weights = 1.0 / fitted_var
            else:
                weights = data[wls_weight_col].values.astype(float)
                if (weights <= 0).any():
                    st.warning("Weight column contains non-positive values. Replacing with small positive value.")
                    weights = np.maximum(weights, 1e-10)

            from statsmodels.regression.linear_model import WLS
            model = WLS(y, X, weights=weights).fit()
        else:
            model = ols_model

        # --- Validation checks ---
        try:
            residuals = model.resid
            X_df = data[x_cols]
            checks = [check_sample_size(n, "regression")]
            checks.append(check_multicollinearity(X_df))
            checks.extend([
                check_residual_normality(residuals),
                check_independence(residuals),
                check_homoscedasticity(residuals, data[x_cols].values),
            ])
            validation_panel(checks)
        except Exception:
            pass

        model_label = "WLS" if use_wls else "OLS"
        section_header(f"Model Summary ({model_label})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{model.rsquared:.4f}")
        c2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
        c3.metric("F-statistic", f"{model.fvalue:.4f}")
        c4.metric("p (F-test)", f"{model.f_pvalue:.6f}")

        # --- Interpretation card for R² ---
        try:
            interpretation_card(interpret_r_squared(model.rsquared, model.rsquared_adj))
        except Exception:
            pass

        coef_names = ["Intercept"] + x_cols
        coef_df = pd.DataFrame({
            "Variable": coef_names,
            "Coefficient": model.params,
            "Std Error": model.bse,
            "t-value": model.tvalues,
            "p-value": model.pvalues,
            "CI Lower": model.conf_int()[0],
            "CI Upper": model.conf_int()[1],
        }).round(6)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.markdown(f"**AIC:** {model.aic:.2f}  |  **BIC:** {model.bic:.2f}  |  **RMSE:** {np.sqrt(model.mse_resid):.4f}")

        # WLS vs OLS comparison
        if use_wls:
            with st.expander("WLS vs OLS Comparison"):
                comp_df = pd.DataFrame({
                    "Variable": coef_names,
                    "OLS Coef": ols_model.params.round(6),
                    "WLS Coef": model.params.round(6),
                    "OLS SE": ols_model.bse.round(6),
                    "WLS SE": model.bse.round(6),
                    "OLS p-value": ols_model.pvalues.round(6),
                    "WLS p-value": model.pvalues.round(6),
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                comp_metrics = pd.DataFrame({
                    "Metric": ["R²", "Adj R²", "AIC", "BIC", "RMSE"],
                    "OLS": [
                        f"{ols_model.rsquared:.4f}", f"{ols_model.rsquared_adj:.4f}",
                        f"{ols_model.aic:.2f}", f"{ols_model.bic:.2f}",
                        f"{np.sqrt(ols_model.mse_resid):.4f}",
                    ],
                    "WLS": [
                        f"{model.rsquared:.4f}", f"{model.rsquared_adj:.4f}",
                        f"{model.aic:.2f}", f"{model.bic:.2f}",
                        f"{np.sqrt(model.mse_resid):.4f}",
                    ],
                })
                st.dataframe(comp_metrics, use_container_width=True, hide_index=True)

        # VIF
        section_header("Variance Inflation Factors")
        X_no_const = data[x_cols].values
        if X_no_const.shape[1] > 1:
            vif_data = []
            for i in range(X_no_const.shape[1]):
                try:
                    vif_val = variance_inflation_factor(X_no_const, i)
                except Exception:
                    vif_val = float("inf")
                vif_data.append({"Variable": x_cols[i], "VIF": round(vif_val, 4)})
            vif_df = pd.DataFrame(vif_data)
            st.dataframe(vif_df, use_container_width=True, hide_index=True)
            high_vif = vif_df[vif_df["VIF"] > 10]
            if not high_vif.empty:
                st.warning(f"High multicollinearity detected (VIF > 10): {', '.join(high_vif['Variable'].tolist())}")
        help_tip("VIF interpretation", "VIF > 5 suggests moderate multicollinearity. VIF > 10 indicates severe multicollinearity — consider removing or combining correlated predictors.")

        # Coefficient plot
        fig = go.Figure()
        coefs = model.params[1:]  # Skip intercept
        ci = model.conf_int()
        ci_lower = ci[1:, 0]
        ci_upper = ci[1:, 1]
        fig.add_trace(go.Scatter(x=coefs, y=x_cols, mode="markers",
                                 marker=dict(size=10),
                                 error_x=dict(type="data",
                                              symmetric=False,
                                              array=ci_upper - coefs,
                                              arrayminus=coefs - ci_lower),
                                 name="Coefficients"))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Coefficient Plot (with 95% CI)", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Residuals
        _plot_residuals(model, "Fitted", y_col)


def _render_polynomial(df: pd.DataFrame):
    """Polynomial regression."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    c1, c2, c3 = st.columns(3)
    x_col = c1.selectbox("X:", num_cols, key="poly_x")
    y_col = c2.selectbox("Y:", [c for c in num_cols if c != x_col], key="poly_y")
    degree = c3.slider("Degree:", 2, 6, 2, key="poly_deg")

    if st.button("Fit Model", key="fit_poly"):
        data = df[[x_col, y_col]].dropna()
        x, y = data[x_col].values, data[y_col].values

        # Fit polynomial
        coeffs = np.polyfit(x, y, degree)
        poly_func = np.poly1d(coeffs)
        y_pred = poly_func(x)

        # R² and adjusted R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        n = len(x)
        if ss_tot == 0:
            st.warning("Constant target variable — R² is undefined.")
            r2 = np.nan
            adj_r2 = np.nan
        elif n <= degree + 1:
            st.warning(f"Too few points ({n}) for polynomial degree {degree}.")
            r2 = 1 - ss_res / ss_tot
            adj_r2 = np.nan
        else:
            r2 = 1 - ss_res / ss_tot
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - degree - 1)
        rmse = np.sqrt(ss_res / n)

        # --- Validation checks ---
        try:
            checks = [check_sample_size(n, "regression")]
            residuals_poly = y - y_pred
            checks.append(check_residual_normality(residuals_poly))
            validation_panel(checks)
        except Exception:
            pass

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{r2:.4f}")
        c2.metric("Adjusted R²", f"{adj_r2:.4f}")
        c3.metric("RMSE", f"{rmse:.4f}")

        # --- Interpretation card for R² ---
        try:
            if not np.isnan(r2):
                interpretation_card(interpret_r_squared(r2, adj_r2 if not np.isnan(adj_r2) else None))
        except Exception:
            pass

        # Build LaTeX equation
        latex_terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power == 0:
                latex_terms.append(f"{c:.4f}")
            elif power == 1:
                latex_terms.append(f"{c:.4f}x")
            else:
                latex_terms.append(f"{c:.4f}x^{{{power}}}")
        st.latex("y = " + " + ".join(latex_terms))

        # Plot
        x_line = np.linspace(x.min(), x.max(), 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                                 marker=dict(size=6, opacity=0.7)))
        fig.add_trace(go.Scatter(x=x_line, y=poly_func(x_line), mode="lines",
                                 name=f"Degree {degree}", line=dict(color="red", width=2)))
        fig.update_layout(title=f"Polynomial Regression (degree={degree})",
                          xaxis_title=x_col, yaxis_title=y_col, height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Compare degrees
        with st.expander("Compare Polynomial Degrees"):
            comp = []
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                                      marker=dict(color="gray", size=4, opacity=0.5)))
            colors = px.colors.qualitative.Set1
            for d in range(1, degree + 1):
                c_d = np.polyfit(x, y, d)
                p_d = np.poly1d(c_d)
                y_d = p_d(x)
                ss_r = np.sum((y - y_d) ** 2)
                r2_d = 1 - ss_r / ss_tot
                adj_r2_d = 1 - (1 - r2_d) * (n - 1) / (n - d - 1)
                comp.append({"Degree": d, "R²": round(r2_d, 4),
                             "Adj R²": round(adj_r2_d, 4),
                             "RMSE": round(np.sqrt(ss_r / n), 4)})
                fig2.add_trace(go.Scatter(x=x_line, y=p_d(x_line), mode="lines",
                                          name=f"Degree {d}",
                                          line=dict(color=colors[d % len(colors)])))
            st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)
            fig2.update_layout(height=400, title="Model Comparison")
            st.plotly_chart(fig2, use_container_width=True)


def _render_logistic(df: pd.DataFrame):
    """Logistic regression - Binary, Multinomial, Ordinal."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    # Find potential target columns (binary or multiclass)
    target_cols = [c for c in all_cols if 2 <= df[c].nunique() <= 50]
    if not target_cols:
        empty_state("No suitable target variable found.", "Need a column with 2-50 unique values for logistic regression.")
        return

    target = st.selectbox("Target variable:", target_cols, key="log_target")
    n_classes = df[target].nunique()

    # Determine logistic regression type
    if n_classes == 2:
        log_type = st.radio("Model type:", ["Binary (2 classes)"], key="log_type", horizontal=True)
    else:
        log_type = st.radio(
            "Model type:",
            ["Binary (2 classes)", "Multinomial (3+ classes)", "Ordinal (ordered classes)"],
            index=1,
            key="log_type",
            horizontal=True,
        )
        if log_type == "Binary (2 classes)" and n_classes > 2:
            st.warning(f"Target has {n_classes} classes. Binary logistic requires exactly 2 classes. Consider Multinomial or Ordinal.")
            return

    features = st.multiselect("Features:", [c for c in num_cols if c != target], key="log_features")

    if not features:
        st.info("Select feature variables.")
        return

    # ---- Multinomial Logistic ----
    if log_type == "Multinomial (3+ classes)":
        _render_multinomial_logistic(df, target, features, n_classes)
        return

    # ---- Ordinal Logistic ----
    if log_type == "Ordinal (ordered classes)":
        _render_ordinal_logistic(df, target, features, n_classes)
        return

    # ---- Binary Logistic (original code) ----
    if st.button("Fit Model", key="fit_log"):
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split

        data = df[[target] + features].dropna()
        X = data[features].values
        y_raw = data[target]

        # Encode target if needed
        if y_raw.dtype == object or y_raw.dtype.name == "category":
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            classes = le.classes_
        else:
            y = y_raw.values.astype(int)
            classes = np.unique(y)

        # --- Validation checks ---
        try:
            n_log = len(y)
            checks = [check_sample_size(n_log, "regression")]
            validation_panel(checks)
        except Exception:
            pass

        # Train/test split for honest evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Fit with statsmodels for proper inference (on training data)
        if HAS_SM:
            X_sm_train = sm.add_constant(X_train)
            X_sm_test = sm.add_constant(X_test)
            try:
                logit_model = sm.Logit(y_train, X_sm_train).fit(disp=0)
                section_header("Model Summary")

                coef_names = ["Intercept"] + features
                coef_df = pd.DataFrame({
                    "Variable": coef_names,
                    "Coefficient": logit_model.params,
                    "Std Error": logit_model.bse,
                    "z-value": logit_model.tvalues,
                    "p-value": logit_model.pvalues,
                    "Odds Ratio": np.exp(logit_model.params),
                }).round(6)
                st.dataframe(coef_df, use_container_width=True, hide_index=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("AIC", f"{logit_model.aic:.2f}")
                c2.metric("BIC", f"{logit_model.bic:.2f}")
                c3.metric("Pseudo R²", f"{logit_model.prsquared:.4f}")

                # --- Interpretation card for Pseudo R² ---
                try:
                    interpretation_card(interpret_r_squared(logit_model.prsquared))
                except Exception:
                    pass

                y_prob = logit_model.predict(X_sm_test)
            except Exception:
                # Fallback to sklearn
                st.warning("Using sklearn for logistic regression — limited statistical output.")
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
        else:
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

        y_pred = (y_prob >= 0.5).astype(int)

        # Confusion matrix (on held-out test data)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           x=[str(c) for c in classes], y=[str(c) for c in classes],
                           labels=dict(x="Predicted", y="Actual"),
                           title="Confusion Matrix (Test Set)")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report
        report = classification_report(y_test, y_pred, target_names=[str(c) for c in classes], output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc:.4f})",
                                     line=dict(color="#6366f1", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                     line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(title="ROC Curve (Test Set)", xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig_roc, use_container_width=True)


def _render_multinomial_logistic(df, target, features, n_classes):
    """Multinomial logistic regression for 3+ classes."""
    if not HAS_SM:
        empty_state("statsmodels required for multinomial logistic regression.", "Install with: pip install statsmodels")
        return

    if st.button("Fit Multinomial Model", key="fit_mnlogit"):
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

        data = df[[target] + features].dropna()
        X = data[features].values
        y_raw = data[target]

        # Encode target
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = le.classes_

        # --- Validation checks ---
        try:
            checks = [check_sample_size(len(y), "regression")]
            min_class_n = pd.Series(y).value_counts().min()
            if min_class_n < 10:
                from modules.validation import ValidationCheck
                checks.append(ValidationCheck(
                    name="Min class size", status="warn",
                    detail=f"Smallest class has {min_class_n} observations",
                    suggestion="Consider merging rare classes or collecting more data",
                ))
            validation_panel(checks)
        except Exception:
            pass

        X_const = sm.add_constant(X)

        try:
            with st.spinner("Fitting multinomial logistic model..."):
                from statsmodels.discrete.discrete_model import MNLogit
                mnlogit_model = MNLogit(y, X_const).fit(disp=0, method="newton", maxiter=100)

            section_header("Multinomial Logistic Regression Summary")

            # Model fit metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("AIC", f"{mnlogit_model.aic:.2f}")
            c2.metric("BIC", f"{mnlogit_model.bic:.2f}")
            c3.metric("Pseudo R²", f"{mnlogit_model.prsquared:.4f}")

            try:
                interpretation_card(interpret_r_squared(mnlogit_model.prsquared))
            except Exception:
                pass

            # Coefficients per class (reference class is 0)
            section_header("Coefficients by Class")
            coef_names = ["Intercept"] + features
            params = mnlogit_model.params
            pvalues = mnlogit_model.pvalues
            bse = mnlogit_model.bse

            # MNLogit params shape: (n_features+1, n_classes-1)
            for j in range(params.shape[1]):
                class_label = classes[j + 1] if j + 1 < len(classes) else f"Class {j + 1}"
                ref_label = classes[0]
                with st.expander(f"Class '{class_label}' vs Reference '{ref_label}'", expanded=(j == 0)):
                    coef_df = pd.DataFrame({
                        "Variable": coef_names,
                        "Coefficient": params[:, j],
                        "Std Error": bse[:, j],
                        "z-value": mnlogit_model.tvalues[:, j],
                        "p-value": pvalues[:, j],
                        "Odds Ratio": np.exp(params[:, j]),
                    }).round(6)
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)

            # Predicted probabilities and confusion matrix
            section_header("Classification Performance")
            y_pred_probs = mnlogit_model.predict(X_const)
            y_pred = y_pred_probs.argmax(axis=1)

            accuracy = accuracy_score(y, y_pred)
            st.metric("Training Accuracy", f"{accuracy:.4f}")

            # Multi-class confusion matrix
            cm = confusion_matrix(y, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               x=[str(c) for c in classes], y=[str(c) for c in classes],
                               labels=dict(x="Predicted", y="Actual"),
                               title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

            # Classification report
            report = classification_report(y, y_pred, target_names=[str(c) for c in classes], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

            # Predicted probability plot (for first feature)
            if len(features) >= 1:
                section_header("Predicted Probabilities")
                plot_feature = st.selectbox("Plot probabilities vs:", features, key="mnlogit_plot_feat")
                feat_idx = features.index(plot_feature)

                x_range = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 100)
                # Hold other features at their means
                X_plot = np.tile(X.mean(axis=0), (100, 1))
                X_plot[:, feat_idx] = x_range
                X_plot_const = sm.add_constant(X_plot)
                probs = mnlogit_model.predict(X_plot_const)

                fig = go.Figure()
                for j in range(probs.shape[1]):
                    class_label = str(classes[j])
                    fig.add_trace(go.Scatter(x=x_range, y=probs[:, j], mode="lines",
                                             name=class_label, line=dict(width=2)))
                fig.update_layout(title=f"Predicted Probabilities vs {plot_feature}",
                                  xaxis_title=plot_feature, yaxis_title="Probability",
                                  height=450)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Multinomial logistic regression failed: {e}")


def _render_ordinal_logistic(df, target, features, n_classes):
    """Ordinal logistic regression for ordered categories."""
    if not HAS_SM:
        empty_state("statsmodels required for ordinal logistic regression.", "Install with: pip install statsmodels")
        return

    # Let user specify the ordering
    unique_vals = sorted(df[target].dropna().unique())
    st.info(f"Target has {n_classes} ordered categories. Verify the ordering below.")
    st.write(f"Current order: {unique_vals}")

    distr = st.selectbox("Distribution:", ["logit", "probit"], key="ordinal_dist")

    if st.button("Fit Ordinal Model", key="fit_ordinal"):
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

        data = df[[target] + features].dropna()
        X = data[features].values
        y_raw = data[target]

        # Encode target to ordered integers
        le = LabelEncoder()
        le.classes_ = np.array(unique_vals)
        y = le.transform(y_raw)
        classes = le.classes_

        # --- Validation checks ---
        try:
            checks = [check_sample_size(len(y), "regression")]
            validation_panel(checks)
        except Exception:
            pass

        try:
            with st.spinner("Fitting ordinal logistic model..."):
                from statsmodels.miscmodels.ordinal_model import OrderedModel
                ordinal_model = OrderedModel(y, X, distr=distr).fit(method="bfgs", disp=0)

            section_header("Ordinal Logistic Regression Summary")

            # Model fit metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("AIC", f"{ordinal_model.aic:.2f}")
            c2.metric("BIC", f"{ordinal_model.bic:.2f}")
            pseudo_r2 = 1 - ordinal_model.llf / ordinal_model.llnull if hasattr(ordinal_model, "llnull") and ordinal_model.llnull != 0 else 0
            c3.metric("Pseudo R²", f"{pseudo_r2:.4f}")

            try:
                interpretation_card(interpret_r_squared(pseudo_r2))
            except Exception:
                pass

            # Coefficients
            section_header("Coefficients")
            # OrderedModel params: first len(features) are coefficients, rest are thresholds
            n_coefs = len(features)
            coef_params = ordinal_model.params[:n_coefs]
            coef_bse = ordinal_model.bse[:n_coefs]
            coef_pvalues = ordinal_model.pvalues[:n_coefs]
            coef_tvalues = ordinal_model.tvalues[:n_coefs]

            coef_df = pd.DataFrame({
                "Variable": features,
                "Coefficient": coef_params,
                "Std Error": coef_bse,
                "z-value": coef_tvalues,
                "p-value": coef_pvalues,
            }).round(6)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

            # Threshold parameters
            section_header("Threshold Parameters")
            n_thresholds = len(ordinal_model.params) - n_coefs
            threshold_params = ordinal_model.params[n_coefs:]
            threshold_bse = ordinal_model.bse[n_coefs:]
            threshold_names = [f"Threshold {i+1} ({classes[i]}|{classes[i+1]})" for i in range(n_thresholds)]

            thresh_df = pd.DataFrame({
                "Threshold": threshold_names,
                "Estimate": threshold_params,
                "Std Error": threshold_bse,
            }).round(6)
            st.dataframe(thresh_df, use_container_width=True, hide_index=True)

            # Predicted cumulative probabilities
            section_header("Classification Performance")
            y_pred_probs = ordinal_model.predict()
            y_pred = y_pred_probs.argmax(axis=1)

            accuracy = accuracy_score(y, y_pred)
            st.metric("Training Accuracy", f"{accuracy:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               x=[str(c) for c in classes], y=[str(c) for c in classes],
                               labels=dict(x="Predicted", y="Actual"),
                               title="Confusion Matrix")
            st.plotly_chart(fig_cm, use_container_width=True)

            # Classification report
            report = classification_report(y, y_pred, target_names=[str(c) for c in classes], output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

            # Predicted cumulative probability plot
            if len(features) >= 1:
                section_header("Predicted Cumulative Probabilities")
                plot_feature = st.selectbox("Plot vs:", features, key="ordinal_plot_feat")
                feat_idx = features.index(plot_feature)

                x_range = np.linspace(X[:, feat_idx].min(), X[:, feat_idx].max(), 100)
                X_plot = np.tile(X.mean(axis=0), (100, 1))
                X_plot[:, feat_idx] = x_range

                # Predict probabilities for each category
                probs = ordinal_model.model.predict(ordinal_model.params, X_plot)

                fig = go.Figure()
                cum_prob = np.zeros(100)
                for j in range(probs.shape[1]):
                    cum_prob = cum_prob + probs[:, j]
                    fig.add_trace(go.Scatter(x=x_range, y=cum_prob, mode="lines",
                                             name=f"P(Y <= {classes[j]})", line=dict(width=2)))
                fig.update_layout(title=f"Cumulative Probabilities vs {plot_feature}",
                                  xaxis_title=plot_feature, yaxis_title="Cumulative Probability",
                                  height=450)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Ordinal logistic regression failed: {e}")


def _render_curve_fitting(df: pd.DataFrame):
    """Non-linear curve fitting (like MATLAB cftool)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X:", num_cols, key="cf_x")
    y_col = c2.selectbox("Y:", [c for c in num_cols if c != x_col], key="cf_y")

    model_type = st.selectbox("Model:", [
        "Linear (a·x + b)",
        "Quadratic (a·x² + b·x + c)",
        "Exponential (a·exp(b·x))",
        "Logarithmic (a·ln(x) + b)",
        "Power (a·x^b)",
        "Sigmoid (L / (1 + exp(-k·(x - x0))))",
        "Gaussian (a·exp(-((x-μ)²)/(2σ²)))",
    ], key="cf_model")

    if st.button("Fit Curve", key="fit_cf"):
        data = df[[x_col, y_col]].dropna()
        x, y = data[x_col].values, data[y_col].values

        try:
            if model_type.startswith("Linear"):
                def func(x, a, b): return a * x + b
                p0 = [1.0, 0.0]
                param_names = ["a", "b"]
            elif model_type.startswith("Quadratic"):
                def func(x, a, b, c): return a * x**2 + b * x + c
                p0 = [1.0, 1.0, 0.0]
                param_names = ["a", "b", "c"]
            elif model_type.startswith("Exponential"):
                def func(x, a, b): return a * np.exp(b * x)
                p0 = [1.0, 0.01]
                param_names = ["a", "b"]
            elif model_type.startswith("Logarithmic"):
                def func(x, a, b): return a * np.log(x) + b
                if (x <= 0).any():
                    st.error("Logarithmic model requires positive X values.")
                    return
                p0 = [1.0, 0.0]
                param_names = ["a", "b"]
            elif model_type.startswith("Power"):
                def func(x, a, b): return a * np.power(x, b)
                if (x <= 0).any():
                    st.error("Power model requires positive X values.")
                    return
                p0 = [1.0, 1.0]
                param_names = ["a", "b"]
            elif model_type.startswith("Sigmoid"):
                def func(x, L, k, x0): return L / (1 + np.exp(-k * (x - x0)))
                p0 = [y.max(), 1.0, x.mean()]
                param_names = ["L", "k", "x0"]
            elif model_type.startswith("Gaussian"):
                def func(x, a, mu, sigma): return a * np.exp(-((x - mu)**2) / (2 * sigma**2))
                p0 = [y.max(), x.mean(), x.std()]
                param_names = ["a", "μ", "σ"]

            popt, pcov = optimize.curve_fit(func, x, y, p0=p0, maxfev=10000)
            perr = np.sqrt(np.diag(pcov))

            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot
            rmse = np.sqrt(ss_res / len(x))

            c1, c2 = st.columns(2)
            c1.metric("R²", f"{r2:.4f}")
            c2.metric("RMSE", f"{rmse:.4f}")

            # --- Interpretation card for R² ---
            try:
                interpretation_card(interpret_r_squared(r2))
            except Exception:
                pass

            # Parameters table
            param_df = pd.DataFrame({
                "Parameter": param_names,
                "Value": popt.round(6),
                "Std Error": perr.round(6),
                "CI Lower": (popt - 1.96 * perr).round(6),
                "CI Upper": (popt + 1.96 * perr).round(6),
            })
            st.dataframe(param_df, use_container_width=True, hide_index=True)

            # Plot
            x_line = np.linspace(x.min(), x.max(), 300)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                                     marker=dict(color="#6366f1", size=6, opacity=0.7)))
            fig.add_trace(go.Scatter(x=x_line, y=func(x_line, *popt), mode="lines",
                                     name="Fit", line=dict(color="red", width=2)))
            fig.update_layout(title=f"Curve Fit: {model_type.split('(')[0].strip()}",
                              xaxis_title=x_col, yaxis_title=y_col, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Residual plot
            residuals = y - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                          marker=dict(color="#6366f1", size=5)))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(title="Residuals vs Fitted", xaxis_title="Fitted",
                                  yaxis_title="Residuals", height=350)
            st.plotly_chart(fig_res, use_container_width=True)

        except RuntimeError as e:
            st.error(f"Curve fitting failed to converge: {e}")
        except Exception as e:
            st.error(f"Error: {e}")


def _render_diagnostics(df: pd.DataFrame):
    """Regression diagnostics panel."""
    if not HAS_SM:
        empty_state("statsmodels required for regression diagnostics.", "Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    y_col = st.selectbox("Y:", num_cols, key="diag_y")
    x_cols = st.multiselect("X:", [c for c in num_cols if c != y_col], key="diag_x")
    if not x_cols:
        st.info("Select predictor variables.")
        return

    if st.button("Run Diagnostics", key="run_diag"):
        data = df[[y_col] + x_cols].dropna()
        y = data[y_col].values
        X = sm.add_constant(data[x_cols].values)
        model = sm.OLS(y, X).fit()

        section_header("Diagnostic Tests")
        results = []

        # Durbin-Watson
        dw = durbin_watson(model.resid)
        results.append({"Test": "Durbin-Watson", "Statistic": dw,
                        "Interpretation": "No autocorrelation" if 1.5 < dw < 2.5 else "Possible autocorrelation"})

        # Breusch-Pagan
        try:
            bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(model.resid, X)
            results.append({"Test": "Breusch-Pagan", "Statistic": bp_stat,
                            "p-value": bp_p,
                            "Interpretation": "Homoscedastic" if bp_p > 0.05 else "Heteroscedastic"})
        except Exception:
            pass

        # Jarque-Bera
        jb_stat, jb_p = stats.jarque_bera(model.resid)
        results.append({"Test": "Jarque-Bera (residuals)", "Statistic": jb_stat,
                        "p-value": jb_p,
                        "Interpretation": "Normal residuals" if jb_p > 0.05 else "Non-normal residuals"})

        # RESET test
        try:
            reset_result = linear_reset(model, power=3, use_f=True)
            results.append({"Test": "Ramsey RESET", "Statistic": reset_result.fvalue,
                            "p-value": reset_result.pvalue,
                            "Interpretation": "Correct functional form" if reset_result.pvalue > 0.05 else "Possible misspecification"})
        except Exception:
            pass

        results_df = pd.DataFrame(results)
        float_cols = results_df.select_dtypes(include=[np.number]).columns
        results_df[float_cols] = results_df[float_cols].round(6)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Influence measures
        section_header("Influence Measures")
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        leverage = influence.hat_matrix_diag

        n = len(y)
        k = len(x_cols) + 1

        c1, c2 = st.columns(2)
        c1.metric("High Leverage Points (>2k/n)", int(np.sum(leverage > 2 * k / n)))
        c2.metric("Influential Points (Cook's D > 4/n)", int(np.sum(cooks_d > 4 / n)))

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Cook's Distance", "Leverage"))
        fig.add_trace(go.Bar(y=cooks_d, name="Cook's D", marker_color="#6366f1"), row=1, col=1)
        fig.add_hline(y=4 / n, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(go.Bar(y=leverage, name="Leverage", marker_color="#6366f1"), row=1, col=2)
        fig.add_hline(y=2 * k / n, line_dash="dash", line_color="red", row=1, col=2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        _plot_residuals(model, "Fitted", y_col)


def _plot_residuals(model, x_label, y_label):
    """Standard residual diagnostic plots."""
    resid = model.resid
    fitted = model.fittedvalues
    std_resid = resid / np.std(resid)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Residuals vs Fitted", "QQ Plot",
                                        "Scale-Location", "Residuals vs Order"))

    # Residuals vs Fitted
    fig.add_trace(go.Scatter(x=fitted, y=resid, mode="markers",
                             marker=dict(color="#6366f1", size=4),
                             showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # QQ Plot
    sorted_resid = np.sort(std_resid)
    n = len(sorted_resid)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_resid, mode="markers",
                             marker=dict(color="#6366f1", size=4),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode="lines",
                             line=dict(color="red", dash="dash"),
                             showlegend=False), row=1, col=2)

    # Scale-Location
    fig.add_trace(go.Scatter(x=fitted, y=np.sqrt(np.abs(std_resid)), mode="markers",
                             marker=dict(color="#6366f1", size=4),
                             showlegend=False), row=2, col=1)

    # Residuals vs Order
    fig.add_trace(go.Scatter(y=resid, mode="lines+markers",
                             marker=dict(color="#6366f1", size=3),
                             line=dict(color="#6366f1", width=1),
                             showlegend=False), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(height=600, title_text="Residual Diagnostics")
    fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
    fig.update_yaxes(title_text="Std Residuals", row=1, col=2)
    fig.update_xaxes(title_text="Fitted Values", row=2, col=1)
    fig.update_yaxes(title_text="√|Std Residuals|", row=2, col=1)
    fig.update_xaxes(title_text="Observation Order", row=2, col=2)
    fig.update_yaxes(title_text="Residuals", row=2, col=2)
    st.plotly_chart(fig, use_container_width=True)


def _render_glm(df: pd.DataFrame):
    """Generalized Linear Models."""
    if not HAS_SM:
        empty_state("statsmodels required for GLM.", "Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    y_col = st.selectbox("Response (Y):", num_cols, key="glm_y")
    x_cols = st.multiselect("Predictors (X):", [c for c in num_cols if c != y_col], key="glm_x")

    if not x_cols:
        st.info("Select predictor variables.")
        return

    family_name = st.selectbox("Family:", [
        "Gaussian", "Poisson", "Negative Binomial", "Gamma", "Binomial",
    ], key="glm_family")

    link_options = {
        "Gaussian": ["identity", "log"],
        "Poisson": ["log", "identity"],
        "Negative Binomial": ["log", "identity"],
        "Gamma": ["inverse", "log", "identity"],
        "Binomial": ["logit", "probit", "cloglog"],
    }
    link_name = st.selectbox("Link function:", link_options[family_name], key="glm_link")

    if st.button("Fit GLM", key="fit_glm"):
        from statsmodels.genmod.generalized_linear_model import GLM as StatsGLM
        import statsmodels.genmod.families as fam
        import statsmodels.genmod.families.links as lnk

        data = df[[y_col] + x_cols].dropna()
        y = data[y_col].values
        X = sm.add_constant(data[x_cols].values)

        link_map = {
            "identity": lnk.Identity(),
            "log": lnk.Log(),
            "inverse": lnk.InversePower(),
            "logit": lnk.Logit(),
            "probit": lnk.Probit(),
            "cloglog": lnk.CLogLog(),
        }
        link = link_map[link_name]

        family_map = {
            "Gaussian": fam.Gaussian(link=link),
            "Poisson": fam.Poisson(link=link),
            "Negative Binomial": fam.NegativeBinomial(link=link),
            "Gamma": fam.Gamma(link=link),
            "Binomial": fam.Binomial(link=link),
        }
        family = family_map[family_name]

        try:
            with st.spinner("Fitting GLM..."):
                model = StatsGLM(y, X, family=family).fit()

            # --- Validation checks ---
            try:
                n_glm = len(y)
                checks = [check_sample_size(n_glm, "regression")]
                validation_panel(checks)
            except Exception:
                pass

            section_header("GLM Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AIC", f"{model.aic:.2f}")
            c2.metric("BIC", f"{model.bic_deviance:.2f}")
            c3.metric("Deviance", f"{model.deviance:.4f}")
            c4.metric("Pearson χ²", f"{model.pearson_chi2:.4f}")

            # --- Interpretation card for deviance-based pseudo R² ---
            try:
                null_dev = model.null_deviance
                resid_dev = model.deviance
                pseudo_r2_glm = 1 - resid_dev / null_dev if null_dev > 0 else 0
                interpretation_card(interpret_r_squared(pseudo_r2_glm))
            except Exception:
                pass

            coef_names = ["Intercept"] + x_cols
            coef_df = pd.DataFrame({
                "Variable": coef_names,
                "Coefficient": model.params,
                "Std Error": model.bse,
                "z-value": model.tvalues,
                "p-value": model.pvalues,
                "CI Lower": model.conf_int()[:, 0],
                "CI Upper": model.conf_int()[:, 1],
            }).round(6)

            if family_name in ("Poisson", "Negative Binomial", "Gamma"):
                coef_df["exp(Coef)"] = np.exp(model.params).round(6)

            st.dataframe(coef_df, use_container_width=True, hide_index=True)

            # Deviance table
            null_dev = model.null_deviance
            resid_dev = model.deviance
            section_header("Deviance Table")
            dev_df = pd.DataFrame({
                "Source": ["Null", "Residual", "Model"],
                "Deviance": [null_dev, resid_dev, null_dev - resid_dev],
                "df": [model.df_model + model.df_resid, model.df_resid, model.df_model],
            }).round(4)
            st.dataframe(dev_df, use_container_width=True, hide_index=True)

            # Deviance residual plot
            resid = model.resid_deviance
            fitted = model.fittedvalues
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("Deviance Residuals vs Fitted", "QQ Plot of Deviance Residuals"))
            fig.add_trace(go.Scatter(x=fitted, y=resid, mode="markers",
                                     marker=dict(color="#6366f1", size=4), showlegend=False),
                          row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            sorted_resid = np.sort(resid)
            n = len(sorted_resid)
            theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            fig.add_trace(go.Scatter(x=theoretical, y=sorted_resid, mode="markers",
                                     marker=dict(color="#6366f1", size=4), showlegend=False),
                          row=1, col=2)
            fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode="lines",
                                     line=dict(color="red", dash="dash"), showlegend=False),
                          row=1, col=2)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"GLM fitting failed: {e}")


def _render_robust_quantile(df: pd.DataFrame):
    """Robust and quantile regression."""
    if not HAS_SM:
        empty_state("statsmodels required for robust & quantile regression.", "Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    reg_type = st.selectbox("Method:", [
        "Robust Regression (RLM)", "Quantile Regression", "RANSAC",
    ], key="rq_type")

    y_col = st.selectbox("Y (response):", num_cols, key="rq_y")
    x_cols = st.multiselect("X (predictors):", [c for c in num_cols if c != y_col], key="rq_x")

    if not x_cols:
        st.info("Select predictor variables.")
        return

    if reg_type == "Robust Regression (RLM)":
        m_estimator = st.selectbox("M-estimator:", ["Huber", "Tukey Bisquare", "Andrew's Wave"], key="rq_mest")

        if st.button("Fit Robust Model", key="fit_rlm"):
            from statsmodels.robust import RLM
            import statsmodels.robust.norms as norms

            data = df[[y_col] + x_cols].dropna()
            y = data[y_col].values
            X = sm.add_constant(data[x_cols].values)

            norm_map = {
                "Huber": norms.HuberT(),
                "Tukey Bisquare": norms.TukeyBiweight(),
                "Andrew's Wave": norms.AndrewWave(),
            }

            try:
                with st.spinner("Fitting robust model..."):
                    model = RLM(y, X, M=norm_map[m_estimator]).fit()

                # --- Validation checks ---
                try:
                    n_rlm = len(y)
                    checks = [check_sample_size(n_rlm, "regression")]
                    checks.append(check_outlier_proportion(y))
                    validation_panel(checks)
                except Exception:
                    pass

                section_header("Robust Regression Summary")
                coef_names = ["Intercept"] + x_cols
                coef_df = pd.DataFrame({
                    "Variable": coef_names,
                    "Coefficient": model.params,
                    "Std Error": model.bse,
                    "z-value": model.tvalues,
                    "p-value": model.pvalues,
                }).round(6)
                st.dataframe(coef_df, use_container_width=True, hide_index=True)

                # Compare with OLS
                ols_model = sm.OLS(y, X).fit()
                comp_df = pd.DataFrame({
                    "Variable": coef_names,
                    "OLS Coef": ols_model.params.round(6),
                    "Robust Coef": model.params.round(6),
                    "Difference": (model.params - ols_model.params).round(6),
                })
                section_header("OLS vs Robust Comparison")
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Weights plot
                weights = model.weights
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=weights, mode="markers",
                                         marker=dict(color="#6366f1", size=4), name="Weights"))
                fig.add_hline(y=1, line_dash="dash", line_color="red")
                fig.update_layout(title="Robustness Weights (low = downweighted outlier)",
                                  xaxis_title="Observation", yaxis_title="Weight", height=350)
                st.plotly_chart(fig, use_container_width=True)

                n_downweighted = (weights < 0.9).sum()
                st.write(f"**{n_downweighted}** observations downweighted (weight < 0.9)")

            except Exception as e:
                st.error(f"Robust regression failed: {e}")

    elif reg_type == "Quantile Regression":
        quantiles_input = st.text_input("Quantiles (comma-separated):", "0.25, 0.50, 0.75", key="rq_quantiles")

        if st.button("Fit Quantile Models", key="fit_quantreg"):
            from statsmodels.regression.quantile_regression import QuantReg

            data = df[[y_col] + x_cols].dropna()
            y = data[y_col].values
            X = sm.add_constant(data[x_cols].values)
            coef_names = ["Intercept"] + x_cols

            try:
                quantiles = [float(q.strip()) for q in quantiles_input.split(",")]
            except ValueError:
                st.error("Invalid quantile format.")
                return

            all_results = []
            for q in quantiles:
                try:
                    model = QuantReg(y, X).fit(q=q)
                    row = {"Quantile": q}
                    for i, name in enumerate(coef_names):
                        row[f"{name}"] = round(model.params[i], 6)
                    row["Pseudo R²"] = round(model.prsquared, 4)
                    all_results.append(row)
                except Exception:
                    pass

            if all_results:
                section_header("Quantile Regression Coefficients")
                st.dataframe(pd.DataFrame(all_results), use_container_width=True, hide_index=True)

                # Plot quantile regression lines (if 1 predictor)
                if len(x_cols) == 1:
                    x_data = data[x_cols[0]].values
                    x_line = np.linspace(x_data.min(), x_data.max(), 200)
                    X_line = sm.add_constant(x_line)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=x_data, y=y, mode="markers", name="Data",
                                             marker=dict(color="gray", size=4, opacity=0.5)))
                    colors = px.colors.qualitative.Set1
                    for i, q in enumerate(quantiles):
                        model = QuantReg(y, X).fit(q=q)
                        y_line = model.predict(X_line)
                        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines",
                                                 name=f"Q={q}",
                                                 line=dict(color=colors[i % len(colors)], width=2)))
                    # OLS for comparison
                    ols_model = sm.OLS(y, X).fit()
                    fig.add_trace(go.Scatter(x=x_line, y=ols_model.predict(X_line), mode="lines",
                                             name="OLS", line=dict(color="black", dash="dash", width=2)))
                    fig.update_layout(title="Quantile Regression Lines",
                                      xaxis_title=x_cols[0], yaxis_title=y_col, height=500)
                    st.plotly_chart(fig, use_container_width=True)

    elif reg_type == "RANSAC":
        from sklearn.linear_model import RANSACRegressor

        residual_threshold = st.slider("Residual threshold:", 0.1, 50.0, 5.0, 0.5, key="rq_ransac_thresh")
        min_samples = st.slider("Min samples (fraction):", 0.1, 0.9, 0.5, 0.05, key="rq_ransac_min")

        if st.button("Fit RANSAC", key="fit_ransac"):
            data = df[[y_col] + x_cols].dropna()
            X = data[x_cols].values
            y = data[y_col].values

            try:
                model = RANSACRegressor(
                    residual_threshold=residual_threshold,
                    min_samples=min_samples,
                    random_state=42,
                )
                model.fit(X, y)
                inlier_mask = model.inlier_mask_
                n_inliers = inlier_mask.sum()
                n_outliers = (~inlier_mask).sum()

                c1, c2, c3 = st.columns(3)
                c1.metric("Inliers", n_inliers)
                c2.metric("Outliers", n_outliers)
                from sklearn.metrics import r2_score as r2s
                c3.metric("R² (inliers)", f"{r2s(y[inlier_mask], model.predict(X[inlier_mask])):.4f}")

                if len(x_cols) == 1:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=X[inlier_mask, 0], y=y[inlier_mask], mode="markers",
                                             name="Inliers", marker=dict(color="#6366f1", size=5)))
                    fig.add_trace(go.Scatter(x=X[~inlier_mask, 0], y=y[~inlier_mask], mode="markers",
                                             name="Outliers", marker=dict(color="red", size=7, symbol="x")))
                    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 200).reshape(-1, 1)
                    fig.add_trace(go.Scatter(x=x_line.ravel(), y=model.predict(x_line), mode="lines",
                                             name="RANSAC", line=dict(color="green", width=2)))
                    from sklearn.linear_model import LinearRegression as LR
                    ols = LR().fit(X, y)
                    fig.add_trace(go.Scatter(x=x_line.ravel(), y=ols.predict(x_line), mode="lines",
                                             name="OLS", line=dict(color="orange", dash="dash", width=2)))
                    fig.update_layout(title="RANSAC vs OLS", xaxis_title=x_cols[0], yaxis_title=y_col, height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(f"**RANSAC Coefficients:** {model.estimator_.coef_.round(6)}")
                    st.write(f"**Intercept:** {model.estimator_.intercept_:.6f}")

            except Exception as e:
                st.error(f"RANSAC failed: {e}")


def _render_mixed_models(df: pd.DataFrame):
    """Mixed / multilevel models."""
    if not HAS_SM:
        empty_state("statsmodels required for mixed models.", "Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols or not cat_cols:
        empty_state("Need numeric and categorical columns for mixed models.")
        return

    y_col = st.selectbox("Dependent variable:", num_cols, key="mm_y")
    fixed_effects = st.multiselect("Fixed effects:", [c for c in num_cols if c != y_col], key="mm_fixed")
    group_col = st.selectbox("Grouping variable (random intercept):", cat_cols, key="mm_group")
    random_slope_col = st.selectbox("Random slope variable (optional):", [None] + [c for c in num_cols if c != y_col], key="mm_rslope")

    if not fixed_effects:
        st.info("Select fixed effect variables.")
        return

    if st.button("Fit Mixed Model", key="fit_mm"):
        from statsmodels.regression.mixed_linear_model import MixedLM

        data = df[[y_col] + fixed_effects + [group_col]].dropna()
        if random_slope_col:
            data = data.dropna(subset=[random_slope_col])

        y = data[y_col]
        X = sm.add_constant(data[fixed_effects])
        groups = data[group_col]

        try:
            with st.spinner("Fitting mixed model..."):
                if random_slope_col:
                    exog_re = data[[random_slope_col]]
                    model = MixedLM(y, X, groups=groups, exog_re=exog_re).fit(reml=True)
                else:
                    model = MixedLM(y, X, groups=groups).fit(reml=True)

            section_header("Mixed Model Summary")

            # Fixed effects
            st.markdown("**Fixed Effects:**")
            fe_df = pd.DataFrame({
                "Variable": model.fe_params.index,
                "Coefficient": model.fe_params.values,
                "Std Error": model.bse_fe.values,
                "z-value": model.tvalues[:len(model.fe_params)].values,
                "p-value": model.pvalues[:len(model.fe_params)].values,
            }).round(6)
            st.dataframe(fe_df, use_container_width=True, hide_index=True)

            # Random effects variance
            st.markdown("**Random Effects Variance:**")
            re_params = model.cov_re
            if hasattr(re_params, 'values'):
                re_df = pd.DataFrame(re_params).round(6)
            else:
                re_df = pd.DataFrame({"Variance": [re_params]}).round(6)
            st.dataframe(re_df, use_container_width=True)

            # Model fit
            c1, c2, c3 = st.columns(3)
            c1.metric("Log-Likelihood", f"{model.llf:.2f}")
            c2.metric("AIC", f"{model.aic:.2f}")
            c3.metric("BIC", f"{model.bic:.2f}")

            # ICC (for random intercept model)
            if not random_slope_col:
                re_var = float(model.cov_re.iloc[0, 0]) if hasattr(model.cov_re, 'iloc') else float(model.cov_re)
                resid_var = model.scale
                icc = re_var / (re_var + resid_var)
                st.metric("ICC (Intraclass Correlation)", f"{icc:.4f}")
                st.write(f"**{icc*100:.1f}%** of variance is explained by group membership ({group_col}).")

            # Compare with OLS
            with st.expander("Compare with OLS"):
                ols_model = sm.OLS(y, X).fit()
                comp_df = pd.DataFrame({
                    "Variable": model.fe_params.index,
                    "OLS Coef": ols_model.params.values[:len(model.fe_params)],
                    "Mixed Coef": model.fe_params.values,
                    "OLS SE": ols_model.bse.values[:len(model.fe_params)],
                    "Mixed SE": model.bse_fe.values,
                }).round(6)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                st.write(f"**OLS AIC:** {ols_model.aic:.2f}  |  **Mixed AIC:** {model.aic:.2f}")

                # LRT
                lr_stat = -2 * (ols_model.llf - model.llf)
                lr_df = 1 if not random_slope_col else 2
                lr_p = 1 - stats.chi2.cdf(max(0, lr_stat), lr_df)
                st.write(f"**Likelihood Ratio Test:** χ² = {lr_stat:.4f}, df = {lr_df}, p = {lr_p:.6f}")
                if lr_p < 0.05:
                    st.write("Mixed model significantly improves over OLS — random effects are warranted.")
                else:
                    st.write("No significant improvement — OLS may be sufficient.")

            # Random effects by group
            with st.expander("Random Effects by Group"):
                re = model.random_effects
                re_rows = []
                for grp, vals in re.items():
                    row = {"Group": grp}
                    for k, v in vals.items():
                        row[k] = round(v, 6)
                    re_rows.append(row)
                re_df = pd.DataFrame(re_rows)
                st.dataframe(re_df, use_container_width=True, hide_index=True)

                fig = go.Figure()
                intercepts = [row.get("Group", 0) for row in re_rows]
                group_names = [row["Group"] for row in re_rows]
                group_var = list(re_rows[0].keys())[1] if len(re_rows[0]) > 1 else "Intercept"
                values = [row.get(group_var, 0) for row in re_rows]
                fig.add_trace(go.Bar(x=group_names, y=values, marker_color="#6366f1"))
                fig.update_layout(title=f"Random {group_var} by Group",
                                  xaxis_title=group_col, yaxis_title=f"Random {group_var}", height=350)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Mixed model failed: {e}")


# ===================================================================
# Tab 10 -- Regularized Regression (LASSO / Ridge / Elastic Net)
# ===================================================================

def _render_regularized(df: pd.DataFrame):
    """Regularized regression with path plots and CV alpha selection."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    section_header("Regularized Regression")
    help_tip("Regularized Regression", """
- **LASSO (L1):** Shrinks some coefficients to exactly zero → feature selection
- **Ridge (L2):** Shrinks all coefficients toward zero → handles multicollinearity
- **Elastic Net:** Combines L1 and L2 penalties
- The regularization path shows how coefficients change as penalty (α) increases
""")

    y_col = st.selectbox("Response (Y):", num_cols, key="reg_reg_y")
    x_cols = st.multiselect("Predictors (X):", [c for c in num_cols if c != y_col],
                             key="reg_reg_x")
    if not x_cols:
        st.info("Select predictor variables.")
        return

    method = st.selectbox("Method:", ["LASSO (L1)", "Ridge (L2)", "Elastic Net"],
                           key="reg_reg_method")

    c1, c2 = st.columns(2)
    n_folds = c1.number_input("CV folds:", 3, 20, 5, key="reg_reg_cv")
    if method == "Elastic Net":
        l1_ratio = c2.slider("L1 ratio:", 0.0, 1.0, 0.5, 0.05, key="reg_reg_l1")

    if st.button("Fit Model", key="reg_reg_fit"):
        data = df[[y_col] + x_cols].dropna()
        X = data[x_cols].values
        y = data[y_col].values

        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, lasso_path

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        with st.spinner("Fitting regularized model..."):
            if method == "LASSO (L1)":
                model = LassoCV(cv=n_folds, random_state=42, max_iter=10000)
                model.fit(X_scaled, y)
                best_alpha = model.alpha_
                coefs = model.coef_
                mse_path = model.mse_path_
                alphas_path = model.alphas_

                # Regularization path
                alphas_lasso, coefs_path, _ = lasso_path(X_scaled, y, alphas=alphas_path)

            elif method == "Ridge (L2)":
                alphas_ridge = np.logspace(-3, 3, 100)
                model = RidgeCV(alphas=alphas_ridge, cv=n_folds,
                                scoring="neg_mean_squared_error")
                model.fit(X_scaled, y)
                best_alpha = model.alpha_
                coefs = model.coef_

                # Compute path manually
                from sklearn.linear_model import Ridge
                coefs_path = []
                for a in alphas_ridge:
                    ridge = Ridge(alpha=a)
                    ridge.fit(X_scaled, y)
                    coefs_path.append(ridge.coef_)
                coefs_path = np.array(coefs_path).T
                alphas_path = alphas_ridge
                mse_path = None

            else:  # Elastic Net
                model = ElasticNetCV(cv=n_folds, l1_ratio=l1_ratio, random_state=42,
                                      max_iter=10000)
                model.fit(X_scaled, y)
                best_alpha = model.alpha_
                coefs = model.coef_
                mse_path = model.mse_path_
                alphas_path = model.alphas_

                alphas_en, coefs_path, _ = lasso_path(X_scaled, y, alphas=alphas_path,
                                                        l1_ratio=l1_ratio if method == "Elastic Net" else 1.0)

        # Results
        section_header("Results")
        r2 = model.score(X_scaled, y)
        y_pred = model.predict(X_scaled)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        c1m, c2m, c3m = st.columns(3)
        c1m.metric("Best α", f"{best_alpha:.6f}")
        c2m.metric("R²", f"{r2:.4f}")
        c3m.metric("RMSE", f"{rmse:.4f}")

        # --- Interpretation card for R² ---
        try:
            interpretation_card(interpret_r_squared(r2))
        except Exception:
            pass

        # Coefficient table
        selected_mask = coefs != 0
        coef_df = pd.DataFrame({
            "Variable": x_cols,
            "Coefficient": coefs.round(6),
            "Selected": selected_mask,
        }).sort_values("Coefficient", key=abs, ascending=False)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        n_selected = selected_mask.sum()
        st.write(f"**Variables selected:** {n_selected} / {len(x_cols)}")

        # Regularization path plot
        section_header("Regularization Path")
        fig = go.Figure()
        for i, name in enumerate(x_cols):
            fig.add_trace(go.Scatter(x=np.log10(alphas_path), y=coefs_path[i],
                                     mode="lines", name=name))
        fig.add_vline(x=np.log10(best_alpha), line_dash="dash", line_color="red",
                      annotation_text=f"Best α={best_alpha:.4f}")
        fig.update_layout(title="Coefficient Path vs log(α)",
                          xaxis_title="log₁₀(α)", yaxis_title="Coefficient",
                          height=500)
        st.plotly_chart(fig, use_container_width=True)

        # MSE vs alpha plot
        if mse_path is not None:
            section_header("Cross-Validation MSE")
            mean_mse = mse_path.mean(axis=1)
            std_mse = mse_path.std(axis=1)
            fig_mse = go.Figure()
            fig_mse.add_trace(go.Scatter(x=np.log10(alphas_path), y=mean_mse,
                                          mode="lines+markers", name="Mean MSE"))
            fig_mse.add_trace(go.Scatter(
                x=np.log10(alphas_path).tolist() + np.log10(alphas_path).tolist()[::-1],
                y=(mean_mse + std_mse).tolist() + (mean_mse - std_mse).tolist()[::-1],
                fill="toself", fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="rgba(99,102,241,0)"), name="±1 Std",
            ))
            fig_mse.add_vline(x=np.log10(best_alpha), line_dash="dash", line_color="red",
                              annotation_text=f"Best α")
            fig_mse.update_layout(title="CV MSE vs log(α)",
                                  xaxis_title="log₁₀(α)", yaxis_title="MSE",
                                  height=400)
            st.plotly_chart(fig_mse, use_container_width=True)

        # Coefficient bar chart
        section_header("Coefficient Magnitudes")
        coef_abs = pd.DataFrame({
            "Variable": x_cols, "Coefficient": coefs
        }).sort_values("Coefficient", key=abs)
        fig_bar = go.Figure(go.Bar(x=coef_abs["Coefficient"], y=coef_abs["Variable"],
                                    orientation="h",
                                    marker_color=["#EF553B" if c == 0 else "#6366f1"
                                                   for c in coef_abs["Coefficient"]]))
        fig_bar.update_layout(title="Coefficients (standardized)", height=max(300, len(x_cols) * 25))
        st.plotly_chart(fig_bar, use_container_width=True)


# ===================================================================
# Tab 11 -- Nonlinear Regression
# ===================================================================

def _render_nonlinear(df: pd.DataFrame):
    """Nonlinear regression with built-in model library."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    section_header("Nonlinear Regression")
    help_tip("Nonlinear Models", """
Built-in models:
- **Exponential Growth:** y = a · exp(b·x)
- **Exponential Decay:** y = a · exp(-b·x)
- **Logistic/Sigmoid:** y = L / (1 + exp(-k·(x-x₀)))
- **Michaelis-Menten:** y = Vmax·x / (Km + x)
- **Power Law:** y = a · x^b
- **Gompertz:** y = a · exp(-b · exp(-c·x))
""")

    c1, c2 = st.columns(2)
    x_col = c1.selectbox("X:", num_cols, key="nl_x")
    y_col = c2.selectbox("Y:", [c for c in num_cols if c != x_col], key="nl_y")

    model_type = st.selectbox("Model:", [
        "Exponential Growth", "Exponential Decay", "Logistic (Sigmoid)",
        "Michaelis-Menten", "Power Law", "Gompertz",
    ], key="nl_model")

    if st.button("Fit Model", key="nl_fit"):
        data = df[[x_col, y_col]].dropna()
        x = data[x_col].values
        y = data[y_col].values

        if len(data) < 3:
            st.error("Need at least 3 data points.")
            return

        # Model definitions: (function, initial parameters, parameter names)
        models = {
            "Exponential Growth": (
                lambda x, a, b: a * np.exp(b * x),
                [1.0, 0.01],
                ["a", "b"],
                "y = a · exp(b·x)",
            ),
            "Exponential Decay": (
                lambda x, a, b: a * np.exp(-b * x),
                [max(y), 0.01],
                ["a", "b"],
                "y = a · exp(-b·x)",
            ),
            "Logistic (Sigmoid)": (
                lambda x, L, k, x0: L / (1 + np.exp(-k * (x - x0))),
                [max(y), 1.0, np.median(x)],
                ["L", "k", "x₀"],
                "y = L / (1 + exp(-k·(x-x₀)))",
            ),
            "Michaelis-Menten": (
                lambda x, Vmax, Km: Vmax * x / (Km + x),
                [max(y), np.median(x)],
                ["Vmax", "Km"],
                "y = Vmax·x / (Km + x)",
            ),
            "Power Law": (
                lambda x, a, b: a * np.power(np.abs(x), b),
                [1.0, 1.0],
                ["a", "b"],
                "y = a · x^b",
            ),
            "Gompertz": (
                lambda x, a, b, c: a * np.exp(-b * np.exp(-c * x)),
                [max(y), 1.0, 0.1],
                ["a", "b", "c"],
                "y = a · exp(-b · exp(-c·x))",
            ),
        }

        func, p0, param_names, formula = models[model_type]

        try:
            with st.spinner("Fitting nonlinear model..."):
                popt, pcov = optimize.curve_fit(func, x, y, p0=p0, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))

            y_pred = func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2_analog = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))

            # Results
            section_header("Model Fit")
            st.markdown(f"**Model:** {formula}")

            c1m, c2m = st.columns(2)
            c1m.metric("Pseudo R²", f"{r2_analog:.4f}")
            c2m.metric("RMSE", f"{rmse:.4f}")

            # --- Interpretation card for Pseudo R² ---
            try:
                interpretation_card(interpret_r_squared(r2_analog))
            except Exception:
                pass

            # Parameter estimates
            section_header("Parameter Estimates")
            param_df = pd.DataFrame({
                "Parameter": param_names,
                "Estimate": popt,
                "Std Error": perr,
                "Lower 95% CI": popt - 1.96 * perr,
                "Upper 95% CI": popt + 1.96 * perr,
            }).round(6)
            st.dataframe(param_df, use_container_width=True, hide_index=True)

            # Fitted equation
            param_str = ", ".join([f"{name}={val:.4f}" for name, val in zip(param_names, popt)])
            st.write(f"**Fitted parameters:** {param_str}")

            # Plot
            section_header("Fitted Curve")
            x_smooth = np.linspace(x.min(), x.max(), 200)
            y_smooth = func(x_smooth, *popt)

            # Confidence bands (delta method approximation)
            from scipy.optimize import approx_fprime
            y_ci_upper = np.zeros(len(x_smooth))
            y_ci_lower = np.zeros(len(x_smooth))
            for idx_s, xi in enumerate(x_smooth):
                def pred_func(params):
                    return func(xi, *params)
                J = approx_fprime(popt, pred_func, 1e-8)
                var_pred = J @ pcov @ J
                se_pred = np.sqrt(max(0, var_pred))
                y_ci_upper[idx_s] = y_smooth[idx_s] + 1.96 * se_pred
                y_ci_lower[idx_s] = y_smooth[idx_s] - 1.96 * se_pred

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                                     marker=dict(size=6, opacity=0.7)))
            fig.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode="lines",
                                     name="Fitted", line=dict(color="red", width=2)))
            fig.add_trace(go.Scatter(
                x=x_smooth.tolist() + x_smooth.tolist()[::-1],
                y=y_ci_upper.tolist() + y_ci_lower.tolist()[::-1],
                fill="toself", fillcolor="rgba(255,0,0,0.1)",
                line=dict(color="rgba(255,0,0,0)"), name="95% CI",
            ))
            fig.update_layout(title=f"Nonlinear Fit: {model_type}",
                              xaxis_title=x_col, yaxis_title=y_col, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Residual plot
            residuals = y - y_pred
            fig_res = make_subplots(rows=1, cols=2,
                                     subplot_titles=("Residuals vs Predicted", "Residual Distribution"))
            fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                          marker=dict(size=5)), row=1, col=1)
            fig_res.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            fig_res.add_trace(go.Histogram(x=residuals, nbinsx=20), row=1, col=2)
            fig_res.update_layout(height=350)
            st.plotly_chart(fig_res, use_container_width=True)

        except RuntimeError as e:
            st.error(f"Curve fitting failed: {e}. Try different initial parameters or a different model.")
        except Exception as e:
            st.error(f"Error: {e}")


# ===================================================================
# Tab 12 -- Prediction Profiler
# ===================================================================

def _render_profiler(df: pd.DataFrame):
    """Interactive prediction profiler for exploring fitted models."""
    if not HAS_SM:
        empty_state("statsmodels required for the prediction profiler.", "Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns.")
        return

    section_header("Prediction Profiler")
    help_tip("Profiler", """
Interactively explore how each predictor affects the response:
- Each subplot shows the marginal effect of one predictor
- Other predictors are held at the values set by the sliders
- The prediction interval shows uncertainty at each setting
""")

    y_col = st.selectbox("Response (Y):", num_cols, key="prof_y")
    x_cols = st.multiselect("Predictors:", [c for c in num_cols if c != y_col],
                             key="prof_x")
    if len(x_cols) < 1:
        st.info("Select at least 1 predictor.")
        return

    model_type = st.selectbox("Model type:", [
        "Linear (main effects)", "Quadratic (with squares & interactions)",
    ], key="prof_model_type")

    if st.button("Fit & Profile", key="prof_fit"):
        data = df[[y_col] + x_cols].dropna()
        X = data[x_cols].values
        y = data[y_col].values

        if len(data) < len(x_cols) + 2:
            st.error("Not enough data points.")
            return

        # Build design matrix
        X_design = [np.ones(len(y))]
        for j in range(len(x_cols)):
            X_design.append(X[:, j])
        if model_type.startswith("Quadratic"):
            for j in range(len(x_cols)):
                X_design.append(X[:, j] ** 2)
            for j1 in range(len(x_cols)):
                for j2 in range(j1 + 1, len(x_cols)):
                    X_design.append(X[:, j1] * X[:, j2])
        X_design = np.column_stack(X_design)

        try:
            model = sm.OLS(y, X_design).fit()
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            return

        section_header("Model Fit")
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("R²", f"{model.rsquared:.4f}")
        c2m.metric("Adj R²", f"{model.rsquared_adj:.4f}")
        c3m.metric("RMSE", f"{np.sqrt(model.mse_resid):.4f}")

        # --- Interpretation card for R² ---
        try:
            interpretation_card(interpret_r_squared(model.rsquared, model.rsquared_adj))
        except Exception:
            pass

        # Interactive sliders
        section_header("Profiler Settings")
        st.caption("Adjust predictor values. Plots show marginal effect of each predictor while others are held constant.")
        current_values = {}
        cols_per_row = min(4, len(x_cols))
        slider_cols = st.columns(cols_per_row)
        for i, col in enumerate(x_cols):
            col_data = data[col]
            c = slider_cols[i % cols_per_row]
            current_values[col] = c.slider(
                f"{col}:", float(col_data.min()), float(col_data.max()),
                float(col_data.mean()), key=f"prof_slider_{col}",
            )

        # Build prediction function
        def predict_at(x_vals):
            x_row = [1.0]
            for j in range(len(x_cols)):
                x_row.append(x_vals[j])
            if model_type.startswith("Quadratic"):
                for j in range(len(x_cols)):
                    x_row.append(x_vals[j] ** 2)
                for j1 in range(len(x_cols)):
                    for j2 in range(j1 + 1, len(x_cols)):
                        x_row.append(x_vals[j1] * x_vals[j2])
            x_arr = np.array([x_row])
            pred = model.get_prediction(x_arr)
            return pred.predicted_mean[0], pred.conf_int(alpha=0.05)[0]

        # Current prediction
        current_x = [current_values[col] for col in x_cols]
        pred_mean, pred_ci = predict_at(current_x)
        st.metric("Predicted Response", f"{pred_mean:.4f}",
                   delta=f"95% CI: [{pred_ci[0]:.4f}, {pred_ci[1]:.4f}]")

        # Marginal effect plots
        section_header("Marginal Effect Plots")
        n_plots = len(x_cols)
        cols_plot = min(n_plots, 3)
        rows_plot = (n_plots + cols_plot - 1) // cols_plot

        fig = make_subplots(rows=rows_plot, cols=cols_plot,
                            subplot_titles=[f"{col}" for col in x_cols])

        for idx, col in enumerate(x_cols):
            row = idx // cols_plot + 1
            c = idx % cols_plot + 1

            x_range = np.linspace(data[col].min(), data[col].max(), 50)
            y_preds = []
            y_lower = []
            y_upper = []

            for x_val in x_range:
                test_x = current_x.copy()
                test_x[idx] = x_val
                pm, ci = predict_at(test_x)
                y_preds.append(pm)
                y_lower.append(ci[0])
                y_upper.append(ci[1])

            # Prediction line
            fig.add_trace(go.Scatter(x=x_range, y=y_preds, mode="lines",
                                     line=dict(color="#6366f1", width=2),
                                     showlegend=False), row=row, col=c)
            # CI band
            fig.add_trace(go.Scatter(
                x=x_range.tolist() + x_range.tolist()[::-1],
                y=y_upper + y_lower[::-1],
                fill="toself", fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="rgba(99,102,241,0)"),
                showlegend=False), row=row, col=c)
            # Current value marker
            fig.add_vline(x=current_values[col], line_dash="dot", line_color="red",
                          row=row, col=c)

            fig.update_xaxes(title_text=col, row=row, col=c)
            if c == 1:
                fig.update_yaxes(title_text=y_col, row=row, col=c)

        fig.update_layout(height=350 * rows_plot,
                          title_text="Marginal Effect Plots (with 95% CI)")
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Tab 13 -- Variable Selection (Stepwise & Best Subsets)
# ===================================================================

def _render_variable_selection(df: pd.DataFrame):
    """Stepwise and best subsets regression for variable selection."""
    if not HAS_SM:
        empty_state("statsmodels required for variable selection.", "Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 3:
        empty_state("Need at least 3 numeric columns.", "Upload a dataset with multiple numeric predictors.")
        return

    section_header("Variable Selection", "Identify the best subset of predictors using stepwise or exhaustive search methods.")
    help_tip("Variable Selection Methods", """
- **Forward Selection:** Start with no predictors. Add the one that improves the criterion most at each step.
- **Backward Elimination:** Start with all predictors. Remove the worst one at each step.
- **Bidirectional Stepwise:** After each forward addition, check if any existing predictor should be removed.
- **Best Subsets:** Exhaustively evaluate all 2^p predictor combinations (limited to p <= 15).
""")

    y_col = st.selectbox("Dependent variable (Y):", num_cols, key="vs_y")
    candidate_cols = [c for c in num_cols if c != y_col]
    x_cols = st.multiselect("Candidate predictors:", candidate_cols, default=candidate_cols[:min(8, len(candidate_cols))], key="vs_x")

    if len(x_cols) < 2:
        st.info("Select at least 2 candidate predictors.")
        return

    p = len(x_cols)
    methods = ["Forward Selection", "Backward Elimination", "Bidirectional Stepwise"]
    if p <= 15:
        methods.append("Best Subsets")

    c1, c2 = st.columns(2)
    method = c1.selectbox("Method:", methods, key="vs_method")
    criterion = c2.selectbox("Criterion:", ["AIC", "BIC", "Adjusted R²", "p-value threshold"], key="vs_criterion")

    p_enter = 0.05
    p_remove = 0.10
    if criterion == "p-value threshold":
        c3, c4 = st.columns(2)
        p_enter = c3.slider("Entry p-value:", 0.001, 0.20, 0.05, 0.005, key="vs_p_enter")
        p_remove = c4.slider("Removal p-value:", 0.01, 0.30, 0.10, 0.005, key="vs_p_remove")

    if st.button("Run Variable Selection", key="vs_run"):
        data = df[[y_col] + x_cols].dropna()
        y = data[y_col].values
        X_all = data[x_cols]
        n = len(y)

        if n < p + 2:
            st.error(f"Not enough observations ({n}) for {p} predictors.")
            return

        def _fit_ols_subset(predictors):
            """Fit OLS with a given subset and return model."""
            if not predictors:
                X_sub = np.ones((n, 1))
            else:
                X_sub = sm.add_constant(X_all[list(predictors)].values)
            return sm.OLS(y, X_sub).fit()

        def _get_criterion(model, predictors, n_total_predictors):
            """Get the selected criterion value for a model."""
            if criterion == "AIC":
                return model.aic
            elif criterion == "BIC":
                return model.bic
            elif criterion == "Adjusted R²":
                return -model.rsquared_adj  # Negate so lower is better
            else:  # p-value threshold - return max p-value
                if not predictors:
                    return 0
                return max(model.pvalues[1:])  # Skip intercept

        with st.spinner("Running variable selection..."):
            step_log = []

            if method == "Forward Selection":
                selected = []
                remaining = list(x_cols)
                step_criteria = []

                # Null model
                null_model = _fit_ols_subset([])
                step_criteria.append({"Step": 0, "Action": "Start (null model)", "Variable": "-",
                                      "AIC": null_model.aic, "BIC": null_model.bic,
                                      "Adj R²": 0.0, "n_predictors": 0})

                for step in range(p):
                    best_crit = float("inf")
                    best_var = None
                    best_model = None

                    for var in remaining:
                        trial = selected + [var]
                        trial_model = _fit_ols_subset(trial)
                        crit_val = _get_criterion(trial_model, trial, p)

                        if crit_val < best_crit:
                            best_crit = crit_val
                            best_var = var
                            best_model = trial_model

                    if best_var is None:
                        break

                    # Check stopping condition
                    if criterion == "p-value threshold":
                        # Check if the new variable's p-value is below threshold
                        var_idx = selected + [best_var]
                        var_pos = len(var_idx)  # Position in params (1-indexed after intercept)
                        if best_model.pvalues[var_pos] > p_enter:
                            step_log.append(f"Step {step+1}: No variable meets p < {p_enter}. Stopping.")
                            break
                    else:
                        # Check if criterion improved
                        current_model = _fit_ols_subset(selected) if selected else null_model
                        current_crit = _get_criterion(current_model, selected, p)
                        if best_crit >= current_crit:
                            step_log.append(f"Step {step+1}: No improvement in {criterion}. Stopping.")
                            break

                    selected.append(best_var)
                    remaining.remove(best_var)
                    step_log.append(f"Step {step+1}: ADD '{best_var}' (AIC={best_model.aic:.2f}, Adj R²={best_model.rsquared_adj:.4f})")
                    step_criteria.append({
                        "Step": step + 1, "Action": f"Add '{best_var}'", "Variable": best_var,
                        "AIC": round(best_model.aic, 2), "BIC": round(best_model.bic, 2),
                        "Adj R²": round(best_model.rsquared_adj, 4),
                        "n_predictors": len(selected),
                    })

                final_predictors = selected
                criteria_df = pd.DataFrame(step_criteria)

            elif method == "Backward Elimination":
                selected = list(x_cols)
                step_criteria = []

                full_model = _fit_ols_subset(selected)
                step_criteria.append({
                    "Step": 0, "Action": "Start (full model)", "Variable": "-",
                    "AIC": full_model.aic, "BIC": full_model.bic,
                    "Adj R²": round(full_model.rsquared_adj, 4),
                    "n_predictors": len(selected),
                })

                for step in range(p):
                    if len(selected) == 0:
                        break

                    current_model = _fit_ols_subset(selected)

                    if criterion == "p-value threshold":
                        # Find the predictor with highest p-value
                        pvals = current_model.pvalues[1:]  # Skip intercept
                        worst_idx = np.argmax(pvals)
                        worst_pval = pvals[worst_idx]
                        if worst_pval <= p_remove:
                            step_log.append(f"Step {step+1}: All p-values <= {p_remove}. Stopping.")
                            break
                        worst_var = selected[worst_idx]
                    else:
                        best_crit = float("inf")
                        worst_var = None
                        for var in selected:
                            trial = [v for v in selected if v != var]
                            if not trial:
                                trial_model = _fit_ols_subset([])
                            else:
                                trial_model = _fit_ols_subset(trial)
                            crit_val = _get_criterion(trial_model, trial, p)
                            if crit_val < best_crit:
                                best_crit = crit_val
                                worst_var = var

                        if worst_var is None:
                            break

                        # Check if removing improves criterion
                        current_crit = _get_criterion(current_model, selected, p)
                        if best_crit >= current_crit:
                            step_log.append(f"Step {step+1}: No improvement from removal. Stopping.")
                            break

                    selected.remove(worst_var)
                    new_model = _fit_ols_subset(selected) if selected else _fit_ols_subset([])
                    step_log.append(f"Step {step+1}: REMOVE '{worst_var}' (AIC={new_model.aic:.2f}, Adj R²={new_model.rsquared_adj:.4f})")
                    step_criteria.append({
                        "Step": step + 1, "Action": f"Remove '{worst_var}'", "Variable": worst_var,
                        "AIC": round(new_model.aic, 2), "BIC": round(new_model.bic, 2),
                        "Adj R²": round(new_model.rsquared_adj, 4),
                        "n_predictors": len(selected),
                    })

                final_predictors = selected
                criteria_df = pd.DataFrame(step_criteria)

            elif method == "Bidirectional Stepwise":
                selected = []
                remaining = list(x_cols)
                step_criteria = []

                null_model = _fit_ols_subset([])
                step_criteria.append({"Step": 0, "Action": "Start (null model)", "Variable": "-",
                                      "AIC": null_model.aic, "BIC": null_model.bic,
                                      "Adj R²": 0.0, "n_predictors": 0})

                step_count = 0
                for _ in range(2 * p):
                    # Forward step: try adding
                    best_add_crit = float("inf")
                    best_add_var = None
                    best_add_model = None

                    for var in remaining:
                        trial = selected + [var]
                        trial_model = _fit_ols_subset(trial)
                        crit_val = _get_criterion(trial_model, trial, p)
                        if crit_val < best_add_crit:
                            best_add_crit = crit_val
                            best_add_var = var
                            best_add_model = trial_model

                    added = False
                    if best_add_var is not None:
                        current_model = _fit_ols_subset(selected) if selected else null_model
                        current_crit = _get_criterion(current_model, selected, p)

                        should_add = False
                        if criterion == "p-value threshold":
                            trial = selected + [best_add_var]
                            trial_model = _fit_ols_subset(trial)
                            var_pos = len(trial)
                            if trial_model.pvalues[var_pos] < p_enter:
                                should_add = True
                        else:
                            if best_add_crit < current_crit:
                                should_add = True

                        if should_add:
                            selected.append(best_add_var)
                            remaining.remove(best_add_var)
                            added = True
                            step_count += 1
                            m = _fit_ols_subset(selected)
                            step_log.append(f"Step {step_count}: ADD '{best_add_var}' (AIC={m.aic:.2f}, Adj R²={m.rsquared_adj:.4f})")
                            step_criteria.append({
                                "Step": step_count, "Action": f"Add '{best_add_var}'", "Variable": best_add_var,
                                "AIC": round(m.aic, 2), "BIC": round(m.bic, 2),
                                "Adj R²": round(m.rsquared_adj, 4),
                                "n_predictors": len(selected),
                            })

                    # Backward step: try removing (only if we have predictors)
                    removed = False
                    if len(selected) > 1:
                        current_model = _fit_ols_subset(selected)
                        current_crit = _get_criterion(current_model, selected, p)

                        best_rem_crit = float("inf")
                        best_rem_var = None

                        for var in selected:
                            trial = [v for v in selected if v != var]
                            trial_model = _fit_ols_subset(trial)
                            crit_val = _get_criterion(trial_model, trial, p)
                            if crit_val < best_rem_crit:
                                best_rem_crit = crit_val
                                best_rem_var = var

                        should_remove = False
                        if criterion == "p-value threshold":
                            pvals = current_model.pvalues[1:]
                            worst_idx = np.argmax(pvals)
                            if pvals[worst_idx] > p_remove:
                                best_rem_var = selected[worst_idx]
                                should_remove = True
                        else:
                            if best_rem_crit < current_crit:
                                should_remove = True

                        if should_remove and best_rem_var is not None:
                            selected.remove(best_rem_var)
                            remaining.append(best_rem_var)
                            removed = True
                            step_count += 1
                            m = _fit_ols_subset(selected) if selected else _fit_ols_subset([])
                            step_log.append(f"Step {step_count}: REMOVE '{best_rem_var}' (AIC={m.aic:.2f}, Adj R²={m.rsquared_adj:.4f})")
                            step_criteria.append({
                                "Step": step_count, "Action": f"Remove '{best_rem_var}'", "Variable": best_rem_var,
                                "AIC": round(m.aic, 2), "BIC": round(m.bic, 2),
                                "Adj R²": round(m.rsquared_adj, 4),
                                "n_predictors": len(selected),
                            })

                    if not added and not removed:
                        step_log.append("No further improvements. Stopping.")
                        break

                final_predictors = selected
                criteria_df = pd.DataFrame(step_criteria)

            else:  # Best Subsets
                from itertools import combinations

                all_results = []
                best_by_size = {}

                total_combos = sum(1 for k in range(1, p + 1) for _ in combinations(x_cols, k))
                progress_bar = st.progress(0)
                combo_count = 0

                for k in range(1, p + 1):
                    for combo in combinations(x_cols, k):
                        combo_count += 1
                        progress_bar.progress(min(combo_count / total_combos, 1.0))

                        predictors = list(combo)
                        model_k = _fit_ols_subset(predictors)

                        # Mallows' Cp
                        full_model = _fit_ols_subset(x_cols)
                        mse_full = full_model.mse_resid
                        sse_p = np.sum(model_k.resid ** 2)
                        cp = sse_p / mse_full - n + 2 * (k + 1)

                        result = {
                            "Predictors": ", ".join(predictors),
                            "n_predictors": k,
                            "R²": round(model_k.rsquared, 4),
                            "Adj R²": round(model_k.rsquared_adj, 4),
                            "AIC": round(model_k.aic, 2),
                            "BIC": round(model_k.bic, 2),
                            "Cp": round(cp, 2),
                        }
                        all_results.append(result)

                        # Track best per size
                        crit_val = _get_criterion(model_k, predictors, p)
                        if k not in best_by_size or crit_val < best_by_size[k]["crit"]:
                            best_by_size[k] = {"crit": crit_val, "predictors": predictors, "result": result}

                progress_bar.empty()

                # Find overall best
                overall_best = min(best_by_size.values(), key=lambda x: x["crit"])
                final_predictors = overall_best["predictors"]

                all_results_df = pd.DataFrame(all_results)

                # Show best subset per size
                section_header("Best Subset per Number of Predictors")
                best_df = pd.DataFrame([v["result"] for v in best_by_size.values()])
                st.dataframe(best_df, use_container_width=True, hide_index=True)

                # Full results in expander
                with st.expander(f"All {len(all_results)} Model Combinations"):
                    st.dataframe(all_results_df.sort_values("AIC"), use_container_width=True, hide_index=True)

                # Criterion vs number of predictors plot
                section_header("Criterion vs Number of Predictors")
                crit_col = criterion if criterion in ["AIC", "BIC"] else "Adj R²"
                if crit_col not in all_results_df.columns:
                    crit_col = "AIC"

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=all_results_df["n_predictors"], y=all_results_df[crit_col],
                    mode="markers", marker=dict(size=5, opacity=0.3, color="#6366f1"),
                    name="All subsets",
                ))
                # Highlight best per size
                best_x = [v["result"]["n_predictors"] for v in best_by_size.values()]
                best_y = [v["result"][crit_col] for v in best_by_size.values()]
                fig.add_trace(go.Scatter(
                    x=best_x, y=best_y,
                    mode="markers+lines", marker=dict(size=10, color="red", symbol="star"),
                    line=dict(color="red", width=2), name="Best subset",
                ))
                fig.update_layout(title=f"{crit_col} vs Number of Predictors",
                                  xaxis_title="Number of Predictors", yaxis_title=crit_col,
                                  height=450)
                st.plotly_chart(fig, use_container_width=True)

                # Mallows' Cp plot
                fig_cp = go.Figure()
                fig_cp.add_trace(go.Scatter(
                    x=all_results_df["n_predictors"], y=all_results_df["Cp"],
                    mode="markers", marker=dict(size=5, opacity=0.3, color="#6366f1"),
                    name="All subsets",
                ))
                # Reference line Cp = p
                cp_ref = np.arange(1, p + 1)
                fig_cp.add_trace(go.Scatter(x=cp_ref, y=cp_ref, mode="lines",
                                             line=dict(color="red", dash="dash"), name="Cp = p"))
                fig_cp.update_layout(title="Mallows' Cp vs Number of Predictors",
                                      xaxis_title="Number of Predictors", yaxis_title="Mallows' Cp",
                                      height=400)
                st.plotly_chart(fig_cp, use_container_width=True)

                criteria_df = None  # Already shown above

        # Step-by-step log (for stepwise methods)
        if step_log:
            with st.expander("Step-by-Step Log", expanded=True):
                for entry in step_log:
                    st.write(entry)

        # Criterion at each step (for stepwise methods)
        if method != "Best Subsets" and criteria_df is not None and len(criteria_df) > 1:
            section_header("Criterion at Each Step")
            st.dataframe(criteria_df, use_container_width=True, hide_index=True)

            fig_step = go.Figure()
            fig_step.add_trace(go.Scatter(
                x=criteria_df["Step"], y=criteria_df["AIC"],
                mode="lines+markers", name="AIC", line=dict(width=2),
            ))
            fig_step.add_trace(go.Scatter(
                x=criteria_df["Step"], y=criteria_df["BIC"],
                mode="lines+markers", name="BIC", line=dict(width=2),
            ))
            fig_step.update_layout(title="Information Criteria at Each Step",
                                   xaxis_title="Step", yaxis_title="Criterion Value",
                                   height=400)
            st.plotly_chart(fig_step, use_container_width=True)

        # Final model summary
        if final_predictors:
            section_header("Selected Model Summary")
            st.write(f"**Selected predictors ({len(final_predictors)}):** {', '.join(final_predictors)}")

            final_model = _fit_ols_subset(final_predictors)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{final_model.rsquared:.4f}")
            c2.metric("Adj R²", f"{final_model.rsquared_adj:.4f}")
            c3.metric("AIC", f"{final_model.aic:.2f}")
            c4.metric("BIC", f"{final_model.bic:.2f}")

            try:
                interpretation_card(interpret_r_squared(final_model.rsquared, final_model.rsquared_adj))
            except Exception:
                pass

            # Coefficients table with VIF
            coef_names = ["Intercept"] + final_predictors
            coef_df = pd.DataFrame({
                "Variable": coef_names,
                "Coefficient": final_model.params,
                "Std Error": final_model.bse,
                "t-value": final_model.tvalues,
                "p-value": final_model.pvalues,
            }).round(6)

            # Add VIF for selected predictors
            if len(final_predictors) > 1:
                X_selected = X_all[final_predictors].values
                vif_vals = [np.nan]  # Intercept
                for i in range(X_selected.shape[1]):
                    try:
                        vif_vals.append(round(variance_inflation_factor(X_selected, i), 4))
                    except Exception:
                        vif_vals.append(float("inf"))
                coef_df["VIF"] = vif_vals

            st.dataframe(coef_df, use_container_width=True, hide_index=True)

            st.markdown(f"**RMSE:** {np.sqrt(final_model.mse_resid):.4f}")
        else:
            st.warning("No predictors were selected. The null model (intercept only) was chosen.")
