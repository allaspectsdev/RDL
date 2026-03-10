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
        st.warning("No data loaded.")
        return

    tabs = st.tabs([
        "Simple Linear", "Multiple Linear", "Polynomial",
        "Logistic", "Curve Fitting", "Diagnostics",
        "GLM", "Robust & Quantile", "Mixed Models",
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


def _render_simple_linear(df: pd.DataFrame):
    """Simple linear regression."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
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

            st.markdown("#### Regression Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R²", f"{model.rsquared:.4f}")
            c2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
            c3.metric("F-statistic", f"{model.fvalue:.4f}")
            c4.metric("p (F-test)", f"{model.f_pvalue:.6f}")

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
            st.markdown(f"**Equation:** y = {b0:.4f} + {b1:.4f} · x")
            st.markdown(f"**AIC:** {model.aic:.2f}  |  **BIC:** {model.bic:.2f}  |  **RMSE:** {np.sqrt(model.mse_resid):.4f}")

            # Plot
            x_line = np.linspace(x.min(), x.max(), 200)
            X_line = sm.add_constant(x_line)
            y_pred = model.predict(X_line)
            predictions = model.get_prediction(X_line)
            ci_frame = predictions.summary_frame(alpha=0.05)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                                     marker=dict(color="steelblue", size=6, opacity=0.7)))
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
            st.metric("R²", f"{r_value**2:.4f}")
            st.write(f"**y = {intercept:.4f} + {slope:.4f} · x**  (p = {p_value:.6f})")
            fig = px.scatter(data, x=x_col, y=y_col, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)


def _render_multiple_linear(df: pd.DataFrame):
    """Multiple linear regression."""
    if not HAS_SM:
        st.warning("statsmodels required for multiple regression.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 3:
        st.warning("Need at least 3 numeric columns.")
        return

    y_col = st.selectbox("Y (response):", num_cols, key="mlr_y")
    x_cols = st.multiselect("X (predictors):", [c for c in num_cols if c != y_col], key="mlr_x")

    if not x_cols:
        st.info("Select predictor variables.")
        return

    if st.button("Fit Model", key="fit_mlr"):
        data = df[[y_col] + x_cols].dropna()
        y = data[y_col].values
        X = sm.add_constant(data[x_cols].values)
        model = sm.OLS(y, X).fit()

        st.markdown("#### Model Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R²", f"{model.rsquared:.4f}")
        c2.metric("Adj R²", f"{model.rsquared_adj:.4f}")
        c3.metric("F-statistic", f"{model.fvalue:.4f}")
        c4.metric("p (F-test)", f"{model.f_pvalue:.6f}")

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

        # VIF
        st.markdown("#### Variance Inflation Factors")
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

        # Coefficient plot
        fig = go.Figure()
        coefs = model.params[1:]  # Skip intercept
        ci = model.conf_int()
        ci_lower = ci[1:, 0]
        ci_upper = ci[1:, 1]
        fig.add_trace(go.Scatter(x=coefs, y=x_cols, mode="markers",
                                 marker=dict(size=10, color="steelblue"),
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
        st.warning("Need at least 2 numeric columns.")
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

        c1, c2, c3 = st.columns(3)
        c1.metric("R²", f"{r2:.4f}")
        c2.metric("Adjusted R²", f"{adj_r2:.4f}")
        c3.metric("RMSE", f"{rmse:.4f}")

        # Equation
        terms = []
        for i, c in enumerate(coeffs):
            power = degree - i
            if power == 0:
                terms.append(f"{c:.4f}")
            elif power == 1:
                terms.append(f"{c:.4f}x")
            else:
                terms.append(f"{c:.4f}x^{power}")
        st.markdown(f"**Equation:** y = {' + '.join(terms)}")

        # Plot
        x_line = np.linspace(x.min(), x.max(), 300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Data",
                                 marker=dict(color="steelblue", size=6, opacity=0.7)))
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
    """Logistic regression."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    # Find potential binary targets
    binary_cols = [c for c in all_cols if df[c].nunique() == 2]
    if not binary_cols:
        st.warning("No binary target variable found (need column with exactly 2 unique values).")
        return

    target = st.selectbox("Target (binary):", binary_cols, key="log_target")
    features = st.multiselect("Features:", [c for c in num_cols if c != target], key="log_features")

    if not features:
        st.info("Select feature variables.")
        return

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

        # Train/test split for honest evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        # Fit with statsmodels for proper inference (on training data)
        if HAS_SM:
            X_sm_train = sm.add_constant(X_train)
            X_sm_test = sm.add_constant(X_test)
            try:
                logit_model = sm.Logit(y_train, X_sm_train).fit(disp=0)
                st.markdown("#### Model Summary")

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
                                     line=dict(color="steelblue", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                     line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(title="ROC Curve (Test Set)", xaxis_title="False Positive Rate",
                              yaxis_title="True Positive Rate", height=400)
        st.plotly_chart(fig_roc, use_container_width=True)


def _render_curve_fitting(df: pd.DataFrame):
    """Non-linear curve fitting (like MATLAB cftool)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
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
                                     marker=dict(color="steelblue", size=6, opacity=0.7)))
            fig.add_trace(go.Scatter(x=x_line, y=func(x_line, *popt), mode="lines",
                                     name="Fit", line=dict(color="red", width=2)))
            fig.update_layout(title=f"Curve Fit: {model_type.split('(')[0].strip()}",
                              xaxis_title=x_col, yaxis_title=y_col, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Residual plot
            residuals = y - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                          marker=dict(color="steelblue", size=5)))
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
        st.warning("statsmodels required for regression diagnostics.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
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

        st.markdown("#### Diagnostic Tests")
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
        st.markdown("#### Influence Measures")
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        leverage = influence.hat_matrix_diag

        n = len(y)
        k = len(x_cols) + 1

        c1, c2 = st.columns(2)
        c1.metric("High Leverage Points (>2k/n)", int(np.sum(leverage > 2 * k / n)))
        c2.metric("Influential Points (Cook's D > 4/n)", int(np.sum(cooks_d > 4 / n)))

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Cook's Distance", "Leverage"))
        fig.add_trace(go.Bar(y=cooks_d, name="Cook's D", marker_color="steelblue"), row=1, col=1)
        fig.add_hline(y=4 / n, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_trace(go.Bar(y=leverage, name="Leverage", marker_color="steelblue"), row=1, col=2)
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
                             marker=dict(color="steelblue", size=4),
                             showlegend=False), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    # QQ Plot
    sorted_resid = np.sort(std_resid)
    n = len(sorted_resid)
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_resid, mode="markers",
                             marker=dict(color="steelblue", size=4),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode="lines",
                             line=dict(color="red", dash="dash"),
                             showlegend=False), row=1, col=2)

    # Scale-Location
    fig.add_trace(go.Scatter(x=fitted, y=np.sqrt(np.abs(std_resid)), mode="markers",
                             marker=dict(color="steelblue", size=4),
                             showlegend=False), row=2, col=1)

    # Residuals vs Order
    fig.add_trace(go.Scatter(y=resid, mode="lines+markers",
                             marker=dict(color="steelblue", size=3),
                             line=dict(color="steelblue", width=1),
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
        st.warning("statsmodels required for GLM.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
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
            model = StatsGLM(y, X, family=family).fit()

            st.markdown("#### GLM Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("AIC", f"{model.aic:.2f}")
            c2.metric("BIC", f"{model.bic_deviance:.2f}")
            c3.metric("Deviance", f"{model.deviance:.4f}")
            c4.metric("Pearson χ²", f"{model.pearson_chi2:.4f}")

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
            st.markdown("#### Deviance Table")
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
                                     marker=dict(color="steelblue", size=4), showlegend=False),
                          row=1, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            sorted_resid = np.sort(resid)
            n = len(sorted_resid)
            theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
            fig.add_trace(go.Scatter(x=theoretical, y=sorted_resid, mode="markers",
                                     marker=dict(color="steelblue", size=4), showlegend=False),
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
        st.warning("statsmodels required.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns.")
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
                model = RLM(y, X, M=norm_map[m_estimator]).fit()

                st.markdown("#### Robust Regression Summary")
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
                st.markdown("#### OLS vs Robust Comparison")
                st.dataframe(comp_df, use_container_width=True, hide_index=True)

                # Weights plot
                weights = model.weights
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=weights, mode="markers",
                                         marker=dict(color="steelblue", size=4), name="Weights"))
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
                st.markdown("#### Quantile Regression Coefficients")
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
                                             name="Inliers", marker=dict(color="steelblue", size=5)))
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
        st.warning("statsmodels required for mixed models.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols or not cat_cols:
        st.warning("Need numeric and categorical columns for mixed models.")
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
            if random_slope_col:
                exog_re = data[[random_slope_col]]
                model = MixedLM(y, X, groups=groups, exog_re=exog_re).fit(reml=True)
            else:
                model = MixedLM(y, X, groups=groups).fit(reml=True)

            st.markdown("#### Mixed Model Summary")

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
                fig.add_trace(go.Bar(x=group_names, y=values, marker_color="steelblue"))
                fig.update_layout(title=f"Random {group_var} by Group",
                                  xaxis_title=group_col, yaxis_title=f"Random {group_var}", height=350)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Mixed model failed: {e}")
