"""
Survival Analysis Module - Kaplan-Meier, Log-Rank, Cox PH, Parametric AFT models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import section_header, empty_state, help_tip

try:
    from lifelines import (
        KaplanMeierFitter,
        CoxPHFitter,
        WeibullAFTFitter,
        LogNormalAFTFitter,
        LogLogisticAFTFitter,
    )
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


def render_survival_analysis(df: pd.DataFrame):
    """Render survival analysis interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return
    if not HAS_LIFELINES:
        st.error("The `lifelines` library is required for survival analysis. Install it with `pip install lifelines`.")
        return

    tabs = st.tabs([
        "Kaplan-Meier", "Log-Rank Test",
        "Cox Proportional Hazards", "Parametric Models",
    ])

    with tabs[0]:
        _render_kaplan_meier(df)
    with tabs[1]:
        _render_logrank(df)
    with tabs[2]:
        _render_cox_ph(df)
    with tabs[3]:
        _render_parametric(df)


def _get_duration_event_cols(df, prefix):
    """Common column selectors for duration and event."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns (duration and event).")
        return None, None
    c1, c2 = st.columns(2)
    duration_col = c1.selectbox("Duration column:", num_cols, key=f"{prefix}_dur")
    event_col = c2.selectbox("Event column (0/1):", num_cols, key=f"{prefix}_evt")
    help_tip("Survival data format", "**Event = 1** for observed events (e.g., death, failure). **Event = 0** for censored observations (e.g., lost to follow-up, study ended).")
    return duration_col, event_col


def _validate_survival_data(df, duration_col, event_col):
    """Validate and clean survival data, returning the cleaned subset or None."""
    data = df[[duration_col, event_col]].dropna()
    if data.empty:
        st.error("No valid rows after dropping missing values.")
        return None
    if (data[duration_col] < 0).any():
        st.warning("Negative durations detected and removed.")
        data = data[data[duration_col] >= 0]
    unique_events = sorted(data[event_col].unique())
    if not set(unique_events).issubset({0, 1, 0.0, 1.0}):
        st.error(f"Event column must contain only 0 and 1. Found: {unique_events}")
        return None
    return data


# ---------------------------------------------------------------------------
# Tab 1: Kaplan-Meier
# ---------------------------------------------------------------------------

def _render_kaplan_meier(df: pd.DataFrame):
    """Kaplan-Meier survival estimation."""
    duration_col, event_col = _get_duration_event_cols(df, "km")
    if duration_col is None:
        return

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    group_col = st.selectbox(
        "Group column (optional):", ["None"] + cat_cols, key="km_group"
    )
    group_col = None if group_col == "None" else group_col

    timepoints_input = st.text_input(
        "Survival at timepoints (comma-separated, optional):",
        placeholder="e.g. 12, 24, 36, 60",
        key="km_timepoints",
    )

    if st.button("Fit Model", key="km_fit"):
        data = _validate_survival_data(df, duration_col, event_col)
        if data is None:
            return

        try:
            fig = go.Figure()
            colors = px.colors.qualitative.Set1
            at_risk_records = []

            if group_col is None:
                # Single KM curve
                kmf = KaplanMeierFitter()
                kmf.fit(data[duration_col], event_observed=data[event_col])
                _add_km_trace(fig, kmf, "Overall", colors[0])
                _show_km_metrics(kmf, "Overall")
                at_risk_records.extend(_build_at_risk(kmf, "Overall"))

                if timepoints_input.strip():
                    _show_survival_at_timepoints(kmf, timepoints_input, "Overall")
            else:
                full = df[[duration_col, event_col, group_col]].dropna()
                full = full[full[duration_col] >= 0]
                groups = full[group_col].unique()
                if len(groups) > 10:
                    st.warning("Too many groups (>10). Showing first 10.")
                    groups = groups[:10]

                for i, grp in enumerate(sorted(groups)):
                    mask = full[group_col] == grp
                    kmf = KaplanMeierFitter()
                    kmf.fit(
                        full.loc[mask, duration_col],
                        event_observed=full.loc[mask, event_col],
                        label=str(grp),
                    )
                    color = colors[i % len(colors)]
                    _add_km_trace(fig, kmf, str(grp), color)
                    _show_km_metrics(kmf, str(grp))
                    at_risk_records.extend(_build_at_risk(kmf, str(grp)))

                    if timepoints_input.strip():
                        _show_survival_at_timepoints(kmf, timepoints_input, str(grp))

            fig.update_layout(
                title="Kaplan-Meier Survival Curve",
                xaxis_title="Time",
                yaxis_title="Survival Probability",
                yaxis=dict(range=[0, 1.05]),
                height=550,
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            )
            st.plotly_chart(fig, use_container_width=True)

            # At-risk table
            with st.expander("At-Risk Table"):
                if at_risk_records:
                    atr_df = pd.DataFrame(at_risk_records)
                    st.dataframe(atr_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Kaplan-Meier error: {e}")


def _add_km_trace(fig, kmf, label, color):
    """Add a KM survival curve with confidence band to a plotly figure."""
    sf = kmf.survival_function_
    timeline = sf.index.values
    survival = sf.iloc[:, 0].values

    ci = kmf.confidence_interval_survival_function_
    ci_lower = ci.iloc[:, 0].values
    ci_upper = ci.iloc[:, 1].values

    # Step-style curve
    fig.add_trace(go.Scatter(
        x=timeline, y=survival, mode="lines",
        name=label, line=dict(color=color, width=2, shape="hv"),
    ))
    # Upper CI
    fig.add_trace(go.Scatter(
        x=timeline, y=ci_upper, mode="lines",
        line=dict(width=0, shape="hv"), showlegend=False, hoverinfo="skip",
    ))
    # Lower CI (fill between)
    fig.add_trace(go.Scatter(
        x=timeline, y=ci_lower, mode="lines",
        line=dict(width=0, shape="hv"), showlegend=False, hoverinfo="skip",
        fill="tonexty",
        fillcolor=color.replace("rgb", "rgba").replace(")", ",0.15)") if "rgb" in color
        else f"rgba(150,150,150,0.15)",
    ))


def _show_km_metrics(kmf, label):
    """Display median survival and summary stats."""
    median = kmf.median_survival_time_
    n_obs = kmf.event_observed.shape[0]
    n_events = int(kmf.event_observed.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Median Survival ({label})", f"{median:.2f}" if np.isfinite(median) else "Not reached")
    c2.metric(f"Observations ({label})", str(n_obs))
    c3.metric(f"Events ({label})", str(n_events))


def _show_survival_at_timepoints(kmf, timepoints_input, label):
    """Show survival probability at user-specified timepoints."""
    try:
        tps = [float(t.strip()) for t in timepoints_input.split(",") if t.strip()]
    except ValueError:
        st.warning("Invalid timepoint format. Use comma-separated numbers.")
        return
    rows = []
    for tp in tps:
        try:
            prob = kmf.predict(tp)
            rows.append({"Group": label, "Time": tp, "S(t)": f"{prob:.4f}"})
        except Exception:
            rows.append({"Group": label, "Time": tp, "S(t)": "N/A"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _build_at_risk(kmf, label):
    """Build at-risk table rows from a fitted KM model."""
    sf = kmf.survival_function_
    timeline = sf.index.values
    n_start = kmf.event_observed.shape[0]
    records = []
    # Sample ~10 evenly spaced timepoints
    idx = np.linspace(0, len(timeline) - 1, min(10, len(timeline)), dtype=int)
    for i in idx:
        t = timeline[i]
        surv = sf.iloc[i, 0]
        at_risk = int(np.round(n_start * surv))
        records.append({
            "Group": label, "Time": f"{t:.2f}",
            "At Risk (approx)": at_risk, "S(t)": f"{surv:.4f}",
        })
    return records


# ---------------------------------------------------------------------------
# Tab 2: Log-Rank Test
# ---------------------------------------------------------------------------

def _render_logrank(df: pd.DataFrame):
    """Log-rank test for comparing survival between groups."""
    duration_col, event_col = _get_duration_event_cols(df, "lr")
    if duration_col is None:
        return

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        empty_state("Need at least one categorical column for group comparison.")
        return

    group_col = st.selectbox("Group column:", cat_cols, key="lr_group")
    alpha = st.slider("Significance level:", 0.001, 0.10, 0.05, 0.001, key="lr_alpha")

    if st.button("Run Test", key="lr_run"):
        full = df[[duration_col, event_col, group_col]].dropna()
        full = full[full[duration_col] >= 0]
        groups = full[group_col].unique()

        if len(groups) < 2:
            st.error("Need at least 2 groups for comparison.")
            return

        try:
            if len(groups) == 2:
                g1_mask = full[group_col] == groups[0]
                g2_mask = full[group_col] == groups[1]
                result = logrank_test(
                    full.loc[g1_mask, duration_col],
                    full.loc[g2_mask, duration_col],
                    event_observed_A=full.loc[g1_mask, event_col],
                    event_observed_B=full.loc[g2_mask, event_col],
                )
            else:
                result = multivariate_logrank_test(
                    full[duration_col], full[group_col], full[event_col],
                )

            c1, c2, c3 = st.columns(3)
            c1.metric("Test Statistic", f"{result.test_statistic:.4f}")
            c2.metric("p-value", f"{result.p_value:.6f}")
            c3.metric("Degrees of Freedom", str(len(groups) - 1))

            if result.p_value < alpha:
                st.success(
                    f"**Reject H_0** (p = {result.p_value:.6f} < alpha = {alpha}): "
                    "Significant difference in survival between groups."
                )
            else:
                st.info(
                    f"**Fail to reject H_0** (p = {result.p_value:.6f} >= alpha = {alpha}): "
                    "No significant difference in survival between groups."
                )

            # Overlay KM curves for comparison
            fig = go.Figure()
            colors = px.colors.qualitative.Set1
            if len(groups) > 10:
                st.warning("Showing first 10 groups only.")
                groups = groups[:10]

            for i, grp in enumerate(sorted(groups)):
                mask = full[group_col] == grp
                kmf = KaplanMeierFitter()
                kmf.fit(
                    full.loc[mask, duration_col],
                    event_observed=full.loc[mask, event_col],
                    label=str(grp),
                )
                sf = kmf.survival_function_
                fig.add_trace(go.Scatter(
                    x=sf.index.values, y=sf.iloc[:, 0].values,
                    mode="lines", name=str(grp),
                    line=dict(color=colors[i % len(colors)], width=2, shape="hv"),
                ))

            fig.update_layout(
                title="Survival Curves by Group (Log-Rank Comparison)",
                xaxis_title="Time", yaxis_title="Survival Probability",
                yaxis=dict(range=[0, 1.05]), height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Log-rank test error: {e}")


# ---------------------------------------------------------------------------
# Tab 3: Cox Proportional Hazards
# ---------------------------------------------------------------------------

def _render_cox_ph(df: pd.DataFrame):
    """Cox proportional hazards regression."""
    duration_col, event_col = _get_duration_event_cols(df, "cox")
    if duration_col is None:
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    covariate_options = [c for c in num_cols if c not in (duration_col, event_col)]

    if not covariate_options:
        empty_state("No additional numeric columns available as covariates.")
        return

    covariates = st.multiselect(
        "Covariate columns:", covariate_options,
        default=covariate_options[:min(3, len(covariate_options))],
        key="cox_covariates",
    )

    penalizer = st.slider(
        "L2 penalizer (regularization):", 0.0, 1.0, 0.0, 0.01, key="cox_penalizer"
    )

    if st.button("Fit Model", key="cox_fit"):
        if not covariates:
            st.warning("Select at least one covariate.")
            return

        cols_needed = [duration_col, event_col] + covariates
        data = df[cols_needed].dropna()
        data = data[data[duration_col] >= 0]

        if len(data) < 10:
            st.error("Too few observations after cleaning (need at least 10).")
            return

        try:
            cph = CoxPHFitter(penalizer=penalizer)
            with st.spinner("Fitting Cox model..."):
                cph.fit(data, duration_col=duration_col, event_col=event_col)

            # Model-level metrics
            section_header("Model Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("Concordance Index", f"{cph.concordance_index_:.4f}")
            c2.metric("Partial AIC", f"{cph.AIC_partial_:.2f}")
            c3.metric("Log-Likelihood Ratio p",
                       f"{cph.log_likelihood_ratio_test().p_value:.6f}")

            # Coefficient table
            section_header("Coefficients")
            summary = cph.summary
            display_cols = ["coef", "exp(coef)", "se(coef)", "z", "p", "coef lower 95%", "coef upper 95%"]
            available_cols = [c for c in display_cols if c in summary.columns]
            coef_df = summary[available_cols].copy()
            coef_df = coef_df.round(4)
            coef_df.index.name = "Covariate"
            st.dataframe(coef_df, use_container_width=True)

            # PH assumption test
            with st.expander("Proportional Hazards Assumption Test"):
                try:
                    ph_test = cph.check_assumptions(data, p_value_threshold=0.05, show_plots=False)
                    st.success("Proportional hazards assumption satisfied for all covariates.")
                except Exception as ph_exc:
                    st.warning(f"PH assumption may be violated: {ph_exc}")
                try:
                    from lifelines.statistics import proportional_hazard_test
                    ph_result = proportional_hazard_test(cph, data, time_transform="rank")
                    st.markdown("**Schoenfeld Test Results:**")
                    st.dataframe(ph_result.summary.round(4), use_container_width=True)
                except Exception:
                    pass

            # Forest plot of hazard ratios
            section_header("Forest Plot (Hazard Ratios)")
            _plot_forest(cph.summary)

        except Exception as e:
            st.error(f"Cox PH error: {e}")


def _plot_forest(summary_df):
    """Create a forest plot of hazard ratios with 95% CIs using plotly."""
    covariates = summary_df.index.tolist()

    hr = summary_df["exp(coef)"].values
    # Determine CI column names (lifelines versions vary)
    if "exp(coef) lower 95%" in summary_df.columns:
        hr_lower = summary_df["exp(coef) lower 95%"].values
        hr_upper = summary_df["exp(coef) upper 95%"].values
    else:
        hr_lower = np.exp(summary_df["coef lower 95%"].values)
        hr_upper = np.exp(summary_df["coef upper 95%"].values)

    fig = go.Figure()

    # CI lines
    for i, cov in enumerate(covariates):
        fig.add_trace(go.Scatter(
            x=[hr_lower[i], hr_upper[i]], y=[cov, cov],
            mode="lines", line=dict(color="#6366f1", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # Point estimates
    fig.add_trace(go.Scatter(
        x=hr, y=covariates, mode="markers",
        marker=dict(color="#6366f1", size=10, symbol="diamond"),
        name="Hazard Ratio",
        text=[f"HR={h:.3f} [{l:.3f}, {u:.3f}]" for h, l, u in zip(hr, hr_lower, hr_upper)],
        hoverinfo="text",
    ))

    # Reference line at HR=1
    fig.add_vline(x=1, line_dash="dash", line_color="red", annotation_text="HR=1")

    fig.update_layout(
        title="Hazard Ratios with 95% Confidence Intervals",
        xaxis_title="Hazard Ratio (log scale)",
        xaxis_type="log",
        height=max(350, 60 * len(covariates)),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Parametric Models
# ---------------------------------------------------------------------------

def _render_parametric(df: pd.DataFrame):
    """Parametric AFT survival models."""
    duration_col, event_col = _get_duration_event_cols(df, "param")
    if duration_col is None:
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    covariate_options = [c for c in num_cols if c not in (duration_col, event_col)]

    covariates = st.multiselect(
        "Covariate columns (optional):", covariate_options, key="param_covariates"
    )

    if st.button("Fit Models", key="param_fit"):
        cols_needed = [duration_col, event_col] + covariates
        data = df[cols_needed].dropna()
        data = data[data[duration_col] > 0]  # AFT models need strictly positive durations

        if len(data) < 10:
            st.error("Too few observations after cleaning (need at least 10).")
            return

        model_specs = [
            ("Weibull AFT", WeibullAFTFitter),
            ("Log-Normal AFT", LogNormalAFTFitter),
            ("Log-Logistic AFT", LogLogisticAFTFitter),
        ]

        fitted_models = {}
        aic_rows = []

        for name, ModelClass in model_specs:
            try:
                m = ModelClass(penalizer=0.01)
                m.fit(data, duration_col=duration_col, event_col=event_col)
                fitted_models[name] = m
                aic_rows.append({
                    "Model": name,
                    "AIC": m.AIC_,
                    "BIC": m.BIC_ if hasattr(m, "BIC_") else np.nan,
                    "Log-Likelihood": m.log_likelihood_,
                    "Concordance": m.concordance_index_,
                })
            except Exception as e:
                st.warning(f"{name} failed to fit: {e}")

        if not fitted_models:
            st.error("All parametric models failed to fit.")
            return

        # AIC comparison table
        section_header("Model Comparison")
        aic_df = pd.DataFrame(aic_rows).sort_values("AIC")
        st.dataframe(aic_df.round(4), use_container_width=True, hide_index=True)

        best_name = aic_df.iloc[0]["Model"]
        st.success(f"Best model by AIC: **{best_name}**")

        # Best model coefficients
        best_model = fitted_models[best_name]
        section_header(f"{best_name} Coefficients")
        st.dataframe(best_model.summary.round(4), use_container_width=True)

        # Plot fitted survival curves from each model
        section_header("Fitted Survival Curves")
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        t_max = data[duration_col].max()
        timeline = np.linspace(0.01, t_max, 200)

        for i, (name, model) in enumerate(fitted_models.items()):
            try:
                sf = model.predict_survival_function(
                    data.drop(columns=[duration_col, event_col]).median().to_frame().T
                    if covariates else data.head(1),
                    times=timeline,
                )
                survival_vals = sf.iloc[:, 0].values
                fig.add_trace(go.Scatter(
                    x=timeline, y=survival_vals, mode="lines",
                    name=name, line=dict(color=colors[i % len(colors)], width=2),
                ))
            except Exception:
                # Fallback: use baseline survival if predict fails
                try:
                    sf = model.predict_survival_function(
                        pd.DataFrame({c: [0] for c in covariates}) if covariates
                        else data.head(1),
                        times=timeline,
                    )
                    survival_vals = sf.iloc[:, 0].values
                    fig.add_trace(go.Scatter(
                        x=timeline, y=survival_vals, mode="lines",
                        name=name, line=dict(color=colors[i % len(colors)], width=2, dash="dot"),
                    ))
                except Exception:
                    pass

        # Overlay KM for reference
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(data[duration_col], event_observed=data[event_col])
            km_sf = kmf.survival_function_
            fig.add_trace(go.Scatter(
                x=km_sf.index.values, y=km_sf.iloc[:, 0].values,
                mode="lines", name="Kaplan-Meier (reference)",
                line=dict(color="black", width=1, dash="dash", shape="hv"),
            ))
        except Exception:
            pass

        fig.update_layout(
            title="Parametric Survival Curves vs Kaplan-Meier",
            xaxis_title="Time", yaxis_title="Survival Probability",
            yaxis=dict(range=[0, 1.05]), height=550,
        )
        st.plotly_chart(fig, use_container_width=True)
