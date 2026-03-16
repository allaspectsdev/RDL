"""
Survival Analysis Module - Kaplan-Meier, Log-Rank, Cox PH, Parametric AFT models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import section_header, empty_state, help_tip, validation_panel, interpretation_card, rdl_plotly_chart
from modules.validation import check_sample_size, interpret_p_value, interpret_effect_size

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

try:
    from lifelines import AalenJohansenFitter
    HAS_AJ = True
except ImportError:
    HAS_AJ = False


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
        "Reliability", "Competing Risks",
    ])

    with tabs[0]:
        _render_kaplan_meier(df)
    with tabs[1]:
        _render_logrank(df)
    with tabs[2]:
        _render_cox_ph(df)
    with tabs[3]:
        _render_parametric(df)
    with tabs[4]:
        _render_reliability(df)
    with tabs[5]:
        _render_competing_risks(df)


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
            rdl_plotly_chart(fig)

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
            rdl_plotly_chart(fig)

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
            empty_state("Select at least one covariate.", "Choose covariates from the list above to fit the Cox model.")
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

            # Sample size check for survival analysis
            try:
                n_events = int(data[event_col].sum())
                ss_check = check_sample_size(n_events, "survival")
                validation_panel([ss_check], title="Sample Size Check")
            except Exception:
                pass

            # PH assumption test — rendered prominently via validation_panel
            try:
                from modules.validation import ValidationCheck
                ph_checks = []
                try:
                    cph.check_assumptions(data, p_value_threshold=0.05, show_plots=False)
                    ph_checks.append(ValidationCheck(
                        name="Proportional Hazards Assumption",
                        status="pass",
                        detail="All covariates satisfy the PH assumption (Schoenfeld test p > 0.05)",
                    ))
                except Exception as ph_exc:
                    ph_checks.append(ValidationCheck(
                        name="Proportional Hazards Assumption",
                        status="warn",
                        detail=f"PH assumption may be violated: {ph_exc}",
                        suggestion="Consider time-varying covariates or stratified Cox model",
                    ))
                try:
                    from lifelines.statistics import proportional_hazard_test
                    ph_result = proportional_hazard_test(cph, data, time_transform="rank")
                    for cov_name in ph_result.summary.index:
                        row = ph_result.summary.loc[cov_name]
                        p_val = row["p"] if "p" in row.index else row.iloc[-1]
                        status = "pass" if p_val >= 0.05 else ("warn" if p_val >= 0.01 else "fail")
                        ph_checks.append(ValidationCheck(
                            name=f"Schoenfeld Test ({cov_name})",
                            status=status,
                            detail=f"p = {p_val:.4f}" + (" -- PH assumption holds" if p_val >= 0.05 else " -- PH assumption violated"),
                            suggestion="" if p_val >= 0.05 else "Consider time-varying coefficient for this covariate",
                        ))
                    with st.expander("Schoenfeld Test Details"):
                        st.dataframe(ph_result.summary.round(4), use_container_width=True)
                except Exception:
                    pass

                if ph_checks:
                    validation_panel(ph_checks, title="Proportional Hazards Diagnostics")
            except Exception:
                pass

            # Forest plot of hazard ratios
            section_header("Forest Plot (Hazard Ratios)")
            _plot_forest(cph.summary)

            # Hazard ratio interpretation cards
            try:
                for cov_name in cph.summary.index:
                    hr = cph.summary.loc[cov_name, "exp(coef)"]
                    if hr > 1:
                        direction = f"a {((hr - 1) * 100):.1f}% increase in the hazard (risk) for each unit increase in {cov_name}"
                    elif hr < 1:
                        direction = f"a {((1 - hr) * 100):.1f}% decrease in the hazard (risk) for each unit increase in {cov_name}"
                    else:
                        direction = f"no change in the hazard for each unit increase in {cov_name}"
                    interpretation_card({
                        "title": "Hazard Ratio",
                        "body": f"A hazard ratio of {hr:.2f} means {direction}.",
                        "detail": f"Covariate: {cov_name}",
                    })
            except Exception:
                pass

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
    rdl_plotly_chart(fig)


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
        rdl_plotly_chart(fig)


# ---------------------------------------------------------------------------
# Tab 5: Reliability Analysis
# ---------------------------------------------------------------------------

def _render_reliability(df: pd.DataFrame):
    """Reliability analysis: Weibull plots, multi-distribution fitting, degradation."""
    from scipy import stats as sp_stats

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
        return

    analysis = st.selectbox("Analysis:", [
        "Weibull Probability Plot",
        "Multi-Distribution Fitting",
        "Degradation Analysis",
    ], key="rel_analysis")

    if analysis == "Weibull Probability Plot":
        section_header("Weibull Probability Plot")
        help_tip("Weibull Analysis", """
The Weibull distribution is the most common reliability model:
- **Shape (\u03b2):** \u03b2 < 1 = early life failures, \u03b2 = 1 = random failures, \u03b2 > 1 = wear-out
- **Scale (\u03b7):** Characteristic life (63.2% of units fail by this time)
- Points following a straight line on Weibull paper indicate a good fit
""")

        col = st.selectbox("Failure time column:", num_cols, key="rel_wb_col")
        data = df[col].dropna().values
        data = data[data > 0]  # Weibull needs positive values

        if len(data) < 3:
            st.error("Need at least 3 positive observations.")
            return

        # Fit Weibull distribution
        shape, loc, scale = sp_stats.weibull_min.fit(data, floc=0)

        # Weibull probability plot (linearized)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        # Median rank approximation for plotting positions
        median_ranks = (np.arange(1, n + 1) - 0.3) / (n + 0.4)

        # Linearized coordinates
        x_plot = np.log(sorted_data)
        y_plot = np.log(-np.log(1 - median_ranks))

        # Fitted line
        slope = shape
        intercept = -shape * np.log(scale)
        x_fit = np.linspace(x_plot.min(), x_plot.max(), 100)
        y_fit = slope * x_fit + intercept

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="markers",
                                 marker=dict(size=8), name="Data"))
        fig.add_trace(go.Scatter(x=x_fit, y=y_fit, mode="lines",
                                 line=dict(color="red", dash="dash"), name="Fitted Line"))
        fig.update_layout(title="Weibull Probability Plot",
                          xaxis_title="ln(Time)", yaxis_title="ln(-ln(1-F(t)))",
                          height=500)
        rdl_plotly_chart(fig)

        # Parameters
        section_header("Weibull Parameters")
        c1, c2, c3 = st.columns(3)
        c1.metric("Shape (\u03b2)", f"{shape:.4f}")
        c2.metric("Scale (\u03b7)", f"{scale:.4f}")

        # Interpretation
        if shape < 1:
            c3.metric("Failure Pattern", "Early Life (\u03b2 < 1)")
        elif abs(shape - 1) < 0.1:
            c3.metric("Failure Pattern", "Random (\u03b2 \u2248 1)")
        else:
            c3.metric("Failure Pattern", "Wear-out (\u03b2 > 1)")

        # Reliability at specific times
        section_header("Reliability Predictions")
        times_input = st.text_input("Predict at times (comma-sep):",
                                     placeholder="e.g. 100, 500, 1000",
                                     key="rel_wb_times")
        if times_input.strip():
            try:
                times = [float(t.strip()) for t in times_input.split(",")]
                pred_rows = []
                for t in times:
                    R_t = 1 - sp_stats.weibull_min.cdf(t, shape, scale=scale)
                    h_t = sp_stats.weibull_min.pdf(t, shape, scale=scale) / R_t if R_t > 0 else np.inf
                    pred_rows.append({
                        "Time": t,
                        "R(t) Reliability": round(R_t, 6),
                        "F(t) Failure Prob": round(1 - R_t, 6),
                        "h(t) Hazard Rate": round(h_t, 6),
                    })
                st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)
            except ValueError:
                st.warning("Invalid time format.")

        # CDF and reliability curves
        t_range = np.linspace(0.01, sorted_data.max() * 1.5, 200)
        cdf_vals = sp_stats.weibull_min.cdf(t_range, shape, scale=scale)
        rel_vals = 1 - cdf_vals
        pdf_vals = sp_stats.weibull_min.pdf(t_range, shape, scale=scale)

        fig2 = make_subplots(rows=1, cols=2,
                              subplot_titles=("CDF & Reliability", "PDF & Hazard"))
        fig2.add_trace(go.Scatter(x=t_range, y=cdf_vals, name="F(t) CDF",
                                   line=dict(color="red")), row=1, col=1)
        fig2.add_trace(go.Scatter(x=t_range, y=rel_vals, name="R(t) Reliability",
                                   line=dict(color="green")), row=1, col=1)
        fig2.add_trace(go.Scatter(x=t_range, y=pdf_vals, name="f(t) PDF",
                                   line=dict(color="blue")), row=1, col=2)
        hazard = pdf_vals / rel_vals
        hazard[rel_vals < 1e-10] = np.nan
        fig2.add_trace(go.Scatter(x=t_range, y=hazard, name="h(t) Hazard",
                                   line=dict(color="orange")), row=1, col=2)
        fig2.update_layout(height=400)
        rdl_plotly_chart(fig2)

    elif analysis == "Multi-Distribution Fitting":
        section_header("Multi-Distribution Fitting")
        col = st.selectbox("Failure time column:", num_cols, key="rel_multi_col")
        data = df[col].dropna().values
        data = data[data > 0]

        if len(data) < 5:
            st.error("Need at least 5 positive observations.")
            return

        distributions = [
            ("Weibull", sp_stats.weibull_min),
            ("Lognormal", sp_stats.lognorm),
            ("Exponential", sp_stats.expon),
            ("Normal", sp_stats.norm),
            ("Gamma", sp_stats.gamma),
        ]

        results = []
        fitted = {}
        for name, dist in distributions:
            try:
                params = dist.fit(data)
                log_lik = np.sum(dist.logpdf(data, *params))
                k = len(params)
                n = len(data)
                aic = 2 * k - 2 * log_lik
                bic = k * np.log(n) - 2 * log_lik
                # KS test
                ks_stat, ks_p = sp_stats.kstest(data, dist.cdf, args=params)

                results.append({
                    "Distribution": name,
                    "AIC": round(aic, 2),
                    "BIC": round(bic, 2),
                    "Log-Likelihood": round(log_lik, 2),
                    "KS Statistic": round(ks_stat, 4),
                    "KS p-value": round(ks_p, 6),
                })
                fitted[name] = (dist, params)
            except Exception:
                pass

        if not results:
            st.error("All distribution fits failed.")
            return

        results_df = pd.DataFrame(results).sort_values("AIC")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        best = results_df.iloc[0]["Distribution"]
        st.success(f"Best fit by AIC: **{best}**")

        # Overlay fitted CDFs
        section_header("Fitted CDFs")
        sorted_data = np.sort(data)
        empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        t_range = np.linspace(data.min() * 0.5, data.max() * 1.2, 200)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sorted_data, y=empirical_cdf, mode="markers",
                                 marker=dict(size=5, color="black"), name="Empirical"))

        colors = px.colors.qualitative.Set1
        for i, (name, (dist, params)) in enumerate(fitted.items()):
            cdf_vals = dist.cdf(t_range, *params)
            fig.add_trace(go.Scatter(x=t_range, y=cdf_vals, mode="lines",
                                     name=name, line=dict(color=colors[i % len(colors)])))

        fig.update_layout(title="Empirical vs Fitted CDFs",
                          xaxis_title="Time", yaxis_title="F(t)", height=500)
        rdl_plotly_chart(fig)

    elif analysis == "Degradation Analysis":
        section_header("Degradation Analysis")
        help_tip("Degradation Analysis", """
Track degradation measurements over time for each unit.
Fit degradation models and extrapolate to a failure threshold.
""")

        c1, c2, c3 = st.columns(3)
        time_col = c1.selectbox("Time column:", num_cols, key="rel_deg_time")
        meas_col = c2.selectbox("Measurement column:", [c for c in num_cols if c != time_col],
                                 key="rel_deg_meas")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        unit_col = c3.selectbox("Unit ID column:", cat_cols + num_cols, key="rel_deg_unit")

        threshold = st.number_input("Failure threshold:", value=0.0, key="rel_deg_threshold")
        model_type = st.selectbox("Degradation model:", ["Linear", "Power Law"], key="rel_deg_model")

        if st.button("Analyze Degradation", key="rel_deg_run"):
            data = df[[time_col, meas_col, unit_col]].dropna()
            units = data[unit_col].unique()

            if len(units) < 2:
                st.warning("Need at least 2 units.")
                return

            # Plot degradation paths
            fig = go.Figure()
            estimated_failures = []
            colors = px.colors.qualitative.Set1

            for i, unit in enumerate(units[:20]):  # Limit to 20 units
                unit_data = data[data[unit_col] == unit].sort_values(time_col)
                t = unit_data[time_col].values
                y = unit_data[meas_col].values

                fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers",
                                         name=str(unit),
                                         line=dict(color=colors[i % len(colors)]),
                                         marker=dict(size=4)))

                # Fit degradation model
                if len(t) >= 2:
                    if model_type == "Linear":
                        coeffs = np.polyfit(t, y, 1)
                        if coeffs[0] != 0:
                            t_fail = (threshold - coeffs[1]) / coeffs[0]
                            if t_fail > 0:
                                estimated_failures.append(t_fail)
                    elif model_type == "Power Law":
                        # log(y) = log(a) + b*log(t)
                        t_pos = t[t > 0]
                        y_pos = y[t > 0]
                        if len(t_pos) >= 2 and np.all(y_pos > 0):
                            coeffs = np.polyfit(np.log(t_pos), np.log(y_pos), 1)
                            b, log_a = coeffs
                            a = np.exp(log_a)
                            if b != 0 and threshold > 0:
                                t_fail = (threshold / a) ** (1 / b)
                                if t_fail > 0:
                                    estimated_failures.append(t_fail)

            fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                          annotation_text=f"Failure Threshold={threshold}")
            fig.update_layout(title="Degradation Paths",
                              xaxis_title=time_col, yaxis_title=meas_col,
                              height=500)
            rdl_plotly_chart(fig)

            if estimated_failures:
                section_header("Estimated Failure Times")
                fail_arr = np.array(estimated_failures)
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean Failure Time", f"{np.mean(fail_arr):.4f}")
                c2.metric("Median Failure Time", f"{np.median(fail_arr):.4f}")
                c3.metric("Std Dev", f"{np.std(fail_arr):.4f}")

                fig_fail = px.histogram(x=fail_arr, nbins=20,
                                        title="Estimated Failure Time Distribution",
                                        labels={"x": "Failure Time"})
                fig_fail.update_layout(height=350)
                rdl_plotly_chart(fig_fail)


# ---------------------------------------------------------------------------
# Tab 6: Competing Risks
# ---------------------------------------------------------------------------

def _render_competing_risks(df: pd.DataFrame):
    """Competing risks analysis using Aalen-Johansen cumulative incidence."""
    section_header("Competing Risks Analysis",
                   "Estimate cumulative incidence functions when multiple event types compete.")

    if not HAS_AJ:
        st.warning(
            "The `AalenJohansenFitter` is required for competing risks analysis. "
            "Please upgrade lifelines to version >= 0.27: `pip install lifelines>=0.27`"
        )
        return

    help_tip("Competing Risks", """
**Competing risks** arise when subjects can experience one of several mutually exclusive event types.
Standard Kaplan-Meier overestimates the probability of each event because it treats competing events as censored.

The **Aalen-Johansen estimator** correctly accounts for competing risks by estimating the
**cumulative incidence function (CIF)** for each event type. The CIF gives the probability of
experiencing a specific event type by time *t*, considering that other events may occur first.

- **Event column:** Must contain 0 for censored and integer codes (1, 2, 3, ...) for each event type.
- The sum of all CIFs at any time point equals the overall event probability.
""")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns (duration and event with multiple types).")
        return

    c1, c2 = st.columns(2)
    duration_col = c1.selectbox("Duration / time column:", num_cols, key="cr_dur")
    event_col = c2.selectbox("Event column (0=censored, 1,2,...=event types):", num_cols, key="cr_evt")

    # Validate event column has multiple event types
    data = df[[duration_col, event_col]].dropna().copy()
    if (data[duration_col] < 0).any():
        st.warning("Negative durations detected and removed.")
        data = data[data[duration_col] >= 0]

    if data.empty:
        st.error("No valid rows after dropping missing values.")
        return

    event_values = sorted(data[event_col].unique())
    event_types = [v for v in event_values if v != 0]

    if len(event_types) < 2:
        empty_state(
            "The event column must contain at least 2 distinct non-zero event types for competing risks.",
            "Use 0 for censored observations and integer codes (1, 2, 3, ...) for different event types."
        )
        return

    st.info(f"Detected event types: {event_types} (0 = censored, n = {len(data)})")

    # Optional grouping variable
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    group_col = st.selectbox("Group column (optional):", ["None"] + cat_cols, key="cr_group")
    group_col = None if group_col == "None" else group_col

    # Time points for table
    timepoints_input = st.text_input(
        "CIF at timepoints (comma-separated, optional):",
        placeholder="e.g. 12, 24, 36, 60",
        key="cr_timepoints",
    )

    if st.button("Estimate Cumulative Incidence", key="cr_fit"):
        try:
            colors = px.colors.qualitative.Set1

            if group_col is None:
                # Single-group analysis
                fig = go.Figure()
                cif_records = []

                for idx, event_of_interest in enumerate(event_types):
                    aj = AalenJohansenFitter(calculate_variance=True)
                    aj.fit(
                        data[duration_col],
                        data[event_col],
                        event_of_interest=event_of_interest,
                    )

                    cif = aj.cumulative_density_
                    timeline = cif.index.values
                    cif_vals = cif.iloc[:, 0].values
                    color = colors[idx % len(colors)]

                    fig.add_trace(go.Scatter(
                        x=timeline, y=cif_vals,
                        mode="lines", name=f"Event {int(event_of_interest)}",
                        line=dict(color=color, width=2, shape="hv"),
                    ))

                    # Build CIF records at time points
                    if timepoints_input.strip():
                        try:
                            tps = [float(t.strip()) for t in timepoints_input.split(",") if t.strip()]
                            for tp in tps:
                                # Find closest time point
                                valid_times = timeline[timeline <= tp]
                                if len(valid_times) > 0:
                                    closest_idx = np.argmin(np.abs(timeline - tp))
                                    cif_val = cif_vals[closest_idx]
                                else:
                                    cif_val = 0.0
                                cif_records.append({
                                    "Event Type": int(event_of_interest),
                                    "Time": tp,
                                    "CIF": f"{cif_val:.4f}",
                                })
                        except ValueError:
                            st.warning("Invalid timepoint format. Use comma-separated numbers.")

                fig.update_layout(
                    title="Cumulative Incidence Functions (Aalen-Johansen)",
                    xaxis_title="Time",
                    yaxis_title="Cumulative Incidence",
                    yaxis=dict(range=[0, 1.05]),
                    height=550,
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                )
                rdl_plotly_chart(fig)

                # Summary metrics
                section_header("Event Summary")
                summary_rows = []
                for event_type in event_types:
                    n_events = int((data[event_col] == event_type).sum())
                    pct = n_events / len(data) * 100
                    summary_rows.append({
                        "Event Type": int(event_type),
                        "Count": n_events,
                        "Percentage": f"{pct:.1f}%",
                    })
                n_censored = int((data[event_col] == 0).sum())
                summary_rows.append({
                    "Event Type": "Censored (0)",
                    "Count": n_censored,
                    "Percentage": f"{n_censored / len(data) * 100:.1f}%",
                })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

                # CIF at timepoints table
                if cif_records:
                    section_header("Cumulative Incidence at Specified Time Points")
                    st.dataframe(pd.DataFrame(cif_records), use_container_width=True, hide_index=True)

            else:
                # Grouped analysis
                full = df[[duration_col, event_col, group_col]].dropna()
                full = full[full[duration_col] >= 0]
                groups = sorted(full[group_col].unique())

                if len(groups) > 10:
                    st.warning("Too many groups (>10). Showing first 10.")
                    groups = groups[:10]

                for event_of_interest in event_types:
                    section_header(f"Event Type {int(event_of_interest)}")
                    fig = go.Figure()

                    for g_idx, grp in enumerate(groups):
                        mask = full[group_col] == grp
                        grp_data = full.loc[mask]

                        if len(grp_data) < 2:
                            continue

                        aj = AalenJohansenFitter(calculate_variance=True)
                        aj.fit(
                            grp_data[duration_col],
                            grp_data[event_col],
                            event_of_interest=event_of_interest,
                        )

                        cif = aj.cumulative_density_
                        timeline = cif.index.values
                        cif_vals = cif.iloc[:, 0].values
                        color = colors[g_idx % len(colors)]

                        fig.add_trace(go.Scatter(
                            x=timeline, y=cif_vals,
                            mode="lines", name=str(grp),
                            line=dict(color=color, width=2, shape="hv"),
                        ))

                    fig.update_layout(
                        title=f"Cumulative Incidence - Event {int(event_of_interest)} by {group_col}",
                        xaxis_title="Time",
                        yaxis_title="Cumulative Incidence",
                        yaxis=dict(range=[0, 1.05]),
                        height=500,
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                    )
                    rdl_plotly_chart(fig)

                # Event summary by group
                section_header("Event Summary by Group")
                summary_rows = []
                for grp in groups:
                    mask = full[group_col] == grp
                    grp_data = full.loc[mask]
                    for event_type in event_types:
                        n_ev = int((grp_data[event_col] == event_type).sum())
                        summary_rows.append({
                            "Group": str(grp),
                            "Event Type": int(event_type),
                            "Count": n_ev,
                            "N": len(grp_data),
                            "Percentage": f"{n_ev / len(grp_data) * 100:.1f}%" if len(grp_data) > 0 else "0.0%",
                        })
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Competing risks error: {e}")
