"""
ICH Q2 Method Validation Module - Linearity, Accuracy, Precision, LOD/LOQ,
and System Suitability for analytical method validation.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import plotly.graph_objects as go
from modules.ui_helpers import (
    section_header, empty_state, help_tip, validation_panel,
    interpretation_card, significance_result, rdl_plotly_chart,
)
from modules.validation import check_normality, ValidationCheck, Interpretation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return list of numeric column names."""
    return list(df.select_dtypes(include="number").columns)


def _pct_rsd(values) -> float:
    """Percentage relative standard deviation (coefficient of variation)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return np.nan
    mean = np.mean(arr)
    if mean == 0:
        return np.nan
    return (np.std(arr, ddof=1) / abs(mean)) * 100.0


def _pass_fail_badge(passed: bool) -> str:
    """Return an HTML badge for pass/fail status."""
    if passed:
        return '<span style="color:#16a34a;font-weight:600;">PASS</span>'
    return '<span style="color:#dc2626;font-weight:600;">FAIL</span>'


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_method_validation(df: pd.DataFrame):
    """Render ICH Q2 analytical method validation interface."""
    if df is None or df.empty:
        empty_state(
            "No data loaded.",
            "Upload a dataset from the sidebar to begin method validation.",
        )
        return

    num_cols = _numeric_columns(df)
    all_cols = list(df.columns)

    if len(num_cols) < 2:
        empty_state(
            "At least two numeric columns are required.",
            "Your dataset needs concentration/response columns for validation analyses.",
        )
        return

    help_tip(
        "ICH Q2 Method Validation",
        "This module supports the key validation characteristics defined in "
        "ICH Q2(R1): Linearity, Accuracy, Precision, Detection/Quantitation "
        "Limits, and System Suitability. Select data columns and configure "
        "acceptance criteria for each tab.",
    )

    tabs = st.tabs([
        "Linearity",
        "Accuracy (% Recovery)",
        "Precision",
        "LOD / LOQ",
        "System Suitability",
    ])

    with tabs[0]:
        _render_linearity(df, num_cols)

    with tabs[1]:
        _render_accuracy(df, num_cols, all_cols)

    with tabs[2]:
        _render_precision(df, num_cols, all_cols)

    with tabs[3]:
        _render_lod_loq(df, num_cols)

    with tabs[4]:
        _render_system_suitability(df, num_cols, all_cols)


# ---------------------------------------------------------------------------
# Tab 1 — Linearity
# ---------------------------------------------------------------------------

def _render_linearity(df: pd.DataFrame, num_cols: list[str]):
    section_header(
        "Linearity",
        "Demonstrate a proportional relationship between analyte concentration "
        "and instrument response across the specified range.",
    )

    c1, c2 = st.columns(2)
    with c1:
        conc_col = st.selectbox(
            "Concentration column", num_cols, key="mv_lin_conc",
        )
    with c2:
        resp_col = st.selectbox(
            "Response column", num_cols,
            index=min(1, len(num_cols) - 1), key="mv_lin_resp",
        )

    r_crit = st.number_input(
        "R criterion (minimum acceptable R)",
        min_value=0.900, max_value=1.000, value=0.999,
        step=0.001, format="%.4f", key="mv_lin_r_crit",
    )

    subset = df[[conc_col, resp_col]].dropna()
    if len(subset) < 3:
        empty_state(
            "Not enough data points for regression.",
            "At least 3 non-missing paired observations are required.",
        )
        return

    x = subset[conc_col].values.astype(float)
    y = subset[resp_col].values.astype(float)

    with st.spinner("Fitting linear model..."):
        X_const = sm.add_constant(x)
        model = sm.OLS(y, X_const).fit()

    intercept, slope = model.params[0], model.params[1]
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    r_value = np.sqrt(r_squared) if r_squared >= 0 else 0.0
    se_slope = model.bse[1]
    intercept_p = model.pvalues[0]
    residuals = model.resid
    predicted = model.fittedvalues

    # --- Results metrics ---
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R", f"{r_value:.6f}")
    m2.metric("R\u00b2", f"{r_squared:.6f}")
    m3.metric("Adj R\u00b2", f"{adj_r_squared:.6f}")
    m4.metric("SE of Slope", f"{se_slope:.4e}")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Slope", f"{slope:.4e}")
    m6.metric("Intercept", f"{intercept:.4e}")
    m7.metric("Intercept p-value", f"{intercept_p:.4f}")
    passed = r_value >= r_crit
    m8.markdown(
        f"**R criterion:** {_pass_fail_badge(passed)}",
        unsafe_allow_html=True,
    )

    # --- Intercept significance ---
    significance_result(
        intercept_p, 0.05, "Intercept t-test (H\u2080: intercept = 0)",
        effect_size=intercept, effect_label="Intercept",
    )

    # --- Calibration curve ---
    section_header("Calibration Curve")
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = intercept + slope * x_line
    eq_text = f"y = {slope:.4e}x + {intercept:.4e}   (R\u00b2 = {r_squared:.6f})"

    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Data",
        marker=dict(size=8, color="#6366f1"),
    ))
    fig_cal.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name="Regression",
        line=dict(color="#ef4444", width=2),
    ))
    fig_cal.add_annotation(
        x=0.05, y=0.95, xref="paper", yref="paper",
        text=eq_text, showarrow=False, font=dict(size=12),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="#6366f1", borderwidth=1,
    )
    fig_cal.update_layout(
        template="plotly+rdl",
        xaxis_title=conc_col, yaxis_title=resp_col,
        title="Calibration Curve",
    )
    rdl_plotly_chart(fig_cal)

    # --- Residual plot ---
    section_header("Residual Analysis")
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=predicted, y=residuals, mode="markers",
        marker=dict(size=7, color="#6366f1"),
        name="Residuals",
    ))
    fig_res.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig_res.update_layout(
        template="plotly+rdl",
        xaxis_title="Predicted", yaxis_title="Residual",
        title="Residuals vs. Predicted",
    )
    rdl_plotly_chart(fig_res)

    # --- Validation panel for residual checks ---
    checks = []
    checks.append(check_normality(residuals, label="residuals"))

    # Homoscedasticity visual note via Breusch-Pagan
    try:
        from statsmodels.stats.diagnostic import het_breuschpagan
        _, bp_p, _, _ = het_breuschpagan(residuals, X_const)
        if bp_p >= 0.05:
            checks.append(ValidationCheck(
                name="Homoscedasticity (Breusch-Pagan)",
                status="pass",
                detail=f"BP p = {bp_p:.4f} (constant variance assumption met)",
            ))
        else:
            checks.append(ValidationCheck(
                name="Homoscedasticity (Breusch-Pagan)",
                status="warn" if bp_p >= 0.01 else "fail",
                detail=f"BP p = {bp_p:.4f} (heteroscedasticity detected)",
                suggestion="Weighted regression or transformation may improve linearity.",
            ))
    except Exception:
        checks.append(ValidationCheck(
            name="Homoscedasticity",
            status="warn",
            detail="Could not run Breusch-Pagan test.",
        ))

    # R criterion check
    checks.append(ValidationCheck(
        name=f"R criterion (\u2265 {r_crit})",
        status="pass" if passed else "fail",
        detail=f"R = {r_value:.6f} {'meets' if passed else 'does not meet'} the criterion",
        suggestion="" if passed else "Review data range or investigate non-linearity.",
    ))

    validation_panel(checks, title="Linearity Checks")


# ---------------------------------------------------------------------------
# Tab 2 — Accuracy (% Recovery)
# ---------------------------------------------------------------------------

def _render_accuracy(df: pd.DataFrame, num_cols: list[str], all_cols: list[str]):
    section_header(
        "Accuracy (% Recovery)",
        "Closeness of test results to the true value, expressed as percent recovery.",
    )

    c1, c2 = st.columns(2)
    with c1:
        known_col = st.selectbox(
            "Known / expected concentration", num_cols, key="mv_acc_known",
        )
    with c2:
        found_col = st.selectbox(
            "Measured / found concentration", num_cols,
            index=min(1, len(num_cols) - 1), key="mv_acc_found",
        )

    group_col = st.selectbox(
        "Grouping column (optional, for concentration levels)",
        ["None"] + all_cols, key="mv_acc_group",
    )
    use_group = group_col != "None"

    acc_range = st.slider(
        "Acceptable recovery range (%)",
        min_value=80.0, max_value=120.0, value=(98.0, 102.0),
        step=0.5, key="mv_acc_range",
    )
    acc_lo, acc_hi = acc_range

    subset = df[[known_col, found_col]].copy()
    if use_group:
        subset[group_col] = df[group_col]
    subset = subset.dropna(subset=[known_col, found_col])

    if len(subset) < 1:
        empty_state(
            "No valid data for accuracy calculation.",
            "Ensure known and found columns have non-missing numeric values.",
        )
        return

    # Guard against division by zero
    mask = subset[known_col] != 0
    if mask.sum() == 0:
        empty_state(
            "All known concentration values are zero.",
            "Cannot compute percent recovery when expected values are zero.",
        )
        return
    subset = subset[mask].copy()
    subset["% Recovery"] = (subset[found_col] / subset[known_col]) * 100.0

    # --- Grouped summary ---
    if use_group:
        groups = subset.groupby(group_col)
    else:
        subset["_all"] = "Overall"
        groups = subset.groupby("_all")

    summary_rows = []
    for name, grp in groups:
        rec = grp["% Recovery"].values
        n = len(rec)
        mean_rec = np.mean(rec)
        rsd = _pct_rsd(rec)
        if n >= 2:
            ci = stats.t.interval(0.95, df=n - 1, loc=mean_rec,
                                  scale=stats.sem(rec))
        else:
            ci = (np.nan, np.nan)
        passed = acc_lo <= mean_rec <= acc_hi
        summary_rows.append({
            "Level": name,
            "n": n,
            "Mean Recovery (%)": round(mean_rec, 2),
            "%RSD": round(rsd, 2) if np.isfinite(rsd) else "N/A",
            "95% CI Lower": round(ci[0], 2) if np.isfinite(ci[0]) else "N/A",
            "95% CI Upper": round(ci[1], 2) if np.isfinite(ci[1]) else "N/A",
            "Status": "PASS" if passed else "FAIL",
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # --- Pass/fail cards per level ---
    for row in summary_rows:
        passed = row["Status"] == "PASS"
        st.markdown(
            f"**{row['Level']}:** Mean Recovery = {row['Mean Recovery (%)']}% "
            f"&mdash; {_pass_fail_badge(passed)}",
            unsafe_allow_html=True,
        )

    # --- Bar chart ---
    section_header("Recovery by Level")
    chart_df = pd.DataFrame(summary_rows)
    chart_df["Mean Recovery (%)"] = pd.to_numeric(
        chart_df["Mean Recovery (%)"], errors="coerce",
    )
    # Compute error (half-width of 95% CI) for error bars
    error_vals = []
    for row in summary_rows:
        ci_lo = row["95% CI Lower"]
        ci_hi = row["95% CI Upper"]
        mean_val = row["Mean Recovery (%)"]
        if isinstance(ci_hi, (int, float)) and isinstance(ci_lo, (int, float)):
            error_vals.append((ci_hi - ci_lo) / 2.0)
        else:
            error_vals.append(0)
    chart_df["Error"] = error_vals

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Bar(
        x=chart_df["Level"].astype(str),
        y=chart_df["Mean Recovery (%)"],
        error_y=dict(type="data", array=chart_df["Error"], visible=True),
        marker_color="#6366f1", name="Mean Recovery",
    ))
    fig_acc.add_hline(y=acc_lo, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Lower limit ({acc_lo}%)")
    fig_acc.add_hline(y=acc_hi, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Upper limit ({acc_hi}%)")
    fig_acc.add_hline(y=100, line_dash="dot", line_color="#94a3b8")
    fig_acc.update_layout(
        template="plotly+rdl",
        xaxis_title="Concentration Level",
        yaxis_title="Mean % Recovery",
        title="Accuracy: Mean Recovery by Level",
    )
    rdl_plotly_chart(fig_acc)


# ---------------------------------------------------------------------------
# Tab 3 — Precision
# ---------------------------------------------------------------------------

def _render_precision(df: pd.DataFrame, num_cols: list[str], all_cols: list[str]):
    section_header(
        "Precision",
        "Closeness of agreement between a series of measurements from "
        "multiple sampling of the same homogeneous sample.",
    )

    rsd_limit = st.number_input(
        "Maximum acceptable %RSD", min_value=0.1, max_value=20.0,
        value=2.0, step=0.1, format="%.1f", key="mv_prec_rsd_limit",
    )

    # --- Repeatability ---
    section_header(
        "Repeatability (Within-Analyst)",
        "Precision under the same operating conditions over a short interval.",
    )

    c1, c2 = st.columns(2)
    with c1:
        val_col = st.selectbox(
            "Value column", num_cols, key="mv_prec_val",
        )
    with c2:
        rep_col = st.selectbox(
            "Replicate/Group column (optional)",
            ["None"] + all_cols, key="mv_prec_rep",
        )

    vals = df[val_col].dropna().values.astype(float)
    if len(vals) < 2:
        empty_state("Need at least 2 values for repeatability.", "")
        return

    overall_rsd = _pct_rsd(vals)
    overall_mean = np.mean(vals)
    overall_sd = np.std(vals, ddof=1)

    m1, m2, m3 = st.columns(3)
    m1.metric("Mean", f"{overall_mean:.4f}")
    m2.metric("SD", f"{overall_sd:.4f}")
    m3.metric("%RSD (Repeatability)", f"{overall_rsd:.2f}%")

    rep_pass = overall_rsd <= rsd_limit if np.isfinite(overall_rsd) else False
    st.markdown(
        f"**Repeatability:** %RSD = {overall_rsd:.2f}% &mdash; "
        f"{_pass_fail_badge(rep_pass)}",
        unsafe_allow_html=True,
    )

    # --- Intermediate Precision ---
    section_header(
        "Intermediate Precision",
        "Within-laboratory variation: different days, analysts, equipment.",
    )

    c1, c2 = st.columns(2)
    with c1:
        analyst_col = st.selectbox(
            "Analyst column (optional)",
            ["None"] + all_cols, key="mv_prec_analyst",
        )
    with c2:
        day_col = st.selectbox(
            "Day column (optional)",
            ["None"] + all_cols, key="mv_prec_day",
        )

    variance_rows = []
    variance_rows.append({
        "Component": "Repeatability",
        "\u03c3": round(overall_sd, 6),
        "\u03c3\u00b2": round(overall_sd ** 2, 6),
        "%RSD": round(overall_rsd, 2) if np.isfinite(overall_rsd) else "N/A",
    })

    # Between-analyst variance
    if analyst_col != "None":
        _compute_between_variance(
            df, val_col, analyst_col, "Between-Analyst",
            variance_rows, overall_mean,
        )

    # Between-day variance
    if day_col != "None":
        _compute_between_variance(
            df, val_col, day_col, "Between-Day",
            variance_rows, overall_mean,
        )

    # Total variance
    total_var = sum(
        r["\u03c3\u00b2"] for r in variance_rows
        if isinstance(r["\u03c3\u00b2"], (int, float))
    )
    total_sd = np.sqrt(total_var) if total_var > 0 else 0.0
    total_rsd = (total_sd / abs(overall_mean) * 100.0) if overall_mean != 0 else np.nan
    variance_rows.append({
        "Component": "Total (Intermediate Precision)",
        "\u03c3": round(total_sd, 6),
        "\u03c3\u00b2": round(total_var, 6),
        "%RSD": round(total_rsd, 2) if np.isfinite(total_rsd) else "N/A",
    })

    var_df = pd.DataFrame(variance_rows)
    st.dataframe(var_df, use_container_width=True, hide_index=True)

    # Interpretation
    total_pass = total_rsd <= rsd_limit if np.isfinite(total_rsd) else False
    interpretation_card(Interpretation(
        title="Precision Assessment",
        body=(
            f"Intermediate precision %RSD = {total_rsd:.2f}% "
            f"{'meets' if total_pass else 'does not meet'} "
            f"the acceptance criterion of \u2264 {rsd_limit}%."
        ),
        detail=(
            "Variance components are estimated using one-way ANOVA decomposition "
            "(SS_between and SS_within). The total intermediate precision variance "
            "is the sum of all components."
        ),
    ))


def _compute_between_variance(
    df: pd.DataFrame, val_col: str, group_col: str,
    label: str, variance_rows: list, grand_mean: float,
):
    """Compute between-group variance via one-way ANOVA decomposition."""
    subset = df[[val_col, group_col]].dropna()
    if len(subset) < 2:
        return

    groups = [g[val_col].values.astype(float) for _, g in subset.groupby(group_col)]
    groups = [g for g in groups if len(g) >= 1]

    if len(groups) < 2:
        return

    n_total = sum(len(g) for g in groups)
    k = len(groups)

    # SS_between
    group_means = [np.mean(g) for g in groups]
    group_sizes = [len(g) for g in groups]
    ss_between = sum(ni * (mi - grand_mean) ** 2 for ni, mi in zip(group_sizes, group_means))

    # SS_within
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    # Mean squares
    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0

    # Between variance component: (MS_between - MS_within) / n0
    # where n0 is the average group size for unbalanced designs
    n0 = (n_total - sum(ni ** 2 for ni in group_sizes) / n_total) / (k - 1) if k > 1 else 1.0
    var_between = max(0, (ms_between - ms_within) / n0)
    sd_between = np.sqrt(var_between)
    rsd_between = (sd_between / abs(grand_mean) * 100.0) if grand_mean != 0 else np.nan

    variance_rows.append({
        "Component": label,
        "\u03c3": round(sd_between, 6),
        "\u03c3\u00b2": round(var_between, 6),
        "%RSD": round(rsd_between, 2) if np.isfinite(rsd_between) else "N/A",
    })


# ---------------------------------------------------------------------------
# Tab 4 — LOD / LOQ
# ---------------------------------------------------------------------------

def _render_lod_loq(df: pd.DataFrame, num_cols: list[str]):
    section_header(
        "LOD / LOQ",
        "Limit of Detection and Limit of Quantitation per ICH Q2.",
    )

    method = st.radio(
        "Calculation method",
        ["Based on calibration residual SD", "Based on S/N ratio"],
        key="mv_lod_method",
    )

    if method == "Based on S/N ratio":
        _render_lod_sn()
    else:
        _render_lod_calibration(df, num_cols)


def _render_lod_sn():
    """LOD/LOQ from signal-to-noise ratio inputs."""
    section_header("Signal-to-Noise Method")

    st.info(
        "Enter the measured signal-to-noise ratios. "
        "LOD is typically the concentration with S/N \u2248 3:1, "
        "LOQ is the concentration with S/N \u2248 10:1."
    )

    c1, c2 = st.columns(2)
    with c1:
        lod_conc = st.number_input(
            "LOD concentration (S/N \u2248 3)", min_value=0.0,
            value=0.0, format="%.4f", key="mv_lod_sn_lod",
        )
        lod_sn = st.number_input(
            "Measured S/N at LOD", min_value=0.0, value=3.0,
            format="%.1f", key="mv_lod_sn_lod_val",
        )
    with c2:
        loq_conc = st.number_input(
            "LOQ concentration (S/N \u2248 10)", min_value=0.0,
            value=0.0, format="%.4f", key="mv_lod_sn_loq",
        )
        loq_sn = st.number_input(
            "Measured S/N at LOQ", min_value=0.0, value=10.0,
            format="%.1f", key="mv_lod_sn_loq_val",
        )

    m1, m2 = st.columns(2)
    m1.metric("LOD", f"{lod_conc:.4f}" if lod_conc > 0 else "Not set")
    m2.metric("LOQ", f"{loq_conc:.4f}" if loq_conc > 0 else "Not set")

    interpretation_card(Interpretation(
        title="Signal-to-Noise Method",
        body=(
            f"LOD is determined at a signal-to-noise ratio of {lod_sn:.0f}:1 "
            f"and LOQ at {loq_sn:.0f}:1. These are visual or instrumental "
            f"estimates based on comparing measured signals to baseline noise."
        ),
        detail=(
            "ICH Q2 recommends S/N of 3:1 for LOD and 10:1 for LOQ as "
            "a practical approach when standard deviation of blanks is "
            "not readily available."
        ),
    ))


def _render_lod_calibration(df: pd.DataFrame, num_cols: list[str]):
    """LOD/LOQ from calibration curve residual standard deviation."""
    section_header("Calibration Residual SD Method")

    c1, c2 = st.columns(2)
    with c1:
        conc_col = st.selectbox(
            "Concentration column", num_cols, key="mv_lod_conc",
        )
    with c2:
        resp_col = st.selectbox(
            "Response column", num_cols,
            index=min(1, len(num_cols) - 1), key="mv_lod_resp",
        )

    subset = df[[conc_col, resp_col]].dropna()
    if len(subset) < 3:
        empty_state(
            "At least 3 data points required for calibration.",
            "Provide concentration-response pairs for regression.",
        )
        return

    x = subset[conc_col].values.astype(float)
    y = subset[resp_col].values.astype(float)

    with st.spinner("Computing regression for LOD/LOQ..."):
        X_const = sm.add_constant(x)
        model = sm.OLS(y, X_const).fit()

    slope = model.params[1]
    intercept = model.params[0]
    residual_se = np.sqrt(model.mse_resid)

    if abs(slope) < 1e-15:
        st.warning("Slope is effectively zero. Cannot compute LOD/LOQ.")
        return

    lod = 3.3 * residual_se / abs(slope)
    loq = 10.0 * residual_se / abs(slope)

    m1, m2, m3 = st.columns(3)
    m1.metric("Residual SE (\u03c3)", f"{residual_se:.4e}")
    m2.metric("LOD (3.3\u03c3/S)", f"{lod:.4f}")
    m3.metric("LOQ (10\u03c3/S)", f"{loq:.4f}")

    interpretation_card(Interpretation(
        title="LOD / LOQ from Calibration Residuals",
        body=(
            f"LOD = 3.3 \u00d7 {residual_se:.4e} / {abs(slope):.4e} = {lod:.4f}. "
            f"LOQ = 10 \u00d7 {residual_se:.4e} / {abs(slope):.4e} = {loq:.4f}."
        ),
        detail=(
            "The residual standard error of the calibration regression serves "
            "as an estimate of \u03c3. ICH Q2 defines LOD = 3.3\u03c3/S and "
            "LOQ = 10\u03c3/S, where S is the slope of the calibration curve."
        ),
    ))

    # --- Visual: LOD/LOQ on calibration curve ---
    section_header("LOD / LOQ on Calibration Curve")

    x_plot = np.linspace(0, max(x.max(), loq * 1.3), 200)
    y_plot = intercept + slope * x_plot

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", name="Calibration Data",
        marker=dict(size=8, color="#6366f1"),
    ))
    fig.add_trace(go.Scatter(
        x=x_plot, y=y_plot, mode="lines", name="Regression Line",
        line=dict(color="#94a3b8", width=2),
    ))

    # LOD line
    lod_response = intercept + slope * lod
    fig.add_trace(go.Scatter(
        x=[lod, lod], y=[0, lod_response], mode="lines",
        line=dict(color="#f59e0b", width=2, dash="dash"), name=f"LOD = {lod:.4f}",
    ))
    fig.add_trace(go.Scatter(
        x=[lod], y=[lod_response], mode="markers",
        marker=dict(size=12, color="#f59e0b", symbol="diamond"),
        showlegend=False,
    ))

    # LOQ line
    loq_response = intercept + slope * loq
    fig.add_trace(go.Scatter(
        x=[loq, loq], y=[0, loq_response], mode="lines",
        line=dict(color="#ef4444", width=2, dash="dash"), name=f"LOQ = {loq:.4f}",
    ))
    fig.add_trace(go.Scatter(
        x=[loq], y=[loq_response], mode="markers",
        marker=dict(size=12, color="#ef4444", symbol="diamond"),
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly+rdl",
        xaxis_title=conc_col, yaxis_title=resp_col,
        title="Calibration Curve with LOD / LOQ",
    )
    rdl_plotly_chart(fig)


# ---------------------------------------------------------------------------
# Tab 5 — System Suitability
# ---------------------------------------------------------------------------

def _render_system_suitability(
    df: pd.DataFrame, num_cols: list[str], all_cols: list[str],
):
    section_header(
        "System Suitability",
        "Verify that the analytical system is performing adequately before "
        "or during an analysis.",
    )

    c1, c2 = st.columns(2)
    with c1:
        inj_col = st.selectbox(
            "Injection / Sequence column", all_cols, key="mv_sst_inj",
        )
    with c2:
        area_col = st.selectbox(
            "Peak area column", num_cols, key="mv_sst_area",
        )

    c3, c4 = st.columns(2)
    with c3:
        rt_col = st.selectbox(
            "Retention time column (optional)",
            ["None"] + num_cols, key="mv_sst_rt",
        )
    with c4:
        tf_col = st.selectbox(
            "Tailing factor column (optional)",
            ["None"] + num_cols, key="mv_sst_tf",
        )

    tp_col = st.selectbox(
        "Theoretical plates column (optional)",
        ["None"] + num_cols, key="mv_sst_tp",
    )

    plates_limit = st.number_input(
        "Minimum theoretical plates", min_value=100, max_value=100000,
        value=2000, step=100, key="mv_sst_plates_limit",
    )

    # Collect available parameters
    params = {}
    subset = df[[inj_col]].copy()

    # Peak area is always required
    subset["Peak Area"] = pd.to_numeric(df[area_col], errors="coerce")
    params["Peak Area"] = {
        "values": subset["Peak Area"].dropna().values,
        "limit_type": "rsd",
        "limit": 1.0,
        "unit": "%RSD",
    }

    if rt_col != "None":
        subset["Retention Time"] = pd.to_numeric(df[rt_col], errors="coerce")
        params["Retention Time"] = {
            "values": subset["Retention Time"].dropna().values,
            "limit_type": "rsd",
            "limit": 1.0,
            "unit": "%RSD",
        }

    if tf_col != "None":
        subset["Tailing Factor"] = pd.to_numeric(df[tf_col], errors="coerce")
        params["Tailing Factor"] = {
            "values": subset["Tailing Factor"].dropna().values,
            "limit_type": "max",
            "limit": 2.0,
            "unit": "max",
        }

    if tp_col != "None":
        subset["Theoretical Plates"] = pd.to_numeric(df[tp_col], errors="coerce")
        params["Theoretical Plates"] = {
            "values": subset["Theoretical Plates"].dropna().values,
            "limit_type": "min",
            "limit": float(plates_limit),
            "unit": "min",
        }

    # --- Summary table ---
    section_header("SST Summary")
    summary_rows = []
    for pname, pdata in params.items():
        v = pdata["values"]
        if len(v) < 1:
            continue
        mean_v = np.mean(v)
        sd_v = np.std(v, ddof=1) if len(v) >= 2 else 0.0
        rsd_v = _pct_rsd(v)
        min_v = np.min(v)
        max_v = np.max(v)

        if pdata["limit_type"] == "rsd":
            test_val = rsd_v
            passed = test_val <= pdata["limit"] if np.isfinite(test_val) else False
            criterion = f"%RSD \u2264 {pdata['limit']:.1f}%"
            result_str = f"{test_val:.3f}%" if np.isfinite(test_val) else "N/A"
        elif pdata["limit_type"] == "max":
            test_val = max_v
            passed = test_val <= pdata["limit"]
            criterion = f"Max \u2264 {pdata['limit']:.1f}"
            result_str = f"{test_val:.3f}"
        else:  # min
            test_val = min_v
            passed = test_val >= pdata["limit"]
            criterion = f"Min \u2265 {pdata['limit']:.0f}"
            result_str = f"{test_val:.0f}"

        summary_rows.append({
            "Parameter": pname,
            "Mean": round(mean_v, 4),
            "SD": round(sd_v, 4),
            "%RSD": round(rsd_v, 3) if np.isfinite(rsd_v) else "N/A",
            "Min": round(min_v, 4),
            "Max": round(max_v, 4),
            "Criterion": criterion,
            "Result": result_str,
            "Status": "PASS" if passed else "FAIL",
        })

    if not summary_rows:
        empty_state("No valid data for system suitability.", "")
        return

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    all_pass = all(r["Status"] == "PASS" for r in summary_rows)
    st.markdown(
        f"**Overall SST Status:** {_pass_fail_badge(all_pass)}",
        unsafe_allow_html=True,
    )

    # --- Trending / control charts ---
    section_header("Trending Charts")

    injection_idx = np.arange(1, len(subset) + 1)

    for pname, pdata in params.items():
        col_name = pname
        if col_name not in subset.columns:
            continue
        plot_data = subset[[inj_col, col_name]].dropna()
        if len(plot_data) < 2:
            continue

        x_vals = np.arange(1, len(plot_data) + 1)
        y_vals = plot_data[col_name].values

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines+markers",
            marker=dict(size=7, color="#6366f1"),
            line=dict(color="#6366f1", width=2),
            name=pname,
        ))

        # Limit lines
        mean_val = np.mean(y_vals)
        fig.add_hline(y=mean_val, line_dash="dot", line_color="#94a3b8",
                      annotation_text=f"Mean = {mean_val:.3f}")

        if pdata["limit_type"] == "max":
            fig.add_hline(
                y=pdata["limit"], line_dash="dash", line_color="#ef4444",
                annotation_text=f"Limit = {pdata['limit']:.1f}",
            )
        elif pdata["limit_type"] == "min":
            fig.add_hline(
                y=pdata["limit"], line_dash="dash", line_color="#ef4444",
                annotation_text=f"Limit = {pdata['limit']:.0f}",
            )
        elif pdata["limit_type"] == "rsd":
            # Show +/- 2 SD as warning zone around the mean
            sd_val = np.std(y_vals, ddof=1) if len(y_vals) >= 2 else 0
            fig.add_hline(
                y=mean_val + 2 * sd_val, line_dash="dash",
                line_color="#f59e0b", annotation_text="+2 SD",
            )
            fig.add_hline(
                y=mean_val - 2 * sd_val, line_dash="dash",
                line_color="#f59e0b", annotation_text="-2 SD",
            )

        fig.update_layout(
            template="plotly+rdl",
            xaxis_title="Injection #",
            yaxis_title=pname,
            title=f"{pname} Trending",
            height=350,
        )
        rdl_plotly_chart(fig)
