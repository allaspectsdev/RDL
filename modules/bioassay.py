"""
Bioassay / Relative Potency Module - 4PL/5PL curve fitting, parallel-line
analysis, relative potency estimation, and dilution linearity.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import (
    section_header, empty_state, help_tip, validation_panel,
    interpretation_card, significance_result,
)
from modules.validation import ValidationCheck, Interpretation


# ---------------------------------------------------------------------------
# Logistic models
# ---------------------------------------------------------------------------

def _logistic_4pl(x, a, b, c, d):
    """4-Parameter Logistic: y = d + (a - d) / (1 + (x / c) ** b)"""
    return d + (a - d) / (1.0 + (x / c) ** b)


def _logistic_5pl(x, a, b, c, d, e):
    """5-Parameter Logistic: y = d + (a - d) / (1 + (x / c) ** b) ** e"""
    return d + (a - d) / (1.0 + (x / c) ** b) ** e


def _r_squared(y_obs, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_obs - y_pred) ** 2)
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ---------------------------------------------------------------------------
# Shared column selectors
# ---------------------------------------------------------------------------

def _column_selectors(df, key_prefix):
    """Render dose / response / sample column selectors. Returns (dose, resp, group)."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    all_cols = df.columns.tolist()
    if len(num_cols) < 2:
        empty_state(
            "At least two numeric columns are required.",
            "Ensure your dataset contains dose/concentration and response columns.",
        )
        return None, None, None
    c1, c2, c3 = st.columns(3)
    with c1:
        dose_col = st.selectbox("Dose / Concentration", num_cols, key=f"{key_prefix}_dose")
    with c2:
        resp_idx = min(1, len(num_cols) - 1)
        resp_col = st.selectbox("Response", num_cols, index=resp_idx, key=f"{key_prefix}_resp")
    with c3:
        cat_cols = [c for c in all_cols if c not in num_cols or df[c].nunique() <= 20]
        group_col = st.selectbox(
            "Sample / Group (optional)", [None] + cat_cols, key=f"{key_prefix}_group"
        )
    return dose_col, resp_col, group_col


# ---------------------------------------------------------------------------
# Curve fitting helpers
# ---------------------------------------------------------------------------

def _fit_4pl(x, y):
    """Fit a 4PL model. Returns (popt, pcov, y_pred, r2) or raises."""
    a0 = float(np.nanmin(y))
    d0 = float(np.nanmax(y))
    c0 = float(np.nanmedian(x))
    b0 = 1.0
    p0 = [a0, b0, c0, d0]
    bounds_lo = [-np.inf, -50, 1e-15, -np.inf]
    bounds_hi = [np.inf, 50, np.inf, np.inf]
    popt, pcov = curve_fit(
        _logistic_4pl, x, y, p0=p0, maxfev=10000,
        method="trf", bounds=(bounds_lo, bounds_hi),
    )
    y_pred = _logistic_4pl(x, *popt)
    r2 = _r_squared(y, y_pred)
    return popt, pcov, y_pred, r2


def _fit_5pl(x, y):
    """Fit a 5PL model. Returns (popt, pcov, y_pred, r2) or raises."""
    a0 = float(np.nanmin(y))
    d0 = float(np.nanmax(y))
    c0 = float(np.nanmedian(x))
    b0 = 1.0
    e0 = 1.0
    p0 = [a0, b0, c0, d0, e0]
    bounds_lo = [-np.inf, -50, 1e-15, -np.inf, 0.01]
    bounds_hi = [np.inf, 50, np.inf, np.inf, 50]
    popt, pcov = curve_fit(
        _logistic_5pl, x, y, p0=p0, maxfev=10000,
        method="trf", bounds=(bounds_lo, bounds_hi),
    )
    y_pred = _logistic_5pl(x, *popt)
    r2 = _r_squared(y, y_pred)
    return popt, pcov, y_pred, r2


def _gof_checks(y_obs, y_pred, n_params, label=""):
    """Goodness-of-fit validation checks."""
    r2 = _r_squared(y_obs, y_pred)
    n = len(y_obs)
    residuals = y_obs - y_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    resp_range = float(np.ptp(y_obs)) if np.ptp(y_obs) > 0 else 1.0
    nrmse = rmse / resp_range

    checks = []
    # R-squared check
    if r2 >= 0.99:
        checks.append(ValidationCheck(f"R-squared{f' ({label})' if label else ''}", "pass",
                                      f"R² = {r2:.5f} — excellent fit"))
    elif r2 >= 0.95:
        checks.append(ValidationCheck(f"R-squared{f' ({label})' if label else ''}", "pass",
                                      f"R² = {r2:.5f} — good fit"))
    elif r2 >= 0.90:
        checks.append(ValidationCheck(f"R-squared{f' ({label})' if label else ''}", "warn",
                                      f"R² = {r2:.5f} — acceptable fit",
                                      "Consider checking for outliers or model suitability"))
    else:
        checks.append(ValidationCheck(f"R-squared{f' ({label})' if label else ''}", "fail",
                                      f"R² = {r2:.5f} — poor fit",
                                      "The logistic model may not be appropriate for this data"))
    # NRMSE check
    if nrmse <= 0.05:
        checks.append(ValidationCheck(f"NRMSE{f' ({label})' if label else ''}", "pass",
                                      f"NRMSE = {nrmse:.4f} — low relative error"))
    elif nrmse <= 0.10:
        checks.append(ValidationCheck(f"NRMSE{f' ({label})' if label else ''}", "warn",
                                      f"NRMSE = {nrmse:.4f} — moderate relative error"))
    else:
        checks.append(ValidationCheck(f"NRMSE{f' ({label})' if label else ''}", "fail",
                                      f"NRMSE = {nrmse:.4f} — high relative error",
                                      "Check data quality or consider a different model"))
    # Sample size
    if n >= n_params * 3:
        checks.append(ValidationCheck(f"Data Points{f' ({label})' if label else ''}", "pass",
                                      f"n = {n} (>= {n_params * 3} recommended for {n_params} params)"))
    else:
        checks.append(ValidationCheck(f"Data Points{f' ({label})' if label else ''}", "warn",
                                      f"n = {n} (recommend >= {n_params * 3} for {n_params} parameters)",
                                      "More data points improve parameter estimates"))
    return checks


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

_COLORS = ["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#3b82f6",
           "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4"]


def _dose_response_plot(groups_data, model_fn, title="Dose-Response Curves"):
    """Build a Plotly figure with data points and fitted curves per group.

    groups_data: list of dicts with keys: name, x, y, popt, color
    """
    fig = go.Figure()
    for gd in groups_data:
        x, y, popt, color = gd["x"], gd["y"], gd["popt"], gd["color"]
        name = gd["name"]
        # Scatter
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=f"{name} (data)",
            marker=dict(color=color, size=8, opacity=0.7),
        ))
        # Fitted curve
        x_curve = np.linspace(x.min() * 0.8, x.max() * 1.2, 300)
        y_curve = model_fn(x_curve, *popt)
        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve, mode="lines", name=f"{name} (fit)",
            line=dict(color=color, width=2),
        ))
        # EC50 vertical marker
        ec50 = popt[2]
        if x.min() * 0.5 <= ec50 <= x.max() * 2:
            fig.add_vline(x=ec50, line_dash="dash", line_color=color, opacity=0.5,
                          annotation_text=f"EC50={ec50:.4g}", annotation_font_color=color)
    fig.update_layout(
        title=title, xaxis_title="Dose / Concentration", yaxis_title="Response",
        template="plotly+rdl", legend=dict(orientation="h", y=-0.15),
    )
    return fig


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_bioassay(df: pd.DataFrame):
    """Render Bioassay / Relative Potency analysis interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    help_tip(
        "Bioassay & Relative Potency",
        "This module fits sigmoidal dose-response curves (4PL / 5PL), tests "
        "parallelism between reference and test samples, computes relative "
        "potency with confidence intervals, and assesses dilution linearity. "
        "Common in pharmaceutical and biological assay development.",
    )

    tabs = st.tabs([
        "4-Parameter Logistic",
        "5-Parameter Logistic",
        "Parallel Line / Relative Potency",
        "Dilution Linearity",
    ])

    with tabs[0]:
        _render_4pl(df)
    with tabs[1]:
        _render_5pl(df)
    with tabs[2]:
        _render_parallel_line(df)
    with tabs[3]:
        _render_dilution_linearity(df)


# ---------------------------------------------------------------------------
# Tab 1 — 4-Parameter Logistic
# ---------------------------------------------------------------------------

def _render_4pl(df):
    section_header("4-Parameter Logistic (4PL) Fit",
                   "Fits y = d + (a-d) / (1 + (x/c)^b) per sample group.")

    dose_col, resp_col, group_col = _column_selectors(df, "ba_4pl")
    if dose_col is None:
        return

    log_dose = st.checkbox("Log-transform dose", value=False, key="ba_4pl_log")

    if st.button("Fit 4PL Model", key="ba_4pl_run", type="primary"):
        _run_logistic_fit(df, dose_col, resp_col, group_col, log_dose, model="4pl")


def _render_5pl(df):
    section_header("5-Parameter Logistic (5PL) Fit",
                   "Adds asymmetry parameter E: y = d + (a-d) / (1 + (x/c)^b)^e.")

    dose_col, resp_col, group_col = _column_selectors(df, "ba_5pl")
    if dose_col is None:
        return

    log_dose = st.checkbox("Log-transform dose", value=False, key="ba_5pl_log")

    if st.button("Fit 5PL Model", key="ba_5pl_run", type="primary"):
        _run_logistic_fit(df, dose_col, resp_col, group_col, log_dose, model="5pl")


def _run_logistic_fit(df, dose_col, resp_col, group_col, log_dose, model="4pl"):
    """Shared fitting logic for 4PL and 5PL tabs."""
    with st.spinner("Fitting curves..."):
        sub = df[[dose_col, resp_col] + ([group_col] if group_col else [])].dropna()
        if sub.empty:
            st.warning("No valid data after removing missing values.")
            return

        x_raw = sub[dose_col].values.astype(float)
        if log_dose:
            pos_mask = x_raw > 0
            if not pos_mask.all():
                st.warning("Non-positive dose values dropped for log transform.")
            sub = sub.loc[pos_mask]
            x_raw = np.log10(sub[dose_col].values.astype(float))

        groups = sub[group_col].unique().tolist() if group_col else ["All Data"]
        all_checks = []
        results_rows = []
        groups_plot_data = []
        fit_fn = _fit_5pl if model == "5pl" else _fit_4pl
        model_fn = _logistic_5pl if model == "5pl" else _logistic_4pl
        n_params = 5 if model == "5pl" else 4
        param_names_4 = ["Bottom (a)", "Hill Slope (b)", "EC50 (c)", "Top (d)"]
        param_names_5 = param_names_4 + ["Asymmetry (e)"]

        for idx, grp in enumerate(groups):
            if group_col:
                mask = sub[group_col] == grp
                x = (np.log10(sub.loc[mask, dose_col].values.astype(float))
                     if log_dose else sub.loc[mask, dose_col].values.astype(float))
                y = sub.loc[mask, resp_col].values.astype(float)
            else:
                x = x_raw
                y = sub[resp_col].values.astype(float)

            if len(x) < n_params + 1:
                st.warning(f"Group '{grp}': not enough data points (n={len(x)}, need >= {n_params + 1}).")
                continue

            try:
                popt, pcov, y_pred, r2 = fit_fn(x, y)
            except Exception as exc:
                st.error(f"Curve fit failed for '{grp}': {exc}")
                continue

            color = _COLORS[idx % len(_COLORS)]
            groups_plot_data.append(dict(name=str(grp), x=x, y=y, popt=popt, color=color))

            row = {"Sample": grp, "Bottom (a)": popt[0], "Hill Slope (b)": popt[1],
                   "EC50 (c)": popt[2], "Top (d)": popt[3]}
            if model == "5pl":
                row["Asymmetry (e)"] = popt[4]
            row["R²"] = r2
            row["n"] = len(x)
            results_rows.append(row)

            all_checks.extend(_gof_checks(y, y_pred, n_params, label=str(grp)))

        if not groups_plot_data:
            return

        # Plot
        fig = _dose_response_plot(groups_plot_data, model_fn,
                                  title=f"{'5PL' if model == '5pl' else '4PL'} Dose-Response")
        if log_dose:
            fig.update_layout(xaxis_title="log10(Dose)")
        st.plotly_chart(fig, use_container_width=True)

        # Results table
        section_header("Parameter Estimates")
        res_df = pd.DataFrame(results_rows)
        fmt = {c: "{:.5g}" for c in res_df.columns if c not in ("Sample", "n")}
        st.dataframe(res_df.style.format(fmt, na_rep="—"), use_container_width=True)

        # 4PL vs 5PL comparison (only shown in 5PL tab)
        if model == "5pl":
            _model_comparison(df, dose_col, resp_col, group_col, log_dose, groups_plot_data)

        # Goodness-of-fit checks
        validation_panel(all_checks, title="Goodness-of-Fit Checks")


# ---------------------------------------------------------------------------
# 4PL vs 5PL model comparison
# ---------------------------------------------------------------------------

def _model_comparison(df, dose_col, resp_col, group_col, log_dose, groups_5pl):
    """Extra sum-of-squares F-test comparing 4PL vs 5PL."""
    section_header("Model Comparison: 4PL vs 5PL",
                   "F-test determines whether the extra asymmetry parameter is justified.")

    sub = df[[dose_col, resp_col] + ([group_col] if group_col else [])].dropna()

    for gd in groups_5pl:
        grp_name = gd["name"]
        x, y = gd["x"], gd["y"]
        n = len(y)
        if n <= 5:
            continue

        # 5PL residuals
        y_pred_5 = _logistic_5pl(x, *gd["popt"])
        ss_5 = float(np.sum((y - y_pred_5) ** 2))
        df_5 = n - 5

        # Fit 4PL for comparison
        try:
            popt4, _, y_pred_4, _ = _fit_4pl(x, y)
            ss_4 = float(np.sum((y - y_pred_4) ** 2))
            df_4 = n - 4
        except Exception:
            st.info(f"Could not fit 4PL for comparison on group '{grp_name}'.")
            continue

        if df_5 <= 0 or ss_5 <= 0:
            continue

        f_stat = ((ss_4 - ss_5) / (df_4 - df_5)) / (ss_5 / df_5)
        p_val = 1 - stats.f.cdf(max(f_stat, 0), dfn=df_4 - df_5, dfd=df_5)

        if p_val < 0.05:
            recommendation = (
                f"For '{grp_name}', the 5PL model provides a significantly better fit "
                f"(F = {f_stat:.3f}, p = {p_val:.4f}). The asymmetry parameter is justified."
            )
        else:
            recommendation = (
                f"For '{grp_name}', the 5PL does not significantly improve over 4PL "
                f"(F = {f_stat:.3f}, p = {p_val:.4f}). The simpler 4PL model is recommended."
            )

        interpretation_card(Interpretation(
            title=f"Model Selection — {grp_name}",
            body=recommendation,
            detail=f"SS(4PL) = {ss_4:.4f}, SS(5PL) = {ss_5:.4f}, df = ({df_4 - df_5}, {df_5})",
        ))


# ---------------------------------------------------------------------------
# Tab 3 — Parallel Line / Relative Potency
# ---------------------------------------------------------------------------

def _render_parallel_line(df):
    section_header("Parallel Line Analysis & Relative Potency",
                   "Tests whether Reference and Test curves are parallel, then "
                   "computes relative potency with Fieller's confidence interval.")

    dose_col, resp_col, group_col = _column_selectors(df, "ba_pl")
    if dose_col is None:
        return

    if group_col is None:
        empty_state(
            "A sample/group column is required to distinguish Reference from Test.",
            "Select a categorical column that identifies your sample groups.",
        )
        return

    log_dose = st.checkbox("Log-transform dose", value=False, key="ba_pl_log")

    groups = df[group_col].dropna().unique().tolist()
    if len(groups) < 2:
        st.warning("Need at least two groups (Reference and Test) for relative potency.")
        return

    c1, c2 = st.columns(2)
    with c1:
        ref_group = st.selectbox("Reference group", groups, index=0, key="ba_pl_ref")
    with c2:
        test_options = [g for g in groups if g != ref_group]
        if not test_options:
            st.warning("Select a different group for Test.")
            return
        test_group = st.selectbox("Test group", test_options, index=0, key="ba_pl_test")

    alpha = st.slider("Significance level", 0.01, 0.10, 0.05, 0.01, key="ba_pl_alpha")
    accept_lo, accept_hi = st.slider(
        "Acceptance range for potency (%)", 50, 200, (80, 125), key="ba_pl_accept",
    )

    if st.button("Run Parallel Line Analysis", key="ba_pl_run", type="primary"):
        _run_parallel_line(df, dose_col, resp_col, group_col, ref_group, test_group,
                           log_dose, alpha, accept_lo / 100.0, accept_hi / 100.0)


def _run_parallel_line(df, dose_col, resp_col, group_col, ref_group, test_group,
                       log_dose, alpha, accept_lo, accept_hi):
    """Core parallel-line / relative-potency computation."""
    with st.spinner("Computing relative potency..."):
        sub = df.loc[df[group_col].isin([ref_group, test_group]),
                     [dose_col, resp_col, group_col]].dropna()

        ref_mask = sub[group_col] == ref_group
        test_mask = sub[group_col] == test_group

        x_ref = sub.loc[ref_mask, dose_col].values.astype(float)
        y_ref = sub.loc[ref_mask, resp_col].values.astype(float)
        x_test = sub.loc[test_mask, dose_col].values.astype(float)
        y_test = sub.loc[test_mask, resp_col].values.astype(float)

        if log_dose:
            pos_ref = x_ref > 0
            pos_test = x_test > 0
            if not (pos_ref.all() and pos_test.all()):
                st.warning("Non-positive dose values dropped for log transform.")
            x_ref, y_ref = np.log10(x_ref[pos_ref]), y_ref[pos_ref]
            x_test, y_test = np.log10(x_test[pos_test]), y_test[pos_test]

        if len(x_ref) < 5 or len(x_test) < 5:
            st.warning("Each group should have at least 5 data points for reliable results.")
            return

        # --- Unconstrained fits (independent 4PL per group) ---
        try:
            popt_ref, pcov_ref, ypred_ref, r2_ref = _fit_4pl(x_ref, y_ref)
            popt_test, pcov_test, ypred_test, r2_test = _fit_4pl(x_test, y_test)
        except Exception as exc:
            st.error(f"Independent curve fitting failed: {exc}")
            return

        ss_unconstrained = (np.sum((y_ref - ypred_ref) ** 2)
                            + np.sum((y_test - ypred_test) ** 2))
        df_unconstrained = len(y_ref) + len(y_test) - 8  # 4 params each

        # --- Constrained fit (shared slope and asymptotes, different EC50) ---
        def _constrained_model(x_both, a, b, c_ref, c_test, d):
            """Shared a, b, d; separate EC50s packed in one call."""
            n_ref = len(x_ref)
            y_out = np.empty_like(x_both)
            y_out[:n_ref] = d + (a - d) / (1.0 + (x_both[:n_ref] / c_ref) ** b)
            y_out[n_ref:] = d + (a - d) / (1.0 + (x_both[n_ref:] / c_test) ** b)
            return y_out

        x_both = np.concatenate([x_ref, x_test])
        y_both = np.concatenate([y_ref, y_test])
        a0 = float(np.nanmin(y_both))
        d0 = float(np.nanmax(y_both))
        b0 = float(np.mean([popt_ref[1], popt_test[1]]))
        c_ref0 = popt_ref[2]
        c_test0 = popt_test[2]
        p0_c = [a0, b0, c_ref0, c_test0, d0]
        bounds_lo_c = [-np.inf, -50, 1e-15, 1e-15, -np.inf]
        bounds_hi_c = [np.inf, 50, np.inf, np.inf, np.inf]

        try:
            popt_c, pcov_c = curve_fit(
                _constrained_model, x_both, y_both, p0=p0_c,
                maxfev=10000, method="trf", bounds=(bounds_lo_c, bounds_hi_c),
            )
        except Exception as exc:
            st.error(f"Constrained (parallel) curve fit failed: {exc}")
            return

        y_pred_c = _constrained_model(x_both, *popt_c)
        ss_constrained = float(np.sum((y_both - y_pred_c) ** 2))
        df_constrained = len(y_both) - 5  # 5 params in constrained model

        # --- Parallelism F-test ---
        df_num = df_constrained - df_unconstrained  # should be 3 (8-5)
        df_den = df_unconstrained
        if df_num > 0 and df_den > 0 and ss_unconstrained > 0:
            f_stat = ((ss_constrained - ss_unconstrained) / df_num) / (ss_unconstrained / df_den)
            p_parallel = 1 - stats.f.cdf(max(f_stat, 0), dfn=df_num, dfd=df_den)
        else:
            f_stat, p_parallel = np.nan, np.nan

        significance_result(
            p_parallel, alpha, "Parallelism Test (F-test)",
            effect_size=f_stat, effect_label="F-statistic",
        )

        if not np.isnan(p_parallel):
            if p_parallel >= alpha:
                interpretation_card(Interpretation(
                    title="Parallelism Assessment",
                    body=(f"The curves are consistent with parallelism (p = {p_parallel:.4f}). "
                          "Relative potency estimation is valid."),
                    detail=(f"Constrained SS = {ss_constrained:.4f}, "
                            f"Unconstrained SS = {ss_unconstrained:.4f}"),
                ))
            else:
                interpretation_card(Interpretation(
                    title="Parallelism Assessment",
                    body=(f"The parallelism assumption is violated (p = {p_parallel:.4f}). "
                          "Relative potency should be interpreted with caution."),
                    detail="Consider reviewing dose-response shapes for systematic departures.",
                ))

        # --- Relative Potency ---
        ec50_ref = popt_c[2]
        ec50_test = popt_c[3]
        potency = ec50_ref / ec50_test if ec50_test != 0 else np.nan

        # Delta-method SE for ratio
        # Indices in popt_c: 0=a, 1=b, 2=c_ref, 3=c_test, 4=d
        var_ref = pcov_c[2, 2] if pcov_c.shape[0] > 2 else 0
        var_test = pcov_c[3, 3] if pcov_c.shape[0] > 3 else 0
        cov_rt = pcov_c[2, 3] if pcov_c.shape[0] > 3 else 0

        if ec50_test != 0 and var_test >= 0 and var_ref >= 0:
            # Fieller-style CI via delta method for ratio
            r = ec50_ref / ec50_test
            se_ratio = abs(r) * np.sqrt(
                var_ref / ec50_ref ** 2 + var_test / ec50_test ** 2
                - 2 * cov_rt / (ec50_ref * ec50_test)
            ) if ec50_ref != 0 else np.nan
            t_crit = stats.t.ppf(1 - alpha / 2, df_constrained)
            ci_lo = r - t_crit * se_ratio if not np.isnan(se_ratio) else np.nan
            ci_hi = r + t_crit * se_ratio if not np.isnan(se_ratio) else np.nan
        else:
            se_ratio, ci_lo, ci_hi = np.nan, np.nan, np.nan

        # --- Display potency ---
        section_header("Relative Potency Estimate")
        m1, m2, m3 = st.columns(3)
        m1.metric("Relative Potency", f"{potency:.4f}" if not np.isnan(potency) else "N/A")
        m2.metric("Potency (%)", f"{potency * 100:.1f}%" if not np.isnan(potency) else "N/A")
        m3.metric("95% CI",
                  f"({ci_lo:.4f}, {ci_hi:.4f})" if not (np.isnan(ci_lo) or np.isnan(ci_hi)) else "N/A")

        st.markdown(
            f"**EC50 Reference:** {ec50_ref:.5g} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**EC50 Test:** {ec50_test:.5g}"
        )

        # --- Forest plot ---
        section_header("Potency Forest Plot")
        fig = go.Figure()

        # Acceptance region
        fig.add_vrect(x0=accept_lo * 100, x1=accept_hi * 100,
                      fillcolor="#22c55e", opacity=0.12, line_width=0,
                      annotation_text="Acceptance", annotation_position="top left",
                      annotation_font_color="#16a34a")

        # Point estimate + CI
        pot_pct = potency * 100 if not np.isnan(potency) else 100
        ci_lo_pct = ci_lo * 100 if not np.isnan(ci_lo) else pot_pct
        ci_hi_pct = ci_hi * 100 if not np.isnan(ci_hi) else pot_pct

        fig.add_trace(go.Scatter(
            x=[pot_pct], y=[test_group], mode="markers",
            marker=dict(size=14, color="#6366f1", symbol="diamond"),
            name="Point Estimate",
        ))
        fig.add_trace(go.Scatter(
            x=[ci_lo_pct, ci_hi_pct], y=[test_group, test_group], mode="lines",
            line=dict(color="#6366f1", width=3), name="95% CI", showlegend=True,
        ))

        fig.add_vline(x=100, line_dash="dot", line_color="#6b7280", opacity=0.5)
        fig.update_layout(
            title="Relative Potency (%) with 95% CI",
            xaxis_title="Relative Potency (%)",
            yaxis_title="", template="plotly+rdl",
            xaxis=dict(range=[min(50, ci_lo_pct - 10), max(200, ci_hi_pct + 10)]),
            height=250,
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Overlay plot ---
        section_header("Fitted Curves Overlay")
        n_r = len(x_ref)
        fig2 = go.Figure()
        for label, x_g, y_g, ec50, color in [
            (str(ref_group), x_ref, y_ref, ec50_ref, _COLORS[0]),
            (str(test_group), x_test, y_test, ec50_test, _COLORS[1]),
        ]:
            fig2.add_trace(go.Scatter(
                x=x_g, y=y_g, mode="markers", name=f"{label} (data)",
                marker=dict(color=color, size=8, opacity=0.7),
            ))
            x_curve = np.linspace(min(x_g.min(), x_g.min()) * 0.8,
                                  max(x_g.max(), x_g.max()) * 1.2, 300)
            y_curve = _logistic_4pl(x_curve, popt_c[0], popt_c[1], ec50, popt_c[4])
            fig2.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode="lines", name=f"{label} (fit)",
                line=dict(color=color, width=2),
            ))
        fig2.update_layout(
            title="Constrained (Parallel) Model Overlay",
            xaxis_title="log10(Dose)" if log_dose else "Dose",
            yaxis_title="Response", template="plotly+rdl",
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Results summary table
        section_header("Summary")
        summary_df = pd.DataFrame([
            {"Parameter": "EC50 (Reference)", "Value": ec50_ref},
            {"Parameter": "EC50 (Test)", "Value": ec50_test},
            {"Parameter": "Shared Hill Slope", "Value": popt_c[1]},
            {"Parameter": "Shared Bottom", "Value": popt_c[0]},
            {"Parameter": "Shared Top", "Value": popt_c[4]},
            {"Parameter": "Relative Potency", "Value": potency},
            {"Parameter": "95% CI Lower", "Value": ci_lo},
            {"Parameter": "95% CI Upper", "Value": ci_hi},
            {"Parameter": "Parallelism F", "Value": f_stat},
            {"Parameter": "Parallelism p", "Value": p_parallel},
        ])
        st.dataframe(
            summary_df.style.format({"Value": "{:.5g}"}, na_rep="N/A"),
            use_container_width=True,
        )

        # Validation
        checks = _gof_checks(y_both, y_pred_c, 5, label="Constrained model")
        if not np.isnan(p_parallel) and p_parallel < alpha:
            checks.append(ValidationCheck(
                "Parallelism", "fail",
                f"F = {f_stat:.3f}, p = {p_parallel:.4f} — curves are not parallel",
                "Potency estimate may be biased; inspect curves visually",
            ))
        else:
            checks.append(ValidationCheck(
                "Parallelism", "pass",
                f"F = {f_stat:.3f}, p = {p_parallel:.4f} — parallelism holds",
            ))
        validation_panel(checks, title="Analysis Quality Checks")


# ---------------------------------------------------------------------------
# Tab 4 — Dilution Linearity
# ---------------------------------------------------------------------------

def _render_dilution_linearity(df):
    section_header("Dilution Linearity",
                   "Assess linearity of response across a dilution series and "
                   "flag non-linear points via percent recovery.")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) < 2:
        empty_state("At least two numeric columns are required.",
                    "Upload a dataset with dose and response columns.")
        return

    c1, c2 = st.columns(2)
    with c1:
        dose_col = st.selectbox("Dose / Dilution", num_cols, key="ba_dl_dose")
    with c2:
        resp_idx = min(1, len(num_cols) - 1)
        resp_col = st.selectbox("Response", num_cols, index=resp_idx, key="ba_dl_resp")

    sub = df[[dose_col, resp_col]].dropna()
    x_raw = sub[dose_col].values.astype(float)
    y_raw = sub[resp_col].values.astype(float)

    pos_mask = x_raw > 0
    if not pos_mask.all():
        st.info("Non-positive dose values will be excluded for log-dose analysis.")
    sub_pos = sub.loc[pos_mask]
    x_pos = x_raw[pos_mask]
    y_pos = y_raw[pos_mask]

    if len(x_pos) < 3:
        st.warning("Need at least 3 positive dose values for linearity assessment.")
        return

    log_x = np.log10(x_pos)

    # Range selector
    x_min_log, x_max_log = float(log_x.min()), float(log_x.max())
    if x_min_log >= x_max_log:
        st.warning("Dose values have no range for analysis.")
        return

    range_vals = st.slider(
        "Select linear range (log10 dose)",
        x_min_log, x_max_log, (x_min_log, x_max_log),
        step=(x_max_log - x_min_log) / 100 if x_max_log > x_min_log else 0.01,
        key="ba_dl_range",
    )

    recovery_thresh = st.slider(
        "Recovery threshold (%)", 1, 30, 10, key="ba_dl_thresh",
    )

    if st.button("Assess Linearity", key="ba_dl_run", type="primary"):
        _run_dilution_linearity(log_x, y_pos, x_pos, range_vals, recovery_thresh, dose_col, resp_col)


def _run_dilution_linearity(log_x, y, x_raw, range_vals, threshold, dose_col, resp_col):
    """Core dilution linearity computation."""
    with st.spinner("Assessing linearity..."):
        in_range = (log_x >= range_vals[0]) & (log_x <= range_vals[1])
        lx_fit = log_x[in_range]
        y_fit = y[in_range]

        if len(lx_fit) < 3:
            st.warning("Not enough points in the selected range (need >= 3).")
            return

        slope, intercept, r_value, p_value, std_err = stats.linregress(lx_fit, y_fit)
        r2 = r_value ** 2

        # Predictions for all points
        y_pred_all = slope * log_x + intercept
        recovery = (y / y_pred_all) * 100
        flagged = np.abs(recovery - 100) > threshold

        # Results table
        section_header("Dilution Linearity Results")

        res_df = pd.DataFrame({
            "Dose": x_raw,
            "log10(Dose)": log_x,
            "Response": y,
            "Predicted": y_pred_all,
            "% Recovery": recovery,
            "Flag": np.where(flagged, "FAIL", "PASS"),
        })

        def _highlight_flag(val):
            if val == "FAIL":
                return "color: #ef4444; font-weight: 600"
            return "color: #22c55e"

        st.dataframe(
            res_df.style
            .format({"Dose": "{:.4g}", "log10(Dose)": "{:.3f}", "Response": "{:.4g}",
                      "Predicted": "{:.4g}", "% Recovery": "{:.1f}%"})
            .map(_highlight_flag, subset=["Flag"]),
            use_container_width=True,
        )

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Slope", f"{slope:.4f}")
        m2.metric("R²", f"{r2:.5f}")
        m3.metric("Points in Range", f"{int(in_range.sum())}")
        m4.metric("Flagged Points", f"{int(flagged.sum())}")

        # Plot
        section_header("Linearity Plot")
        fig = go.Figure()

        # Passing points
        pass_mask = ~flagged
        fig.add_trace(go.Scatter(
            x=log_x[pass_mask], y=y[pass_mask], mode="markers",
            marker=dict(color="#6366f1", size=9), name="Within limits",
        ))
        # Flagged points
        if flagged.any():
            fig.add_trace(go.Scatter(
                x=log_x[flagged], y=y[flagged], mode="markers",
                marker=dict(color="#ef4444", size=11, symbol="x"),
                name="Outside limits",
            ))
        # Regression line
        lx_line = np.linspace(log_x.min(), log_x.max(), 200)
        fig.add_trace(go.Scatter(
            x=lx_line, y=slope * lx_line + intercept, mode="lines",
            line=dict(color="#6366f1", width=2, dash="dash"), name="Linear fit",
        ))
        # Threshold bands
        fig.add_trace(go.Scatter(
            x=lx_line, y=(slope * lx_line + intercept) * (1 + threshold / 100),
            mode="lines", line=dict(color="#f59e0b", width=1, dash="dot"),
            name=f"+{threshold}%", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=lx_line, y=(slope * lx_line + intercept) * (1 - threshold / 100),
            mode="lines", line=dict(color="#f59e0b", width=1, dash="dot"),
            name=f"-{threshold}%", showlegend=True,
        ))

        fig.update_layout(
            title="Dilution Linearity: log(Dose) vs Response",
            xaxis_title=f"log10({dose_col})", yaxis_title=resp_col,
            template="plotly+rdl",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recovery plot
        section_header("% Recovery Plot")
        fig_rec = go.Figure()
        fig_rec.add_hrect(y0=100 - threshold, y1=100 + threshold,
                          fillcolor="#22c55e", opacity=0.1, line_width=0)
        fig_rec.add_hline(y=100, line_dash="dot", line_color="#6b7280", opacity=0.5)

        fig_rec.add_trace(go.Scatter(
            x=log_x[pass_mask], y=recovery[pass_mask], mode="markers",
            marker=dict(color="#6366f1", size=9), name="Within limits",
        ))
        if flagged.any():
            fig_rec.add_trace(go.Scatter(
                x=log_x[flagged], y=recovery[flagged], mode="markers",
                marker=dict(color="#ef4444", size=11, symbol="x"), name="Outside limits",
            ))

        fig_rec.update_layout(
            title="Percent Recovery by Dilution",
            xaxis_title=f"log10({dose_col})", yaxis_title="% Recovery",
            template="plotly+rdl",
            yaxis=dict(range=[max(0, recovery.min() - 15), recovery.max() + 15]),
        )
        st.plotly_chart(fig_rec, use_container_width=True)

        # Validation
        checks = []
        if r2 >= 0.99:
            checks.append(ValidationCheck("Linearity R²", "pass",
                                          f"R² = {r2:.5f} — excellent linearity"))
        elif r2 >= 0.95:
            checks.append(ValidationCheck("Linearity R²", "pass",
                                          f"R² = {r2:.5f} — good linearity"))
        elif r2 >= 0.90:
            checks.append(ValidationCheck("Linearity R²", "warn",
                                          f"R² = {r2:.5f} — marginal linearity",
                                          "Consider narrowing the dose range"))
        else:
            checks.append(ValidationCheck("Linearity R²", "fail",
                                          f"R² = {r2:.5f} — poor linearity",
                                          "Linear model may not be appropriate for this range"))

        n_flagged = int(flagged.sum())
        pct_flagged = n_flagged / len(flagged) * 100 if len(flagged) > 0 else 0
        if n_flagged == 0:
            checks.append(ValidationCheck("Recovery", "pass",
                                          f"All points within +/-{threshold}% of predicted"))
        elif pct_flagged <= 20:
            checks.append(ValidationCheck("Recovery", "warn",
                                          f"{n_flagged} point(s) ({pct_flagged:.0f}%) outside +/-{threshold}%",
                                          "Review flagged dilutions for pipetting or matrix effects"))
        else:
            checks.append(ValidationCheck("Recovery", "fail",
                                          f"{n_flagged} point(s) ({pct_flagged:.0f}%) outside +/-{threshold}%",
                                          "Significant non-linearity detected"))

        validation_panel(checks, title="Linearity Assessment")
