"""
ICH Stability Analysis Module - Stability trending, poolability testing (ICH Q1E),
shelf-life estimation, and multi-attribute dashboards for pharmaceutical stability data.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    HAS_SM = True
except ImportError:
    HAS_SM = False

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import (
    section_header, empty_state, help_tip, validation_panel, interpretation_card,
)
from modules.validation import (
    check_normality, check_sample_size, ValidationCheck, Interpretation,
)

# RDL colorway (mirrors ui_helpers._RDL_COLORWAY)
_COLORS = [
    "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#3b82f6",
    "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4",
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def render_stability(df: pd.DataFrame):
    """Render the ICH Stability Analysis module."""
    if df is None or df.empty:
        empty_state(
            "No data loaded.",
            "Upload a stability dataset from the sidebar to begin.",
        )
        return

    if not HAS_SM:
        st.error(
            "This module requires **statsmodels**. "
            "Install it with `pip install statsmodels`."
        )
        return

    tabs = st.tabs([
        "Stability Trending",
        "Poolability (ICH Q1E)",
        "Shelf-Life Estimation",
        "Multi-Attribute Dashboard",
    ])

    with tabs[0]:
        _render_trending(df)
    with tabs[1]:
        _render_poolability(df)
    with tabs[2]:
        _render_shelf_life(df)
    with tabs[3]:
        _render_multi_attribute(df)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _safe_col_name(col: str) -> str:
    """Return a column name safe for statsmodels formulas."""
    return col.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")


def _prepare_formula_df(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Return a copy with formula-safe column names and a rename mapping."""
    rename_map = {}
    for c in cols:
        safe = _safe_col_name(c)
        if safe != c:
            rename_map[c] = safe
    return df.rename(columns=rename_map), rename_map


def _get_safe(name: str, rename_map: dict) -> str:
    return rename_map.get(name, name)


# ---------------------------------------------------------------------------
# Tab 1: Stability Trending
# ---------------------------------------------------------------------------

def _render_trending(df: pd.DataFrame):
    section_header(
        "Stability Trending",
        "Fit per-batch linear regressions of response vs. time and overlay "
        "specification limits to visualise degradation trends.",
    )

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    if len(num_cols) < 2:
        empty_state(
            "Need at least 2 numeric columns (time + response).",
            "Ensure your dataset has a numeric time column and at least one numeric response.",
        )
        return

    # --- Column selectors ---------------------------------------------------
    c1, c2 = st.columns(2)
    time_col = c1.selectbox(
        "Time column (numeric):", num_cols, key="stability_trend_time",
    )
    response_options = [c for c in num_cols if c != time_col]
    response_cols = c2.multiselect(
        "Response column(s):", response_options,
        default=response_options[:1] if response_options else [],
        key="stability_trend_resp",
    )

    batch_col = None
    if cat_cols:
        batch_col = st.selectbox(
            "Batch column (categorical, optional):",
            ["(none)"] + cat_cols,
            key="stability_trend_batch",
        )
        if batch_col == "(none)":
            batch_col = None

    c3, c4 = st.columns(2)
    lsl = c3.number_input("Lower Spec Limit (LSL):", value=None, key="stability_trend_lsl")
    usl = c4.number_input("Upper Spec Limit (USL):", value=None, key="stability_trend_usl")

    if not response_cols:
        empty_state("Select at least one response column.")
        return

    if st.button("Run Trending Analysis", key="stability_trend_run"):
        with st.spinner("Fitting regression models..."):
            _trending_analysis(df, time_col, response_cols, batch_col, lsl, usl)


def _trending_analysis(
    df: pd.DataFrame,
    time_col: str,
    response_cols: list[str],
    batch_col: str | None,
    lsl: float | None,
    usl: float | None,
):
    """Core trending analysis: per-batch OLS + overlay plot + validation."""
    for resp_col in response_cols:
        section_header(f"Trend: {resp_col}")
        work = df[[time_col, resp_col]].copy()
        if batch_col:
            work[batch_col] = df[batch_col].astype(str)
        work = work.dropna(subset=[time_col, resp_col])

        if len(work) < 3:
            st.warning(f"Too few data points for {resp_col} after dropping NAs.")
            continue

        batches = work[batch_col].unique().tolist() if batch_col else ["All"]
        fig = go.Figure()
        results_rows = []
        all_checks: list[ValidationCheck] = []

        for i, batch in enumerate(batches):
            color = _COLORS[i % len(_COLORS)]
            subset = work[work[batch_col] == batch] if batch_col else work
            x = subset[time_col].values.astype(float)
            y = subset[resp_col].values.astype(float)

            if len(x) < 3:
                st.warning(f"Batch '{batch}': fewer than 3 points, skipping.")
                continue

            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            slope = model.params[1]
            intercept = model.params[0]
            r2 = model.rsquared
            p_slope = model.pvalues[1]
            residuals = model.resid

            # Data points
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                name=f"{batch} (data)",
                marker=dict(color=color, size=7),
                legendgroup=batch,
            ))

            # Regression line
            x_line = np.linspace(x.min(), x.max(), 200)
            y_line = intercept + slope * x_line
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                name=f"{batch} (fit)",
                line=dict(color=color, width=2),
                legendgroup=batch,
            ))

            results_rows.append({
                "Batch": batch,
                "Slope": round(slope, 6),
                "Intercept": round(intercept, 4),
                "R\u00b2": round(r2, 4),
                "p-value (slope)": round(p_slope, 6),
            })

            # Validation checks per batch
            label = f"Batch {batch}" if batch != "All" else ""
            r2_check = ValidationCheck(
                name=f"Linearity{f' ({label})' if label else ''}",
                status="pass" if r2 >= 0.8 else ("warn" if r2 >= 0.5 else "fail"),
                detail=f"R\u00b2 = {r2:.4f}",
                suggestion="Consider non-linear models if R\u00b2 is low." if r2 < 0.8 else "",
            )
            all_checks.append(r2_check)
            all_checks.append(check_normality(residuals, label=f"residuals{f' {label}' if label else ''}"))

        # Spec-limit lines
        if lsl is not None:
            fig.add_hline(y=lsl, line_dash="dash", line_color="#ef4444",
                          annotation_text="LSL", annotation_position="top left")
        if usl is not None:
            fig.add_hline(y=usl, line_dash="dash", line_color="#ef4444",
                          annotation_text="USL", annotation_position="bottom left")

        fig.update_layout(
            template="plotly+rdl",
            xaxis_title=time_col,
            yaxis_title=resp_col,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Results table
        if results_rows:
            res_df = pd.DataFrame(results_rows)
            st.dataframe(res_df, use_container_width=True, hide_index=True)

        # Validation
        if all_checks:
            validation_panel(all_checks, title="Trending Assumption Checks")

    help_tip(
        "Stability Trending",
        "Per-batch linear regression of each response against time. "
        "The slope represents the rate of change per unit of time. "
        "A statistically significant slope (p < 0.05) indicates a meaningful trend. "
        "Specification limit lines help visualise when the product may go out of spec.",
    )


# ---------------------------------------------------------------------------
# Tab 2: Poolability (ICH Q1E)
# ---------------------------------------------------------------------------

def _render_poolability(df: pd.DataFrame):
    section_header(
        "Poolability Testing (ICH Q1E)",
        "ANCOVA-based test to determine whether batches can be pooled for "
        "shelf-life estimation, following ICH Q1E guidelines.",
    )

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    if len(num_cols) < 2 or not cat_cols:
        empty_state(
            "Need at least 2 numeric columns and 1 categorical batch column.",
            "Ensure your dataset has time (numeric), response (numeric), and batch (categorical) columns.",
        )
        return

    c1, c2, c3 = st.columns(3)
    time_col = c1.selectbox("Time column:", num_cols, key="stability_pool_time")
    resp_options = [c for c in num_cols if c != time_col]
    resp_col = c2.selectbox("Response column:", resp_options, key="stability_pool_resp")
    batch_col = c3.selectbox("Batch column:", cat_cols, key="stability_pool_batch")

    if st.button("Run Poolability Test", key="stability_pool_run"):
        with st.spinner("Fitting ANCOVA models..."):
            _poolability_analysis(df, time_col, resp_col, batch_col)


def _poolability_analysis(
    df: pd.DataFrame, time_col: str, resp_col: str, batch_col: str,
):
    """ICH Q1E poolability: compare three nested ANCOVA models."""
    work = df[[time_col, resp_col, batch_col]].dropna()
    work[batch_col] = work[batch_col].astype(str)

    batches = work[batch_col].unique()
    if len(batches) < 2:
        st.warning("Need at least 2 batches to test poolability.")
        return

    if len(work) < 6:
        st.warning("Not enough data points for ANCOVA (need at least 6).")
        return

    # Build formula-safe names
    needed = [time_col, resp_col, batch_col]
    safe_df, rmap = _prepare_formula_df(work, needed)
    t = _get_safe(time_col, rmap)
    r = _get_safe(resp_col, rmap)
    b = _get_safe(batch_col, rmap)

    try:
        # Model A: common slope + common intercept (fully pooled)
        formula_a = f"Q('{r}') ~ Q('{t}')"
        model_a = smf.ols(formula_a, data=safe_df).fit()

        # Model B: common slope + different intercepts
        formula_b = f"Q('{r}') ~ Q('{t}') + C(Q('{b}'))"
        model_b = smf.ols(formula_b, data=safe_df).fit()

        # Model C: different slopes + different intercepts (separate lines)
        formula_c = f"Q('{r}') ~ Q('{t}') * C(Q('{b}'))"
        model_c = smf.ols(formula_c, data=safe_df).fit()
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        return

    # F-tests via sequential anova_lm
    section_header("ANCOVA Model Comparison")

    try:
        # Test 1: slopes equal? (Model B vs Model C)
        anova_bc = anova_lm(model_b, model_c)
        f_slopes = anova_bc["F"].iloc[1]
        p_slopes = anova_bc["Pr(>F)"].iloc[1]

        # Test 2: intercepts equal? (Model A vs Model B)
        anova_ab = anova_lm(model_a, model_b)
        f_intercepts = anova_ab["F"].iloc[1]
        p_intercepts = anova_ab["Pr(>F)"].iloc[1]
    except Exception as e:
        st.error(f"ANCOVA comparison failed: {e}")
        return

    # Display comparison table
    comp_data = {
        "Comparison": [
            "Slopes equal? (B vs C)",
            "Intercepts equal? (A vs B)",
        ],
        "F-statistic": [
            round(f_slopes, 4) if pd.notna(f_slopes) else "N/A",
            round(f_intercepts, 4) if pd.notna(f_intercepts) else "N/A",
        ],
        "p-value": [
            round(p_slopes, 6) if pd.notna(p_slopes) else "N/A",
            round(p_intercepts, 6) if pd.notna(p_intercepts) else "N/A",
        ],
        "Significant (p < 0.25)": [
            "Yes" if pd.notna(p_slopes) and p_slopes < 0.25 else "No",
            "Yes" if pd.notna(p_intercepts) and p_intercepts < 0.25 else "No",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    st.caption(
        "ICH Q1E uses a significance level of 0.25 for poolability tests "
        "to be conservative about detecting batch differences."
    )

    # Model summary table
    section_header("Model Details")
    model_info = pd.DataFrame({
        "Model": ["A (pooled)", "B (common slope)", "C (separate)"],
        "Formula": [formula_a, formula_b, formula_c],
        "RSS": [round(model_a.ssr, 4), round(model_b.ssr, 4), round(model_c.ssr, 4)],
        "df (resid)": [int(model_a.df_resid), int(model_b.df_resid), int(model_c.df_resid)],
        "R\u00b2": [round(model_a.rsquared, 4), round(model_b.rsquared, 4), round(model_c.rsquared, 4)],
    })
    st.dataframe(model_info, use_container_width=True, hide_index=True)

    # ICH Q1E decision tree
    slopes_differ = pd.notna(p_slopes) and p_slopes < 0.25
    intercepts_differ = pd.notna(p_intercepts) and p_intercepts < 0.25

    if slopes_differ:
        decision = "separate"
        interp = Interpretation(
            title="ICH Q1E Poolability Decision",
            body=(
                "Slopes are statistically different across batches (p < 0.25). "
                "Each batch should be analysed separately for shelf-life estimation."
            ),
            detail=(
                f"Slope test: F = {f_slopes:.4f}, p = {p_slopes:.6f}. "
                "Different degradation rates suggest batch-specific factors are at play."
            ),
        )
    elif intercepts_differ:
        decision = "common_slope"
        interp = Interpretation(
            title="ICH Q1E Poolability Decision",
            body=(
                "Slopes are similar across batches, but intercepts differ (p < 0.25). "
                "Use a common slope with separate intercepts for shelf-life estimation."
            ),
            detail=(
                f"Slope test: F = {f_slopes:.4f}, p = {p_slopes:.6f} (not significant). "
                f"Intercept test: F = {f_intercepts:.4f}, p = {p_intercepts:.6f} (significant). "
                "Batches start at different levels but degrade at the same rate."
            ),
        )
    else:
        decision = "pooled"
        interp = Interpretation(
            title="ICH Q1E Poolability Decision",
            body=(
                "Neither slopes nor intercepts differ significantly across batches. "
                "All batch data can be pooled for a single shelf-life estimate."
            ),
            detail=(
                f"Slope test: F = {f_slopes:.4f}, p = {p_slopes:.6f}. "
                f"Intercept test: F = {f_intercepts:.4f}, p = {p_intercepts:.6f}. "
                "Batches behave consistently and pooling maximises statistical power."
            ),
        )

    interpretation_card(interp)

    # Store decision and model info for Tab 3
    st.session_state["stability_pool_decision"] = decision
    st.session_state["stability_pool_models"] = {
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c,
    }
    st.session_state["stability_pool_config"] = {
        "time_col": time_col,
        "resp_col": resp_col,
        "batch_col": batch_col,
    }

    help_tip(
        "ICH Q1E Poolability",
        "ICH Q1E recommends testing batch poolability using ANCOVA with a "
        "significance level of 0.25 (not the usual 0.05). This is deliberately "
        "conservative to avoid masking real batch differences.\n\n"
        "**Three nested models are compared:**\n"
        "- **Model A**: Single regression line for all batches (fully pooled)\n"
        "- **Model B**: Same slope but different intercepts per batch\n"
        "- **Model C**: Different slopes and intercepts per batch (fully separate)\n\n"
        "The decision tree proceeds top-down: first test whether slopes differ "
        "(C vs B), then whether intercepts differ (B vs A).",
    )


# ---------------------------------------------------------------------------
# Tab 3: Shelf-Life Estimation
# ---------------------------------------------------------------------------

def _render_shelf_life(df: pd.DataFrame):
    section_header(
        "Shelf-Life Estimation",
        "Estimate shelf life as the time at which the 95% one-sided confidence "
        "interval of the mean regression crosses a specification limit.",
    )

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    if len(num_cols) < 2:
        empty_state(
            "Need at least 2 numeric columns (time + response).",
            "Ensure your dataset has a numeric time column and at least one numeric response.",
        )
        return

    c1, c2 = st.columns(2)
    time_col = c1.selectbox("Time column:", num_cols, key="stability_sl_time")
    resp_options = [c for c in num_cols if c != time_col]
    resp_col = c2.selectbox("Response column:", resp_options, key="stability_sl_resp")

    batch_col = None
    if cat_cols:
        batch_col = st.selectbox(
            "Batch column (optional):",
            ["(none)"] + cat_cols,
            key="stability_sl_batch",
        )
        if batch_col == "(none)":
            batch_col = None

    c3, c4 = st.columns(2)
    lsl = c3.number_input("Lower Spec Limit (LSL):", value=None, key="stability_sl_lsl")
    usl = c4.number_input("Upper Spec Limit (USL):", value=None, key="stability_sl_usl")

    if lsl is None and usl is None:
        st.info("Enter at least one specification limit (LSL or USL) to estimate shelf life.")
        return

    # Retrieve poolability decision if available
    decision = st.session_state.get("stability_pool_decision", "pooled")
    st.caption(f"Using poolability decision: **{decision.replace('_', ' ').title()}**")

    if st.button("Estimate Shelf Life", key="stability_sl_run"):
        with st.spinner("Computing shelf-life estimate..."):
            _shelf_life_analysis(df, time_col, resp_col, batch_col, lsl, usl, decision)


def _shelf_life_analysis(
    df: pd.DataFrame,
    time_col: str,
    resp_col: str,
    batch_col: str | None,
    lsl: float | None,
    usl: float | None,
    decision: str,
):
    """Shelf-life estimation via CI intersection with spec limits."""
    work = df[[time_col, resp_col]].copy()
    if batch_col:
        work[batch_col] = df[batch_col].astype(str)
    work = work.dropna(subset=[time_col, resp_col])

    if len(work) < 4:
        st.warning("Not enough data points for shelf-life estimation.")
        return

    batches = work[batch_col].unique().tolist() if batch_col else ["All"]
    should_pool = (decision == "pooled") or batch_col is None

    shelf_lives: list[dict] = []

    if should_pool:
        # Fit a single model on all data
        sl_info = _estimate_single_shelf_life(work, time_col, resp_col, "All (pooled)", lsl, usl)
        if sl_info:
            shelf_lives.append(sl_info)
    elif decision == "common_slope":
        # Fit Model B (common slope, separate intercepts) via formula API
        needed = [time_col, resp_col, batch_col]
        safe_df, rmap = _prepare_formula_df(work, needed)
        t = _get_safe(time_col, rmap)
        r = _get_safe(resp_col, rmap)
        b = _get_safe(batch_col, rmap)

        try:
            formula = f"Q('{r}') ~ Q('{t}') + C(Q('{b}'))"
            model = smf.ols(formula, data=safe_df).fit()
        except Exception as e:
            st.error(f"Common-slope model failed: {e}")
            return

        for batch in batches:
            sl = _shelf_life_from_formula_model(
                model, safe_df, t, r, b, batch, time_col, lsl, usl,
            )
            if sl is not None:
                shelf_lives.append({"Batch": batch, "Shelf Life": sl})
    else:
        # Separate per batch
        for batch in batches:
            subset = work[work[batch_col] == batch] if batch_col else work
            if len(subset) < 3:
                st.warning(f"Batch '{batch}': too few points, skipping.")
                continue
            sl_info = _estimate_single_shelf_life(subset, time_col, resp_col, batch, lsl, usl)
            if sl_info:
                shelf_lives.append(sl_info)

    if not shelf_lives:
        st.info("Could not determine shelf life from the data and specification limits provided.")
        return

    # Display results
    section_header("Shelf-Life Results")

    sl_df = pd.DataFrame(shelf_lives)
    if "Shelf Life" in sl_df.columns:
        valid_sl = sl_df[sl_df["Shelf Life"] != "Beyond data range"]
        numeric_sl = pd.to_numeric(valid_sl["Shelf Life"], errors="coerce").dropna()

        if len(numeric_sl) > 0:
            min_sl = numeric_sl.min()
            if len(shelf_lives) > 1:
                st.metric(
                    "Estimated Shelf Life (worst-case batch)",
                    f"{min_sl:.1f} {time_col}",
                )
            else:
                st.metric(
                    "Estimated Shelf Life",
                    f"{min_sl:.1f} {time_col}",
                )

        st.dataframe(sl_df, use_container_width=True, hide_index=True)

    help_tip(
        "Shelf-Life Estimation",
        "Shelf life is the time at which the **95% one-sided confidence interval** "
        "of the mean regression line crosses the specification limit.\n\n"
        "This uses `summary_frame(alpha=0.10)` to obtain the 90% two-sided CI, "
        "which is equivalent to a 95% one-sided bound. The lower CI bound is "
        "compared against the LSL and the upper CI bound against the USL.\n\n"
        "When multiple batches are analysed separately, the minimum shelf life "
        "across batches is reported as the worst case.",
    )


def _estimate_single_shelf_life(
    data: pd.DataFrame,
    time_col: str,
    resp_col: str,
    batch_label: str,
    lsl: float | None,
    usl: float | None,
) -> dict | None:
    """Estimate shelf life for a single batch (or pooled data) using OLS + CI."""
    x = data[time_col].values.astype(float)
    y = data[resp_col].values.astype(float)

    if len(x) < 3:
        return None

    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    # Generate prediction range extending well beyond observed data
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min
    x_pred = np.linspace(x_min, x_max + 2 * max(x_range, 1), 500)
    X_pred = sm.add_constant(x_pred)

    try:
        pred = model.get_prediction(X_pred)
        # alpha=0.10 for 90% two-sided = 95% one-sided
        summary = pred.summary_frame(alpha=0.10)
    except Exception:
        return {"Batch": batch_label, "Shelf Life": "Error"}

    ci_lower = summary["obs_ci_lower"].values
    ci_upper = summary["obs_ci_upper"].values
    mean_pred = summary["mean"].values

    shelf_life = None

    # Check LSL crossing (lower CI drops below LSL)
    if lsl is not None:
        crossings = np.where(ci_lower < lsl)[0]
        if len(crossings) > 0:
            idx = crossings[0]
            shelf_life = x_pred[idx]

    # Check USL crossing (upper CI rises above USL)
    if usl is not None:
        crossings = np.where(ci_upper > usl)[0]
        if len(crossings) > 0:
            idx = crossings[0]
            sl_usl = x_pred[idx]
            if shelf_life is None or sl_usl < shelf_life:
                shelf_life = sl_usl

    # Plot
    fig = go.Figure()

    # CI band
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_pred, x_pred[::-1]]),
        y=np.concatenate([ci_upper, ci_lower[::-1]]),
        fill="toself",
        fillcolor="rgba(99,102,241,0.15)",
        line=dict(width=0),
        name="95% One-sided CI",
        hoverinfo="skip",
    ))

    # Regression line
    fig.add_trace(go.Scatter(
        x=x_pred, y=mean_pred, mode="lines",
        name="Mean Prediction",
        line=dict(color=_COLORS[0], width=2),
    ))

    # Data points
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        name=f"{batch_label} (data)",
        marker=dict(color=_COLORS[0], size=7),
    ))

    # Spec limits
    if lsl is not None:
        fig.add_hline(y=lsl, line_dash="dash", line_color="#ef4444",
                      annotation_text="LSL", annotation_position="top left")
    if usl is not None:
        fig.add_hline(y=usl, line_dash="dash", line_color="#ef4444",
                      annotation_text="USL", annotation_position="bottom left")

    # Mark intersection
    if shelf_life is not None:
        fig.add_vline(x=shelf_life, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"Shelf Life = {shelf_life:.1f}",
                      annotation_position="top right")
        fig.add_trace(go.Scatter(
            x=[shelf_life], y=[model.params[0] + model.params[1] * shelf_life],
            mode="markers",
            marker=dict(color="#f59e0b", size=12, symbol="diamond"),
            name="Shelf-Life Point",
            showlegend=True,
        ))

    fig.update_layout(
        template="plotly+rdl",
        title=f"Shelf-Life Estimation: {batch_label}",
        xaxis_title=time_col,
        yaxis_title=resp_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    if shelf_life is not None:
        return {"Batch": batch_label, "Shelf Life": round(shelf_life, 2)}
    return {"Batch": batch_label, "Shelf Life": "Beyond data range"}


def _shelf_life_from_formula_model(
    model,
    safe_df: pd.DataFrame,
    t_col: str,
    r_col: str,
    b_col: str,
    batch: str,
    orig_time_col: str,
    lsl: float | None,
    usl: float | None,
) -> float | None:
    """Compute shelf life for a specific batch from a common-slope formula model."""
    batch_data = safe_df[safe_df[b_col] == batch] if b_col in safe_df.columns else safe_df
    x_vals = batch_data[t_col].values.astype(float) if t_col in batch_data.columns else batch_data.iloc[:, 0].values.astype(float)

    if len(x_vals) < 2:
        return None

    x_min, x_max = x_vals.min(), x_vals.max()
    x_range = x_max - x_min
    x_pred = np.linspace(x_min, x_max + 2 * max(x_range, 1), 500)

    # Build prediction dataframe matching the model formula
    pred_df = pd.DataFrame({t_col: x_pred})
    pred_df[b_col] = batch

    try:
        pred = model.get_prediction(pred_df)
        summary = pred.summary_frame(alpha=0.10)
    except Exception:
        return None

    ci_lower = summary["obs_ci_lower"].values
    ci_upper = summary["obs_ci_upper"].values

    shelf_life = None

    if lsl is not None:
        crossings = np.where(ci_lower < lsl)[0]
        if len(crossings) > 0:
            shelf_life = x_pred[crossings[0]]

    if usl is not None:
        crossings = np.where(ci_upper > usl)[0]
        if len(crossings) > 0:
            sl_usl = x_pred[crossings[0]]
            if shelf_life is None or sl_usl < shelf_life:
                shelf_life = sl_usl

    return round(shelf_life, 2) if shelf_life is not None else None


# ---------------------------------------------------------------------------
# Tab 4: Multi-Attribute Dashboard
# ---------------------------------------------------------------------------

def _render_multi_attribute(df: pd.DataFrame):
    section_header(
        "Multi-Attribute Dashboard",
        "View degradation trends for multiple quality attributes side-by-side "
        "and identify which attributes are approaching specification limits.",
    )

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)

    if len(num_cols) < 2:
        empty_state(
            "Need at least 2 numeric columns (time + responses).",
            "Ensure your dataset has a numeric time column and at least one response.",
        )
        return

    time_col = st.selectbox(
        "Time column:", num_cols, key="stability_ma_time",
    )
    attr_options = [c for c in num_cols if c != time_col]
    attr_cols = st.multiselect(
        "Attribute columns:",
        attr_options,
        default=attr_options[:min(4, len(attr_options))],
        key="stability_ma_attrs",
    )

    batch_col = None
    if cat_cols:
        batch_col = st.selectbox(
            "Batch column (optional):",
            ["(none)"] + cat_cols,
            key="stability_ma_batch",
        )
        if batch_col == "(none)":
            batch_col = None

    # Per-attribute spec limits
    spec_limits: dict[str, dict] = {}
    if attr_cols:
        with st.expander("Specification Limits (per attribute)", expanded=False):
            for attr in attr_cols:
                c1, c2 = st.columns(2)
                attr_lsl = c1.number_input(
                    f"{attr} - LSL:", value=None,
                    key=f"stability_ma_lsl_{attr}",
                )
                attr_usl = c2.number_input(
                    f"{attr} - USL:", value=None,
                    key=f"stability_ma_usl_{attr}",
                )
                spec_limits[attr] = {"lsl": attr_lsl, "usl": attr_usl}

    if not attr_cols:
        empty_state("Select at least one attribute column.")
        return

    if st.button("Generate Dashboard", key="stability_ma_run"):
        with st.spinner("Building multi-attribute dashboard..."):
            _multi_attribute_analysis(df, time_col, attr_cols, batch_col, spec_limits)


def _multi_attribute_analysis(
    df: pd.DataFrame,
    time_col: str,
    attr_cols: list[str],
    batch_col: str | None,
    spec_limits: dict,
):
    """Grid of trend plots and summary table for multiple attributes."""
    n_attrs = len(attr_cols)
    n_cols = min(n_attrs, 2)
    n_rows = (n_attrs + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=attr_cols,
        shared_xaxes=True,
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    summary_rows: list[dict] = []

    for idx, attr in enumerate(attr_cols):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        work = df[[time_col, attr]].copy()
        if batch_col:
            work[batch_col] = df[batch_col].astype(str)
        work = work.dropna(subset=[time_col, attr])

        if len(work) < 3:
            summary_rows.append({
                "Attribute": attr,
                "Slope": np.nan,
                "R\u00b2": np.nan,
                "Projected Time to Spec": "Insufficient data",
            })
            continue

        batches = work[batch_col].unique().tolist() if batch_col else ["All"]

        for b_idx, batch in enumerate(batches):
            color = _COLORS[b_idx % len(_COLORS)]
            subset = work[work[batch_col] == batch] if batch_col else work
            x = subset[time_col].values.astype(float)
            y = subset[attr].values.astype(float)

            if len(x) < 2:
                continue

            X = sm.add_constant(x)
            model = sm.OLS(y, X).fit()
            slope = model.params[1]
            intercept = model.params[0]
            r2 = model.rsquared

            # Data points
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(color=color, size=5),
                name=f"{batch}" if batch != "All" else attr,
                showlegend=(idx == 0),
                legendgroup=batch,
            ), row=row, col=col)

            # Regression line
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = intercept + slope * x_line
            fig.add_trace(go.Scatter(
                x=x_line, y=y_line, mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                legendgroup=batch,
            ), row=row, col=col)

            # Projected time to spec
            attr_lsl = spec_limits.get(attr, {}).get("lsl")
            attr_usl = spec_limits.get(attr, {}).get("usl")
            proj_time = _projected_time_to_spec(slope, intercept, attr_lsl, attr_usl, x.max())

            if b_idx == 0 or batch == "All":
                summary_rows.append({
                    "Attribute": attr if batch == "All" else f"{attr} ({batch})",
                    "Slope": round(slope, 6),
                    "R\u00b2": round(r2, 4),
                    "Projected Time to Spec": proj_time,
                })

        # Spec limit lines on subplot
        attr_lsl = spec_limits.get(attr, {}).get("lsl")
        attr_usl = spec_limits.get(attr, {}).get("usl")
        x_range = work[time_col].agg(["min", "max"]).tolist()

        if attr_lsl is not None:
            fig.add_trace(go.Scatter(
                x=x_range, y=[attr_lsl, attr_lsl],
                mode="lines", line=dict(color="#ef4444", dash="dash", width=1),
                showlegend=False,
            ), row=row, col=col)
        if attr_usl is not None:
            fig.add_trace(go.Scatter(
                x=x_range, y=[attr_usl, attr_usl],
                mode="lines", line=dict(color="#ef4444", dash="dash", width=1),
                showlegend=False,
            ), row=row, col=col)

    fig.update_layout(
        template="plotly+rdl",
        height=300 * n_rows,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    # Label shared x-axis on bottom row only
    for c in range(1, n_cols + 1):
        fig.update_xaxes(title_text=time_col, row=n_rows, col=c)

    st.plotly_chart(fig, use_container_width=True)

    # Summary table with color coding
    if summary_rows:
        section_header("Attribute Summary")
        sum_df = pd.DataFrame(summary_rows)

        def _style_projection(val):
            if isinstance(val, str):
                if "approaching" in val.lower() or "within" in val.lower():
                    return "background-color: rgba(239,68,68,0.15); color: #b91c1c;"
                if "beyond" in val.lower() or "stable" in val.lower():
                    return "background-color: rgba(34,197,94,0.15); color: #166534;"
            return ""

        styled = sum_df.style.map(
            _style_projection, subset=["Projected Time to Spec"]
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    help_tip(
        "Multi-Attribute Dashboard",
        "This view provides a quick overview of all quality attributes on a "
        "single time axis. The summary table shows the rate of change (slope) "
        "and projected time to hit specification limits for each attribute.\n\n"
        "Rows highlighted in red are approaching limits within the observed "
        "time range; green rows are stable or far from limits.",
    )


def _projected_time_to_spec(
    slope: float,
    intercept: float,
    lsl: float | None,
    usl: float | None,
    current_max_time: float,
) -> str:
    """Project when the mean trend line will cross a spec limit."""
    if abs(slope) < 1e-12:
        return "Stable (no trend)"

    crossings: list[float] = []

    if lsl is not None:
        # intercept + slope * t = lsl  =>  t = (lsl - intercept) / slope
        t_cross = (lsl - intercept) / slope
        if t_cross > 0:
            crossings.append(t_cross)

    if usl is not None:
        t_cross = (usl - intercept) / slope
        if t_cross > 0:
            crossings.append(t_cross)

    if not crossings:
        return "Beyond foreseeable range"

    earliest = min(crossings)
    if earliest <= current_max_time:
        return f"Within data range ({earliest:.1f})"
    elif earliest <= current_max_time * 2:
        return f"Approaching limit ({earliest:.1f})"
    else:
        return f"Beyond 2x observed ({earliest:.1f})"
