"""
Quality / SPC Module - Statistical Process Control charts and Process Capability analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import section_header, empty_state, help_tip, validation_panel, interpretation_card
from modules.validation import check_normality, interpret_capability


# ---------------------------------------------------------------------------
# SPC constants for subgroup sizes n = 2..10
# ---------------------------------------------------------------------------
SPC_CONSTANTS = {
    2:  {'A2': 1.880, 'A3': 2.659, 'B3': 0,     'B4': 3.267, 'D3': 0,     'D4': 3.267, 'd2': 1.128},
    3:  {'A2': 1.023, 'A3': 1.954, 'B3': 0,     'B4': 2.568, 'D3': 0,     'D4': 2.574, 'd2': 1.693},
    4:  {'A2': 0.729, 'A3': 1.628, 'B3': 0,     'B4': 2.266, 'D3': 0,     'D4': 2.282, 'd2': 2.059},
    5:  {'A2': 0.577, 'A3': 1.427, 'B3': 0,     'B4': 2.089, 'D3': 0,     'D4': 2.114, 'd2': 2.326},
    6:  {'A2': 0.483, 'A3': 1.287, 'B3': 0.030, 'B4': 1.970, 'D3': 0,     'D4': 2.004, 'd2': 2.534},
    7:  {'A2': 0.419, 'A3': 1.182, 'B3': 0.118, 'B4': 1.882, 'D3': 0.076, 'D4': 1.924, 'd2': 2.704},
    8:  {'A2': 0.373, 'A3': 1.099, 'B3': 0.185, 'B4': 1.815, 'D3': 0.136, 'D4': 1.864, 'd2': 2.847},
    9:  {'A2': 0.337, 'A3': 1.032, 'B3': 0.239, 'B4': 1.761, 'D3': 0.184, 'D4': 1.816, 'd2': 2.970},
    10: {'A2': 0.308, 'A3': 0.975, 'B3': 0.284, 'B4': 1.716, 'D3': 0.223, 'D4': 1.777, 'd2': 3.078},
}


# ---------------------------------------------------------------------------
# Western Electric Rules
# ---------------------------------------------------------------------------

def _western_electric_rules(values, cl, ucl, lcl):
    """Apply Western Electric rules and return a boolean array of violations.

    Rules implemented:
      1. One point beyond 3-sigma (UCL/LCL).
      2. Two of three consecutive points beyond 2-sigma on the same side.
      3. Four of five consecutive points beyond 1-sigma on the same side.
      4. Eight consecutive points on one side of the centre line.
    """
    n = len(values)
    violations = np.zeros(n, dtype=bool)
    sigma = (ucl - cl) / 3.0 if ucl != cl else 1.0

    one_sigma_upper = cl + sigma
    one_sigma_lower = cl - sigma
    two_sigma_upper = cl + 2 * sigma
    two_sigma_lower = cl - 2 * sigma

    for i in range(n):
        # Rule 1 -- beyond 3-sigma
        if values[i] > ucl or values[i] < lcl:
            violations[i] = True

        # Rule 2 -- 2 of 3 beyond 2-sigma (same side)
        if i >= 2:
            window = values[i - 2: i + 1]
            if np.sum(window > two_sigma_upper) >= 2 or np.sum(window < two_sigma_lower) >= 2:
                violations[i - 2: i + 1] = True

        # Rule 3 -- 4 of 5 beyond 1-sigma (same side)
        if i >= 4:
            window = values[i - 4: i + 1]
            if np.sum(window > one_sigma_upper) >= 4 or np.sum(window < one_sigma_lower) >= 4:
                violations[i - 4: i + 1] = True

        # Rule 4 -- 8 consecutive on one side
        if i >= 7:
            window = values[i - 7: i + 1]
            if np.all(window > cl) or np.all(window < cl):
                violations[i - 7: i + 1] = True

    return violations


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_control_chart(fig, x, values, cl, ucl, lcl, violations, title,
                       row=1, col=1, y_label="Value"):
    """Add a control chart trace set to a plotly subplot."""
    sigma = (ucl - cl) / 3.0 if ucl != cl else 0.0

    # Data trace (normal points)
    normal_mask = ~violations
    fig.add_trace(go.Scatter(
        x=x[normal_mask], y=values[normal_mask], mode="markers+lines",
        marker=dict(size=5),
        line=dict(width=1),
        name=title, showlegend=False,
    ), row=row, col=col)

    # Violation points
    if violations.any():
        fig.add_trace(go.Scatter(
            x=x[violations], y=values[violations], mode="markers",
            marker=dict(color="red", size=8, symbol="x"),
            name="OOC", showlegend=False,
        ), row=row, col=col)

    # Control lines
    fig.add_hline(y=cl, line_dash="solid", line_color="green",
                  annotation_text=f"CL={cl:.4f}", row=row, col=col)
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL={ucl:.4f}", row=row, col=col)
    fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                  annotation_text=f"LCL={lcl:.4f}", row=row, col=col)

    # Zone lines (1-sigma, 2-sigma)
    if sigma > 0:
        for mult in [1, 2]:
            fig.add_hline(y=cl + mult * sigma, line_dash="dot",
                          line_color="rgba(150,150,150,0.4)", row=row, col=col)
            fig.add_hline(y=cl - mult * sigma, line_dash="dot",
                          line_color="rgba(150,150,150,0.4)", row=row, col=col)


# ===================================================================
# Public entry point
# ===================================================================

def render_quality(df: pd.DataFrame):
    """Render quality / SPC interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    tabs = st.tabs([
        "Variables Charts", "Attributes Charts", "Process Capability",
        "Multivariate Charts", "Gage R&R / MSA", "Acceptance Sampling",
        "Multi-Vari Analysis", "Fishbone Diagram",
    ])

    with tabs[0]:
        _render_variables_charts(df)
    with tabs[1]:
        _render_attributes_charts(df)
    with tabs[2]:
        _render_process_capability(df)
    with tabs[3]:
        _render_multivariate_charts(df)
    with tabs[4]:
        _render_gage_rr(df)
    with tabs[5]:
        _render_acceptance_sampling(df)
    with tabs[6]:
        _render_multi_vari(df)
    with tabs[7]:
        _render_fishbone(df)


# ===================================================================
# Tab 1 -- Variables Charts (I-MR, X-bar & R, X-bar & S)
# ===================================================================

def _render_variables_charts(df: pd.DataFrame):
    """I-MR, X-bar & R, X-bar & S control charts."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
        return

    col_name = st.selectbox("Measurement column:", num_cols, key="spc_var_col")

    chart_type = st.selectbox("Chart type:", [
        "I-MR (Individuals & Moving Range)",
        "X-bar & R",
        "X-bar & S",
        "EWMA",
        "CUSUM",
    ], key="spc_var_chart_type")

    # Subgroup configuration
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    subgroup_col = None
    subgroup_size = 5

    if chart_type not in ("I-MR (Individuals & Moving Range)", "EWMA", "CUSUM"):
        sg_mode = st.radio(
            "Define subgroups by:",
            ["Fixed subgroup size", "Subgroup column"],
            horizontal=True, key="spc_sg_mode",
        )
        if sg_mode == "Subgroup column" and cat_cols:
            subgroup_col = st.selectbox("Subgroup column:", cat_cols, key="spc_sg_col")
        else:
            subgroup_size = st.slider("Subgroup size (n):", 2, 10, 5, key="spc_sg_size")
    elif chart_type == "EWMA":
        st.caption("EWMA uses individual observations with exponential weighting.")
    elif chart_type == "CUSUM":
        st.caption("CUSUM uses individual observations to detect small sustained shifts.")
    else:
        st.caption("I-MR uses individual observations (subgroup size = 1).")

    if st.button("Generate Chart", key="spc_var_generate"):
        with st.spinner("Generating control chart..."):
            data = df[col_name].dropna().values

            if chart_type == "I-MR (Individuals & Moving Range)":
                _imr_chart(data, col_name)
            elif chart_type == "X-bar & R":
                _xbar_r_chart(data, col_name, subgroup_col, subgroup_size, df)
            elif chart_type == "X-bar & S":
                _xbar_s_chart(data, col_name, subgroup_col, subgroup_size, df)
            elif chart_type == "EWMA":
                _ewma_chart(data, col_name)
            elif chart_type == "CUSUM":
                _cusum_chart(data, col_name)


def _build_subgroups(data, df, col_name, subgroup_col, subgroup_size):
    """Split data into subgroups.  Returns list of arrays."""
    if subgroup_col is not None:
        groups = df.groupby(subgroup_col)[col_name].apply(lambda s: s.dropna().values)
        return [g for g in groups if len(g) > 0]
    # Fixed-size subgroups
    n = len(data)
    return [data[i:i + subgroup_size] for i in range(0, n - subgroup_size + 1, subgroup_size)]


def _imr_chart(data, col_name):
    """Individuals & Moving Range chart."""
    n = len(data)
    if n < 2:
        empty_state("Need at least 2 observations for I-MR chart.")
        return

    mr = np.abs(np.diff(data))
    mr_bar = np.mean(mr)
    d2 = SPC_CONSTANTS[2]['d2']
    D3 = SPC_CONSTANTS[2]['D3']
    D4 = SPC_CONSTANTS[2]['D4']

    x_bar = np.mean(data)
    i_ucl = x_bar + 3 * mr_bar / d2
    i_lcl = x_bar - 3 * mr_bar / d2

    mr_ucl = D4 * mr_bar
    mr_lcl = D3 * mr_bar

    x_idx = np.arange(1, n + 1)
    mr_idx = np.arange(2, n + 1)

    i_violations = _western_electric_rules(data, x_bar, i_ucl, i_lcl)
    mr_violations = _western_electric_rules(mr, mr_bar, mr_ucl, mr_lcl)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"Individuals Chart: {col_name}",
                                        f"Moving Range Chart: {col_name}"),
                        vertical_spacing=0.12)

    _add_control_chart(fig, x_idx, data, x_bar, i_ucl, i_lcl,
                       i_violations, "Individuals", row=1, col=1)
    _add_control_chart(fig, mr_idx, mr, mr_bar, mr_ucl, mr_lcl,
                       mr_violations, "MR", row=2, col=1)

    fig.update_layout(height=700, title_text=f"I-MR Chart: {col_name}")
    fig.update_xaxes(title_text="Observation", row=2, col=1)
    fig.update_yaxes(title_text="Individual Value", row=1, col=1)
    fig.update_yaxes(title_text="Moving Range", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(data, x_idx, i_violations, "Individuals")
    _show_ooc_summary(mr, mr_idx, mr_violations, "Moving Range")


def _xbar_r_chart(data, col_name, subgroup_col, subgroup_size, df):
    """X-bar & R chart."""
    subgroups = _build_subgroups(data, df, col_name, subgroup_col, subgroup_size)
    if len(subgroups) < 2:
        empty_state("Need at least 2 subgroups.")
        return

    n = len(subgroups[0])
    if n < 2 or n > 10:
        st.warning("Subgroup size must be between 2 and 10.")
        return

    means = np.array([np.mean(g) for g in subgroups])
    ranges = np.array([np.ptp(g) for g in subgroups])

    x_bar_bar = np.mean(means)
    r_bar = np.mean(ranges)

    A2 = SPC_CONSTANTS[n]['A2']
    D3 = SPC_CONSTANTS[n]['D3']
    D4 = SPC_CONSTANTS[n]['D4']

    xbar_ucl = x_bar_bar + A2 * r_bar
    xbar_lcl = x_bar_bar - A2 * r_bar
    r_ucl = D4 * r_bar
    r_lcl = D3 * r_bar

    sg_idx = np.arange(1, len(subgroups) + 1)

    xbar_v = _western_electric_rules(means, x_bar_bar, xbar_ucl, xbar_lcl)
    r_v = _western_electric_rules(ranges, r_bar, r_ucl, r_lcl)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"X-bar Chart: {col_name} (n={n})",
                                        f"R Chart: {col_name} (n={n})"),
                        vertical_spacing=0.12)

    _add_control_chart(fig, sg_idx, means, x_bar_bar, xbar_ucl, xbar_lcl,
                       xbar_v, "X-bar", row=1, col=1)
    _add_control_chart(fig, sg_idx, ranges, r_bar, r_ucl, r_lcl,
                       r_v, "R", row=2, col=1)

    fig.update_layout(height=700, title_text=f"X-bar & R Chart: {col_name}")
    fig.update_xaxes(title_text="Subgroup", row=2, col=1)
    fig.update_yaxes(title_text="Subgroup Mean", row=1, col=1)
    fig.update_yaxes(title_text="Range", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(means, sg_idx, xbar_v, "X-bar")
    _show_ooc_summary(ranges, sg_idx, r_v, "R")


def _xbar_s_chart(data, col_name, subgroup_col, subgroup_size, df):
    """X-bar & S chart."""
    subgroups = _build_subgroups(data, df, col_name, subgroup_col, subgroup_size)
    if len(subgroups) < 2:
        empty_state("Need at least 2 subgroups.")
        return

    n = len(subgroups[0])
    if n < 2 or n > 10:
        st.warning("Subgroup size must be between 2 and 10.")
        return

    means = np.array([np.mean(g) for g in subgroups])
    stds = np.array([np.std(g, ddof=1) for g in subgroups])

    x_bar_bar = np.mean(means)
    s_bar = np.mean(stds)

    A3 = SPC_CONSTANTS[n]['A3']
    B3 = SPC_CONSTANTS[n]['B3']
    B4 = SPC_CONSTANTS[n]['B4']

    xbar_ucl = x_bar_bar + A3 * s_bar
    xbar_lcl = x_bar_bar - A3 * s_bar
    s_ucl = B4 * s_bar
    s_lcl = B3 * s_bar

    sg_idx = np.arange(1, len(subgroups) + 1)

    xbar_v = _western_electric_rules(means, x_bar_bar, xbar_ucl, xbar_lcl)
    s_v = _western_electric_rules(stds, s_bar, s_ucl, s_lcl)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"X-bar Chart: {col_name} (n={n})",
                                        f"S Chart: {col_name} (n={n})"),
                        vertical_spacing=0.12)

    _add_control_chart(fig, sg_idx, means, x_bar_bar, xbar_ucl, xbar_lcl,
                       xbar_v, "X-bar", row=1, col=1)
    _add_control_chart(fig, sg_idx, stds, s_bar, s_ucl, s_lcl,
                       s_v, "S", row=2, col=1)

    fig.update_layout(height=700, title_text=f"X-bar & S Chart: {col_name}")
    fig.update_xaxes(title_text="Subgroup", row=2, col=1)
    fig.update_yaxes(title_text="Subgroup Mean", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(means, sg_idx, xbar_v, "X-bar")
    _show_ooc_summary(stds, sg_idx, s_v, "S")


def _show_ooc_summary(values, indices, violations, chart_label):
    """Display out-of-control summary in an expander."""
    ooc_count = int(violations.sum())
    total = len(values)
    with st.expander(f"{chart_label}: {ooc_count} / {total} out-of-control points"):
        if ooc_count == 0:
            st.success("All points are in control.")
        else:
            ooc_df = pd.DataFrame({
                "Index": indices[violations],
                "Value": values[violations],
            })
            st.dataframe(ooc_df, use_container_width=True, hide_index=True)


def _ewma_chart(data, col_name):
    """EWMA (Exponentially Weighted Moving Average) chart."""
    n = len(data)
    if n < 2:
        empty_state("Need at least 2 observations for EWMA chart.")
        return

    c1, c2 = st.columns(2)
    lam = c1.slider("Lambda (\u03bb):", 0.05, 0.40, 0.20, 0.05, key="ewma_lambda")
    L = c2.slider("Control limit width (L):", 2.0, 3.5, 2.7, 0.1, key="ewma_L")

    mu = np.mean(data)
    sigma = np.std(data, ddof=1)

    # Compute EWMA
    z = np.zeros(n)
    z[0] = lam * data[0] + (1 - lam) * mu
    for i in range(1, n):
        z[i] = lam * data[i] + (1 - lam) * z[i - 1]

    # Control limits (exact, time-varying)
    idx = np.arange(1, n + 1)
    factor = sigma * L * np.sqrt(lam / (2 - lam) * (1 - (1 - lam) ** (2 * idx)))
    ucl = mu + factor
    lcl = mu - factor

    violations = (z > ucl) | (z < lcl)

    fig = go.Figure()
    normal = ~violations
    fig.add_trace(go.Scatter(x=idx[normal], y=z[normal], mode="markers+lines",
                             marker=dict(size=5), line=dict(width=1), name="EWMA"))
    if violations.any():
        fig.add_trace(go.Scatter(x=idx[violations], y=z[violations], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"), name="OOC"))
    fig.add_trace(go.Scatter(x=idx, y=ucl, mode="lines",
                             line=dict(color="red", dash="dash", width=1), name="UCL"))
    fig.add_trace(go.Scatter(x=idx, y=lcl, mode="lines",
                             line=dict(color="red", dash="dash", width=1), name="LCL"))
    fig.add_hline(y=mu, line_dash="solid", line_color="green",
                  annotation_text=f"CL={mu:.4f}")

    fig.update_layout(title=f"EWMA Chart: {col_name} (\u03bb={lam}, L={L})", height=500,
                      xaxis_title="Observation", yaxis_title="EWMA")
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(z, idx, violations, "EWMA")

    with st.expander("EWMA Parameters"):
        st.write(f"**Process mean (\u03bc\u2080):** {mu:.4f}")
        st.write(f"**Process \u03c3:** {sigma:.4f}")
        st.write(f"**\u03bb:** {lam}")
        st.write(f"**L:** {L}")
        st.write(f"**Steady-state UCL:** {mu + L * sigma * np.sqrt(lam / (2 - lam)):.4f}")
        st.write(f"**Steady-state LCL:** {mu - L * sigma * np.sqrt(lam / (2 - lam)):.4f}")


def _cusum_chart(data, col_name):
    """Tabular CUSUM chart for detecting shifts from target."""
    n = len(data)
    if n < 2:
        empty_state("Need at least 2 observations for CUSUM chart.")
        return

    mu = np.mean(data)
    sigma = np.std(data, ddof=1)

    c1, c2, c3 = st.columns(3)
    target = c1.number_input("Target mean (\u03bc\u2080):", value=float(mu), key="cusum_target")
    K = c2.number_input("Slack value K (in \u03c3 units):", value=0.5, min_value=0.01, key="cusum_K")
    H = c3.number_input("Decision interval H (in \u03c3 units):", value=5.0, min_value=0.1, key="cusum_H")

    k = K * sigma
    h = H * sigma

    # Upper and lower CUSUM
    S_H = np.zeros(n)  # upper
    S_L = np.zeros(n)  # lower
    for i in range(n):
        if i == 0:
            S_H[i] = max(0, data[i] - (target + k))
            S_L[i] = max(0, (target - k) - data[i])
        else:
            S_H[i] = max(0, data[i] - (target + k) + S_H[i - 1])
            S_L[i] = max(0, (target - k) - data[i] + S_L[i - 1])

    idx = np.arange(1, n + 1)
    upper_signal = S_H > h
    lower_signal = S_L > h

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(f"Upper CUSUM (C\u207a): {col_name}",
                                        f"Lower CUSUM (C\u207b): {col_name}"),
                        vertical_spacing=0.12)

    # Upper CUSUM
    normal_u = ~upper_signal
    fig.add_trace(go.Scatter(x=idx[normal_u], y=S_H[normal_u], mode="markers+lines",
                             marker=dict(size=5), line=dict(width=1), name="C\u207a",
                             showlegend=True), row=1, col=1)
    if upper_signal.any():
        fig.add_trace(go.Scatter(x=idx[upper_signal], y=S_H[upper_signal], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"),
                                 name="Upper Signal", showlegend=True), row=1, col=1)
    fig.add_hline(y=h, line_dash="dash", line_color="red",
                  annotation_text=f"H={h:.4f}", row=1, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="green", row=1, col=1)

    # Lower CUSUM
    normal_l = ~lower_signal
    fig.add_trace(go.Scatter(x=idx[normal_l], y=S_L[normal_l], mode="markers+lines",
                             marker=dict(size=5), line=dict(width=1), name="C\u207b",
                             showlegend=True), row=2, col=1)
    if lower_signal.any():
        fig.add_trace(go.Scatter(x=idx[lower_signal], y=S_L[lower_signal], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"),
                                 name="Lower Signal", showlegend=True), row=2, col=1)
    fig.add_hline(y=h, line_dash="dash", line_color="red",
                  annotation_text=f"H={h:.4f}", row=2, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="green", row=2, col=1)

    fig.update_layout(height=700, title_text=f"CUSUM Chart: {col_name}")
    fig.update_xaxes(title_text="Observation", row=2, col=1)
    fig.update_yaxes(title_text="C\u207a", row=1, col=1)
    fig.update_yaxes(title_text="C\u207b", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    total_signals = int(upper_signal.sum() + lower_signal.sum())
    st.write(f"**Signals detected:** {total_signals} ({int(upper_signal.sum())} upper, {int(lower_signal.sum())} lower)")

    with st.expander("CUSUM Parameters"):
        st.write(f"**Target (\u03bc\u2080):** {target:.4f}")
        st.write(f"**Process \u03c3:** {sigma:.4f}")
        st.write(f"**K (slack):** {K}\u03c3 = {k:.4f}")
        st.write(f"**H (decision interval):** {H}\u03c3 = {h:.4f}")


# ===================================================================
# Tab 2 -- Attributes Charts (p, np, c, u)
# ===================================================================

def _render_attributes_charts(df: pd.DataFrame):
    """p-chart, np-chart, c-chart, u-chart."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
        return

    chart_type = st.selectbox("Attributes chart type:", [
        "p-chart (proportion defective)",
        "np-chart (number defective)",
        "c-chart (defects per unit)",
        "u-chart (defects per unit, variable sample)",
        "g-chart (count between events)",
        "h-chart (time between events)",
    ], key="spc_attr_type")

    if chart_type.startswith("p") or chart_type.startswith("np"):
        defectives_col = st.selectbox("Defectives (count) column:", num_cols,
                                      key="spc_attr_def_col")
        sample_col = st.selectbox("Sample size column:", num_cols,
                                  key="spc_attr_n_col")

        if st.button("Generate Chart", key="spc_attr_generate"):
            defectives = df[defectives_col].dropna().values.astype(float)
            sample_n = df[sample_col].dropna().values.astype(float)
            k = min(len(defectives), len(sample_n))
            defectives = defectives[:k]
            sample_n = sample_n[:k]

            if chart_type.startswith("p"):
                _p_chart(defectives, sample_n)
            else:
                _np_chart(defectives, sample_n)

    elif chart_type.startswith("c"):
        defects_col = st.selectbox("Defects count column:", num_cols,
                                   key="spc_attr_c_col")
        if st.button("Generate Chart", key="spc_attr_c_generate"):
            defects = df[defects_col].dropna().values.astype(float)
            _c_chart(defects)

    elif chart_type.startswith("u"):
        defects_col = st.selectbox("Defects column:", num_cols,
                                   key="spc_attr_u_def_col")
        units_col = st.selectbox("Inspection units column:", num_cols,
                                 key="spc_attr_u_units_col")
        if st.button("Generate Chart", key="spc_attr_u_generate"):
            defects = df[defects_col].dropna().values.astype(float)
            units = df[units_col].dropna().values.astype(float)
            k = min(len(defects), len(units))
            _u_chart(defects[:k], units[:k])

    elif chart_type.startswith("g"):
        events_col = st.selectbox("Count between events column:", num_cols,
                                   key="spc_attr_g_col")
        if st.button("Generate Chart", key="spc_attr_g_generate"):
            counts = df[events_col].dropna().values.astype(float)
            _g_chart(counts)

    elif chart_type.startswith("h"):
        time_col = st.selectbox("Time between events column:", num_cols,
                                 key="spc_attr_h_col")
        if st.button("Generate Chart", key="spc_attr_h_generate"):
            times = df[time_col].dropna().values.astype(float)
            _h_chart(times)


def _p_chart(defectives, sample_n):
    """Proportion defective chart."""
    p = defectives / sample_n
    p_bar = defectives.sum() / sample_n.sum()

    ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / sample_n)
    lcl = np.maximum(p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / sample_n), 0)

    idx = np.arange(1, len(p) + 1)
    violations = (p > ucl) | (p < lcl)

    fig = go.Figure()
    normal = ~violations
    fig.add_trace(go.Scatter(x=idx[normal], y=p[normal], mode="markers+lines",
                             marker=dict(size=5),
                             line=dict(width=1), name="p"))
    if violations.any():
        fig.add_trace(go.Scatter(x=idx[violations], y=p[violations], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"), name="OOC"))
    fig.add_trace(go.Scatter(x=idx, y=ucl, mode="lines",
                             line=dict(color="red", dash="dash", width=1), name="UCL"))
    fig.add_trace(go.Scatter(x=idx, y=lcl, mode="lines",
                             line=dict(color="red", dash="dash", width=1), name="LCL"))
    fig.add_hline(y=p_bar, line_dash="solid", line_color="green",
                  annotation_text=f"CL={p_bar:.4f}")

    fig.update_layout(title="p-Chart (Proportion Defective)", height=500,
                      xaxis_title="Sample", yaxis_title="Proportion")
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(p, idx, violations, "p-chart")


def _np_chart(defectives, sample_n):
    """Number defective chart (constant sample size assumed as average)."""
    n_bar = np.mean(sample_n)
    p_bar = defectives.sum() / sample_n.sum()
    np_bar = n_bar * p_bar

    ucl = np_bar + 3 * np.sqrt(np_bar * (1 - p_bar))
    lcl = max(np_bar - 3 * np.sqrt(np_bar * (1 - p_bar)), 0)

    idx = np.arange(1, len(defectives) + 1)
    violations = (defectives > ucl) | (defectives < lcl)

    fig = go.Figure()
    normal = ~violations
    fig.add_trace(go.Scatter(x=idx[normal], y=defectives[normal], mode="markers+lines",
                             marker=dict(size=5),
                             line=dict(width=1), name="np"))
    if violations.any():
        fig.add_trace(go.Scatter(x=idx[violations], y=defectives[violations], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"), name="OOC"))
    fig.add_hline(y=np_bar, line_dash="solid", line_color="green",
                  annotation_text=f"CL={np_bar:.4f}")
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL={ucl:.4f}")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                  annotation_text=f"LCL={lcl:.4f}")

    fig.update_layout(title="np-Chart (Number Defective)", height=500,
                      xaxis_title="Sample", yaxis_title="Defectives")
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(defectives, idx, violations, "np-chart")


def _c_chart(defects):
    """Defects count chart (constant opportunity)."""
    c_bar = np.mean(defects)
    ucl = c_bar + 3 * np.sqrt(c_bar)
    lcl = max(c_bar - 3 * np.sqrt(c_bar), 0)

    idx = np.arange(1, len(defects) + 1)
    violations = (defects > ucl) | (defects < lcl)

    fig = go.Figure()
    normal = ~violations
    fig.add_trace(go.Scatter(x=idx[normal], y=defects[normal], mode="markers+lines",
                             marker=dict(size=5),
                             line=dict(width=1), name="c"))
    if violations.any():
        fig.add_trace(go.Scatter(x=idx[violations], y=defects[violations], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"), name="OOC"))
    fig.add_hline(y=c_bar, line_dash="solid", line_color="green",
                  annotation_text=f"CL={c_bar:.4f}")
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL={ucl:.4f}")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                  annotation_text=f"LCL={lcl:.4f}")

    fig.update_layout(title="c-Chart (Defects per Unit)", height=500,
                      xaxis_title="Sample", yaxis_title="Defects")
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(defects, idx, violations, "c-chart")


def _u_chart(defects, units):
    """Defects per unit chart (variable inspection size)."""
    u = defects / units
    u_bar = defects.sum() / units.sum()

    ucl = u_bar + 3 * np.sqrt(u_bar / units)
    lcl = np.maximum(u_bar - 3 * np.sqrt(u_bar / units), 0)

    idx = np.arange(1, len(u) + 1)
    violations = (u > ucl) | (u < lcl)

    fig = go.Figure()
    normal = ~violations
    fig.add_trace(go.Scatter(x=idx[normal], y=u[normal], mode="markers+lines",
                             marker=dict(size=5),
                             line=dict(width=1), name="u"))
    if violations.any():
        fig.add_trace(go.Scatter(x=idx[violations], y=u[violations], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"), name="OOC"))
    fig.add_trace(go.Scatter(x=idx, y=ucl, mode="lines",
                             line=dict(color="red", dash="dash", width=1), name="UCL"))
    fig.add_trace(go.Scatter(x=idx, y=lcl, mode="lines",
                             line=dict(color="red", dash="dash", width=1), name="LCL"))
    fig.add_hline(y=u_bar, line_dash="solid", line_color="green",
                  annotation_text=f"CL={u_bar:.4f}")

    fig.update_layout(title="u-Chart (Defects per Unit)", height=500,
                      xaxis_title="Sample", yaxis_title="Defects / Unit")
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(u, idx, violations, "u-chart")


def _g_chart(counts):
    """g-chart: count between events (geometric distribution)."""
    g_bar = np.mean(counts)
    # CL and limits from geometric distribution
    p_hat = 1 / (g_bar + 1) if g_bar > 0 else 0.5
    ucl = (1 - p_hat) / p_hat + 3 * np.sqrt((1 - p_hat) / p_hat ** 2)
    lcl = max(0, (1 - p_hat) / p_hat - 3 * np.sqrt((1 - p_hat) / p_hat ** 2))

    idx = np.arange(1, len(counts) + 1)
    violations = (counts > ucl) | (counts < lcl)

    fig = go.Figure()
    normal = ~violations
    fig.add_trace(go.Scatter(x=idx[normal], y=counts[normal], mode="markers+lines",
                             marker=dict(size=5), line=dict(width=1), name="g"))
    if violations.any():
        fig.add_trace(go.Scatter(x=idx[violations], y=counts[violations], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"), name="OOC"))
    fig.add_hline(y=g_bar, line_dash="solid", line_color="green",
                  annotation_text=f"CL={g_bar:.4f}")
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL={ucl:.4f}")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                  annotation_text=f"LCL={lcl:.4f}")
    fig.update_layout(title="g-Chart (Count Between Events)", height=500,
                      xaxis_title="Sample", yaxis_title="Count Between Events")
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(counts, idx, violations, "g-chart")


def _h_chart(times):
    """h-chart: time between events."""
    h_bar = np.mean(times)
    # Exponential distribution based limits
    ucl = h_bar + 3 * h_bar  # Approximate 3-sigma for exponential
    lcl = max(0, h_bar - 3 * h_bar)

    # Better limits using gamma distribution
    # For exponential with mean h_bar, variance = h_bar^2
    ucl = h_bar + 3 * np.sqrt(h_bar ** 2)
    lcl = max(0, h_bar - 3 * np.sqrt(h_bar ** 2))

    idx = np.arange(1, len(times) + 1)
    violations = (times > ucl) | (times < lcl)

    fig = go.Figure()
    normal = ~violations
    fig.add_trace(go.Scatter(x=idx[normal], y=times[normal], mode="markers+lines",
                             marker=dict(size=5), line=dict(width=1), name="h"))
    if violations.any():
        fig.add_trace(go.Scatter(x=idx[violations], y=times[violations], mode="markers",
                                 marker=dict(color="red", size=8, symbol="x"), name="OOC"))
    fig.add_hline(y=h_bar, line_dash="solid", line_color="green",
                  annotation_text=f"CL={h_bar:.4f}")
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL={ucl:.4f}")
    if lcl > 0:
        fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                      annotation_text=f"LCL={lcl:.4f}")
    fig.update_layout(title="h-Chart (Time Between Events)", height=500,
                      xaxis_title="Sample", yaxis_title="Time Between Events")
    st.plotly_chart(fig, use_container_width=True)

    _show_ooc_summary(times, idx, violations, "h-chart")


# ===================================================================
# Tab 3 -- Process Capability
# ===================================================================

def _render_process_capability(df: pd.DataFrame):
    """Process capability analysis: Cp, Cpk, Pp, Ppk, Cpm."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
        return

    col_name = st.selectbox("Measurement column:", num_cols, key="spc_cap_col")
    data = df[col_name].dropna().values

    c1, c2 = st.columns(2)
    lsl = c1.number_input("Lower Spec Limit (LSL):", value=float(np.mean(data) - 3 * np.std(data)),
                          key="spc_cap_lsl")
    usl = c2.number_input("Upper Spec Limit (USL):", value=float(np.mean(data) + 3 * np.std(data)),
                          key="spc_cap_usl")
    target = st.number_input("Target (optional, for Cpm):", value=float((lsl + usl) / 2),
                             key="spc_cap_target")

    if st.button("Analyse Capability", key="spc_cap_run"):
        if len(data) < 2:
            empty_state("Need at least 2 observations.")
            return
        if usl <= lsl:
            st.error("USL must be greater than LSL.")
            return

        # Normality check — capability indices assume normal data
        try:
            normality_check = check_normality(data, label=col_name)
            validation_panel([normality_check], title="Normality Assumption")
        except Exception:
            pass

        mean = np.mean(data)
        std_within = np.std(data, ddof=1)  # within / short-term proxy
        std_overall = np.std(data, ddof=0)  # overall / long-term

        # -- Potential capability (within) --
        cp = (usl - lsl) / (6 * std_within) if std_within > 0 else np.inf
        cpu = (usl - mean) / (3 * std_within) if std_within > 0 else np.inf
        cpl = (mean - lsl) / (3 * std_within) if std_within > 0 else np.inf
        cpk = min(cpu, cpl)

        # -- Performance (overall) --
        pp = (usl - lsl) / (6 * std_overall) if std_overall > 0 else np.inf
        ppu = (usl - mean) / (3 * std_overall) if std_overall > 0 else np.inf
        ppl = (mean - lsl) / (3 * std_overall) if std_overall > 0 else np.inf
        ppk = min(ppu, ppl)

        # -- Cpm (Taguchi) --
        tau = np.sqrt(std_overall ** 2 + (mean - target) ** 2)
        cpm = (usl - lsl) / (6 * tau) if tau > 0 else np.inf

        # -- Metrics display --
        section_header("Capability Indices")
        help_tip("Capability index interpretation", """
- **Cpk >= 2.0:** Six Sigma capable
- **Cpk >= 1.67:** Excellent capability
- **Cpk >= 1.33:** Capable process
- **Cpk >= 1.00:** Barely capable
- **Cpk < 1.00:** Not capable — process improvement needed
""")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cp", f"{cp:.3f}")
        m2.metric("Cpk", f"{cpk:.3f}")
        m3.metric("Pp", f"{pp:.3f}")
        m4.metric("Ppk", f"{ppk:.3f}")
        m5.metric("Cpm", f"{cpm:.3f}")

        # Capability interpretation card
        try:
            interpretation_card(interpret_capability(cpk))
        except Exception:
            pass

        # Interpretation
        def _interpret_index(name, value):
            if value >= 2.0:
                return f"**{name} = {value:.3f}** -- Six Sigma capable"
            elif value >= 1.67:
                return f"**{name} = {value:.3f}** -- Excellent"
            elif value >= 1.33:
                return f"**{name} = {value:.3f}** -- Capable"
            elif value >= 1.00:
                return f"**{name} = {value:.3f}** -- Barely capable"
            else:
                return f"**{name} = {value:.3f}** -- Not capable"

        with st.expander("Interpretation"):
            for name, val in [("Cp", cp), ("Cpk", cpk), ("Pp", pp), ("Ppk", ppk), ("Cpm", cpm)]:
                st.markdown(_interpret_index(name, val))

        # -- Detail table --
        with st.expander("Detailed Statistics"):
            detail = pd.DataFrame({
                "Statistic": ["N", "Mean", "Std Dev (within)", "Std Dev (overall)",
                              "Min", "Max", "LSL", "USL", "Target",
                              "Cp", "Cpl", "Cpu", "Cpk",
                              "Pp", "Ppl", "Ppu", "Ppk", "Cpm"],
                "Value": [len(data), mean, std_within, std_overall,
                          np.min(data), np.max(data), lsl, usl, target,
                          cp, cpl, cpu, cpk,
                          pp, ppl, ppu, ppk, cpm],
            })
            detail["Value"] = detail["Value"].apply(
                lambda v: f"{v:.6f}" if isinstance(v, float) else str(v))
            st.dataframe(detail, use_container_width=True, hide_index=True)

        # -- Sixpack-style chart --
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("I Chart", "Moving Range", "Histogram + Specs",
                            "Normal Probability Plot", "Capability Plot", "Summary"),
            vertical_spacing=0.15, horizontal_spacing=0.08,
        )

        # 1. Individuals chart (top-left)
        x_idx = np.arange(1, len(data) + 1)
        mr = np.abs(np.diff(data))
        mr_bar = np.mean(mr) if len(mr) > 0 else 0
        d2 = SPC_CONSTANTS[2]['d2']
        i_ucl = mean + 3 * mr_bar / d2
        i_lcl = mean - 3 * mr_bar / d2

        fig.add_trace(go.Scatter(x=x_idx, y=data, mode="lines+markers",
                                 marker=dict(size=3),
                                 line=dict(width=1),
                                 showlegend=False), row=1, col=1)
        fig.add_hline(y=mean, line_color="green", row=1, col=1)
        fig.add_hline(y=i_ucl, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=i_lcl, line_dash="dash", line_color="red", row=1, col=1)

        # 2. Moving Range (top-middle)
        mr_idx = np.arange(2, len(data) + 1)
        D4 = SPC_CONSTANTS[2]['D4']
        fig.add_trace(go.Scatter(x=mr_idx, y=mr, mode="lines+markers",
                                 marker=dict(size=3),
                                 line=dict(width=1),
                                 showlegend=False), row=1, col=2)
        fig.add_hline(y=mr_bar, line_color="green", row=1, col=2)
        fig.add_hline(y=D4 * mr_bar, line_dash="dash", line_color="red", row=1, col=2)

        # 3. Histogram with spec limits (top-right)
        hist_vals, bin_edges = np.histogram(data, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig.add_trace(go.Bar(x=bin_centers, y=hist_vals,
                             opacity=0.7, showlegend=False), row=1, col=3)
        # Normal overlay
        x_norm = np.linspace(min(data.min(), lsl), max(data.max(), usl), 200)
        y_norm = stats.norm.pdf(x_norm, mean, std_within) * len(data) * (bin_edges[1] - bin_edges[0])
        fig.add_trace(go.Scatter(x=x_norm, y=y_norm, mode="lines",
                                 line=dict(color="darkblue", width=2),
                                 showlegend=False), row=1, col=3)
        fig.add_vline(x=lsl, line_dash="dash", line_color="red", row=1, col=3)
        fig.add_vline(x=usl, line_dash="dash", line_color="red", row=1, col=3)
        fig.add_vline(x=target, line_dash="dot", line_color="green", row=1, col=3)

        # 4. Normal probability plot (bottom-left)
        sorted_data = np.sort(data)
        n = len(sorted_data)
        theoretical_q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        fig.add_trace(go.Scatter(x=theoretical_q, y=sorted_data, mode="markers",
                                 marker=dict(size=3),
                                 showlegend=False), row=2, col=1)
        slope, intercept = np.polyfit(theoretical_q, sorted_data, 1)
        fig.add_trace(go.Scatter(x=theoretical_q, y=slope * theoretical_q + intercept,
                                 mode="lines", line=dict(color="red", dash="dash"),
                                 showlegend=False), row=2, col=1)

        # 5. Capability bar plot (bottom-middle)
        cap_names = ["Cp", "Cpk", "Pp", "Ppk", "Cpm"]
        cap_values = [cp, cpk, pp, ppk, cpm]
        colors = ["green" if v >= 1.33 else "orange" if v >= 1.0 else "red" for v in cap_values]
        fig.add_trace(go.Bar(x=cap_names, y=cap_values, marker_color=colors,
                             showlegend=False), row=2, col=2)
        fig.add_hline(y=1.33, line_dash="dash", line_color="green", row=2, col=2)
        fig.add_hline(y=1.00, line_dash="dot", line_color="orange", row=2, col=2)

        # 6. Summary text via annotations (bottom-right) -- use invisible scatter
        summary_text = (
            f"N = {len(data)}<br>"
            f"Mean = {mean:.4f}<br>"
            f"StDev = {std_within:.4f}<br>"
            f"LSL = {lsl:.4f}<br>"
            f"USL = {usl:.4f}<br>"
            f"Cp = {cp:.3f}<br>"
            f"Cpk = {cpk:.3f}<br>"
            f"Pp = {pp:.3f}<br>"
            f"Ppk = {ppk:.3f}<br>"
            f"Cpm = {cpm:.3f}"
        )
        fig.add_trace(go.Scatter(x=[0.5], y=[0.5], mode="text",
                                 text=[summary_text], textposition="middle center",
                                 showlegend=False), row=2, col=3)
        fig.update_xaxes(visible=False, row=2, col=3)
        fig.update_yaxes(visible=False, row=2, col=3)

        fig.update_layout(height=800, title_text=f"Process Capability Sixpack: {col_name}")
        fig.update_xaxes(title_text="Observation", row=1, col=1)
        fig.update_xaxes(title_text="Observation", row=1, col=2)
        fig.update_xaxes(title_text="Value", row=1, col=3)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Individual", row=1, col=1)
        fig.update_yaxes(title_text="MR", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=3)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Index Value", row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Tab 4 -- Multivariate Charts (Hotelling T²)
# ===================================================================

def _render_multivariate_charts(df: pd.DataFrame):
    """Hotelling T² multivariate control chart."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns for multivariate charts.")
        return

    section_header("Hotelling T\u00b2 Chart")
    help_tip("Hotelling T\u00b2 Chart", "Monitors multiple correlated quality variables simultaneously. "
             "A single chart replaces multiple univariate charts when variables are correlated.")

    selected = st.multiselect("Select variables (2+):", num_cols, default=num_cols[:3],
                               key="tsq_cols")
    if len(selected) < 2:
        st.info("Select at least 2 variables.")
        return

    alpha = st.slider("Significance level (\u03b1):", 0.001, 0.10, 0.05, 0.001, key="tsq_alpha")

    if st.button("Generate T\u00b2 Chart", key="tsq_generate"):
        data = df[selected].dropna()
        n = len(data)
        p = len(selected)

        if n <= p:
            st.error(f"Need more observations ({n}) than variables ({p}).")
            return

        X = data.values
        x_bar = np.mean(X, axis=0)
        S = np.cov(X, rowvar=False)

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            st.error("Covariance matrix is singular. Check for collinear variables.")
            return

        # Compute T² for each observation
        diff = X - x_bar
        t_sq = np.array([d @ S_inv @ d for d in diff])

        # UCL from F-distribution
        ucl = p * (n + 1) * (n - 1) / (n * (n - p)) * stats.f.ppf(1 - alpha, p, n - p)

        idx = np.arange(1, n + 1)
        violations = t_sq > ucl

        fig = go.Figure()
        normal = ~violations
        fig.add_trace(go.Scatter(x=idx[normal], y=t_sq[normal], mode="markers+lines",
                                 marker=dict(size=5), line=dict(width=1), name="T\u00b2"))
        if violations.any():
            fig.add_trace(go.Scatter(x=idx[violations], y=t_sq[violations], mode="markers",
                                     marker=dict(color="red", size=8, symbol="x"), name="OOC"))
        fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                      annotation_text=f"UCL={ucl:.4f}")
        fig.update_layout(title=f"Hotelling T\u00b2 Chart (p={p}, \u03b1={alpha})", height=500,
                          xaxis_title="Observation", yaxis_title="T\u00b2")
        st.plotly_chart(fig, use_container_width=True)

        ooc_count = int(violations.sum())
        st.write(f"**Out-of-control points:** {ooc_count} / {n}")

        # Variable contribution decomposition for OOC points
        if violations.any():
            section_header("Variable Contributions (OOC Points)")
            ooc_indices = np.where(violations)[0]
            contrib_rows = []
            for oi in ooc_indices[:10]:  # Limit to first 10
                d = X[oi] - x_bar
                contributions = d ** 2 * np.diag(S_inv)
                row = {"Observation": oi + 1, "T\u00b2": round(t_sq[oi], 4)}
                for vi, var_name in enumerate(selected):
                    row[var_name] = round(contributions[vi], 4)
                contrib_rows.append(row)
            contrib_df = pd.DataFrame(contrib_rows)
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)

            # Contribution bar chart for worst point
            worst = ooc_indices[np.argmax(t_sq[ooc_indices])]
            d = X[worst] - x_bar
            contributions = d ** 2 * np.diag(S_inv)
            fig_c = go.Figure(go.Bar(x=selected, y=contributions,
                                      marker_color=["red" if c > ucl / p else "#6366f1"
                                                     for c in contributions]))
            fig_c.update_layout(title=f"Variable Contributions (Obs #{worst + 1}, T\u00b2={t_sq[worst]:.4f})",
                                xaxis_title="Variable", yaxis_title="Contribution", height=350)
            st.plotly_chart(fig_c, use_container_width=True)


# ===================================================================
# Tab 5 -- Gage R&R / MSA
# ===================================================================

def _render_gage_rr(df: pd.DataFrame):
    """Measurement Systems Analysis / Gage R&R."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        empty_state("No numeric columns found.")
        return
    if len(cat_cols) < 2:
        empty_state("Need at least 2 categorical columns (operator and part).",
                     "Your data should have: measurement values, operator ID, and part ID.")
        return

    section_header("Crossed Gage R&R Study")
    help_tip("Gage R&R", """
A Gage R&R study assesses measurement system variability:
- **Repeatability (EV):** Variation when same operator measures same part multiple times
- **Reproducibility (AV):** Variation between different operators
- **GRR:** Total measurement system variation (EV + AV)
- **%GRR < 10%:** Measurement system is acceptable
- **%GRR 10-30%:** May be acceptable depending on application
- **%GRR > 30%:** Measurement system needs improvement
- **NDC \u2265 5:** Measurement system can distinguish enough categories
""")

    c1, c2, c3 = st.columns(3)
    meas_col = c1.selectbox("Measurement column:", num_cols, key="grr_meas")
    operator_col = c2.selectbox("Operator column:", cat_cols, key="grr_operator")
    part_col = c3.selectbox("Part column:", cat_cols, key="grr_part")

    tolerance = st.number_input("Process tolerance (USL - LSL, optional, 0 to skip):",
                                 value=0.0, min_value=0.0, key="grr_tolerance")

    if st.button("Run Gage R&R Analysis", key="grr_run"):
        data = df[[meas_col, operator_col, part_col]].dropna()
        if len(data) < 6:
            st.error("Need at least 6 observations for Gage R&R.")
            return

        operators = data[operator_col].unique()
        parts = data[part_col].unique()
        n_operators = len(operators)
        n_parts = len(parts)

        # Count replicates
        reps = data.groupby([operator_col, part_col]).size()
        n_replicates = int(reps.min())
        if n_replicates < 2:
            st.error("Need at least 2 replicates per operator-part combination.")
            return

        N = len(data)

        # ANOVA decomposition
        grand_mean = data[meas_col].mean()

        # Part means
        part_means = data.groupby(part_col)[meas_col].mean()
        # Operator means
        operator_means = data.groupby(operator_col)[meas_col].mean()
        # Cell means
        cell_means = data.groupby([operator_col, part_col])[meas_col].mean()

        # Sum of squares
        SS_parts = n_operators * n_replicates * np.sum((part_means - grand_mean) ** 2)
        SS_operators = n_parts * n_replicates * np.sum((operator_means - grand_mean) ** 2)

        SS_interaction = n_replicates * sum(
            (cell_means.loc[(op, pt)] - operator_means.loc[op] - part_means.loc[pt] + grand_mean) ** 2
            for op in operators for pt in parts
            if (op, pt) in cell_means.index
        )

        SS_total = np.sum((data[meas_col] - grand_mean) ** 2)
        SS_repeat = SS_total - SS_parts - SS_operators - SS_interaction

        # Degrees of freedom
        df_parts = n_parts - 1
        df_operators = n_operators - 1
        df_interaction = df_parts * df_operators
        df_repeat = N - n_operators * n_parts
        df_total = N - 1

        # Mean squares
        MS_parts = SS_parts / df_parts if df_parts > 0 else 0
        MS_operators = SS_operators / df_operators if df_operators > 0 else 0
        MS_interaction = SS_interaction / df_interaction if df_interaction > 0 else 0
        MS_repeat = SS_repeat / df_repeat if df_repeat > 0 else 0

        # Variance components
        sigma2_repeat = MS_repeat
        sigma2_interaction = max(0, (MS_interaction - MS_repeat) / n_replicates)
        sigma2_operator = max(0, (MS_operators - MS_interaction) / (n_parts * n_replicates))
        sigma2_part = max(0, (MS_parts - MS_interaction) / (n_operators * n_replicates))

        # GRR components
        EV = np.sqrt(sigma2_repeat)  # Repeatability
        AV = np.sqrt(sigma2_operator + sigma2_interaction)  # Reproducibility
        GRR = np.sqrt(EV ** 2 + AV ** 2)
        PV = np.sqrt(sigma2_part)  # Part variation
        TV = np.sqrt(GRR ** 2 + PV ** 2)  # Total variation

        # Percentages
        pct_EV = (EV / TV * 100) if TV > 0 else 0
        pct_AV = (AV / TV * 100) if TV > 0 else 0
        pct_GRR = (GRR / TV * 100) if TV > 0 else 0
        pct_PV = (PV / TV * 100) if TV > 0 else 0

        # Number of distinct categories
        NDC = int(np.floor(1.41 * PV / GRR)) if GRR > 0 else 0

        # ANOVA table
        section_header("ANOVA Table")
        anova_data = pd.DataFrame({
            "Source": ["Part", "Operator", "Part \u00d7 Operator", "Repeatability", "Total"],
            "SS": [SS_parts, SS_operators, SS_interaction, SS_repeat, SS_total],
            "df": [df_parts, df_operators, df_interaction, df_repeat, df_total],
            "MS": [MS_parts, MS_operators, MS_interaction, MS_repeat, np.nan],
        })
        # F-values
        f_parts = MS_parts / MS_interaction if MS_interaction > 0 else np.nan
        f_operators = MS_operators / MS_interaction if MS_interaction > 0 else np.nan
        f_interaction = MS_interaction / MS_repeat if MS_repeat > 0 else np.nan
        anova_data["F"] = [f_parts, f_operators, f_interaction, np.nan, np.nan]
        p_parts = 1 - stats.f.cdf(f_parts, df_parts, df_interaction) if not np.isnan(f_parts) and df_interaction > 0 else np.nan
        p_operators = 1 - stats.f.cdf(f_operators, df_operators, df_interaction) if not np.isnan(f_operators) and df_interaction > 0 else np.nan
        p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_repeat) if not np.isnan(f_interaction) and df_repeat > 0 else np.nan
        anova_data["p-value"] = [p_parts, p_operators, p_interaction, np.nan, np.nan]
        st.dataframe(anova_data.round(4), use_container_width=True, hide_index=True)

        # Variance components table
        section_header("Variance Components")
        var_data = pd.DataFrame({
            "Source": ["Repeatability (EV)", "Reproducibility (AV)", "  Operator", "  Operator \u00d7 Part",
                       "GRR (EV + AV)", "Part Variation (PV)", "Total Variation (TV)"],
            "Variance": [sigma2_repeat, sigma2_operator + sigma2_interaction, sigma2_operator, sigma2_interaction,
                         GRR ** 2, PV ** 2, TV ** 2],
            "Std Dev": [EV, AV, np.sqrt(sigma2_operator), np.sqrt(sigma2_interaction),
                        GRR, PV, TV],
            "% Study Var": [pct_EV, pct_AV, np.nan, np.nan, pct_GRR, pct_PV, 100.0],
        })
        st.dataframe(var_data.round(4), use_container_width=True, hide_index=True)

        # Key metrics
        section_header("Key Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("%GRR (of TV)", f"{pct_GRR:.1f}%")
        m2.metric("NDC", str(NDC))
        if tolerance > 0:
            pct_tol = GRR / tolerance * 6 * 100
            m3.metric("%GRR (of Tolerance)", f"{pct_tol:.1f}%")

        # Interpretation
        if pct_GRR < 10:
            st.success(f"**%GRR = {pct_GRR:.1f}%** \u2014 Measurement system is acceptable.")
        elif pct_GRR < 30:
            st.warning(f"**%GRR = {pct_GRR:.1f}%** \u2014 May be acceptable depending on application.")
        else:
            st.error(f"**%GRR = {pct_GRR:.1f}%** \u2014 Measurement system needs improvement.")

        if NDC >= 5:
            st.success(f"**NDC = {NDC}** \u2014 Adequate discrimination (\u2265 5 required).")
        else:
            st.warning(f"**NDC = {NDC}** \u2014 Insufficient discrimination (need \u2265 5).")

        # Visualizations
        section_header("Gage R&R Charts")

        # Components of variation bar chart
        fig_comp = go.Figure()
        sources = ["GRR", "Repeatability", "Reproducibility", "Part-to-Part"]
        pcts = [pct_GRR, pct_EV, pct_AV, pct_PV]
        colors_bar = ["#EF553B" if p > 30 else "#F59E0B" if p > 10 else "#22C55E" for p in pcts]
        fig_comp.add_trace(go.Bar(x=sources, y=pcts, marker_color=colors_bar))
        fig_comp.add_hline(y=10, line_dash="dot", line_color="green", annotation_text="10%")
        fig_comp.add_hline(y=30, line_dash="dot", line_color="red", annotation_text="30%")
        fig_comp.update_layout(title="Components of Variation (% Study Var)", height=400,
                               yaxis_title="% of Study Variation")
        st.plotly_chart(fig_comp, use_container_width=True)

        # Measurement by Operator
        fig_op = px.box(data, x=operator_col, y=meas_col, color=operator_col,
                        title="Measurement by Operator")
        fig_op.update_layout(height=400)
        st.plotly_chart(fig_op, use_container_width=True)

        # Measurement by Part
        fig_pt = px.box(data, x=part_col, y=meas_col, color=part_col,
                        title="Measurement by Part")
        fig_pt.update_layout(height=400)
        st.plotly_chart(fig_pt, use_container_width=True)

        # Interaction plot (Operator x Part)
        interaction = data.groupby([operator_col, part_col])[meas_col].mean().reset_index()
        fig_int = px.line(interaction, x=part_col, y=meas_col, color=operator_col,
                          markers=True, title="Operator \u00d7 Part Interaction")
        fig_int.update_layout(height=400)
        st.plotly_chart(fig_int, use_container_width=True)

        # X-bar chart by operator
        op_means = data.groupby([operator_col, part_col])[meas_col].mean().reset_index()
        fig_xbar = px.scatter(op_means, x=part_col, y=meas_col, color=operator_col,
                              title="X-bar by Operator", symbol=operator_col)
        fig_xbar.add_hline(y=grand_mean, line_dash="solid", line_color="green",
                           annotation_text=f"Grand Mean={grand_mean:.4f}")
        fig_xbar.update_layout(height=400)
        st.plotly_chart(fig_xbar, use_container_width=True)


# ===================================================================
# Tab 6 -- Acceptance Sampling
# ===================================================================

def _render_acceptance_sampling(df: pd.DataFrame):
    """Acceptance sampling plans for lot inspection."""
    section_header("Acceptance Sampling Plan")
    help_tip("Acceptance Sampling", """
Design sampling plans for lot-by-lot inspection:
- **AQL (Acceptable Quality Level):** Maximum defect rate considered acceptable
- **LTPD (Lot Tolerance Percent Defective):** Defect rate that should be rejected
- **Producer's risk (\u03b1):** Probability of rejecting a good lot (Type I error)
- **Consumer's risk (\u03b2):** Probability of accepting a bad lot (Type II error)
- **OC Curve:** Shows probability of acceptance vs true defect rate
""")

    plan_type = st.selectbox("Plan type:", ["Single Sampling", "Double Sampling"], key="as_plan_type")

    c1, c2 = st.columns(2)
    N = c1.number_input("Lot size (N):", 50, 1000000, 1000, key="as_N")
    AQL = c2.number_input("AQL (fraction, e.g. 0.01):", 0.001, 0.10, 0.01, 0.001,
                           format="%.3f", key="as_AQL")

    c1, c2 = st.columns(2)
    LTPD = c1.number_input("LTPD (fraction):", 0.01, 0.50, 0.05, 0.01,
                            format="%.2f", key="as_LTPD")
    alpha = c2.number_input("Producer's risk (\u03b1):", 0.01, 0.20, 0.05, 0.01, key="as_alpha")
    beta = st.number_input("Consumer's risk (\u03b2):", 0.01, 0.20, 0.10, 0.01, key="as_beta")

    if AQL >= LTPD:
        st.error("AQL must be less than LTPD.")
        return

    if st.button("Compute Sampling Plan", key="as_compute"):
        if plan_type == "Single Sampling":
            # Search for n and c that satisfy both risks
            best_n, best_c = None, None
            for c_val in range(0, 30):
                for n_val in range(c_val + 1, min(N, 1000)):
                    # P(accept | p=AQL) >= 1-alpha (producer's risk)
                    pa_aql = stats.binom.cdf(c_val, n_val, AQL)
                    # P(accept | p=LTPD) <= beta (consumer's risk)
                    pa_ltpd = stats.binom.cdf(c_val, n_val, LTPD)

                    if pa_aql >= 1 - alpha and pa_ltpd <= beta:
                        if best_n is None or n_val < best_n:
                            best_n = n_val
                            best_c = c_val
                        break  # Found smallest n for this c

            if best_n is None:
                st.warning("Could not find a plan satisfying both risks. Try relaxing constraints.")
                # Use approximate solution
                from scipy.optimize import brentq
                for c_val in range(0, 20):
                    try:
                        def f(n_try):
                            return stats.binom.cdf(c_val, int(n_try), LTPD) - beta
                        n_approx = int(brentq(f, c_val + 1, N))
                        pa_aql = stats.binom.cdf(c_val, n_approx, AQL)
                        if pa_aql >= 1 - alpha:
                            best_n, best_c = n_approx, c_val
                            break
                    except Exception:
                        continue

            if best_n is not None:
                section_header("Sampling Plan")
                c1m, c2m, c3m = st.columns(3)
                c1m.metric("Sample size (n)", best_n)
                c2m.metric("Acceptance number (c)", best_c)
                c3m.metric("Rejection number", best_c + 1)

                pa_at_aql = stats.binom.cdf(best_c, best_n, AQL)
                pa_at_ltpd = stats.binom.cdf(best_c, best_n, LTPD)
                st.write(f"**P(accept | AQL={AQL}):** {pa_at_aql:.4f} (need \u2265 {1-alpha:.2f})")
                st.write(f"**P(accept | LTPD={LTPD}):** {pa_at_ltpd:.4f} (need \u2264 {beta:.2f})")

                # OC Curve
                section_header("Operating Characteristic (OC) Curve")
                p_range = np.linspace(0, min(LTPD * 3, 0.5), 200)
                pa = [stats.binom.cdf(best_c, best_n, p) if p > 0 else 1.0 for p in p_range]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=p_range * 100, y=pa, mode="lines",
                                         name="OC Curve", line=dict(width=2)))
                fig.add_vline(x=AQL * 100, line_dash="dash", line_color="green",
                              annotation_text=f"AQL={AQL*100:.1f}%")
                fig.add_vline(x=LTPD * 100, line_dash="dash", line_color="red",
                              annotation_text=f"LTPD={LTPD*100:.1f}%")
                fig.add_hline(y=1 - alpha, line_dash="dot", line_color="green",
                              annotation_text=f"1-\u03b1={1-alpha:.2f}")
                fig.add_hline(y=beta, line_dash="dot", line_color="red",
                              annotation_text=f"\u03b2={beta:.2f}")
                fig.update_layout(title=f"OC Curve (n={best_n}, c={best_c})",
                                  xaxis_title="Lot Defective (%)",
                                  yaxis_title="P(Accept)", height=500)
                st.plotly_chart(fig, use_container_width=True)

                # ASN and AOQ curves
                section_header("AOQ and ATI Curves")
                aoq = [p * stats.binom.cdf(best_c, best_n, p) * (N - best_n) / N
                       for p in p_range]
                ati = [best_n * stats.binom.cdf(best_c, best_n, p) +
                       N * (1 - stats.binom.cdf(best_c, best_n, p))
                       for p in p_range]

                fig2 = make_subplots(rows=1, cols=2,
                                     subplot_titles=("Average Outgoing Quality (AOQ)",
                                                     "Average Total Inspection (ATI)"))
                fig2.add_trace(go.Scatter(x=p_range * 100, y=aoq, mode="lines",
                                          name="AOQ", line=dict(width=2)), row=1, col=1)
                fig2.add_trace(go.Scatter(x=p_range * 100, y=ati, mode="lines",
                                          name="ATI", line=dict(width=2)), row=1, col=2)
                fig2.update_xaxes(title_text="Incoming Quality (%)", row=1, col=1)
                fig2.update_xaxes(title_text="Incoming Quality (%)", row=1, col=2)
                fig2.update_yaxes(title_text="AOQ", row=1, col=1)
                fig2.update_yaxes(title_text="ATI", row=1, col=2)
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

                # AOQL
                aoql = max(aoq)
                aoql_p = p_range[np.argmax(aoq)] * 100
                st.write(f"**AOQL (Average Outgoing Quality Limit):** {aoql:.4f} at incoming {aoql_p:.2f}%")
            else:
                st.error("Could not find a feasible sampling plan.")

        elif plan_type == "Double Sampling":
            st.info("Double sampling uses two stages: if the first sample is inconclusive, a second sample is drawn.")
            c1, c2 = st.columns(2)
            n1 = c1.number_input("First sample size (n\u2081):", 5, N // 2, 50, key="as_n1")
            n2 = c2.number_input("Second sample size (n\u2082):", 5, N // 2, 50, key="as_n2")
            c1a, c2a, c3a = st.columns(3)
            c1_accept = c1a.number_input("Accept after 1st (c\u2081):", 0, 20, 1, key="as_c1")
            c1_reject = c2a.number_input("Reject after 1st (r\u2081):", 1, 30, 4, key="as_r1")
            c2_accept = c3a.number_input("Accept after 2nd (c\u2082):", 0, 30, 4, key="as_c2")

            section_header("Double Sampling OC Curve")
            p_range = np.linspace(0.001, min(LTPD * 3, 0.5), 200)
            pa_double = []
            asn_vals = []
            for p in p_range:
                # P(accept on first sample)
                pa1 = stats.binom.cdf(c1_accept, n1, p)
                # P(reject on first sample)
                pr1 = 1 - stats.binom.cdf(c1_reject - 1, n1, p)
                # P(go to second sample)
                p_second = 1 - pa1 - pr1
                # P(accept on second sample) = sum over d1 in [c1+1, r1-1] of P(X1=d1)*P(X1+X2<=c2)
                pa2 = 0
                for d1 in range(c1_accept + 1, c1_reject):
                    p_d1 = stats.binom.pmf(d1, n1, p)
                    p_accept_combined = stats.binom.cdf(c2_accept - d1, n2, p)
                    pa2 += p_d1 * p_accept_combined
                pa_double.append(pa1 + pa2)
                asn_vals.append(n1 + n2 * p_second)

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("OC Curve (Double Sampling)", "ASN Curve"))
            fig.add_trace(go.Scatter(x=p_range * 100, y=pa_double, mode="lines",
                                     name="P(Accept)", line=dict(width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=p_range * 100, y=asn_vals, mode="lines",
                                     name="ASN", line=dict(width=2)), row=1, col=2)
            fig.add_vline(x=AQL * 100, line_dash="dash", line_color="green", row=1, col=1)
            fig.add_vline(x=LTPD * 100, line_dash="dash", line_color="red", row=1, col=1)
            fig.update_xaxes(title_text="Lot Defective (%)")
            fig.update_yaxes(title_text="P(Accept)", row=1, col=1)
            fig.update_yaxes(title_text="Avg Sample Number", row=1, col=2)
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Tab 7 -- Multi-Vari Analysis
# ===================================================================

def _render_multi_vari(df: pd.DataFrame):
    """Multi-Vari chart to visualise variation at multiple levels."""
    section_header("Multi-Vari Analysis")
    help_tip("Multi-Vari Charts", """
Multi-Vari charts display variation from multiple sources simultaneously:
- **Between groups (Factor 1):** variation across primary categories
- **Within groups (Factor 2):** variation within each primary group
- **Temporal / nested (Factor 3):** additional nesting or faceting

The chart shows individual data points, within-group connections, and
group means so you can visually identify the dominant source of variation.
""")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        empty_state("No numeric columns found for the response variable.")
        return
    if not cat_cols:
        empty_state("Need at least one categorical column for grouping.",
                     "Multi-Vari analysis requires categorical factor columns.")
        return

    response_col = st.selectbox("Response variable (numeric):", num_cols,
                                 key="mv_response")

    factor1 = st.selectbox("Factor 1 (primary grouping, required):", cat_cols,
                            key="mv_factor1")

    remaining_cats = [c for c in cat_cols if c != factor1]
    factor2 = st.selectbox("Factor 2 (secondary nesting, optional):",
                            ["(none)"] + remaining_cats, key="mv_factor2")
    if factor2 == "(none)":
        factor2 = None

    remaining_cats2 = [c for c in remaining_cats if c != factor2]
    factor3 = st.selectbox("Factor 3 (tertiary nesting, optional):",
                            ["(none)"] + remaining_cats2, key="mv_factor3")
    if factor3 == "(none)":
        factor3 = None

    if st.button("Generate Multi-Vari Chart", key="mv_run"):
        cols_needed = [response_col, factor1]
        if factor2:
            cols_needed.append(factor2)
        if factor3:
            cols_needed.append(factor3)
        data = df[cols_needed].dropna()

        if len(data) < 3:
            st.error("Need at least 3 observations for Multi-Vari analysis.")
            return

        f1_levels = sorted(data[factor1].unique())
        if len(f1_levels) > 30:
            st.warning("Factor 1 has more than 30 levels. Showing the first 30.")
            f1_levels = f1_levels[:30]
            data = data[data[factor1].isin(f1_levels)]

        # RDL color palette
        _palette = [
            "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#3b82f6",
            "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4",
        ]

        # --- Build the Multi-Vari chart ---
        if factor3:
            # Use faceted subplots for factor 3
            f3_levels = sorted(data[factor3].unique())
            n_panels = len(f3_levels)
            fig = make_subplots(
                rows=1, cols=n_panels,
                subplot_titles=[f"{factor3} = {lv}" for lv in f3_levels],
                shared_yaxes=True,
            )
            for pi, f3_lv in enumerate(f3_levels):
                sub = data[data[factor3] == f3_lv]
                _add_multi_vari_traces(
                    fig, sub, response_col, factor1, factor2,
                    f1_levels, _palette, row=1, col=pi + 1,
                    show_legend=(pi == 0),
                )
            fig.update_layout(
                height=550, title_text="Multi-Vari Chart",
                yaxis_title=response_col,
            )
            for pi in range(n_panels):
                fig.update_xaxes(title_text=factor1, row=1, col=pi + 1)
        else:
            fig = go.Figure()
            _add_multi_vari_traces(
                fig, data, response_col, factor1, factor2,
                f1_levels, _palette, row=None, col=None,
                show_legend=True,
            )
            fig.update_layout(
                height=550, title_text="Multi-Vari Chart",
                xaxis_title=factor1, yaxis_title=response_col,
            )

        st.plotly_chart(fig, use_container_width=True)

        # --- Variation decomposition ---
        section_header("Variation Decomposition")
        _multi_vari_decomposition(data, response_col, factor1, factor2, factor3)


def _add_multi_vari_traces(fig, data, response_col, factor1, factor2,
                           f1_levels, palette, row, col, show_legend):
    """Add Multi-Vari traces (points, within-group lines, group means) to a figure."""

    def _add(trace, r, c):
        if r is not None:
            fig.add_trace(trace, row=r, col=c)
        else:
            fig.add_trace(trace)

    f1_strings = [str(lv) for lv in f1_levels]

    if factor2 is None:
        # Simple case: scatter individual points + connect group means
        for xi, lv in enumerate(f1_levels):
            sub = data[data[factor1] == lv]
            y_vals = sub[response_col].values
            x_vals = [f1_strings[xi]] * len(y_vals)
            _add(go.Scatter(
                x=x_vals, y=y_vals, mode="markers",
                marker=dict(size=6, color=palette[0], opacity=0.5),
                showlegend=False,
                name="Observations",
            ), row, col)

        # Group means line
        means = [data.loc[data[factor1] == lv, response_col].mean() for lv in f1_levels]
        _add(go.Scatter(
            x=f1_strings, y=means, mode="lines+markers",
            marker=dict(size=12, color=palette[0], symbol="diamond"),
            line=dict(width=2, color=palette[0]),
            name="Group Mean",
            showlegend=show_legend,
        ), row, col)

    else:
        # Factor 2 grouping: color by factor 2 levels, connect within groups
        f2_levels = sorted(data[factor2].unique())

        for f2i, f2_lv in enumerate(f2_levels):
            color = palette[f2i % len(palette)]
            f2_sub = data[data[factor2] == f2_lv]

            # Individual points
            for xi, f1_lv in enumerate(f1_levels):
                cell = f2_sub[f2_sub[factor1] == f1_lv]
                if cell.empty:
                    continue
                y_vals = cell[response_col].values
                x_vals = [f1_strings[xi]] * len(y_vals)
                _add(go.Scatter(
                    x=x_vals, y=y_vals, mode="markers",
                    marker=dict(size=5, color=color, opacity=0.5),
                    showlegend=False,
                    name=f"{factor2}={f2_lv}",
                ), row, col)

            # Connect means within this Factor 2 level
            f2_means = []
            f2_x = []
            for xi, f1_lv in enumerate(f1_levels):
                cell = f2_sub[f2_sub[factor1] == f1_lv]
                if not cell.empty:
                    f2_means.append(cell[response_col].mean())
                    f2_x.append(f1_strings[xi])

            if f2_means:
                _add(go.Scatter(
                    x=f2_x, y=f2_means, mode="lines+markers",
                    marker=dict(size=9, color=color, symbol="diamond"),
                    line=dict(width=2, color=color),
                    name=f"{factor2}={f2_lv}",
                    showlegend=show_legend,
                ), row, col)

        # Overall Factor 1 means
        overall_means = [data.loc[data[factor1] == lv, response_col].mean()
                         for lv in f1_levels]
        _add(go.Scatter(
            x=f1_strings, y=overall_means, mode="lines+markers",
            marker=dict(size=14, color="black", symbol="diamond-open"),
            line=dict(width=3, color="black", dash="dash"),
            name="Overall Mean",
            showlegend=show_legend,
        ), row, col)


def _multi_vari_decomposition(data, response_col, factor1, factor2, factor3):
    """Compute and display variation components for the Multi-Vari analysis."""
    grand_mean = data[response_col].mean()
    ss_total = np.sum((data[response_col] - grand_mean) ** 2)

    if ss_total == 0:
        st.warning("No variation in the response variable.")
        return

    # SS for Factor 1 (between Factor 1 levels)
    f1_groups = data.groupby(factor1)[response_col]
    ss_f1 = sum(len(g) * (g.mean() - grand_mean) ** 2 for _, g in f1_groups)

    rows = [
        {"Source": f"Between {factor1}", "SS": ss_f1,
         "% of Total": round(ss_f1 / ss_total * 100, 2)},
    ]

    ss_explained = ss_f1

    if factor2:
        # SS for Factor 2 within Factor 1
        ss_f2_within = 0.0
        for f1_lv, f1_group in data.groupby(factor1):
            f1_mean = f1_group[response_col].mean()
            for f2_lv, cell in f1_group.groupby(factor2):
                ss_f2_within += len(cell) * (cell[response_col].mean() - f1_mean) ** 2
        rows.append({
            "Source": f"Within {factor1}, Between {factor2}",
            "SS": ss_f2_within,
            "% of Total": round(ss_f2_within / ss_total * 100, 2),
        })
        ss_explained += ss_f2_within

    if factor3:
        # SS for Factor 3 within Factor 2 (within Factor 1)
        ss_f3_within = 0.0
        group_keys = [factor1]
        if factor2:
            group_keys.append(factor2)
        for keys, cell_group in data.groupby(group_keys):
            cell_mean = cell_group[response_col].mean()
            for f3_lv, sub_cell in cell_group.groupby(factor3):
                ss_f3_within += len(sub_cell) * (sub_cell[response_col].mean() - cell_mean) ** 2
        label_within = f"Within {factor2}" if factor2 else f"Within {factor1}"
        rows.append({
            "Source": f"{label_within}, Between {factor3}",
            "SS": ss_f3_within,
            "% of Total": round(ss_f3_within / ss_total * 100, 2),
        })
        ss_explained += ss_f3_within

    ss_residual = ss_total - ss_explained
    rows.append({
        "Source": "Residual (Within-cell)",
        "SS": ss_residual,
        "% of Total": round(ss_residual / ss_total * 100, 2),
    })
    rows.append({
        "Source": "Total",
        "SS": ss_total,
        "% of Total": 100.0,
    })

    decomp_df = pd.DataFrame(rows)
    st.dataframe(decomp_df.round(4), use_container_width=True, hide_index=True)

    # Identify dominant source
    non_total = [r for r in rows if r["Source"] != "Total"]
    dominant = max(non_total, key=lambda r: r["% of Total"])
    st.success(
        f"Dominant variation source: **{dominant['Source']}** "
        f"({dominant['% of Total']:.1f}% of total variation)"
    )

    # Bar chart of variation components
    sources = [r["Source"] for r in non_total]
    pcts = [r["% of Total"] for r in non_total]
    fig_bar = go.Figure(go.Bar(x=sources, y=pcts))
    fig_bar.update_layout(
        title="Variation Breakdown",
        yaxis_title="% of Total Variation",
        height=350,
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ===================================================================
# Tab 8 -- Fishbone (Cause & Effect) Diagram
# ===================================================================

def _render_fishbone(df: pd.DataFrame):
    """Interactive fishbone (Ishikawa / cause-and-effect) diagram builder."""
    section_header("Fishbone (Cause & Effect) Diagram")
    help_tip("Fishbone Diagram", """
Also called an Ishikawa or cause-and-effect diagram. Used to identify
potential root causes of a problem or effect.

**Default 6M categories:**
- **Man:** people, skills, training
- **Machine:** equipment, tools, technology
- **Material:** raw materials, consumables
- **Method:** procedures, processes
- **Measurement:** gauges, data collection
- **Environment:** conditions, regulations

Customize the categories and enter causes (one per line) for each.
The diagram is rendered as a Plotly figure that you can download.
""")

    # Initialize session state for fishbone
    if "fishbone_categories" not in st.session_state:
        st.session_state["fishbone_categories"] = [
            "Man", "Machine", "Material", "Method", "Measurement", "Environment",
        ]
    if "fishbone_causes" not in st.session_state:
        st.session_state["fishbone_causes"] = {
            cat: "" for cat in st.session_state["fishbone_categories"]
        }
    if "fishbone_effect" not in st.session_state:
        st.session_state["fishbone_effect"] = "Quality Problem"

    # Effect input
    effect = st.text_input(
        "Effect (problem / outcome):",
        value=st.session_state["fishbone_effect"],
        key="fb_effect_input",
    )
    st.session_state["fishbone_effect"] = effect

    # Category management
    section_header("Categories and Causes")

    cat_text = st.text_input(
        "Categories (comma-separated):",
        value=", ".join(st.session_state["fishbone_categories"]),
        key="fb_cats_input",
    )
    new_cats = [c.strip() for c in cat_text.split(",") if c.strip()]
    if new_cats != st.session_state["fishbone_categories"]:
        # Preserve existing causes for categories that still exist
        old_causes = st.session_state["fishbone_causes"]
        st.session_state["fishbone_categories"] = new_cats
        st.session_state["fishbone_causes"] = {
            cat: old_causes.get(cat, "") for cat in new_cats
        }

    categories = st.session_state["fishbone_categories"]
    causes_dict = st.session_state["fishbone_causes"]

    if not categories:
        st.info("Enter at least one category above.")
        return

    # Cause entry for each category
    n_cats = len(categories)
    cols_per_row = min(n_cats, 3)
    for row_start in range(0, n_cats, cols_per_row):
        cols = st.columns(cols_per_row)
        for ci in range(cols_per_row):
            idx = row_start + ci
            if idx >= n_cats:
                break
            cat = categories[idx]
            with cols[ci]:
                val = st.text_area(
                    f"{cat} (one cause per line):",
                    value=causes_dict.get(cat, ""),
                    height=120,
                    key=f"fb_cause_{cat}",
                )
                causes_dict[cat] = val

    st.session_state["fishbone_causes"] = causes_dict

    if st.button("Generate Fishbone Diagram", key="fb_generate"):
        # Parse causes
        parsed_causes = {}
        for cat in categories:
            raw = causes_dict.get(cat, "")
            causes = [line.strip() for line in raw.split("\n") if line.strip()]
            parsed_causes[cat] = causes

        fig = _build_fishbone_figure(effect, categories, parsed_causes)
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Use the Plotly toolbar (top-right of chart) to download as PNG.")


def _build_fishbone_figure(effect: str, categories: list,
                           causes: dict) -> go.Figure:
    """Build a fishbone diagram using Plotly shapes and annotations."""
    _palette = [
        "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#3b82f6",
        "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4",
    ]

    n_cats = len(categories)

    # Layout dimensions
    spine_y = 0.5
    x_start = 0.05
    x_end = 0.82
    effect_x = 0.88

    # Distribute branch attachment points evenly along spine
    if n_cats > 0:
        branch_xs = np.linspace(x_start + 0.08, x_end - 0.04, n_cats)
    else:
        branch_xs = []

    # Alternate branches above and below
    branch_len_y = 0.32
    shapes = []
    annotations = []

    # --- Spine line ---
    shapes.append(dict(
        type="line",
        x0=x_start, y0=spine_y, x1=x_end, y1=spine_y,
        line=dict(color="black", width=3),
        xref="paper", yref="paper",
    ))

    # --- Arrow head to effect box ---
    shapes.append(dict(
        type="line",
        x0=x_end, y0=spine_y, x1=effect_x - 0.02, y1=spine_y,
        line=dict(color="black", width=3),
        xref="paper", yref="paper",
    ))
    # Arrow head
    shapes.append(dict(
        type="line",
        x0=effect_x - 0.04, y0=spine_y + 0.02,
        x1=effect_x - 0.02, y1=spine_y,
        line=dict(color="black", width=2),
        xref="paper", yref="paper",
    ))
    shapes.append(dict(
        type="line",
        x0=effect_x - 0.04, y0=spine_y - 0.02,
        x1=effect_x - 0.02, y1=spine_y,
        line=dict(color="black", width=2),
        xref="paper", yref="paper",
    ))

    # --- Effect box ---
    shapes.append(dict(
        type="rect",
        x0=effect_x - 0.02, y0=spine_y - 0.06,
        x1=1.0, y1=spine_y + 0.06,
        line=dict(color="black", width=2),
        fillcolor="rgba(99,102,241,0.12)",
        xref="paper", yref="paper",
    ))
    annotations.append(dict(
        x=(effect_x - 0.02 + 1.0) / 2,
        y=spine_y,
        text=f"<b>{effect}</b>",
        showarrow=False,
        font=dict(size=13, color="black"),
        xref="paper", yref="paper",
        xanchor="center", yanchor="middle",
    ))

    # --- Branches ---
    for i, cat in enumerate(categories):
        bx = branch_xs[i]
        color = _palette[i % len(_palette)]
        above = (i % 2 == 0)
        branch_tip_y = spine_y + branch_len_y if above else spine_y - branch_len_y

        # Branch line (diagonal from spine to category label)
        shapes.append(dict(
            type="line",
            x0=bx, y0=spine_y,
            x1=bx, y1=branch_tip_y,
            line=dict(color=color, width=2.5),
            xref="paper", yref="paper",
        ))

        # Category label at the tip
        label_y = branch_tip_y + (0.04 if above else -0.04)
        annotations.append(dict(
            x=bx, y=label_y,
            text=f"<b>{cat}</b>",
            showarrow=False,
            font=dict(size=12, color=color),
            xref="paper", yref="paper",
            xanchor="center",
            yanchor="bottom" if above else "top",
        ))

        # Cause sub-branches
        cause_list = causes.get(cat, [])
        n_causes = len(cause_list)
        if n_causes > 0:
            # Space causes along the branch
            for ci, cause_text in enumerate(cause_list):
                frac = (ci + 1) / (n_causes + 1)
                cy = spine_y + frac * (branch_tip_y - spine_y)
                # Small horizontal tick for cause
                tick_dir = 0.06 if (i < n_cats / 2) else -0.06
                shapes.append(dict(
                    type="line",
                    x0=bx, y0=cy,
                    x1=bx + tick_dir, y1=cy,
                    line=dict(color=color, width=1.5),
                    xref="paper", yref="paper",
                ))
                # Cause text
                annotations.append(dict(
                    x=bx + tick_dir * 1.1,
                    y=cy,
                    text=cause_text,
                    showarrow=False,
                    font=dict(size=9, color="#444"),
                    xref="paper", yref="paper",
                    xanchor="left" if tick_dir > 0 else "right",
                    yanchor="middle",
                ))

    # --- Build figure ---
    fig = go.Figure()
    # Invisible scatter to create the plotting area
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode="markers",
        marker=dict(size=0.1, opacity=0),
        showlegend=False,
    ))
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, 1], scaleanchor="x", scaleratio=0.6),
        title=dict(text="Fishbone (Cause & Effect) Diagram", x=0.5),
        height=600,
        width=1000,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig
