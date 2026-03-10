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
        marker=dict(color="steelblue", size=5),
        line=dict(color="steelblue", width=1),
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
        st.warning("No data loaded.")
        return

    tabs = st.tabs([
        "Variables Charts", "Attributes Charts", "Process Capability",
    ])

    with tabs[0]:
        _render_variables_charts(df)
    with tabs[1]:
        _render_attributes_charts(df)
    with tabs[2]:
        _render_process_capability(df)


# ===================================================================
# Tab 1 -- Variables Charts (I-MR, X-bar & R, X-bar & S)
# ===================================================================

def _render_variables_charts(df: pd.DataFrame):
    """I-MR, X-bar & R, X-bar & S control charts."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    col_name = st.selectbox("Measurement column:", num_cols, key="spc_var_col")

    chart_type = st.selectbox("Chart type:", [
        "I-MR (Individuals & Moving Range)",
        "X-bar & R",
        "X-bar & S",
    ], key="spc_var_chart_type")

    # Subgroup configuration
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    subgroup_col = None
    subgroup_size = 5

    if chart_type != "I-MR (Individuals & Moving Range)":
        sg_mode = st.radio(
            "Define subgroups by:",
            ["Fixed subgroup size", "Subgroup column"],
            horizontal=True, key="spc_sg_mode",
        )
        if sg_mode == "Subgroup column" and cat_cols:
            subgroup_col = st.selectbox("Subgroup column:", cat_cols, key="spc_sg_col")
        else:
            subgroup_size = st.slider("Subgroup size (n):", 2, 10, 5, key="spc_sg_size")
    else:
        st.caption("I-MR uses individual observations (subgroup size = 1).")

    if st.button("Generate Chart", key="spc_var_generate"):
        data = df[col_name].dropna().values

        if chart_type == "I-MR (Individuals & Moving Range)":
            _imr_chart(data, col_name)
        elif chart_type == "X-bar & R":
            _xbar_r_chart(data, col_name, subgroup_col, subgroup_size, df)
        elif chart_type == "X-bar & S":
            _xbar_s_chart(data, col_name, subgroup_col, subgroup_size, df)


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
        st.warning("Need at least 2 observations for I-MR chart.")
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
        st.warning("Need at least 2 subgroups.")
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
        st.warning("Need at least 2 subgroups.")
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


# ===================================================================
# Tab 2 -- Attributes Charts (p, np, c, u)
# ===================================================================

def _render_attributes_charts(df: pd.DataFrame):
    """p-chart, np-chart, c-chart, u-chart."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    chart_type = st.selectbox("Attributes chart type:", [
        "p-chart (proportion defective)",
        "np-chart (number defective)",
        "c-chart (defects per unit)",
        "u-chart (defects per unit, variable sample)",
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
                             marker=dict(color="steelblue", size=5),
                             line=dict(color="steelblue", width=1), name="p"))
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
                             marker=dict(color="steelblue", size=5),
                             line=dict(color="steelblue", width=1), name="np"))
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
                             marker=dict(color="steelblue", size=5),
                             line=dict(color="steelblue", width=1), name="c"))
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
                             marker=dict(color="steelblue", size=5),
                             line=dict(color="steelblue", width=1), name="u"))
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


# ===================================================================
# Tab 3 -- Process Capability
# ===================================================================

def _render_process_capability(df: pd.DataFrame):
    """Process capability analysis: Cp, Cpk, Pp, Ppk, Cpm."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
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
            st.warning("Need at least 2 observations.")
            return
        if usl <= lsl:
            st.error("USL must be greater than LSL.")
            return

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
        st.markdown("#### Capability Indices")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cp", f"{cp:.3f}")
        m2.metric("Cpk", f"{cpk:.3f}")
        m3.metric("Pp", f"{pp:.3f}")
        m4.metric("Ppk", f"{ppk:.3f}")
        m5.metric("Cpm", f"{cpm:.3f}")

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
                                 marker=dict(size=3, color="steelblue"),
                                 line=dict(width=1, color="steelblue"),
                                 showlegend=False), row=1, col=1)
        fig.add_hline(y=mean, line_color="green", row=1, col=1)
        fig.add_hline(y=i_ucl, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=i_lcl, line_dash="dash", line_color="red", row=1, col=1)

        # 2. Moving Range (top-middle)
        mr_idx = np.arange(2, len(data) + 1)
        D4 = SPC_CONSTANTS[2]['D4']
        fig.add_trace(go.Scatter(x=mr_idx, y=mr, mode="lines+markers",
                                 marker=dict(size=3, color="steelblue"),
                                 line=dict(width=1, color="steelblue"),
                                 showlegend=False), row=1, col=2)
        fig.add_hline(y=mr_bar, line_color="green", row=1, col=2)
        fig.add_hline(y=D4 * mr_bar, line_dash="dash", line_color="red", row=1, col=2)

        # 3. Histogram with spec limits (top-right)
        hist_vals, bin_edges = np.histogram(data, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig.add_trace(go.Bar(x=bin_centers, y=hist_vals, marker_color="steelblue",
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
                                 marker=dict(size=3, color="steelblue"),
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
