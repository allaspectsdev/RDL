"""
RDL UI Helpers - Shared Plotly template, significance cards, help tips,
section headers, empty states, and grouped chart selector.
"""

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

# ─── RDL Plotly Template ────────────────────────────────────────────────────
# Registered at import time so every chart inherits the theme automatically.

_RDL_COLORWAY = [
    "#6366f1", "#818cf8", "#a78bfa", "#22c55e", "#f59e0b",
    "#ef4444", "#3b82f6", "#ec4899", "#14b8a6", "#f97316",
]

_rdl_template = go.layout.Template()
_rdl_template.layout = go.Layout(
    font=dict(family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif"),
    colorway=_RDL_COLORWAY,
    plot_bgcolor="#fafbfe",
    paper_bgcolor="#ffffff",
    xaxis=dict(gridcolor="#e2e8f0", zeroline=False),
    yaxis=dict(gridcolor="#e2e8f0", zeroline=False),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_color="#f1f5f9",
        font_size=13,
        bordercolor="#334155",
    ),
    title=dict(font=dict(size=16, color="#1e293b")),
    legend=dict(
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e2e8f0",
        borderwidth=1,
    ),
    margin=dict(t=48, b=40, l=48, r=24),
)

pio.templates["rdl"] = _rdl_template
pio.templates.default = "plotly+rdl"


# ─── Significance Result Card ──────────────────────────────────────────────

def significance_result(p_value, alpha, test_name, effect_size=None, effect_label=None):
    """Render a colored significance result card.

    Green border = fail to reject H0 (not significant).
    Red border   = reject H0 (significant).
    """
    reject = p_value < alpha
    css_class = "rdl-sig-card--reject" if reject else "rdl-sig-card--accept"
    verdict = "Reject H\u2080" if reject else "Fail to reject H\u2080"
    badge = "Significant" if reject else "Not Significant"
    comparator = "<" if reject else "\u2265"

    effect_html = ""
    if effect_size is not None and effect_label:
        effect_html = f'<span class="rdl-sig-effect">{effect_label} = {effect_size:.4f}</span>'

    html = f"""
    <div class="rdl-sig-card {css_class}">
        <div class="rdl-sig-header">
            <span class="rdl-sig-test">{test_name}</span>
            <span class="rdl-sig-badge">{badge}</span>
        </div>
        <div class="rdl-sig-body">
            <strong>{verdict}</strong> &mdash; p = {p_value:.6f} {comparator} \u03b1 = {alpha}
            {effect_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ─── Help Tip ──────────────────────────────────────────────────────────────

def help_tip(title, body):
    """Consistent info expander, collapsed by default."""
    with st.expander(f"\u2139\ufe0f {title}", expanded=False):
        st.markdown(body)


# ─── Section Header ───────────────────────────────────────────────────────

def section_header(title, help_text=None):
    """Standardised #### heading with optional inline help tooltip."""
    if help_text:
        st.markdown(f"#### {title}", help=help_text)
    else:
        st.markdown(f"#### {title}")


# ─── Empty State ──────────────────────────────────────────────────────────

def empty_state(message, suggestion=None):
    """Branded empty-state block replacing generic st.warning calls."""
    suggestion_html = f'<p class="rdl-empty-suggestion">{suggestion}</p>' if suggestion else ""
    html = f"""
    <div class="rdl-empty-state">
        <p class="rdl-empty-msg">{message}</p>
        {suggestion_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ─── Grouped Chart Selector ──────────────────────────────────────────────

CHART_CATEGORIES = {
    "Distribution": ["Histogram", "Box Plot", "Violin Plot", "Strip Plot"],
    "Comparison": ["Bar Chart", "Radar Chart", "Funnel Chart", "Waterfall Chart"],
    "Relationship": ["Scatter Plot", "Bubble Chart", "Joint Plot", "Heatmap", "Mosaic Plot"],
    "Composition": ["Pie / Donut", "Treemap", "Sunburst", "Area Chart"],
    "Trend": ["Line Chart", "Candlestick (OHLC)"],
    "3D & Spatial": ["3D Scatter", "3D Surface", "Contour Plot"],
    "Multivariate": ["Parallel Coordinates", "Variability Chart"],
}


def grouped_chart_selector(key_prefix="viz"):
    """Two-level chart selector: category radio -> type selectbox."""
    categories = list(CHART_CATEGORIES.keys())
    category = st.radio(
        "Category:", categories, horizontal=True, key=f"{key_prefix}_cat"
    )
    chart_types = CHART_CATEGORIES[category]
    chart_type = st.selectbox(
        "Chart Type:", chart_types, key=f"{key_prefix}_chart_type"
    )
    return chart_type
