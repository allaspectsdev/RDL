"""
RDL UI Helpers - Shared Plotly template, significance cards, help tips,
section headers, empty states, grouped chart selector, and validation components.
"""

from __future__ import annotations

import html as _html

import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go


# ─── RDL Plotly Template ────────────────────────────────────────────────────
# Registered at import time so every chart inherits the theme automatically.

_RDL_COLORWAY = [
    "#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#3b82f6",
    "#ec4899", "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4",
]

_rdl_template = go.layout.Template()
_rdl_template.layout = go.Layout(
    font=dict(
        family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif",
        size=13,
    ),
    colorway=_RDL_COLORWAY,
    plot_bgcolor="#fafafa",
    paper_bgcolor="#ffffff",
    xaxis=dict(gridcolor="#ebebeb", zeroline=False),
    yaxis=dict(gridcolor="#ebebeb", zeroline=False),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_color="#f8fafc",
        font_size=13,
        bordercolor="rgba(255,255,255,0.1)",
    ),
    title=dict(font=dict(size=16, color="#1e293b")),
    legend=dict(
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.06)",
        borderwidth=1,
    ),
    margin=dict(t=40, b=40, l=48, r=24, pad=4),
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

    html = (
        f'<div class="rdl-sig-card {css_class}">'
        f'<div class="rdl-sig-header">'
        f'<span class="rdl-sig-test">{test_name}</span>'
        f'<span class="rdl-sig-badge">{badge}</span>'
        f'</div>'
        f'<div class="rdl-sig-body">'
        f'<strong>{verdict}</strong> &mdash; p = {p_value:.6f} {comparator} \u03b1 = {alpha}'
        f'{effect_html}'
        f'</div></div>'
    )
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
    html = (
        f'<div class="rdl-empty-state">'
        f'<p class="rdl-empty-msg">{message}</p>'
        f'{suggestion_html}'
        f'</div>'
    )
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


# ─── Validation Panel ────────────────────────────────────────────────────

def validation_panel(checks, title="Assumption Checks"):
    """Render a prominent validation status panel.

    Parameters
    ----------
    checks : list[validation.ValidationCheck]
        Each check has .name, .status ('pass'/'warn'/'fail'),
        .detail, and optional .suggestion.
    title : str
        Panel heading text.
    """
    if not checks:
        return

    # Sort: failures first, then warnings, then passes
    _order = {"fail": 0, "warn": 1, "pass": 2}
    sorted_checks = sorted(checks, key=lambda c: _order.get(c.status, 2))

    n_fail = sum(1 for c in checks if c.status == "fail")
    n_warn = sum(1 for c in checks if c.status == "warn")

    if n_fail > 0:
        badge_class = "rdl-vp-badge--fail"
        badge_text = f"{n_fail} Issue{'s' if n_fail > 1 else ''}"
    elif n_warn > 0:
        badge_class = "rdl-vp-badge--warn"
        badge_text = f"{n_warn} Warning{'s' if n_warn > 1 else ''}"
    else:
        badge_class = "rdl-vp-badge--pass"
        badge_text = "All Passed"

    _icons = {"pass": ("&#10003;", "pass"), "warn": ("&#9888;", "warn"), "fail": ("&#10007;", "fail")}

    rows_parts = []
    for c in sorted_checks:
        icon_char, icon_class = _icons.get(c.status, ("&#8226;", "pass"))
        suggestion_html = ""
        if c.suggestion:
            suggestion_html = (
                f'<span class="rdl-vp-suggestion">'
                f'{_html.escape(c.suggestion)}</span>'
            )
        rows_parts.append(
            f'<div class="rdl-vp-check">'
            f'<span class="rdl-vp-icon rdl-vp-icon--{icon_class}">{icon_char}</span>'
            f'<div class="rdl-vp-content">'
            f'<span class="rdl-vp-name">{_html.escape(c.name)}</span>'
            f'<span class="rdl-vp-detail">&mdash; {_html.escape(c.detail)}</span>'
            f'{suggestion_html}'
            f'</div></div>'
        )

    rows_html = "".join(rows_parts)
    html = (
        f'<div class="rdl-validation-panel">'
        f'<div class="rdl-vp-header">'
        f'<span class="rdl-vp-title">{_html.escape(title)}</span>'
        f'<span class="rdl-vp-badge {badge_class}">{badge_text}</span>'
        f'</div>{rows_html}</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─── Confidence Badge ────────────────────────────────────────────────────

def confidence_badge(checks):
    """Return an HTML badge string indicating confidence level based on checks.

    Parameters
    ----------
    checks : list[validation.ValidationCheck]

    Returns
    -------
    str : HTML string to embed inline (e.g. after a section header).
    """
    if not checks:
        return ""
    n_fail = sum(1 for c in checks if c.status == "fail")
    n_warn = sum(1 for c in checks if c.status == "warn")
    if n_fail > 0:
        return '<span class="rdl-confidence-badge rdl-confidence-badge--low">Low Confidence</span>'
    if n_warn > 0:
        return '<span class="rdl-confidence-badge rdl-confidence-badge--moderate">Moderate</span>'
    return '<span class="rdl-confidence-badge rdl-confidence-badge--high">High Confidence</span>'


# ─── Interpretation Card ─────────────────────────────────────────────────

def interpretation_card(interp):
    """Render a plain-language interpretation card.

    Parameters
    ----------
    interp : validation.Interpretation | dict
        Has .title, .body, and optional .detail (or dict with same keys).
    """
    if interp is None:
        return
    if isinstance(interp, dict):
        title = interp.get("title", "Interpretation")
        body = interp.get("body", "")
        detail = interp.get("detail", "")
    else:
        title, body, detail = interp.title, interp.body, interp.detail

    detail_html = ""
    if detail:
        detail_html = f'<div class="rdl-interp-detail">{_html.escape(detail)}</div>'

    html = (
        f'<div class="rdl-interp-card">'
        f'<div class="rdl-interp-title">{_html.escape(title)}</div>'
        f'{_html.escape(body)}'
        f'{detail_html}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─── Alternative Suggestion ──────────────────────────────────────────────

def alternative_suggestion(issue, alternatives):
    """Render an alternative-test recommendation card.

    Parameters
    ----------
    issue : str
        Why the current approach may be problematic.
    alternatives : list[str]
        Suggested alternative methods/tests.
    """
    if not alternatives:
        return
    tags = " ".join(f'<span class="rdl-alt-tag">{_html.escape(a)}</span>' for a in alternatives)
    html = (
        f'<div class="rdl-alt-card">'
        f'{_html.escape(issue)} &mdash; consider: {tags}'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─── Sample Size Indicator ───────────────────────────────────────────────

def sample_size_indicator(n, min_recommended):
    """Render an inline sample-size status indicator.

    Parameters
    ----------
    n : int
        Actual sample size.
    min_recommended : int
        Minimum recommended sample size for this analysis.
    """
    if n >= min_recommended:
        css = "rdl-sample-size--ok"
        text = f"n = {n:,}"
    elif n >= min_recommended * 0.5:
        css = "rdl-sample-size--low"
        text = f"n = {n:,} (below recommended {min_recommended})"
    else:
        css = "rdl-sample-size--critical"
        text = f"n = {n:,} (well below recommended {min_recommended})"
    st.markdown(f'<span class="rdl-sample-size {css}">{text}</span>', unsafe_allow_html=True)
