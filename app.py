"""
Ryan's Data Lab (RDL) - Visual Data Analysis Tool
A comprehensive, interactive data analysis platform modeled after
JMP, MATLAB, and other leading statistical analysis tools.

Run with: streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as _components
import pandas as pd
import numpy as np
import html as _html
from datetime import datetime
import io

st.set_page_config(
    page_title="Ryan's Data Lab",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── FONT IMPORT ────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ─── DESIGN SYSTEM — Variables, Typography, Background ─────────────────────
st.markdown("""<style>
:root {
    --rdl-font: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --rdl-bg: #f7f8fb;
    --rdl-bg-card: rgba(255,255,255,0.72);
    --rdl-sidebar-bg: #111827;
    --rdl-sidebar-text: #cbd5e1;
    --rdl-sidebar-bright: #f1f5f9;
    --rdl-sidebar-hover: rgba(255,255,255,0.06);
    --rdl-sidebar-active: rgba(99,102,241,0.15);
    --rdl-sidebar-border: rgba(255,255,255,0.06);
    --rdl-text: #1e293b;
    --rdl-text-secondary: #4b5563;
    --rdl-text-muted: #6b7280;
    --rdl-accent: #6366f1;
    --rdl-accent-light: #eef2ff;
    --rdl-accent-hover: #4f46e5;
    --rdl-accent-subtle: rgba(99,102,241,0.08);
    --rdl-success: #22c55e;
    --rdl-warning: #f59e0b;
    --rdl-warning-dark: #b45309;
    --rdl-error: #ef4444;
    --rdl-info: #3b82f6;
    --rdl-border: rgba(0,0,0,0.06);
    --rdl-shadow-xs: 0 1px 3px rgba(0,0,0,0.03);
    --rdl-shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
    --rdl-shadow-md: 0 4px 16px rgba(0,0,0,0.05);
    --rdl-shadow-lg: 0 8px 32px rgba(0,0,0,0.06);
    --rdl-shadow-xl: 0 16px 48px rgba(0,0,0,0.08);
    --rdl-accent-gradient: linear-gradient(135deg, #6366f1, #8b5cf6, #a78bfa);
    --rdl-accent-gradient-warm: linear-gradient(135deg, #6366f1, #8b5cf6, #ec4899);
    --rdl-radius-lg: 20px;
    --rdl-radius: 14px;
    --rdl-radius-sm: 10px;
    --rdl-radius-xs: 8px;
    --rdl-ease: cubic-bezier(0.22, 0.61, 0.36, 1);
    --rdl-dur: 0.15s;
    --rdl-glass-blur: 16px;
    --rdl-glass-border: rgba(255,255,255,0.18);
    --rdl-glass-bg: rgba(255,255,255,0.72);
}

html, body,
.stMarkdown, .stText, p, label, li, td, th,
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
button, input, select, textarea,
.stSelectbox label, .stMultiSelect label,
.stRadio label, .stCheckbox label,
[data-baseweb="select"],
[data-baseweb="radio"],
[data-baseweb="tab"] {
    font-family: var(--rdl-font) !important;
}

h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: var(--rdl-font) !important;
    letter-spacing: -0.02em;
}

.stApp {
    background: var(--rdl-bg) !important;
}
[data-testid="stAppViewContainer"] {
    background: var(--rdl-bg) !important;
}

[data-testid="stHeader"] {
    background: rgba(247,248,251,0.7) !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--rdl-glass-border) !important;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

# ─── SIDEBAR STYLES ─────────────────────────────────────────────────────────
st.markdown("""<style>
[data-testid="stSidebar"] {
    background: var(--rdl-sidebar-bg) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: var(--rdl-sidebar-text) !important;
}

[data-testid="stSidebar"] [data-testid="stHeading"] h2,
[data-testid="stSidebar"] [data-testid="stHeading"] h3 {
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    font-weight: 600 !important;
    color: var(--rdl-text-muted) !important;
    margin-top: 0.25rem !important;
    margin-bottom: 0.25rem !important;
    border-bottom: none !important;
    display: block !important;
    padding-bottom: 0 !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--rdl-sidebar-border) !important;
    margin: 0.75rem 0 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 1px !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] {
    padding: 0.5rem 0.75rem !important;
    border-radius: var(--rdl-radius-xs) !important;
    transition: background var(--rdl-dur) var(--rdl-ease),
                transform var(--rdl-dur) var(--rdl-ease) !important;
    margin-bottom: 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: var(--rdl-sidebar-hover) !important;
    transform: translateX(2px);
}
[data-testid="stSidebar"] [data-baseweb="radio"]:focus-visible,
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:focus-visible) {
    outline: 2px solid var(--rdl-accent);
    outline-offset: -2px;
    background: var(--rdl-sidebar-hover) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {
    background: var(--rdl-sidebar-active) !important;
    border-left: 3px solid var(--rdl-accent) !important;
    padding-left: calc(0.75rem - 3px) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) div:last-child {
    color: #a5b4fc !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(34,197,94,0.1) !important;
    border: 1px solid rgba(34,197,94,0.2) !important;
    border-radius: var(--rdl-radius-xs) !important;
    padding: 0.5rem 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] p {
    color: #86efac !important;
    font-size: 0.8rem !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: var(--rdl-radius-xs) !important;
    color: var(--rdl-sidebar-bright) !important;
}

[data-testid="stSidebar"] .stButton > button {
    background: var(--rdl-accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--rdl-radius-xs) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1rem !important;
    transition: background var(--rdl-dur) var(--rdl-ease),
                transform var(--rdl-dur) var(--rdl-ease),
                box-shadow var(--rdl-dur) var(--rdl-ease) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: var(--rdl-accent-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(99,102,241,0.25) !important;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: var(--rdl-radius-sm) !important;
    padding: 0.75rem !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    padding: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: var(--rdl-sidebar-bright) !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    color: var(--rdl-sidebar-text) !important;
}

.rdl-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0 0.5rem;
    margin-bottom: 0.25rem;
    cursor: pointer;
    border-radius: var(--rdl-radius-sm);
    transition: background var(--rdl-dur) var(--rdl-ease);
}
.rdl-brand:hover {
    background: var(--rdl-sidebar-hover);
}
.rdl-brand-mark {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: var(--rdl-accent-gradient);
    border-radius: var(--rdl-radius-sm);
    font-weight: 800;
    font-size: 0.78rem;
    color: #ffffff;
    letter-spacing: 0.03em;
    flex-shrink: 0;
    box-shadow: 0 2px 12px rgba(99,102,241,0.2);
}
.rdl-brand-name {
    display: block;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--rdl-sidebar-bright);
    line-height: 1.2;
}
.rdl-brand-tag {
    display: block;
    font-size: 0.68rem;
    color: var(--rdl-text-muted);
    font-weight: 400;
    margin-top: 2px;
}

.rdl-dataset-info {
    background: rgba(255,255,255,0.04);
    border-radius: var(--rdl-radius-xs);
    padding: 0.6rem 0.75rem;
}
.rdl-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    font-size: 0.78rem;
}
.rdl-info-row:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.rdl-info-row:first-child {
    padding-top: 0;
}
.rdl-info-row span:first-child {
    color: var(--rdl-text-muted);
    font-weight: 400;
}
.rdl-info-row span:last-child {
    color: var(--rdl-sidebar-bright);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
</style>""", unsafe_allow_html=True)

# ─── COMPONENT STYLES — Tabs, Metrics, Buttons, Inputs, etc. ───────────────
st.markdown("""<style>
[data-testid="stAppViewContainer"] h2 {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--rdl-text) !important;
    padding-bottom: 0.6rem !important;
    margin-bottom: 1rem !important;
    border-bottom: 2px solid transparent !important;
    border-image: linear-gradient(90deg, #6366f1, #8b5cf6, transparent) 1 !important;
    display: inline-block !important;
}
[data-testid="stAppViewContainer"] h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--rdl-text) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: var(--rdl-glass-bg) !important;
    backdrop-filter: blur(var(--rdl-glass-blur));
    -webkit-backdrop-filter: blur(var(--rdl-glass-blur));
    border-radius: var(--rdl-radius-sm) !important;
    padding: 4px !important;
    border: 1px solid var(--rdl-glass-border) !important;
    box-shadow: var(--rdl-shadow-xs) !important;
    overflow-x: auto !important;
    scrollbar-width: thin !important;
    -webkit-overflow-scrolling: touch !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 0.5rem 1.1rem !important;
    border-radius: var(--rdl-radius-xs) !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: var(--rdl-text-secondary) !important;
    background: transparent !important;
    border: none !important;
    transition: background var(--rdl-dur) var(--rdl-ease),
                color var(--rdl-dur) var(--rdl-ease),
                box-shadow var(--rdl-dur) var(--rdl-ease) !important;
    white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: var(--rdl-accent-subtle) !important;
    color: var(--rdl-accent) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--rdl-accent) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 12px rgba(99,102,241,0.3) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.25rem !important;
}

[data-testid="stMetric"] {
    background: var(--rdl-glass-bg) !important;
    backdrop-filter: blur(var(--rdl-glass-blur));
    -webkit-backdrop-filter: blur(var(--rdl-glass-blur));
    border: 1px solid var(--rdl-glass-border) !important;
    border-top: 3px solid var(--rdl-accent) !important;
    border-radius: var(--rdl-radius-sm) !important;
    padding: 1rem 1.25rem !important;
    box-shadow: var(--rdl-shadow-sm) !important;
    transition: box-shadow var(--rdl-dur) var(--rdl-ease),
                transform var(--rdl-dur) var(--rdl-ease) !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--rdl-shadow-md) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
    color: var(--rdl-text-muted) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    color: var(--rdl-text) !important;
    font-variant-numeric: tabular-nums;
}

.stButton > button {
    border-radius: var(--rdl-radius-xs) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.25rem !important;
    border: 1px solid transparent !important;
    background: var(--rdl-bg-card) !important;
    color: var(--rdl-text) !important;
    box-shadow: var(--rdl-shadow-xs) !important;
    transition: background var(--rdl-dur) var(--rdl-ease),
                color var(--rdl-dur) var(--rdl-ease),
                box-shadow var(--rdl-dur) var(--rdl-ease) !important;
}
.stButton > button:hover {
    background: var(--rdl-accent-subtle) !important;
    color: var(--rdl-accent) !important;
    box-shadow: var(--rdl-shadow-sm) !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
    transition: transform 0.1s ease !important;
}
.stDownloadButton > button:active {
    transform: scale(0.98) !important;
    transition: transform 0.1s ease !important;
}
.stDownloadButton > button {
    background: var(--rdl-accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--rdl-radius-xs) !important;
}
.stDownloadButton > button:hover {
    background: var(--rdl-accent-hover) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.2) !important;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
.stTextInput > div > div,
.stNumberInput > div > div,
.stTextArea > div > div {
    border-radius: var(--rdl-radius-xs) !important;
    border-color: var(--rdl-border) !important;
    background: var(--rdl-bg-card) !important;
    transition: border-color var(--rdl-dur) var(--rdl-ease),
                box-shadow var(--rdl-dur) var(--rdl-ease) !important;
}
[data-baseweb="select"] > div:focus-within,
[data-baseweb="input"] > div:focus-within,
.stTextInput > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: var(--rdl-accent) !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.12) !important;
}

[data-testid="stExpander"] {
    background: var(--rdl-glass-bg) !important;
    backdrop-filter: blur(var(--rdl-glass-blur));
    -webkit-backdrop-filter: blur(var(--rdl-glass-blur));
    border: 1px solid var(--rdl-glass-border) !important;
    border-radius: var(--rdl-radius-sm) !important;
    box-shadow: var(--rdl-shadow-xs) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: var(--rdl-text) !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--rdl-accent) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--rdl-border) !important;
    border-radius: var(--rdl-radius-sm) !important;
    overflow: hidden !important;
    box-shadow: var(--rdl-shadow-sm) !important;
    transition: box-shadow var(--rdl-dur) var(--rdl-ease) !important;
}
[data-testid="stDataFrame"]:hover {
    box-shadow: var(--rdl-shadow-sm) !important;
}

[data-testid="stAlert"] {
    border-radius: var(--rdl-radius-sm) !important;
    font-size: 0.88rem !important;
}

[data-testid="stAppViewContainer"] hr {
    border-color: var(--rdl-border) !important;
    opacity: 0.35 !important;
}

html {
    scrollbar-width: thin;
    scrollbar-color: var(--rdl-text-muted) transparent;
}
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--rdl-text-muted);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--rdl-text-secondary);
}

.js-plotly-plot, .plotly {
    border-radius: var(--rdl-radius-sm) !important;
    overflow: hidden !important;
    transition: box-shadow var(--rdl-dur) var(--rdl-ease) !important;
}
.js-plotly-plot:hover, .plotly:hover {
    box-shadow: var(--rdl-shadow-sm) !important;
}

[data-testid="stSpinner"] {
    background: rgba(244,245,249,0.8) !important;
    backdrop-filter: blur(4px) !important;
    -webkit-backdrop-filter: blur(4px) !important;
    border-radius: var(--rdl-radius-sm) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--rdl-accent) !important;
}

/* ── Significance Result Cards ── */
.rdl-sig-card {
    background: var(--rdl-glass-bg);
    backdrop-filter: blur(var(--rdl-glass-blur));
    -webkit-backdrop-filter: blur(var(--rdl-glass-blur));
    border: none;
    border-left: 4px solid var(--rdl-success);
    border-radius: var(--rdl-radius-sm);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    box-shadow: var(--rdl-shadow-sm);
    transition: box-shadow var(--rdl-dur) var(--rdl-ease);
}
.rdl-sig-card:hover {
    box-shadow: var(--rdl-shadow-md);
}
.rdl-sig-card--reject {
    border-left-color: var(--rdl-error);
    background: linear-gradient(90deg, rgba(239,68,68,0.03) 0%, var(--rdl-bg-card) 30%);
}
.rdl-sig-card--accept {
    border-left-color: var(--rdl-success);
    background: linear-gradient(90deg, rgba(34,197,94,0.03) 0%, var(--rdl-bg-card) 30%);
}
.rdl-sig-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.rdl-sig-test {
    font-weight: 700;
    font-size: 0.88rem;
    color: var(--rdl-text);
}
.rdl-sig-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.rdl-sig-card--reject .rdl-sig-badge {
    background: rgba(239,68,68,0.1);
    color: var(--rdl-error);
}
.rdl-sig-card--accept .rdl-sig-badge {
    background: rgba(34,197,94,0.1);
    color: var(--rdl-success);
}
.rdl-sig-body {
    font-size: 0.88rem;
    color: var(--rdl-text-secondary);
    line-height: 1.6;
}
.rdl-sig-effect {
    display: inline-block;
    margin-left: 0.75rem;
    padding: 0.15rem 0.5rem;
    background: var(--rdl-accent-subtle);
    border-radius: var(--rdl-radius-xs);
    font-size: 0.8rem;
    color: var(--rdl-accent);
    font-weight: 500;
}

/* ── Empty State ── */
.rdl-empty-state {
    text-align: center;
    padding: 2.5rem 1rem;
    margin: 1rem 0;
    border: none;
    border-radius: var(--rdl-radius-sm);
    background: rgba(99,102,241,0.03);
}
.rdl-empty-msg {
    font-size: 0.92rem;
    font-weight: 500;
    color: var(--rdl-text-secondary);
    margin: 0 0 0.25rem 0;
}
.rdl-empty-suggestion {
    font-size: 0.8rem;
    color: var(--rdl-text-muted);
    margin: 0;
}

/* ── Validation Panel ── */
.rdl-validation-panel {
    background: var(--rdl-glass-bg);
    backdrop-filter: blur(var(--rdl-glass-blur));
    -webkit-backdrop-filter: blur(var(--rdl-glass-blur));
    border: 1px solid var(--rdl-glass-border);
    border-radius: var(--rdl-radius-sm);
    padding: 0.75rem 1rem;
    margin: 0.75rem 0;
    box-shadow: var(--rdl-shadow-sm);
}
.rdl-vp-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,0,0,0.04);
}
.rdl-vp-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--rdl-text);
}
.rdl-vp-badge {
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 99px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.rdl-vp-badge--pass { background: rgba(34,197,94,0.1); color: var(--rdl-success); }
.rdl-vp-badge--warn { background: rgba(245,158,11,0.1); color: var(--rdl-warning); }
.rdl-vp-badge--fail { background: rgba(239,68,68,0.1); color: var(--rdl-error); }
.rdl-vp-check {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.35rem 0;
    font-size: 0.82rem;
}
.rdl-vp-icon { font-size: 0.9rem; flex-shrink: 0; margin-top: 0.1rem; }
.rdl-vp-icon--pass { color: var(--rdl-success); }
.rdl-vp-icon--warn { color: var(--rdl-warning); }
.rdl-vp-icon--fail { color: var(--rdl-error); }
.rdl-vp-content { flex: 1; }
.rdl-vp-name { font-weight: 500; color: var(--rdl-text); }
.rdl-vp-detail { color: var(--rdl-text-secondary); margin-left: 0.25rem; }
.rdl-vp-suggestion {
    display: block;
    font-size: 0.78rem;
    color: var(--rdl-text-muted);
    font-style: italic;
    margin-top: 0.15rem;
}

/* ── Confidence Badge ── */
.rdl-confidence-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.1rem 0.5rem;
    border-radius: 99px;
    vertical-align: middle;
    margin-left: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
.rdl-confidence-badge--high { background: rgba(34,197,94,0.1); color: var(--rdl-success); }
.rdl-confidence-badge--moderate { background: rgba(245,158,11,0.1); color: var(--rdl-warning); }
.rdl-confidence-badge--low { background: rgba(239,68,68,0.1); color: var(--rdl-error); }

/* ── Interpretation Card ── */
.rdl-interp-card {
    background: var(--rdl-accent-subtle);
    border-left: 2px solid var(--rdl-accent);
    border-radius: var(--rdl-radius-xs);
    padding: 0.65rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: var(--rdl-text);
    line-height: 1.6;
}
.rdl-interp-title {
    font-weight: 600;
    font-size: 0.82rem;
    margin-bottom: 0.25rem;
    color: var(--rdl-accent);
}
.rdl-interp-detail {
    font-size: 0.78rem;
    color: var(--rdl-text-secondary);
    margin-top: 0.25rem;
}

/* ── Data Readiness Gauge ── */
.rdl-readiness-gauge {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1rem 0;
    margin-bottom: 0.75rem;
}
.rdl-rg-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}
.rdl-rg-inner {
    width: 76px;
    height: 76px;
    border-radius: 50%;
    background: var(--rdl-bg, #0e1117);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.rdl-rg-grade {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--rdl-text, #fafafa);
    line-height: 1;
}
.rdl-rg-score {
    font-size: 0.7rem;
    color: var(--rdl-text-muted, #a1a1aa);
    margin-top: 2px;
}
.rdl-rg-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--rdl-text, #fafafa);
    margin-top: 0.5rem;
}
.rdl-rg-breakdown {
    display: flex;
    gap: 0.75rem;
    margin-top: 0.35rem;
    font-size: 0.72rem;
}
.rdl-rg-stat { color: var(--rdl-text-muted, #a1a1aa); }
.rdl-rg-stat--pass { color: var(--rdl-success, #22c55e); }
.rdl-rg-stat--warn { color: var(--rdl-warning, #f59e0b); }
.rdl-rg-stat--fail { color: var(--rdl-error, #ef4444); }

/* ── Alternative Suggestion Card ── */
.rdl-alt-card {
    background: rgba(245,158,11,0.04);
    border-left: 3px solid var(--rdl-warning);
    border-radius: var(--rdl-radius-xs);
    padding: 0.55rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.82rem;
    color: var(--rdl-text);
    line-height: 1.6;
}
.rdl-alt-tag {
    display: inline-block;
    background: rgba(245,158,11,0.12);
    color: var(--rdl-warning-dark);
    font-weight: 600;
    font-size: 0.75rem;
    padding: 0.1rem 0.45rem;
    border-radius: var(--rdl-radius-xs);
    margin: 0 0.15rem;
}

/* ── Sample Size Indicator ── */
.rdl-sample-size { font-size: 0.82rem; font-weight: 500; }
.rdl-sample-size--ok { color: var(--rdl-success); }
.rdl-sample-size--low { color: var(--rdl-warning); }
.rdl-sample-size--critical { color: var(--rdl-error); }
</style>""", unsafe_allow_html=True)

# ─── LANDING PAGE STYLES ────────────────────────────────────────────────────
st.markdown("""<style>
.rdl-hero {
    text-align: center;
    padding: 3.5rem 1rem 1.5rem;
    position: relative;
    animation: rdl-enter 0.3s ease both;
}
.rdl-hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 100%;
    height: 100%;
    background: radial-gradient(ellipse at 30% 20%, rgba(99,102,241,0.05), transparent 50%), radial-gradient(ellipse at 70% 30%, rgba(139,92,246,0.04), transparent 50%);
    pointer-events: none;
    z-index: 0;
}
.rdl-hero > * { position: relative; z-index: 1; }
.rdl-hero h1 {
    font-size: 3.25rem !important;
    font-weight: 800 !important;
    color: var(--rdl-text) !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.75rem !important;
    line-height: 1.1 !important;
    border-bottom: none !important;
    display: block !important;
    padding-bottom: 0 !important;
}
.rdl-hero h1::before {
    display: none;
}
.rdl-accent-text {
    background: var(--rdl-accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.rdl-hero p {
    font-size: 1.1rem;
    color: var(--rdl-text-secondary);
    font-weight: 400;
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.65;
}

.rdl-stats-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    padding: 1.5rem 2rem;
    margin: 0.5rem auto 2rem;
    max-width: 700px;
    background: var(--rdl-glass-bg);
    backdrop-filter: blur(var(--rdl-glass-blur));
    -webkit-backdrop-filter: blur(var(--rdl-glass-blur));
    border: 1px solid var(--rdl-glass-border);
    border-radius: var(--rdl-radius);
    box-shadow: var(--rdl-shadow-sm);
    animation: rdl-enter 0.3s ease both;
}
.rdl-stat {
    text-align: center;
}
.rdl-stat-value {
    display: block;
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--rdl-accent);
    letter-spacing: -0.02em;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.rdl-stat-label {
    display: block;
    font-size: 0.72rem;
    color: var(--rdl-text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.4rem;
}

.rdl-features-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    max-width: 1100px;
    margin: 0 auto 2.5rem;
    padding: 0 1rem;
    animation: rdl-enter 0.3s ease both;
}
.rdl-feature-card {
    background: var(--rdl-glass-bg);
    backdrop-filter: blur(var(--rdl-glass-blur));
    -webkit-backdrop-filter: blur(var(--rdl-glass-blur));
    border: 1px solid var(--rdl-glass-border);
    border-radius: var(--rdl-radius);
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow var(--rdl-dur) var(--rdl-ease),
                transform var(--rdl-dur) var(--rdl-ease),
                border-color var(--rdl-dur) var(--rdl-ease);
}
.rdl-feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cat-color, var(--rdl-accent)), rgba(167,139,250,0.6));
    opacity: 0.5;
    transition: opacity var(--rdl-dur) var(--rdl-ease);
}
.rdl-feature-card:hover {
    box-shadow: var(--rdl-shadow-lg);
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.2);
}
.rdl-feature-card:hover::before {
    opacity: 1;
}
.rdl-feature-card h3 {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: var(--rdl-text) !important;
    margin: 0 0 0.5rem 0 !important;
    line-height: 1.3 !important;
}
.rdl-feature-card p {
    font-size: 0.82rem;
    color: var(--rdl-text-secondary);
    line-height: 1.6;
    margin: 0;
}

.rdl-modules-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    max-width: 1100px;
    margin: 0 auto 2rem;
    padding: 0 1rem;
    animation: rdl-enter 0.3s ease both;
}
.rdl-module-item {
    background: var(--rdl-glass-bg);
    backdrop-filter: blur(var(--rdl-glass-blur));
    -webkit-backdrop-filter: blur(var(--rdl-glass-blur));
    border: 1px solid var(--rdl-glass-border);
    border-radius: var(--rdl-radius);
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow var(--rdl-dur) var(--rdl-ease),
                transform var(--rdl-dur) var(--rdl-ease),
                border-color var(--rdl-dur) var(--rdl-ease);
}
.rdl-module-item::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cat-color, var(--rdl-accent)), rgba(167,139,250,0.6));
    opacity: 0.5;
    transition: opacity var(--rdl-dur) var(--rdl-ease);
}
.rdl-module-item:hover {
    box-shadow: var(--rdl-shadow-lg);
    transform: translateY(-2px);
    border-color: rgba(99,102,241,0.2);
}
.rdl-module-item:hover::before {
    opacity: 1;
}
.rdl-module-item h4 {
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    color: var(--rdl-text) !important;
    margin: 0 0 0.25rem 0 !important;
}
.rdl-module-item h4::before {
    content: '';
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--cat-color, var(--rdl-accent));
    margin-right: 0.5rem;
    vertical-align: middle;
}
.rdl-module-item p {
    font-size: 0.78rem;
    color: var(--rdl-text-secondary);
    line-height: 1.55;
    margin: 0;
}

.rdl-cta {
    text-align: center;
    padding: 1.5rem 1rem 3rem;
    animation: rdl-enter 0.3s ease both;
}
.rdl-cta-card {
    display: inline-block;
    background: var(--rdl-accent-gradient);
    color: #fff;
    padding: 0.85rem 2rem;
    border-radius: var(--rdl-radius);
    font-size: 0.92rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3);
    animation: rdl-pulse-border 3s ease-in-out infinite;
    transition: transform var(--rdl-dur) var(--rdl-ease),
                box-shadow var(--rdl-dur) var(--rdl-ease);
}
.rdl-cta-card:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}
.rdl-cta-card::after {
    content: ' \\2192';
    margin-left: 0.5rem;
    font-size: 1rem;
}

.rdl-section-label {
    text-align: center;
    margin-bottom: 1.25rem;
    animation: rdl-enter 0.3s ease both;
}
.rdl-section-label span {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    color: var(--rdl-text-muted);
    position: relative;
    padding: 0 1rem;
}

@keyframes rdl-enter {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes rdl-pulse-border {
    0%, 100% { box-shadow: 0 4px 14px rgba(99,102,241,0.3); }
    50% { box-shadow: 0 4px 14px rgba(99,102,241,0.3), 0 0 0 4px rgba(99,102,241,0.06); }
}

@media (max-width: 1024px) {
    .rdl-features-grid,
    .rdl-modules-grid {
        grid-template-columns: repeat(2, 1fr);
    }
    .rdl-hero h1 {
        font-size: 2.5rem !important;
    }
}
@media (max-width: 768px) {
    .rdl-features-grid,
    .rdl-modules-grid {
        grid-template-columns: 1fr;
    }
    .rdl-stats-bar {
        flex-wrap: wrap;
        gap: 1.5rem;
        padding: 1rem 1.25rem;
    }
    .rdl-hero h1 {
        font-size: 2rem !important;
    }
    .rdl-hero p {
        font-size: 0.95rem;
    }
}
@media (max-width: 480px) {
    .rdl-hero h1 {
        font-size: 1.75rem !important;
    }
    .rdl-stats-bar {
        flex-direction: column;
        gap: 0.75rem;
    }
}
</style>""", unsafe_allow_html=True)


# ─── MODULE IMPORTS ──────────────────────────────────────────────────────────
import modules.ui_helpers  # noqa: F401  — registers RDL Plotly template
from modules.data_manager import render_upload, render_data_manager
from modules.dataset_editor import render_dataset_editor
from modules.descriptive_stats import render_descriptive_stats
from modules.hypothesis_testing import render_hypothesis_testing
from modules.regression import render_regression
from modules.anova import render_anova
from modules.correlation import render_correlation
from modules.visualization import render_visualization
from modules.time_series import render_time_series
from modules.machine_learning import render_machine_learning
from modules.survival_analysis import render_survival_analysis
from modules.quality import render_quality
from modules.doe import render_doe
from modules.text_analytics import render_text_analytics
from modules.monte_carlo import render_monte_carlo
from modules.report import render_report_builder
from modules.templates import render_templates
from modules.experimental import render_experimental
from modules.stability import render_stability
from modules.method_validation import render_method_validation
from modules.bioassay import render_bioassay


@st.cache_data
def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a built-in sample dataset for demonstration."""
    if name == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["species"] = pd.Categorical([data.target_names[t] for t in data.target])
        return df
    elif name == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["wine_class"] = pd.Categorical([str(t) for t in data.target])
        return df
    elif name == "Boston-style Housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["MedianValue"] = data.target
        return df
    elif name == "Diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["progression"] = data.target
        return df
    elif name == "Tips":
        import plotly.express as px
        return px.data.tips()
    elif name == "Gapminder":
        import plotly.express as px
        return px.data.gapminder()
    elif name == "Stocks":
        import plotly.express as px
        return px.data.stocks()
    elif name == "Bioprocess":
        rng = np.random.default_rng(42)
        organisms = ["E. coli", "S. cerevisiae", "CHO", "P. pastoris"]
        media = ["LB", "YPD", "DMEM", "BMGY"]
        scales = ["Flask", "Bench (2L)", "Pilot (50L)"]
        rows = []
        run_id = 0
        for org_idx, organism in enumerate(organisms):
            preferred_media = media[org_idx]
            for scale in scales:
                for replicate in range(1, 4):
                    run_id += 1
                    # Growth parameters vary by organism
                    mu_max = {"E. coli": 0.7, "S. cerevisiae": 0.45,
                              "CHO": 0.03, "P. pastoris": 0.25}[organism]
                    temp_set = {"E. coli": 37.0, "S. cerevisiae": 30.0,
                                "CHO": 37.0, "P. pastoris": 28.0}[organism]
                    ph_set = {"E. coli": 7.0, "S. cerevisiae": 5.5,
                              "CHO": 7.2, "P. pastoris": 5.0}[organism]
                    max_biomass = {"E. coli": 12, "S. cerevisiae": 8,
                                   "CHO": 6, "P. pastoris": 90}[organism]
                    hours = 48 if organism != "CHO" else 168
                    n_points = hours // 2
                    scale_factor = {"Flask": 0.8, "Bench (2L)": 1.0,
                                    "Pilot (50L)": 1.1}[scale]
                    for i in range(n_points):
                        t = i * 2
                        # Logistic growth with noise
                        x0 = 0.1
                        K = max_biomass * scale_factor * rng.normal(1, 0.05)
                        biomass = K / (1 + ((K - x0) / x0) * np.exp(-mu_max * t))
                        biomass *= rng.normal(1, 0.03)
                        biomass = max(0.05, biomass)
                        # Substrate depletion (Monod-like)
                        substrate_0 = 20.0
                        substrate = max(0, substrate_0 * (1 - biomass / K) + rng.normal(0, 0.3))
                        # Product formation (growth-associated + maintenance)
                        product = 0.35 * biomass + 0.01 * t + rng.normal(0, 0.1)
                        product = max(0, product)
                        # Process parameters with noise
                        temp = temp_set + rng.normal(0, 0.3)
                        ph = ph_set + rng.normal(0, 0.1) - 0.005 * t
                        do2 = max(0, min(100, 80 - 2.5 * biomass + rng.normal(0, 3)))
                        agitation = min(1200, 200 + 15 * biomass + rng.normal(0, 10))
                        viability = min(100, max(0, 98 - 0.3 * t + rng.normal(0, 2)))
                        rows.append({
                            "run_id": f"R{run_id:03d}",
                            "organism": organism,
                            "media": preferred_media,
                            "scale": scale,
                            "replicate": replicate,
                            "time_h": t,
                            "temperature_C": round(temp, 1),
                            "pH": round(ph, 2),
                            "dissolved_O2_pct": round(do2, 1),
                            "agitation_rpm": int(agitation),
                            "substrate_g_L": round(substrate, 2),
                            "biomass_g_L": round(biomass, 2),
                            "product_g_L": round(product, 2),
                            "viability_pct": round(viability, 1),
                        })
        df = pd.DataFrame(rows)
        df["organism"] = pd.Categorical(df["organism"])
        df["media"] = pd.Categorical(df["media"])
        df["scale"] = pd.Categorical(df["scale"])
        return df
    elif name == "Lung Cancer (Survival)":
        rng = np.random.default_rng(123)
        n = 228
        treatment = rng.choice(["Drug A", "Drug B", "Placebo"], size=n, p=[0.35, 0.35, 0.30])
        age = rng.normal(62, 10, n).astype(int).clip(30, 90)
        sex = rng.choice(["Male", "Female"], size=n, p=[0.6, 0.4])
        ecog = rng.choice([0, 1, 2, 3], size=n, p=[0.15, 0.45, 0.30, 0.10])

        # Survival times depend on treatment and covariates
        base_time = np.where(treatment == "Drug A", 420, np.where(treatment == "Drug B", 360, 250))
        time_mod = -2 * age + -80 * ecog + np.where(sex == "Female", 40, 0) + rng.normal(0, 80, n)
        time = (base_time + time_mod).clip(10, 1000).astype(int)
        event = rng.binomial(1, 0.72, n)

        df = pd.DataFrame({
            "time": time, "event": event, "treatment": pd.Categorical(treatment),
            "age": age, "sex": pd.Categorical(sex), "ecog_score": ecog,
        })
        return df
    elif name == "DOE Reactor":
        rng = np.random.default_rng(99)
        temps = [-1, 1]
        pressures = [-1, 1]
        catalysts = [-1, 1]
        concs = [-1, 1]
        rows = []
        for t in temps:
            for p in pressures:
                for cat in catalysts:
                    for conc in concs:
                        for rep in range(2):
                            yield_val = (70 + 5*t + 3*p + 4*cat + 2*conc
                                         + 2*t*p + 1.5*t*cat - 1*p*conc
                                         + rng.normal(0, 1.5))
                            rows.append({
                                "Temperature": t, "Pressure": p,
                                "Catalyst": cat, "Concentration": conc,
                                "Yield": round(yield_val, 2),
                            })
        return pd.DataFrame(rows)
    elif name == "SPC Manufacturing":
        rng = np.random.default_rng(77)
        n_samples = 100
        # Process with a shift at sample 60
        values = np.concatenate([
            rng.normal(50.0, 1.0, 60),
            rng.normal(51.2, 1.2, 40),
        ])
        df = pd.DataFrame({
            "Sample": np.arange(1, n_samples + 1),
            "Measurement": values.round(3),
            "Batch": pd.Categorical([f"B{i//10+1:02d}" for i in range(n_samples)]),
            "Operator": pd.Categorical(rng.choice(["Op_A", "Op_B", "Op_C"], n_samples)),
        })
        return df
    elif name == "Stability (ICH)":
        rng = np.random.default_rng(55)
        rows = []
        batches = [("B001", "25C/60%RH"), ("B002", "25C/60%RH"),
                    ("B003", "25C/60%RH"), ("B004", "40C/75%RH")]
        time_points = [0, 3, 6, 9, 12, 18, 24, 36]
        for batch, condition in batches:
            batch_offset = rng.normal(0, 0.3)
            is_accelerated = "40C" in condition
            for t in time_points:
                # Potency declines ~0.5%/year, accelerated ~1.5%/year
                rate = -1.5 / 12 if is_accelerated else -0.5 / 12
                potency = 100.0 + batch_offset + rate * t + rng.normal(0, 0.2)
                # Purity declines slightly
                purity = 99.5 + batch_offset * 0.3 + (rate * 0.4) * t + rng.normal(0, 0.1)
                # Degradation increases
                deg = 0.1 + abs(rate * 0.3) * t + rng.normal(0, 0.05)
                deg = max(0, deg)
                # Moisture
                moisture = 2.0 + 0.02 * t + rng.normal(0, 0.1)
                if is_accelerated:
                    moisture += 0.03 * t
                rows.append({
                    "batch": batch,
                    "time_months": t,
                    "condition": condition,
                    "potency_pct": round(potency, 2),
                    "purity_pct": round(purity, 2),
                    "degradation_A_pct": round(deg, 3),
                    "moisture_pct": round(moisture, 2),
                })
        df = pd.DataFrame(rows)
        df["batch"] = pd.Categorical(df["batch"])
        df["condition"] = pd.Categorical(df["condition"])
        return df
    elif name == "Method Validation":
        rng = np.random.default_rng(56)
        rows = []
        # Calibration: 5 levels x 3 reps
        true_slope = 2.5
        true_intercept = 15.0
        levels = [10, 50, 100, 200, 500]
        for level in levels:
            for rep in range(1, 4):
                response = true_slope * level + true_intercept + rng.normal(0, level * 0.01 + 2)
                rows.append({
                    "sample_type": "Calibration",
                    "level": level,
                    "concentration_known": level,
                    "response": round(response, 2),
                    "analyst": "Analyst_1",
                    "day": "Day_1",
                    "replicate": rep,
                })
        # Accuracy: 3 levels x 3 reps x 2 analysts x 2 days
        accuracy_levels = [80, 100, 120]
        for level in accuracy_levels:
            conc = level  # as % of 100 target
            for analyst in ["Analyst_1", "Analyst_2"]:
                analyst_bias = rng.normal(0, 0.3)
                for day in ["Day_1", "Day_2"]:
                    day_bias = rng.normal(0, 0.2)
                    for rep in range(1, 4):
                        response = true_slope * conc + true_intercept + analyst_bias + day_bias + rng.normal(0, 1.5)
                        rows.append({
                            "sample_type": "Accuracy",
                            "level": level,
                            "concentration_known": conc,
                            "response": round(response, 2),
                            "analyst": analyst,
                            "day": day,
                            "replicate": rep,
                        })
        df = pd.DataFrame(rows)
        df["sample_type"] = pd.Categorical(df["sample_type"])
        df["analyst"] = pd.Categorical(df["analyst"])
        df["day"] = pd.Categorical(df["day"])
        return df
    elif name == "Bioassay (Potency)":
        rng = np.random.default_rng(57)
        rows = []
        # 4PL parameters for reference
        a_ref, b_ref, c_ref, d_ref = 10.0, 1.2, 100.0, 95.0
        # Test sample: ~110% potency (EC50 shifted left)
        c_test = c_ref / 1.10
        doses = [1000, 500, 250, 125, 62.5, 31.25, 15.625, 7.8125]
        for sample, c_param in [("Reference", c_ref), ("Test", c_test)]:
            for dose in doses:
                for rep in range(1, 4):
                    # 4PL: y = d + (a - d) / (1 + (x/c)^b)
                    y = d_ref + (a_ref - d_ref) / (1 + (dose / c_param) ** b_ref)
                    y += rng.normal(0, 1.5)
                    rows.append({
                        "sample": sample,
                        "dose_ng_mL": dose,
                        "log_dose": round(np.log10(dose), 4),
                        "response": round(y, 2),
                        "replicate": rep,
                    })
        df = pd.DataFrame(rows)
        df["sample"] = pd.Categorical(df["sample"])
        return df
    return None


@st.cache_data
def _dataset_info(df):
    """Compute dataset info for sidebar display (cached)."""
    n_numeric = len(df.select_dtypes(include=[np.number]).columns)
    n_cat = len(df.select_dtypes(include=["object", "category"]).columns)
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0.0
    mem_kb = df.memory_usage(deep=True).sum() / 1024
    return n_numeric, n_cat, missing_pct, mem_kb


def _apply_data_filters(df):
    """Interactive filter bar at top of main content. Returns filtered DataFrame.

    Modules receive the filtered df — zero changes needed in module code.
    Filter state lives in widget keys so it survives reruns naturally.
    """
    with st.expander("Data Filters", expanded=False):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        filterable = num_cols + cat_cols + date_cols

        if not filterable:
            st.caption("No filterable columns available.")
            return df

        filter_cols = st.multiselect(
            "Filter by:", filterable, key="fbar_cols",
            help="Select columns to add filter controls",
        )

        if not filter_cols:
            return df

        filtered = df
        for col in filter_cols:
            if col in num_cols:
                col_data = df[col].dropna()
                if col_data.empty:
                    continue
                col_min, col_max = float(col_data.min()), float(col_data.max())
                if col_min >= col_max:
                    continue
                vals = st.slider(
                    f"{col}:", col_min, col_max, (col_min, col_max),
                    key=f"fbar_n_{col}",
                )
                filtered = filtered[
                    (filtered[col] >= vals[0]) & (filtered[col] <= vals[1])
                ]
            elif col in cat_cols:
                unique = sorted(df[col].dropna().unique().tolist(), key=str)
                if not unique:
                    continue
                selected = st.multiselect(
                    f"{col}:", unique, default=unique, key=f"fbar_c_{col}",
                )
                if selected:
                    filtered = filtered[filtered[col].isin(selected)]
            elif col in date_cols:
                try:
                    min_d = df[col].min().date()
                    max_d = df[col].max().date()
                    date_range = st.date_input(
                        f"{col}:", value=(min_d, max_d), key=f"fbar_d_{col}",
                    )
                    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                        filtered = filtered[
                            (filtered[col].dt.date >= date_range[0])
                            & (filtered[col].dt.date <= date_range[1])
                        ]
                except Exception:
                    pass

    n_total = len(df)
    n_filtered = len(filtered)
    if n_filtered < n_total:
        st.markdown(
            f'<div style="background:var(--rdl-accent-light);color:var(--rdl-accent);'
            f'padding:0.35rem 0.75rem;border-radius:var(--rdl-radius-xs);'
            f'display:inline-block;font-size:0.8rem;font-weight:600;'
            f'margin-bottom:0.5rem;">'
            f'Filtered: {n_filtered:,} of {n_total:,} rows</div>',
            unsafe_allow_html=True,
        )

    return filtered


def main():
    with st.sidebar:
        st.markdown("""
            <div class="rdl-brand">
                <div class="rdl-brand-mark">RDL</div>
                <div class="rdl-brand-text">
                    <span class="rdl-brand-name">Ryan's Data Lab</span>
                    <span class="rdl-brand-tag">Visual Analysis Platform</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        _components.html("""
        <script>
        (function() {
            const doc = window.parent.document;
            const brand = doc.querySelector('.rdl-brand');
            if (brand && !brand.dataset.homeWired) {
                brand.dataset.homeWired = '1';
                brand.addEventListener('click', function() {
                    const labels = doc.querySelectorAll(
                        '[data-testid="stSidebar"] [role="radiogroup"] label'
                    );
                    for (const lbl of labels) {
                        if (lbl.textContent.trim() === 'Home') {
                            lbl.click();
                            break;
                        }
                    }
                });
            }
        })();
        </script>
        """, height=0)

        dark_mode = st.toggle("Dark Mode", key="dark_mode")

        if dark_mode:
            st.markdown("""<style>
            :root {
                --rdl-bg: #0f172a !important;
                --rdl-bg-card: rgba(30,41,59,0.72) !important;
                --rdl-text: #e2e8f0 !important;
                --rdl-text-secondary: #94a3b8 !important;
                --rdl-text-muted: #64748b !important;
                --rdl-glass-bg: rgba(30,41,59,0.72) !important;
                --rdl-border: rgba(255,255,255,0.08) !important;
                --rdl-shadow-xs: 0 1px 3px rgba(0,0,0,0.2) !important;
                --rdl-shadow-sm: 0 2px 8px rgba(0,0,0,0.25) !important;
                --rdl-shadow-md: 0 4px 16px rgba(0,0,0,0.3) !important;
            }
            .stApp {
                background: #0f172a !important;
                color: #e2e8f0 !important;
            }
            [data-testid="stSidebar"] {
                background: #0a0f1a !important;
            }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p,
            .stApp label, .stApp span, .stMarkdown {
                color: #e2e8f0 !important;
            }
            .rdl-hero h1, .rdl-hero p {
                color: #e2e8f0 !important;
            }
            .rdl-feature-card, .rdl-stat {
                background: rgba(30,41,59,0.72) !important;
                border-color: rgba(255,255,255,0.08) !important;
            }
            .rdl-feature-card h3, .rdl-feature-card p,
            .rdl-stat .rdl-stat-label {
                color: #e2e8f0 !important;
            }
            .rdl-dataset-info {
                background: rgba(30,41,59,0.72) !important;
            }
            .rdl-info-row span {
                color: #94a3b8 !important;
            }
            .rdl-info-row span:last-child {
                color: #e2e8f0 !important;
            }
            [data-testid="stExpander"] {
                background: rgba(30,41,59,0.5) !important;
                border-color: rgba(255,255,255,0.08) !important;
            }
            .js-plotly-plot .plotly .main-svg {
                background: #0f172a !important;
            }
            .js-plotly-plot .plotly .bg {
                fill: #1e293b !important;
            }
            </style>""", unsafe_allow_html=True)

        st.divider()

        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Upload File", "Sample Dataset"],
            label_visibility="collapsed",
        )

        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your dataset",
                type=["csv", "xlsx", "xls", "tsv", "json"],
                help="Supported formats: CSV, Excel, TSV, JSON",
            )
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        st.session_state["df"] = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith((".xlsx", ".xls")):
                        st.session_state["df"] = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith(".tsv"):
                        st.session_state["df"] = pd.read_csv(uploaded_file, sep="\t")
                    elif uploaded_file.name.endswith(".json"):
                        st.session_state["df"] = pd.read_json(uploaded_file)
                    st.session_state["data_name"] = uploaded_file.name
                    st.success(f"Loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            sample = st.selectbox(
                "Select dataset:",
                ["Iris", "Wine", "Boston-style Housing", "Diabetes",
                 "Tips", "Gapminder", "Stocks", "Bioprocess",
                 "Lung Cancer (Survival)", "DOE Reactor", "SPC Manufacturing",
                 "Stability (ICH)", "Method Validation", "Bioassay (Potency)"],
            )
            if st.button("Load Dataset", use_container_width=True):
                st.session_state["df"] = load_sample_dataset(sample)
                st.session_state["data_name"] = sample
                st.success(f"Loaded: {sample}")

        st.divider()

        st.subheader("Analysis Module")

        _MODULE_DESCRIPTIONS = {
            "Home": "Overview of Ryan's Data Lab and available modules.",
            "Data Manager": "Upload, clean, reshape, merge, string ops, date extraction, and export.",
            "Dataset Editor": "Spreadsheet-like cell editing, find & replace, validation rules, and undo history.",
            "Descriptive Statistics": "Summary stats, distribution fitting, normality tests, tolerance intervals.",
            "Visualization Builder": "34+ interactive chart types: Sankey, ridgeline, hexbin, and more.",
            "Hypothesis Testing": "t-tests, chi-square, McNemar, Bartlett, runs test, expanded power analysis.",
            "Correlation & Multivariate": "Correlation matrices, PCA, t-SNE, correspondence analysis, MDS.",
            "Regression Analysis": "Linear, GLM, stepwise, multinomial/ordinal logistic, WLS, mixed models.",
            "ANOVA": "One-way, two-way, MANOVA, Games-Howell, Dunnett, repeated measures.",
            "Time Series Analysis": "Decomposition, ARIMA/SARIMA, change point detection, forecasting.",
            "Machine Learning": "XGBoost, LightGBM, SHAP, GMM, precision-recall, model comparison.",
            "Survival Analysis": "Kaplan-Meier, log-rank test, Cox PH, parametric AFT models.",
            "Quality & SPC": "Control charts, multi-vari analysis, fishbone diagrams, capability, EM monitoring.",
            "Stability Analysis (ICH)": "ICH Q1E stability trending, poolability, shelf-life estimation.",
            "Method Validation (ICH)": "ICH Q2 linearity, accuracy, precision, LOD/LOQ, system suitability.",
            "Bioassay & Potency": "4PL/5PL dose-response, parallel line assay, relative potency (Fieller's CI).",
            "Design of Experiments": "Factorial, CCD, Taguchi, Latin hypercube, response surface.",
            "Text Analytics": "Text exploration, TF-IDF, word clouds, sentiment analysis.",
            "Monte Carlo Simulation": "Distribution simulation, process propagation, risk analysis.",
            "Report Builder": "Build and download HTML reports from your analyses.",
            "Templates": "Save and load analysis configurations as JSON templates.",
            "Experimental": "Visual workflow builder and AI-powered data analysis assistant.",
        }

        _ALL_MODULES = [
            "Home", "Data Manager", "Dataset Editor", "Descriptive Statistics",
            "Visualization Builder", "Hypothesis Testing",
            "Correlation & Multivariate", "Regression Analysis", "ANOVA",
            "Time Series Analysis", "Machine Learning", "Survival Analysis",
            "Quality & SPC", "Stability Analysis (ICH)",
            "Method Validation (ICH)", "Bioassay & Potency",
            "Design of Experiments", "Text Analytics",
            "Monte Carlo Simulation", "Report Builder", "Templates",
            "Experimental",
        ]

        module_search = st.text_input("", placeholder="Search modules...", key="module_search")

        st.session_state.setdefault("_recent_modules", [])

        if module_search:
            _search_lower = module_search.lower()
            _filtered_modules = [
                m for m in _ALL_MODULES
                if _search_lower in m.lower()
                or _search_lower in _MODULE_DESCRIPTIONS.get(m, "").lower()
            ]
            if _filtered_modules:
                module = st.radio(
                    "Select module:",
                    _filtered_modules,
                    key="module_radio",
                    label_visibility="collapsed",
                )
            else:
                st.info("No matching modules")
                module = "Home"
        else:
            # Show recent modules section
            _recent = st.session_state["_recent_modules"]
            if _recent:
                st.caption("Recent")
                _rcols = st.columns(min(len(_recent), 3))
                for _ri, _rm in enumerate(_recent):
                    with _rcols[_ri % min(len(_recent), 3)]:
                        if st.button(_rm, key=f"_recent_{_ri}", use_container_width=True):
                            st.session_state["module_radio"] = _rm
                            st.rerun()

            module = st.radio(
                "Select module:",
                _ALL_MODULES,
                key="module_radio",
                label_visibility="collapsed",
            )

        # Track recently used modules
        if module and module != "Home":
            _recent = st.session_state["_recent_modules"]
            if module in _recent:
                _recent.remove(module)
            _recent.insert(0, module)
            st.session_state["_recent_modules"] = _recent[:5]

        desc = _MODULE_DESCRIPTIONS.get(module, "")
        if desc:
            st.caption(desc)

        if "df" in st.session_state and st.session_state["df"] is not None:
            st.divider()
            st.subheader("Active Dataset")
            df = st.session_state["df"]
            data_name = _html.escape(
                st.session_state.get("data_name", "Unknown")
            )
            n_numeric, n_cat, missing_pct, mem_kb = _dataset_info(df)

            st.markdown(f"""
                <div class="rdl-dataset-info">
                    <div class="rdl-info-row">
                        <span>Dataset</span><span>{data_name}</span>
                    </div>
                    <div class="rdl-info-row">
                        <span>Rows</span><span>{df.shape[0]:,}</span>
                    </div>
                    <div class="rdl-info-row">
                        <span>Columns</span><span>{df.shape[1]}</span>
                    </div>
                    <div class="rdl-info-row">
                        <span>Numeric</span><span>{n_numeric}</span>
                    </div>
                    <div class="rdl-info-row">
                        <span>Categorical</span><span>{n_cat}</span>
                    </div>
                    <div class="rdl-info-row">
                        <span>Missing</span><span>{missing_pct:.1f}%</span>
                    </div>
                    <div class="rdl-info-row">
                        <span>Memory</span><span>{mem_kb:.0f} KB</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # ── Column Roles ──
            st.session_state.setdefault("column_roles", {})
            with st.expander("Column Roles"):
                _role_options = [None, "Y (Response)", "X (Predictor)", "Group", "ID"]
                _cr_cols_list = list(df.columns)
                for _ci in range(0, len(_cr_cols_list), 2):
                    _cr_pair = st.columns(2)
                    for _cj, _cr_col in enumerate(_cr_cols_list[_ci:_ci + 2]):
                        with _cr_pair[_cj]:
                            _current_role = st.session_state["column_roles"].get(
                                _cr_col, {}
                            ).get("role", None)
                            _role_idx = (
                                _role_options.index(_current_role)
                                if _current_role in _role_options
                                else 0
                            )
                            _selected = st.selectbox(
                                _cr_col,
                                _role_options,
                                index=_role_idx,
                                key=f"_colrole_{_cr_col}",
                                format_func=lambda x: "—" if x is None else x,
                            )
                            st.session_state["column_roles"][_cr_col] = {
                                "role": _selected
                            }

            # ── Analysis History ──
            st.divider()
            with st.expander("History"):
                _log = st.session_state.get("analysis_log", [])
                if _log:
                    _display_log = _log[-10:][::-1]
                    for _entry in _display_log:
                        st.markdown(
                            f"**{_entry['timestamp']}** | "
                            f"*{_entry['module']}* — {_entry['action']}"
                        )
                        if _entry.get("summary"):
                            st.caption(_entry["summary"])
                    # Export Log button
                    _log_df = pd.DataFrame(_log)
                    _csv_buf = io.StringIO()
                    _log_df.to_csv(_csv_buf, index=False)
                    st.download_button(
                        "Export Log",
                        data=_csv_buf.getvalue(),
                        file_name="analysis_log.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                else:
                    st.caption("No analysis history yet.")

    # Main content
    if module == "Home" or "df" not in st.session_state or st.session_state["df"] is None:
        st.markdown("""
            <div class="rdl-hero">
                <h1>Ryan's <span class="rdl-accent-text">Data Lab</span></h1>
                <p>
                    Professional-grade visual data analysis. Upload a dataset
                    and explore with interactive charts, statistical tests,
                    machine learning, and more.
                </p>
            </div>

            <div class="rdl-stats-bar">
                <div class="rdl-stat">
                    <span class="rdl-stat-value">34+</span>
                    <span class="rdl-stat-label">Chart Types</span>
                </div>
                <div class="rdl-stat">
                    <span class="rdl-stat-value">21</span>
                    <span class="rdl-stat-label">Modules</span>
                </div>
                <div class="rdl-stat">
                    <span class="rdl-stat-value">100+</span>
                    <span class="rdl-stat-label">Statistical Tests</span>
                </div>
                <div class="rdl-stat">
                    <span class="rdl-stat-value">9+</span>
                    <span class="rdl-stat-label">ML Algorithms</span>
                </div>
            </div>

            <div class="rdl-features-grid">
                <div class="rdl-feature-card" style="--cat-color: #3b82f6">
                    <h3>Upload, Explore & Export</h3>
                    <p>
                        Import CSV, Excel, TSV, or JSON files. Preview, clean,
                        transform, and export data in multiple formats.
                    </p>
                </div>
                <div class="rdl-feature-card" style="--cat-color: #6366f1">
                    <h3>Analyze & Test</h3>
                    <p>
                        Comprehensive statistical suite: hypothesis testing,
                        ANOVA, GLM, mixed models, bootstrap, survival analysis,
                        and DOE — with built-in assumption validation.
                    </p>
                </div>
                <div class="rdl-feature-card" style="--cat-color: #8b5cf6">
                    <h3>Visualize & Model</h3>
                    <p>
                        22+ chart types, XGBoost, SHAP, UMAP, SPC control charts,
                        response surfaces, and model comparison.
                    </p>
                </div>
            </div>

            <div class="rdl-section-label">
                <span>All Modules</span>
            </div>

            <div class="rdl-modules-grid">
                <div class="rdl-module-item" style="--cat-color: #3b82f6">
                    <h4>Data Management</h4>
                    <p>Upload, clean, transform, filter, encode, export CSV/Excel/JSON.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #6366f1">
                    <h4>Descriptive Statistics</h4>
                    <p>Summary stats, distributions, normality tests, outlier detection.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #6366f1">
                    <h4>Visualization Builder</h4>
                    <p>Scatter, bar, heatmap, 3D, treemap, radar, candlestick, and more.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #6366f1">
                    <h4>Hypothesis Testing</h4>
                    <p>t-tests, chi-square, bootstrap CIs, permutation tests, power analysis.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #6366f1">
                    <h4>Correlation & PCA</h4>
                    <p>Correlation matrices, PCA, t-SNE, UMAP, factor analysis.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #8b5cf6">
                    <h4>Regression Analysis</h4>
                    <p>Linear, GLM, robust, quantile, mixed models, curve fitting.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #6366f1">
                    <h4>ANOVA</h4>
                    <p>One-way, two-way, repeated measures, ANCOVA, Kruskal-Wallis.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #8b5cf6">
                    <h4>Time Series</h4>
                    <p>Decomposition, ARIMA/SARIMA, smoothing, forecasting.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #8b5cf6">
                    <h4>Machine Learning</h4>
                    <p>XGBoost, LightGBM, SHAP, clustering, model comparison.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #10b981">
                    <h4>Survival Analysis</h4>
                    <p>Kaplan-Meier, log-rank test, Cox PH, parametric AFT models.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #10b981">
                    <h4>Quality & SPC</h4>
                    <p>Control charts (I-MR, X-bar), attributes charts, process capability.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #10b981">
                    <h4>Design of Experiments</h4>
                    <p>Factorial, CCD, Box-Behnken, response surface, effect analysis.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #06b6d4">
                    <h4>Stability Analysis (ICH)</h4>
                    <p>ICH Q1E trending, poolability testing, shelf-life estimation.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #06b6d4">
                    <h4>Method Validation (ICH)</h4>
                    <p>Linearity, accuracy, precision, LOD/LOQ, system suitability.</p>
                </div>
                <div class="rdl-module-item" style="--cat-color: #06b6d4">
                    <h4>Bioassay & Potency</h4>
                    <p>4PL/5PL dose-response, parallel line assay, relative potency.</p>
                </div>
            </div>

            <div class="rdl-cta">
                <span class="rdl-cta-card">
                    Select a dataset from the sidebar to begin
                </span>
            </div>
        """, unsafe_allow_html=True)

        # Show active dataset summary on Home if data is loaded
        if "df" in st.session_state and st.session_state["df"] is not None:
            df_home = st.session_state["df"]
            data_name_home = st.session_state.get("data_name", "Unknown")
            st.info(f"**{data_name_home}** is loaded ({df_home.shape[0]:,} rows, {df_home.shape[1]} columns). Select a module from the sidebar to analyze.")
        return

    df_raw = st.session_state["df"]

    # Data Manager operates on the raw (unfiltered) df so filters
    # don't permanently delete rows when the user saves changes.
    if module == "Data Manager":
        st.markdown("## Data Manager")
        try:
            result = render_data_manager(df_raw)
            if result is not None:
                st.session_state["df"] = result
        except Exception as e:
            st.error(f"An error occurred in Data Manager: {e}")
        return

    # Dataset Editor also operates on raw df (it mutates data).
    if module == "Dataset Editor":
        st.markdown("## Dataset Editor")
        try:
            render_dataset_editor(df_raw)
        except Exception as e:
            st.error(f"An error occurred in Dataset Editor: {e}")
        return

    # ── Interactive filter bar (all other modules) ──
    df = _apply_data_filters(df_raw)

    if module == "Descriptive Statistics":
        st.markdown("## Descriptive Statistics")
        try:
            render_descriptive_stats(df)
        except Exception as e:
            st.error(f"An error occurred in Descriptive Statistics: {e}")
    elif module == "Visualization Builder":
        st.markdown("## Visualization Builder")
        try:
            render_visualization(df)
        except Exception as e:
            st.error(f"An error occurred in Visualization Builder: {e}")
    elif module == "Hypothesis Testing":
        st.markdown("## Hypothesis Testing")
        try:
            render_hypothesis_testing(df)
        except Exception as e:
            st.error(f"An error occurred in Hypothesis Testing: {e}")
    elif module == "Correlation & Multivariate":
        st.markdown("## Correlation & Multivariate Analysis")
        try:
            render_correlation(df)
        except Exception as e:
            st.error(f"An error occurred in Correlation & Multivariate: {e}")
    elif module == "Regression Analysis":
        st.markdown("## Regression Analysis")
        try:
            render_regression(df)
        except Exception as e:
            st.error(f"An error occurred in Regression Analysis: {e}")
    elif module == "ANOVA":
        st.markdown("## ANOVA")
        try:
            render_anova(df)
        except Exception as e:
            st.error(f"An error occurred in ANOVA: {e}")
    elif module == "Time Series Analysis":
        st.markdown("## Time Series Analysis")
        try:
            render_time_series(df)
        except Exception as e:
            st.error(f"An error occurred in Time Series Analysis: {e}")
    elif module == "Machine Learning":
        st.markdown("## Machine Learning")
        try:
            render_machine_learning(df)
        except Exception as e:
            st.error(f"An error occurred in Machine Learning: {e}")
    elif module == "Survival Analysis":
        st.markdown("## Survival Analysis")
        try:
            render_survival_analysis(df)
        except Exception as e:
            st.error(f"An error occurred in Survival Analysis: {e}")
    elif module == "Quality & SPC":
        st.markdown("## Quality & Statistical Process Control")
        try:
            render_quality(df)
        except Exception as e:
            st.error(f"An error occurred in Quality & SPC: {e}")
    elif module == "Stability Analysis (ICH)":
        st.markdown("## Stability Analysis (ICH Q1E)")
        try:
            render_stability(df)
        except Exception as e:
            st.error(f"An error occurred in Stability Analysis: {e}")
    elif module == "Method Validation (ICH)":
        st.markdown("## Method Validation (ICH Q2)")
        try:
            render_method_validation(df)
        except Exception as e:
            st.error(f"An error occurred in Method Validation: {e}")
    elif module == "Bioassay & Potency":
        st.markdown("## Bioassay & Relative Potency")
        try:
            render_bioassay(df)
        except Exception as e:
            st.error(f"An error occurred in Bioassay & Potency: {e}")
    elif module == "Design of Experiments":
        st.markdown("## Design of Experiments")
        try:
            render_doe(df)
        except Exception as e:
            st.error(f"An error occurred in Design of Experiments: {e}")
    elif module == "Text Analytics":
        st.markdown("## Text Analytics")
        try:
            render_text_analytics(df)
        except Exception as e:
            st.error(f"An error occurred in Text Analytics: {e}")
    elif module == "Monte Carlo Simulation":
        st.markdown("## Monte Carlo Simulation")
        try:
            render_monte_carlo(df)
        except Exception as e:
            st.error(f"An error occurred in Monte Carlo Simulation: {e}")
    elif module == "Report Builder":
        st.markdown("## Report Builder")
        try:
            render_report_builder(df)
        except Exception as e:
            st.error(f"An error occurred in Report Builder: {e}")
    elif module == "Templates":
        st.markdown("## Analysis Templates")
        try:
            render_templates()
        except Exception as e:
            st.error(f"An error occurred in Templates: {e}")
    elif module == "Experimental":
        st.markdown("## Experimental")
        try:
            render_experimental(df_raw)
        except Exception as e:
            st.error(f"An error occurred in Experimental: {e}")


if __name__ == "__main__":
    main()
