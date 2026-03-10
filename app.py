"""
Ryan's Data Lab (RDL) - Visual Data Analysis Tool
A comprehensive, interactive data analysis platform modeled after
JMP, MATLAB, and other leading statistical analysis tools.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import html as _html

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
    --rdl-bg: #f4f5f9;
    --rdl-bg-card: #ffffff;
    --rdl-sidebar-bg: #111827;
    --rdl-sidebar-text: #cbd5e1;
    --rdl-sidebar-bright: #f1f5f9;
    --rdl-sidebar-hover: rgba(255,255,255,0.06);
    --rdl-sidebar-active: rgba(99,102,241,0.15);
    --rdl-sidebar-border: rgba(255,255,255,0.06);
    --rdl-text: #1e293b;
    --rdl-text-secondary: #64748b;
    --rdl-text-muted: #94a3b8;
    --rdl-accent: #6366f1;
    --rdl-accent-light: #eef2ff;
    --rdl-accent-hover: #4f46e5;
    --rdl-accent-subtle: rgba(99,102,241,0.08);
    --rdl-success: #22c55e;
    --rdl-warning: #f59e0b;
    --rdl-error: #ef4444;
    --rdl-info: #3b82f6;
    --rdl-border: #e2e8f0;
    --rdl-shadow-xs: 0 1px 2px rgba(0,0,0,0.03);
    --rdl-shadow-sm: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03);
    --rdl-shadow-md: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -2px rgba(0,0,0,0.03);
    --rdl-shadow-lg: 0 10px 25px -5px rgba(0,0,0,0.06), 0 4px 6px -4px rgba(0,0,0,0.03);
    --rdl-radius: 14px;
    --rdl-radius-sm: 10px;
    --rdl-radius-xs: 6px;
    --rdl-ease: cubic-bezier(0.4, 0, 0.2, 1);
    --rdl-dur: 0.2s;
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
    background: rgba(244,245,249,0.8) !important;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

# ─── SIDEBAR STYLES ─────────────────────────────────────────────────────────
st.markdown("""<style>
[data-testid="stSidebar"] {
    background: var(--rdl-sidebar-bg) !important;
    border-right: 1px solid rgba(255,255,255,0.04) !important;
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
    transition: background var(--rdl-dur) var(--rdl-ease) !important;
    margin-bottom: 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
    background: var(--rdl-sidebar-hover) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {
    background: var(--rdl-sidebar-active) !important;
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
    box-shadow: 0 4px 12px rgba(99,102,241,0.35) !important;
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
}
.rdl-brand-mark {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #6366f1, #818cf8);
    border-radius: 11px;
    font-weight: 800;
    font-size: 0.78rem;
    color: #ffffff;
    letter-spacing: 0.03em;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(99,102,241,0.3);
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
    border: 1px solid rgba(255,255,255,0.06);
}
.rdl-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.3rem 0;
    font-size: 0.78rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
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
    border-bottom: 2px solid var(--rdl-accent) !important;
    display: inline-block !important;
}
[data-testid="stAppViewContainer"] h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: var(--rdl-text) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: var(--rdl-bg-card) !important;
    border-radius: var(--rdl-radius-sm) !important;
    padding: 4px !important;
    border: 1px solid var(--rdl-border) !important;
    box-shadow: var(--rdl-shadow-xs) !important;
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
    box-shadow: 0 2px 8px rgba(99,102,241,0.25) !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.25rem !important;
}

[data-testid="stMetric"] {
    background: var(--rdl-bg-card) !important;
    border: 1px solid var(--rdl-border) !important;
    border-radius: var(--rdl-radius-sm) !important;
    padding: 1rem 1.25rem !important;
    box-shadow: var(--rdl-shadow-sm) !important;
    transition: box-shadow var(--rdl-dur) var(--rdl-ease),
                transform var(--rdl-dur) var(--rdl-ease) !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--rdl-shadow-md) !important;
    transform: translateY(-1px);
}
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    font-weight: 600 !important;
    color: var(--rdl-text-secondary) !important;
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
    border: 1px solid var(--rdl-border) !important;
    background: var(--rdl-bg-card) !important;
    color: var(--rdl-text) !important;
    transition: border-color var(--rdl-dur) var(--rdl-ease),
                color var(--rdl-dur) var(--rdl-ease),
                box-shadow var(--rdl-dur) var(--rdl-ease) !important;
}
.stButton > button:hover {
    border-color: var(--rdl-accent) !important;
    color: var(--rdl-accent) !important;
    box-shadow: var(--rdl-shadow-sm) !important;
}
.stButton > button:active {
    transform: translateY(0.5px);
}
.stDownloadButton > button {
    background: var(--rdl-accent) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: var(--rdl-radius-xs) !important;
}
.stDownloadButton > button:hover {
    background: var(--rdl-accent-hover) !important;
    box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
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
    box-shadow: 0 0 0 3px var(--rdl-accent-subtle) !important;
}

[data-testid="stExpander"] {
    background: var(--rdl-bg-card) !important;
    border: 1px solid var(--rdl-border) !important;
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
    box-shadow: var(--rdl-shadow-xs) !important;
}

[data-testid="stAlert"] {
    border-radius: var(--rdl-radius-sm) !important;
    font-size: 0.88rem !important;
}

[data-testid="stAppViewContainer"] hr {
    border-color: var(--rdl-border) !important;
    opacity: 0.5 !important;
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
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: var(--rdl-accent) !important;
}

/* ── Significance Result Cards ── */
.rdl-sig-card {
    background: var(--rdl-bg-card);
    border: 1px solid var(--rdl-border);
    border-left: 4px solid var(--rdl-success);
    border-radius: var(--rdl-radius-sm);
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    box-shadow: var(--rdl-shadow-sm);
}
.rdl-sig-card--reject {
    border-left-color: var(--rdl-error);
}
.rdl-sig-card--accept {
    border-left-color: var(--rdl-success);
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
    padding: 2rem 1rem;
    margin: 1rem 0;
    border: 1px dashed var(--rdl-border);
    border-radius: var(--rdl-radius-sm);
    background: var(--rdl-bg-card);
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

/* ── Metric hover enhancement ── */
[data-testid="stMetric"]:hover {
    transform: translateY(-3px) !important;
    border-left: 3px solid var(--rdl-accent) !important;
}
</style>""", unsafe_allow_html=True)

# ─── LANDING PAGE STYLES ────────────────────────────────────────────────────
st.markdown("""<style>
.rdl-hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
    animation: rdl-fade-up 0.5s var(--rdl-ease) both;
}
.rdl-hero h1 {
    font-size: 2.75rem !important;
    font-weight: 800 !important;
    color: var(--rdl-text) !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.75rem !important;
    line-height: 1.1 !important;
    border-bottom: none !important;
    display: block !important;
    padding-bottom: 0 !important;
}
.rdl-accent-text {
    background: linear-gradient(135deg, #6366f1, #a78bfa, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.rdl-hero p {
    font-size: 1.1rem;
    color: var(--rdl-text-secondary);
    font-weight: 400;
    max-width: 540px;
    margin: 0 auto;
    line-height: 1.65;
}

.rdl-stats-bar {
    display: flex;
    justify-content: center;
    gap: 3rem;
    padding: 2rem 1rem;
    margin: 0.5rem auto 2rem;
    max-width: 700px;
    animation: rdl-fade-up 0.5s 0.08s var(--rdl-ease) both;
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
    animation: rdl-fade-up 0.5s 0.16s var(--rdl-ease) both;
}
.rdl-feature-card {
    background: var(--rdl-bg-card);
    border: 1px solid var(--rdl-border);
    border-radius: var(--rdl-radius);
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.25s var(--rdl-ease),
                transform 0.25s var(--rdl-ease),
                border-color 0.25s var(--rdl-ease);
}
.rdl-feature-card-bar {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--rdl-accent), #a78bfa);
    opacity: 0;
    transition: opacity 0.25s var(--rdl-ease);
}
.rdl-feature-card:hover {
    box-shadow: var(--rdl-shadow-lg);
    transform: translateY(-3px);
    border-color: rgba(99,102,241,0.2);
}
.rdl-feature-card:hover .rdl-feature-card-bar {
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
    animation: rdl-fade-up 0.5s 0.24s var(--rdl-ease) both;
}
.rdl-module-item {
    background: var(--rdl-bg-card);
    border: 1px solid var(--rdl-border);
    border-radius: var(--rdl-radius);
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.25s var(--rdl-ease),
                transform 0.25s var(--rdl-ease),
                border-color 0.25s var(--rdl-ease);
}
.rdl-module-item::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--rdl-accent), #a78bfa);
    opacity: 0;
    transition: opacity 0.25s var(--rdl-ease);
}
.rdl-module-item:hover {
    box-shadow: var(--rdl-shadow-lg);
    transform: translateY(-3px);
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
.rdl-module-item p {
    font-size: 0.78rem;
    color: var(--rdl-text-secondary);
    line-height: 1.55;
    margin: 0;
}

.rdl-cta {
    text-align: center;
    padding: 1.5rem 1rem 3rem;
    animation: rdl-fade-up 0.5s 0.3s var(--rdl-ease) both;
}
.rdl-cta-pill {
    display: inline-block;
    background: var(--rdl-accent-light);
    padding: 0.7rem 1.5rem;
    border-radius: 999px;
    font-size: 0.88rem;
    font-weight: 500;
    color: var(--rdl-accent);
    border: 1px solid rgba(99,102,241,0.15);
}

.rdl-section-label {
    text-align: center;
    margin-bottom: 1.25rem;
    animation: rdl-fade-up 0.5s 0.2s var(--rdl-ease) both;
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

@keyframes rdl-fade-up {
    from {
        opacity: 0;
        transform: translateY(12px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
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
    }
    .rdl-hero h1 {
        font-size: 2rem !important;
    }
}
</style>""", unsafe_allow_html=True)


# ─── MODULE IMPORTS ──────────────────────────────────────────────────────────
import modules.ui_helpers  # noqa: F401  — registers RDL Plotly template
from modules.data_manager import render_upload, render_data_manager
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
                 "Lung Cancer (Survival)", "DOE Reactor", "SPC Manufacturing"],
            )
            if st.button("Load Dataset", use_container_width=True):
                st.session_state["df"] = load_sample_dataset(sample)
                st.session_state["data_name"] = sample
                st.success(f"Loaded: {sample}")

        st.divider()

        st.subheader("Analysis Module")
        module = st.radio(
            "Select module:",
            [
                "Data Manager",
                "Descriptive Statistics",
                "Visualization Builder",
                "Hypothesis Testing",
                "Correlation & Multivariate",
                "Regression Analysis",
                "ANOVA",
                "Time Series Analysis",
                "Machine Learning",
                "Survival Analysis",
                "Quality & SPC",
                "Design of Experiments",
            ],
            label_visibility="collapsed",
        )

        _MODULE_DESCRIPTIONS = {
            "Data Manager": "Upload, clean, transform, filter, encode, and export datasets.",
            "Descriptive Statistics": "Summary statistics, distributions, normality tests, outlier detection.",
            "Visualization Builder": "22+ interactive chart types with full customization.",
            "Hypothesis Testing": "t-tests, chi-square, bootstrap CIs, permutation tests, power analysis.",
            "Correlation & Multivariate": "Correlation matrices, PCA, t-SNE, UMAP, factor analysis.",
            "Regression Analysis": "Linear, GLM, robust, quantile, mixed models, and curve fitting.",
            "ANOVA": "One-way, two-way, repeated measures, ANCOVA, Kruskal-Wallis.",
            "Time Series Analysis": "Decomposition, ARIMA/SARIMA, smoothing, and forecasting.",
            "Machine Learning": "XGBoost, LightGBM, SHAP, clustering, and model comparison.",
            "Survival Analysis": "Kaplan-Meier, log-rank test, Cox PH, parametric AFT models.",
            "Quality & SPC": "Control charts (I-MR, X-bar), attributes charts, process capability.",
            "Design of Experiments": "Factorial, CCD, Box-Behnken, response surface, effect analysis.",
        }
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

    # Main content
    if "df" not in st.session_state or st.session_state["df"] is None:
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
                    <span class="rdl-stat-value">22+</span>
                    <span class="rdl-stat-label">Chart Types</span>
                </div>
                <div class="rdl-stat">
                    <span class="rdl-stat-value">12</span>
                    <span class="rdl-stat-label">Modules</span>
                </div>
                <div class="rdl-stat">
                    <span class="rdl-stat-value">70+</span>
                    <span class="rdl-stat-label">Statistical Tests</span>
                </div>
                <div class="rdl-stat">
                    <span class="rdl-stat-value">9+</span>
                    <span class="rdl-stat-label">ML Algorithms</span>
                </div>
            </div>

            <div class="rdl-features-grid">
                <div class="rdl-feature-card">
                    <div class="rdl-feature-card-bar"></div>
                    <h3>Upload, Explore & Export</h3>
                    <p>
                        Import CSV, Excel, TSV, or JSON files. Preview, clean,
                        transform, and export data in multiple formats.
                    </p>
                </div>
                <div class="rdl-feature-card">
                    <div class="rdl-feature-card-bar"></div>
                    <h3>Analyze & Test</h3>
                    <p>
                        Comprehensive statistical suite: hypothesis testing,
                        ANOVA, GLM, mixed models, bootstrap, survival analysis,
                        and DOE.
                    </p>
                </div>
                <div class="rdl-feature-card">
                    <div class="rdl-feature-card-bar"></div>
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
                <div class="rdl-module-item">
                    <h4>📂 Data Management</h4>
                    <p>Upload, clean, transform, filter, encode, export CSV/Excel/JSON.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>📊 Descriptive Statistics</h4>
                    <p>Summary stats, distributions, normality tests, outlier detection.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>📈 Visualization Builder</h4>
                    <p>Scatter, bar, heatmap, 3D, treemap, radar, candlestick, and more.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>🧪 Hypothesis Testing</h4>
                    <p>t-tests, chi-square, bootstrap CIs, permutation tests, power analysis.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>🔗 Correlation & PCA</h4>
                    <p>Correlation matrices, PCA, t-SNE, UMAP, factor analysis.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>📐 Regression Analysis</h4>
                    <p>Linear, GLM, robust, quantile, mixed models, curve fitting.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>⚖️ ANOVA</h4>
                    <p>One-way, two-way, repeated measures, ANCOVA, Kruskal-Wallis.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>⏳ Time Series</h4>
                    <p>Decomposition, ARIMA/SARIMA, smoothing, forecasting.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>🤖 Machine Learning</h4>
                    <p>XGBoost, LightGBM, SHAP, clustering, model comparison.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>🏥 Survival Analysis</h4>
                    <p>Kaplan-Meier, log-rank test, Cox PH, parametric AFT models.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>🏭 Quality & SPC</h4>
                    <p>Control charts (I-MR, X-bar), attributes charts, process capability.</p>
                </div>
                <div class="rdl-module-item">
                    <h4>🔬 Design of Experiments</h4>
                    <p>Factorial, CCD, Box-Behnken, response surface, effect analysis.</p>
                </div>
            </div>

            <div class="rdl-cta">
                <span class="rdl-cta-pill">
                    Select a dataset from the sidebar to begin
                </span>
            </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state["df"]

    if module == "Data Manager":
        st.markdown("## Data Manager")
        try:
            result = render_data_manager(df)
            if result is not None:
                st.session_state["df"] = result
        except Exception as e:
            st.error(f"An error occurred in Data Manager: {e}")
    elif module == "Descriptive Statistics":
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
    elif module == "Design of Experiments":
        st.markdown("## Design of Experiments")
        try:
            render_doe(df)
        except Exception as e:
            st.error(f"An error occurred in Design of Experiments: {e}")


if __name__ == "__main__":
    main()
