"""
Experimental Module - Visual workflow builder with node-based canvas
and Claude AI-powered data analysis assistant.
"""

from __future__ import annotations

import json
import io
from collections import deque
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
import plotly.express as px
import plotly.graph_objects as go

from modules.ui_helpers import section_header, empty_state, help_tip

# ─── Optional Dependencies ───────────────────────────────────────────────

try:
    from streamlit_flow import streamlit_flow
    from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
    from streamlit_flow.state import StreamlitFlowState
    from streamlit_flow.layouts import TreeLayout
    HAS_FLOW = True
except ImportError:
    HAS_FLOW = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ─── Constants ────────────────────────────────────────────────────────────

NODE_TYPES = {
    "Data Source": {
        "color": "#6366f1",
        "icon": "\u25cf",
        "desc": "Load dataset or filtered subset",
        "default_config": {"columns": [], "filters": []},
    },
    "Transform": {
        "color": "#22c55e",
        "icon": "\u2699",
        "desc": "Pandas data operations",
        "default_config": {"operation": "select_columns", "params": {}},
    },
    "Analysis": {
        "color": "#f59e0b",
        "icon": "\u2605",
        "desc": "Statistical tests & models",
        "default_config": {"analysis_type": "descriptive", "params": {}},
    },
    "Visualization": {
        "color": "#3b82f6",
        "icon": "\u25c6",
        "desc": "Plotly charts",
        "default_config": {"chart_type": "scatter", "x": None, "y": None, "color": None, "title": ""},
    },
    "AI Prompt": {
        "color": "#8b5cf6",
        "icon": "\u2728",
        "desc": "Claude AI analysis",
        "default_config": {"prompt": "", "include_data_summary": True, "temperature": 0.3, "max_tokens": 1024},
    },
    "Validation": {
        "color": "#14b8a6",
        "icon": "\u2714",
        "desc": "Assumption & data quality checks",
        "default_config": {"checks": ["normality", "missing_data", "outliers"], "columns": [], "alpha": 0.05},
    },
    "Output": {
        "color": "#ec4899",
        "icon": "\u270e",
        "desc": "Display results",
        "default_config": {"format": "table"},
    },
}

TRANSFORM_OPS = [
    "select_columns", "filter_rows", "sort", "group_aggregate",
    "add_computed_column", "sample", "drop_duplicates", "rename_columns",
]

ANALYSIS_TYPES = [
    "descriptive", "t_test", "paired_t_test", "anova", "correlation",
    "linear_regression", "chi_square", "mann_whitney", "normality_test",
]

_QUICK_ACTIONS = [
    {"label": "Summarize Dataset", "icon": "\U0001f4ca",
     "prompt": "Provide a comprehensive summary of this dataset. Describe the key variables, their distributions, and any notable patterns."},
    {"label": "Suggest Tests", "icon": "\U0001f9ea",
     "prompt": "Based on the data types and distributions, suggest appropriate statistical tests. Explain why each test is suitable."},
    {"label": "Find Patterns", "icon": "\U0001f50d",
     "prompt": "Analyze this dataset for interesting patterns, anomalies, correlations, and relationships worth investigating further."},
    {"label": "Explain Distributions", "icon": "\U0001f4c8",
     "prompt": "Describe the distribution of each numeric column. Comment on skewness, outliers, and normality."},
    {"label": "Recommend Charts", "icon": "\U0001f3a8",
     "prompt": "Suggest the most insightful visualizations for this data. Explain what each chart would reveal."},
    {"label": "Write Report", "icon": "\U0001f4dd",
     "prompt": "Write a professional data analysis report based on this dataset. Include key findings and recommendations."},
]

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert data analyst assistant in Ryan's Data Lab (RDL). "
    "You help users understand their data, suggest analyses, and interpret results. "
    "Be concise but thorough. Use markdown formatting and include code examples when relevant. "
    "Reference specific column names and statistics from the data context provided."
)


# ─── Session State ────────────────────────────────────────────────────────

def _init_session_state():
    """Initialize all exp_ session state keys if absent."""
    defaults = {
        "exp_flow_state": None,
        "exp_node_configs": {},
        "exp_node_outputs": {},
        "exp_node_status": {},
        "exp_selected_node": None,
        "exp_node_counter": 0,
        "exp_saved_workflows": [],
        "exp_workflow_name": "Untitled Workflow",
        "exp_api_key": "",
        "exp_model": "claude-opus-4-6",
        "exp_temperature": 0.3,
        "exp_max_tokens": 1024,
        "exp_system_prompt": _DEFAULT_SYSTEM_PROMPT,
        "exp_chat_history": [],
        "exp_auto_context": True,
        "exp_stop_on_error": True,
        "exp_canvas_height": 450,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ─── CSS ──────────────────────────────────────────────────────────────────

def _inject_css():
    """Inject experimental module CSS."""
    st.markdown("""
    <style>
    .rdl-exp-node-card {
        background: var(--rdl-glass-bg, rgba(255,255,255,0.72));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--rdl-glass-border, rgba(255,255,255,0.18));
        border-left: 4px solid var(--node-color, #6366f1);
        border-radius: 10px;
        padding: 0.55rem 0.75rem;
        margin-bottom: 0.35rem;
        transition: box-shadow 0.15s cubic-bezier(0.22,0.61,0.36,1),
                    transform 0.15s cubic-bezier(0.22,0.61,0.36,1);
    }
    .rdl-exp-node-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        transform: translateY(-1px);
    }
    .rdl-exp-node-card .node-icon {
        font-size: 1.1rem;
        margin-right: 0.4rem;
    }
    .rdl-exp-node-card .node-name {
        font-weight: 600;
        font-size: 0.82rem;
        color: #1e293b;
    }
    .rdl-exp-node-card .node-desc {
        font-size: 0.7rem;
        color: #64748b;
        margin-top: 0.1rem;
    }
    .rdl-exp-config-panel {
        background: var(--rdl-glass-bg, rgba(255,255,255,0.72));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid var(--rdl-glass-border, rgba(255,255,255,0.18));
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin-top: 0.75rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .rdl-exp-status {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 600;
        padding: 0.12rem 0.5rem;
        border-radius: 99px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .rdl-exp-status--pending { background: rgba(107,114,128,0.1); color: #6b7280; }
    .rdl-exp-status--running { background: rgba(99,102,241,0.1); color: #6366f1; animation: rdl-exp-pulse 1.5s ease-in-out infinite; }
    .rdl-exp-status--success { background: rgba(34,197,94,0.1); color: #22c55e; }
    .rdl-exp-status--error   { background: rgba(239,68,68,0.1); color: #ef4444; }
    @keyframes rdl-exp-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .rdl-exp-result-card {
        background: var(--rdl-glass-bg, rgba(255,255,255,0.72));
        border: 1px solid var(--rdl-glass-border, rgba(255,255,255,0.18));
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .rdl-exp-result-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
        font-weight: 600;
        font-size: 0.85rem;
        color: #1e293b;
    }
    /* Contain the canvas in a fixed viewport */
    .rdl-exp-canvas-area {
        position: relative;
        border-radius: 14px;
        overflow: hidden;
        margin-bottom: 0.75rem;
    }
    /* Compact node palette buttons */
    .rdl-exp-palette-btn button {
        font-size: 0.75rem !important;
        padding: 0.3rem 0.5rem !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ─── DAG Utilities ────────────────────────────────────────────────────────

def _build_adjacency(nodes, edges):
    """Build adjacency dict from edges. Returns {source_id: [target_ids]}."""
    adj = {n.id if hasattr(n, "id") else n: [] for n in nodes}
    for e in edges:
        src = e.source if hasattr(e, "source") else e.get("source")
        tgt = e.target if hasattr(e, "target") else e.get("target")
        if src in adj:
            adj[src].append(tgt)
        else:
            adj[src] = [tgt]
    return adj


def _detect_cycle(adj):
    """DFS-based cycle detection. Returns True if cycle exists."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in adj}

    def dfs(u):
        color[u] = GRAY
        for v in adj.get(u, []):
            if v not in color:
                continue
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    return any(color[n] == WHITE and dfs(n) for n in adj)


def _topological_sort(nodes, edges):
    """Kahn's algorithm. Returns ordered list of node IDs."""
    adj = _build_adjacency(nodes, edges)
    node_ids = list(adj.keys())
    in_degree = {n: 0 for n in node_ids}
    for src, tgts in adj.items():
        for t in tgts:
            if t in in_degree:
                in_degree[t] += 1

    queue = deque(n for n in node_ids if in_degree[n] == 0)
    order = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for t in adj.get(n, []):
            if t in in_degree:
                in_degree[t] -= 1
                if in_degree[t] == 0:
                    queue.append(t)

    if len(order) != len(node_ids):
        raise ValueError("Workflow contains a cycle.")
    return order


def _get_upstream_outputs(node_id, edges, outputs):
    """Collect output objects from all nodes connected to this node as inputs."""
    results = []
    for e in edges:
        tgt = e.target if hasattr(e, "target") else e.get("target")
        src = e.source if hasattr(e, "source") else e.get("source")
        if tgt == node_id and src in outputs:
            results.append(outputs[src])
    return results


# ─── Node Executors ───────────────────────────────────────────────────────

def _get_input_df(inputs, df):
    """Extract first DataFrame from upstream inputs, or fall back to session df."""
    for inp in inputs:
        if isinstance(inp, pd.DataFrame):
            return inp
    return df


def _exec_data_source(config, inputs, df):
    """Load dataset with optional column selection and filters."""
    result = df.copy()
    cols = config.get("columns", [])
    if cols:
        valid = [c for c in cols if c in result.columns]
        if valid:
            result = result[valid]

    for f in config.get("filters", []):
        col, op, val = f.get("column"), f.get("op"), f.get("value", "")
        if col not in result.columns:
            continue
        try:
            if op == "==":
                if pd.api.types.is_numeric_dtype(result[col]):
                    result = result[result[col] == float(val)]
                else:
                    result = result[result[col].astype(str) == val]
            elif op == "!=":
                if pd.api.types.is_numeric_dtype(result[col]):
                    result = result[result[col] != float(val)]
                else:
                    result = result[result[col].astype(str) != val]
            elif op == ">":
                result = result[pd.to_numeric(result[col], errors="coerce") > float(val)]
            elif op == "<":
                result = result[pd.to_numeric(result[col], errors="coerce") < float(val)]
            elif op == ">=":
                result = result[pd.to_numeric(result[col], errors="coerce") >= float(val)]
            elif op == "<=":
                result = result[pd.to_numeric(result[col], errors="coerce") <= float(val)]
            elif op == "contains":
                result = result[result[col].astype(str).str.contains(val, case=False, na=False)]
        except (ValueError, TypeError):
            pass
    return result.reset_index(drop=True)


def _exec_transform(config, inputs, df):
    """Apply a pandas transformation."""
    data = _get_input_df(inputs, df)
    op = config.get("operation", "select_columns")
    params = config.get("params", {})

    if op == "select_columns":
        cols = params.get("columns", [])
        valid = [c for c in cols if c in data.columns]
        return data[valid] if valid else data

    if op == "filter_rows":
        query = params.get("query", "")
        if query:
            return data.query(query).reset_index(drop=True)
        return data

    if op == "sort":
        col = params.get("sort_by")
        asc = params.get("ascending", True)
        if col and col in data.columns:
            return data.sort_values(col, ascending=asc).reset_index(drop=True)
        return data

    if op == "group_aggregate":
        group_cols = params.get("group_by", [])
        agg = params.get("agg", "mean")
        val_col = params.get("value_col")
        valid_groups = [c for c in group_cols if c in data.columns]
        if valid_groups and val_col and val_col in data.columns:
            return data.groupby(valid_groups)[val_col].agg(agg).reset_index()
        return data

    if op == "add_computed_column":
        name = params.get("name", "new_col")
        expr = params.get("expression", "")
        if expr:
            data = data.copy()
            data[name] = data.eval(expr)
        return data

    if op == "sample":
        n = params.get("n")
        frac = params.get("frac")
        if n:
            return data.sample(n=min(int(n), len(data)), random_state=42).reset_index(drop=True)
        if frac:
            return data.sample(frac=float(frac), random_state=42).reset_index(drop=True)
        return data

    if op == "drop_duplicates":
        subset = params.get("columns")
        valid = [c for c in (subset or []) if c in data.columns] or None
        return data.drop_duplicates(subset=valid).reset_index(drop=True)

    if op == "rename_columns":
        mapping = params.get("mapping", {})
        valid = {k: v for k, v in mapping.items() if k in data.columns and v}
        return data.rename(columns=valid) if valid else data

    return data


def _exec_analysis(config, inputs, df):
    """Run a statistical analysis and return results dict."""
    data = _get_input_df(inputs, df)
    atype = config.get("analysis_type", "descriptive")
    params = config.get("params", {})
    result = {"analysis_type": atype}

    if atype == "descriptive":
        cols = params.get("columns", [])
        subset = data[cols] if cols else data.select_dtypes(include=[np.number])
        result["summary"] = subset.describe().to_dict()
        return result

    if atype == "t_test":
        col = params.get("column")
        group_col = params.get("group_column")
        alpha = params.get("alpha", 0.05)
        if col and group_col and col in data.columns and group_col in data.columns:
            groups = data[group_col].dropna().unique()
            if len(groups) >= 2:
                g1 = data[data[group_col] == groups[0]][col].dropna()
                g2 = data[data[group_col] == groups[1]][col].dropna()
                stat, pval = sp_stats.ttest_ind(g1, g2)
                result.update({"test": "Independent t-test", "statistic": float(stat),
                               "p_value": float(pval), "alpha": alpha,
                               "significant": bool(pval < alpha),
                               "groups": [str(groups[0]), str(groups[1])],
                               "means": [float(g1.mean()), float(g2.mean())]})
        return result

    if atype == "paired_t_test":
        col1, col2 = params.get("column1"), params.get("column2")
        alpha = params.get("alpha", 0.05)
        if col1 and col2 and col1 in data.columns and col2 in data.columns:
            clean = data[[col1, col2]].dropna()
            stat, pval = sp_stats.ttest_rel(clean[col1], clean[col2])
            result.update({"test": "Paired t-test", "statistic": float(stat),
                           "p_value": float(pval), "alpha": alpha,
                           "significant": bool(pval < alpha)})
        return result

    if atype == "anova":
        response = params.get("response")
        factor = params.get("factor")
        if response and factor and response in data.columns and factor in data.columns:
            groups = [g[response].dropna().values for _, g in data.groupby(factor)]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                stat, pval = sp_stats.f_oneway(*groups)
                result.update({"test": "One-way ANOVA", "statistic": float(stat),
                               "p_value": float(pval), "significant": bool(pval < 0.05)})
        return result

    if atype == "correlation":
        cols = params.get("columns", [])
        method = params.get("method", "pearson")
        valid = [c for c in cols if c in data.columns]
        if len(valid) >= 2:
            corr = data[valid].corr(method=method)
            result["correlation_matrix"] = corr.to_dict()
            result["method"] = method
        return result

    if atype == "linear_regression":
        target = params.get("target")
        predictors = params.get("predictors", [])
        valid_pred = [c for c in predictors if c in data.columns]
        if target and target in data.columns and valid_pred:
            clean = data[[target] + valid_pred].dropna()
            if len(clean) > len(valid_pred) + 1:
                from sklearn.linear_model import LinearRegression
                X = clean[valid_pred].values
                y = clean[target].values
                model = LinearRegression().fit(X, y)
                r2 = model.score(X, y)
                result.update({"test": "Linear Regression", "r_squared": float(r2),
                               "coefficients": dict(zip(valid_pred, model.coef_.tolist())),
                               "intercept": float(model.intercept_)})
        return result

    if atype == "chi_square":
        col1, col2 = params.get("column1"), params.get("column2")
        if col1 and col2 and col1 in data.columns and col2 in data.columns:
            ct = pd.crosstab(data[col1], data[col2])
            chi2, pval, dof, _ = sp_stats.chi2_contingency(ct)
            result.update({"test": "Chi-Square Independence", "statistic": float(chi2),
                           "p_value": float(pval), "dof": int(dof),
                           "significant": bool(pval < 0.05)})
        return result

    if atype == "mann_whitney":
        col = params.get("column")
        group_col = params.get("group_column")
        if col and group_col and col in data.columns and group_col in data.columns:
            groups = data[group_col].dropna().unique()
            if len(groups) >= 2:
                g1 = data[data[group_col] == groups[0]][col].dropna()
                g2 = data[data[group_col] == groups[1]][col].dropna()
                stat, pval = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
                result.update({"test": "Mann-Whitney U", "statistic": float(stat),
                               "p_value": float(pval), "significant": bool(pval < 0.05)})
        return result

    if atype == "normality_test":
        col = params.get("column")
        if col and col in data.columns:
            vals = data[col].dropna()
            if len(vals) >= 8:
                stat, pval = sp_stats.shapiro(vals[:5000])
                result.update({"test": "Shapiro-Wilk", "statistic": float(stat),
                               "p_value": float(pval), "normal": bool(pval >= 0.05)})
        return result

    return result


def _exec_visualization(config, inputs, df):
    """Create a Plotly figure."""
    data = _get_input_df(inputs, df)
    chart = config.get("chart_type", "scatter")
    x = config.get("x")
    y = config.get("y")
    color = config.get("color")
    title = config.get("title", "")

    kwargs = {}
    if x and x in data.columns:
        kwargs["x"] = x
    if y and y in data.columns:
        kwargs["y"] = y
    if color and color in data.columns:
        kwargs["color"] = color
    if title:
        kwargs["title"] = title

    chart_funcs = {
        "scatter": px.scatter,
        "line": px.line,
        "bar": px.bar,
        "histogram": px.histogram,
        "box": px.box,
        "violin": px.violin,
        "heatmap": None,
    }

    if chart == "heatmap":
        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        corr = data[num_cols].corr() if num_cols else pd.DataFrame()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(),
                                        y=corr.index.tolist(), colorscale="RdBu_r"))
        fig.update_layout(title=title or "Correlation Heatmap")
        return fig

    func = chart_funcs.get(chart, px.scatter)
    if func is None:
        func = px.scatter
    return func(data, **kwargs)


def _exec_ai_prompt(config, inputs, df):
    """Send prompt to Claude with upstream context."""
    prompt = config.get("prompt", "")
    if not prompt:
        return "[No prompt configured]"

    api_key = st.session_state.get("exp_api_key", "")
    if not api_key or not HAS_ANTHROPIC:
        return "[AI unavailable \u2014 configure API key in Settings]"

    # Build context from upstream outputs
    context_parts = []
    if config.get("include_data_summary", True):
        input_df = _get_input_df(inputs, df)
        context_parts.append(_build_data_context(input_df))

    for inp in inputs:
        if isinstance(inp, dict):
            context_parts.append(f"Analysis result: {json.dumps(inp, indent=2, default=str)}")
        elif isinstance(inp, str):
            context_parts.append(f"Previous output: {inp}")

    system = st.session_state.get("exp_system_prompt", _DEFAULT_SYSTEM_PROMPT)
    if context_parts:
        system += "\n\n--- DATA CONTEXT ---\n" + "\n\n".join(context_parts)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=st.session_state.get("exp_model", "claude-opus-4-6"),
            max_tokens=config.get("max_tokens", 1024),
            temperature=config.get("temperature", 0.3),
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as e:
        return f"[AI Error: {e}]"


def _exec_validation(config, inputs, df):
    """Run validation engine checks on the data and return structured results."""
    from modules.validation import (
        check_normality, check_equal_variance, check_sample_size,
        check_missing_data, check_outlier_proportion, check_independence,
        check_homoscedasticity, check_multicollinearity, check_residual_normality,
    )

    data = _get_input_df(inputs, df)
    checks_to_run = config.get("checks", [])
    columns = config.get("columns", [])
    alpha = config.get("alpha", 0.05)

    # If no columns specified, use all numeric
    if not columns:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    results = []

    for check_name in checks_to_run:
        if check_name == "normality":
            for col in columns:
                if col in data.columns:
                    vals = data[col].dropna().values
                    if len(vals) >= 3:
                        c = check_normality(vals, label=col)
                        results.append({"check": c.name, "status": c.status,
                                        "detail": c.detail, "suggestion": c.suggestion})

        elif check_name == "missing_data":
            subset = data[columns] if columns else data
            c = check_missing_data(subset)
            results.append({"check": c.name, "status": c.status,
                            "detail": c.detail, "suggestion": c.suggestion})

        elif check_name == "outliers":
            for col in columns:
                if col in data.columns:
                    vals = data[col].dropna().values
                    if len(vals) >= 4:
                        c = check_outlier_proportion(vals)
                        results.append({"check": f"Outliers ({col})", "status": c.status,
                                        "detail": c.detail, "suggestion": c.suggestion})

        elif check_name == "sample_size":
            test_type = config.get("params", {}).get("test_type", "t-test")
            for col in columns:
                if col in data.columns:
                    n = data[col].dropna().shape[0]
                    c = check_sample_size(n, test_type)
                    results.append({"check": f"Sample Size ({col})", "status": c.status,
                                    "detail": c.detail, "suggestion": c.suggestion})

        elif check_name == "equal_variance":
            group_col = config.get("params", {}).get("group_column")
            if group_col and group_col in data.columns and columns:
                col = columns[0]
                if col in data.columns:
                    groups = [g[col].dropna().values for _, g in data.groupby(group_col)]
                    groups = [g for g in groups if len(g) >= 2]
                    if len(groups) >= 2:
                        c = check_equal_variance(*groups)
                        results.append({"check": c.name, "status": c.status,
                                        "detail": c.detail, "suggestion": c.suggestion})

        elif check_name == "multicollinearity":
            valid = [c for c in columns if c in data.columns]
            if len(valid) >= 2:
                subset = data[valid].dropna()
                if len(subset) > len(valid) + 1:
                    c = check_multicollinearity(subset, valid)
                    results.append({"check": c.name, "status": c.status,
                                    "detail": c.detail, "suggestion": c.suggestion})

        elif check_name == "independence":
            # Run on residuals from linear fit of first two columns
            if len(columns) >= 2:
                col_y, col_x = columns[0], columns[1]
                if col_y in data.columns and col_x in data.columns:
                    clean = data[[col_y, col_x]].dropna()
                    if len(clean) >= 3:
                        coeffs = np.polyfit(clean[col_x], clean[col_y], 1)
                        residuals = clean[col_y].values - np.polyval(coeffs, clean[col_x].values)
                        c = check_independence(residuals)
                        results.append({"check": c.name, "status": c.status,
                                        "detail": c.detail, "suggestion": c.suggestion})

        elif check_name == "homoscedasticity":
            if len(columns) >= 2:
                col_y, col_x = columns[0], columns[1]
                if col_y in data.columns and col_x in data.columns:
                    clean = data[[col_y, col_x]].dropna()
                    if len(clean) > 3:
                        coeffs = np.polyfit(clean[col_x], clean[col_y], 1)
                        residuals = clean[col_y].values - np.polyval(coeffs, clean[col_x].values)
                        c = check_homoscedasticity(residuals, clean[col_x].values)
                        results.append({"check": c.name, "status": c.status,
                                        "detail": c.detail, "suggestion": c.suggestion})

    # Summary
    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_warn = sum(1 for r in results if r["status"] == "warn")
    n_fail = sum(1 for r in results if r["status"] == "fail")

    return {
        "validation_results": results,
        "summary": {"total": len(results), "pass": n_pass, "warn": n_warn, "fail": n_fail},
        "columns_checked": columns,
    }


def _exec_output(config, inputs, df):
    """Collect upstream inputs for display."""
    return {"format": config.get("format", "table"), "data": inputs}


_EXECUTORS = {
    "Data Source": _exec_data_source,
    "Transform": _exec_transform,
    "Analysis": _exec_analysis,
    "Visualization": _exec_visualization,
    "AI Prompt": _exec_ai_prompt,
    "Validation": _exec_validation,
    "Output": _exec_output,
}


# ─── Workflow Runner ──────────────────────────────────────────────────────

def _run_workflow(df):
    """Execute the full workflow DAG."""
    flow_state = st.session_state.get("exp_flow_state")
    if flow_state is None:
        st.warning("No workflow to run.")
        return

    nodes = flow_state.nodes
    edges = flow_state.edges

    if not nodes:
        st.warning("Add some nodes first.")
        return

    adj = _build_adjacency(nodes, edges)
    if _detect_cycle(adj):
        st.error("Workflow contains a cycle. Remove the circular connection before running.")
        return

    try:
        order = _topological_sort(nodes, edges)
    except ValueError as e:
        st.error(str(e))
        return

    configs = st.session_state.get("exp_node_configs", {})
    outputs = {}
    statuses = {}
    stop_on_error = st.session_state.get("exp_stop_on_error", True)

    n_success, n_error = 0, 0
    progress = st.progress(0, text="Running workflow...")

    for i, node_id in enumerate(order):
        statuses[node_id] = "running"
        config = configs.get(node_id, {})
        node_type = config.get("type", "Output")
        upstream = _get_upstream_outputs(node_id, edges, outputs)

        try:
            executor = _EXECUTORS.get(node_type, _exec_output)
            result = executor(config, upstream, df)
            outputs[node_id] = result
            statuses[node_id] = "success"
            n_success += 1
        except Exception as e:
            outputs[node_id] = f"Error: {e}"
            statuses[node_id] = "error"
            n_error += 1
            if stop_on_error:
                # Mark remaining as pending
                for remaining in order[i + 1:]:
                    statuses[remaining] = "pending"
                break

        progress.progress((i + 1) / len(order), text=f"Executed {i + 1}/{len(order)} nodes...")

    progress.empty()
    st.session_state["exp_node_outputs"] = outputs
    st.session_state["exp_node_status"] = statuses

    if n_error == 0:
        st.success(f"Workflow complete: {n_success} node(s) executed successfully.")
    else:
        st.warning(f"Workflow finished: {n_success} succeeded, {n_error} failed.")


# ─── AI Helpers ───────────────────────────────────────────────────────────

def _build_data_context(df):
    """Build a token-efficient data summary string."""
    if df is None or df.empty:
        return "No data loaded."

    parts = [f"Dataset: {st.session_state.get('data_name', 'Unknown')}"]
    parts.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    parts.append(f"Columns: {', '.join(df.columns.tolist())}")

    # Dtypes
    dtype_summary = df.dtypes.value_counts().to_dict()
    parts.append(f"Types: {', '.join(f'{v} {k}' for k, v in dtype_summary.items())}")

    # Head
    head_str = df.head(5).to_string()
    if len(head_str) > 1000:
        head_str = head_str[:1000] + "..."
    parts.append(f"First 5 rows:\n{head_str}")

    # Describe
    desc = df.describe(include="all")
    desc_str = desc.to_string()
    if len(desc_str) > 800:
        desc_str = desc_str[:800] + "..."
    parts.append(f"Summary statistics:\n{desc_str}")

    # Nulls
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0]
    if not null_cols.empty:
        parts.append(f"Missing values: {', '.join(f'{c}={n}' for c, n in null_cols.items())}")

    return "\n\n".join(parts)


def _get_client():
    """Return anthropic client or None."""
    key = st.session_state.get("exp_api_key", "")
    if not key or not HAS_ANTHROPIC:
        return None
    return anthropic.Anthropic(api_key=key)


def _stream_claude(user_message, df, placeholder):
    """Stream a Claude response into a Streamlit placeholder."""
    client = _get_client()
    if client is None:
        placeholder.error("No API key configured. Add your Anthropic API key in Settings.")
        return None

    system = st.session_state.get("exp_system_prompt", _DEFAULT_SYSTEM_PROMPT)
    if st.session_state.get("exp_auto_context", True) and df is not None:
        system += "\n\n--- DATA CONTEXT ---\n" + _build_data_context(df)

    messages = []
    for msg in st.session_state.get("exp_chat_history", []):
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        with client.messages.stream(
            model=st.session_state.get("exp_model", "claude-opus-4-6"),
            max_tokens=st.session_state.get("exp_max_tokens", 1024),
            temperature=st.session_state.get("exp_temperature", 0.3),
            system=system,
            messages=messages,
        ) as stream:
            accumulated = ""
            for text in stream.text_stream:
                accumulated += text
                placeholder.markdown(accumulated + "\u258c")
            placeholder.markdown(accumulated)
            return accumulated
    except anthropic.AuthenticationError:
        placeholder.error("Invalid API key. Check your key in Settings.")
        return None
    except anthropic.RateLimitError:
        placeholder.error("Rate limit exceeded. Please wait and try again.")
        return None
    except Exception as e:
        placeholder.error(f"AI Error: {e}")
        return None


# ─── Workflow Tab UI ──────────────────────────────────────────────────────

def _render_workflow_tab(df):
    """Render the visual workflow builder."""

    if not HAS_FLOW:
        empty_state(
            "streamlit-flow-component is not installed.",
            "Install with: pip install streamlit-flow-component>=1.6.0",
        )
        return

    # ── Canvas FIRST (always visible) ──
    _render_canvas()

    # ── Compact Node Palette + Controls ──
    node_items = list(NODE_TYPES.items())
    cols = st.columns([1, 1, 1, 1, 1, 1, 1, 2])
    for i, (ntype, info) in enumerate(node_items):
        with cols[i]:
            if st.button(f"{info['icon']} {ntype}", key=f"exp_add_{ntype}", use_container_width=True):
                _add_node(ntype)
                st.rerun()

    with cols[7]:
        bc1, bc2 = st.columns(2)
        if bc1.button("Run Workflow", type="primary", key="exp_run", use_container_width=True):
            _run_workflow(df)

        if bc2.button("Clear Canvas", key="exp_clear", use_container_width=True):
            st.session_state["exp_flow_state"] = StreamlitFlowState(nodes=[], edges=[])
            st.session_state["exp_node_configs"] = {}
            st.session_state["exp_node_outputs"] = {}
            st.session_state["exp_node_status"] = {}
            st.session_state["exp_selected_node"] = None
            st.rerun()

    # ── Workflow Management (Save / Load / Export / Import) ──
    with st.expander("Workflow Management"):
        wm1, wm2 = st.columns(2)
        with wm1:
            wf_name = st.text_input("Workflow name:", value=st.session_state.get("exp_workflow_name", ""),
                                    key="exp_wf_name_input")
            sc1, sc2 = st.columns(2)
            if sc1.button("Save", key="exp_save_wf", use_container_width=True):
                _save_workflow(wf_name)
                st.success(f"Saved: {wf_name}")

            saved = st.session_state.get("exp_saved_workflows", [])
            if saved:
                names = [w["name"] for w in saved]
                sel = st.selectbox("Load workflow:", names, key="exp_load_sel")
                if sc2.button("Load", key="exp_load_wf", use_container_width=True):
                    _load_workflow(sel)
                    st.rerun()

        with wm2:
            if st.button("Export to JSON", key="exp_export"):
                data = _serialize_workflow()
                st.download_button("Download JSON", json.dumps(data, indent=2, default=str),
                                   "workflow.json", "application/json", key="exp_dl_json")

            uploaded = st.file_uploader("Import JSON:", type=["json"], key="exp_import_json")
            if uploaded:
                try:
                    data = json.load(uploaded)
                    _deserialize_workflow(data)
                    st.success("Workflow imported.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Import failed: {e}")

    # ── Help (collapsed to save space) ──
    with st.expander("Help"):
        st.markdown("""
**Build analysis pipelines visually:**
1. **Add nodes** from the palette above.
2. **Connect nodes** by dragging from one handle to another.
3. **Click a node** to configure its settings.
4. **Run** the workflow to execute all nodes in order.

**Node types:** Data Source (load data), Transform (reshape), Analysis (stats),
Visualization (charts), AI Prompt (Claude), Output (display results).
""")

    # ── Config Panel (only if node selected) ──
    _render_config_panel(df)

    # ── Execution Results ──
    _render_execution_results()


def _add_node(ntype):
    """Add a new node to the canvas."""
    counter = st.session_state.get("exp_node_counter", 0)
    counter += 1
    st.session_state["exp_node_counter"] = counter

    node_id = f"node_{counter}"
    info = NODE_TYPES[ntype]

    # Stagger position
    x = 100 + ((counter - 1) % 4) * 220
    y = 80 + ((counter - 1) // 4) * 140

    node = StreamlitFlowNode(
        id=node_id,
        pos=(x, y),
        data={"content": f"{info['icon']} {ntype}"},
        node_type="default",
        source_position="right",
        target_position="left",
        style={"border": f"2px solid {info['color']}", "borderRadius": "10px",
               "padding": "8px 12px", "fontSize": "13px", "fontWeight": "600",
               "background": "white", "minWidth": "120px"},
    )

    flow_state = st.session_state.get("exp_flow_state")
    if flow_state is None:
        flow_state = StreamlitFlowState(nodes=[], edges=[])

    flow_state.nodes.append(node)
    st.session_state["exp_flow_state"] = flow_state

    config = dict(info["default_config"])
    config["type"] = ntype
    st.session_state["exp_node_configs"][node_id] = config
    st.session_state["exp_node_status"][node_id] = "pending"


def _render_canvas():
    """Render the streamlit_flow canvas."""
    flow_state = st.session_state.get("exp_flow_state")
    if flow_state is None:
        flow_state = StreamlitFlowState(nodes=[], edges=[])
        st.session_state["exp_flow_state"] = flow_state

    height = st.session_state.get("exp_canvas_height", 450)

    st.markdown('<div class="rdl-exp-canvas-area">', unsafe_allow_html=True)
    updated = streamlit_flow(
        "exp_canvas",
        state=flow_state,
        height=height,
        fit_view=True,
        show_minimap=True,
        show_controls=True,
        get_node_on_click=True,
        get_edge_on_click=False,
        enable_node_menu=True,
        enable_edge_menu=True,
        enable_pane_menu=True,
        allow_new_edges=True,
        animate_new_edges=True,
        min_zoom=0.3,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if not flow_state.nodes:
        st.caption("Add nodes from the palette below, then connect them to build a workflow.")

    if updated is not None:
        # Check for node click
        if hasattr(updated, "selected_id") and updated.selected_id:
            st.session_state["exp_selected_node"] = updated.selected_id
        st.session_state["exp_flow_state"] = updated

        # Clean up configs for deleted nodes
        current_ids = {n.id for n in updated.nodes}
        configs = st.session_state.get("exp_node_configs", {})
        st.session_state["exp_node_configs"] = {k: v for k, v in configs.items() if k in current_ids}


def _render_config_panel(df):
    """Render configuration panel for the selected node."""
    selected = st.session_state.get("exp_selected_node")
    if not selected:
        return

    configs = st.session_state.get("exp_node_configs", {})
    config = configs.get(selected)
    if not config:
        return

    ntype = config.get("type", "Unknown")
    status = st.session_state.get("exp_node_status", {}).get(selected, "pending")
    info = NODE_TYPES.get(ntype, {})

    st.markdown(f'<div class="rdl-exp-config-panel">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.markdown(f"**{info.get('icon', '')} {ntype}** — `{selected}`")
    c2.markdown(f'<span class="rdl-exp-status rdl-exp-status--{status}">{status}</span>',
                unsafe_allow_html=True)
    if c3.button("Deselect", key="exp_deselect"):
        st.session_state["exp_selected_node"] = None
        st.rerun()

    # Type-specific config
    if ntype == "Data Source":
        _config_data_source(selected, config, df)
    elif ntype == "Transform":
        _config_transform(selected, config, df)
    elif ntype == "Analysis":
        _config_analysis(selected, config, df)
    elif ntype == "Visualization":
        _config_visualization(selected, config, df)
    elif ntype == "AI Prompt":
        _config_ai_prompt(selected, config, df)
    elif ntype == "Validation":
        _config_validation(selected, config, df)
    elif ntype == "Output":
        _config_output(selected, config, df)

    st.session_state["exp_node_configs"][selected] = config
    st.markdown("</div>", unsafe_allow_html=True)


def _config_data_source(node_id, config, df):
    """Configure Data Source node."""
    config["columns"] = st.multiselect("Select columns (empty = all):",
                                       df.columns.tolist(),
                                       default=config.get("columns", []),
                                       key=f"exp_ds_cols_{node_id}")

    section_header("Filters")
    filters = config.get("filters", [])
    new_filters = []
    for i, f in enumerate(filters):
        fc1, fc2, fc3, fc4 = st.columns([3, 2, 3, 1])
        f_col = fc1.selectbox("Col:", df.columns.tolist(),
                              index=df.columns.tolist().index(f["column"]) if f["column"] in df.columns else 0,
                              key=f"exp_ds_fcol_{node_id}_{i}")
        f_op = fc2.selectbox("Op:", ["==", "!=", ">", "<", ">=", "<=", "contains"],
                             index=["==", "!=", ">", "<", ">=", "<=", "contains"].index(f.get("op", "==")),
                             key=f"exp_ds_fop_{node_id}_{i}")
        f_val = fc3.text_input("Value:", value=f.get("value", ""),
                               key=f"exp_ds_fval_{node_id}_{i}")
        if not fc4.button("\u2715", key=f"exp_ds_fdel_{node_id}_{i}"):
            new_filters.append({"column": f_col, "op": f_op, "value": f_val})

    if st.button("+ Add Filter", key=f"exp_ds_fadd_{node_id}"):
        new_filters.append({"column": df.columns[0] if len(df.columns) > 0 else "", "op": "==", "value": ""})
    config["filters"] = new_filters


def _config_transform(node_id, config, df):
    """Configure Transform node."""
    op = st.selectbox("Operation:", TRANSFORM_OPS,
                      index=TRANSFORM_OPS.index(config.get("operation", "select_columns")),
                      key=f"exp_tr_op_{node_id}")
    config["operation"] = op
    params = config.get("params", {})

    if op == "select_columns":
        params["columns"] = st.multiselect("Columns:", df.columns.tolist(),
                                           default=params.get("columns", []),
                                           key=f"exp_tr_cols_{node_id}")
    elif op == "filter_rows":
        params["query"] = st.text_input("Query expression (e.g. `age > 30`):",
                                        value=params.get("query", ""),
                                        key=f"exp_tr_query_{node_id}")
    elif op == "sort":
        params["sort_by"] = st.selectbox("Sort by:", df.columns.tolist(),
                                         key=f"exp_tr_sortby_{node_id}")
        params["ascending"] = st.checkbox("Ascending", value=params.get("ascending", True),
                                          key=f"exp_tr_asc_{node_id}")
    elif op == "group_aggregate":
        params["group_by"] = st.multiselect("Group by:", df.columns.tolist(),
                                            default=params.get("group_by", []),
                                            key=f"exp_tr_groupby_{node_id}")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        params["value_col"] = st.selectbox("Value column:", num_cols if num_cols else df.columns.tolist(),
                                           key=f"exp_tr_valcol_{node_id}")
        params["agg"] = st.selectbox("Aggregation:", ["mean", "sum", "count", "min", "max", "median", "std"],
                                     key=f"exp_tr_agg_{node_id}")
    elif op == "add_computed_column":
        params["name"] = st.text_input("New column name:", value=params.get("name", "new_col"),
                                       key=f"exp_tr_newname_{node_id}")
        params["expression"] = st.text_input("Expression (e.g. `col_a + col_b`):",
                                             value=params.get("expression", ""),
                                             key=f"exp_tr_expr_{node_id}")
    elif op == "sample":
        mode = st.radio("Sample by:", ["Count", "Fraction"], horizontal=True,
                        key=f"exp_tr_smode_{node_id}")
        if mode == "Count":
            params["n"] = st.number_input("N:", min_value=1, value=params.get("n", 100),
                                          key=f"exp_tr_sn_{node_id}")
            params["frac"] = None
        else:
            params["frac"] = st.slider("Fraction:", 0.01, 1.0, value=params.get("frac", 0.5),
                                       key=f"exp_tr_sfrac_{node_id}")
            params["n"] = None
    elif op == "drop_duplicates":
        params["columns"] = st.multiselect("Subset columns (empty = all):", df.columns.tolist(),
                                           default=params.get("columns", []),
                                           key=f"exp_tr_dedup_{node_id}")
    elif op == "rename_columns":
        mapping = params.get("mapping", {})
        st.caption("Enter new names (leave blank to keep original):")
        new_mapping = {}
        for col in df.columns[:20]:
            new_name = st.text_input(f"{col} \u2192", value=mapping.get(col, ""),
                                     key=f"exp_tr_rename_{node_id}_{col}")
            if new_name:
                new_mapping[col] = new_name
        params["mapping"] = new_mapping

    config["params"] = params


def _config_analysis(node_id, config, df):
    """Configure Analysis node."""
    atype = st.selectbox("Analysis type:", ANALYSIS_TYPES,
                         index=ANALYSIS_TYPES.index(config.get("analysis_type", "descriptive")),
                         key=f"exp_an_type_{node_id}")
    config["analysis_type"] = atype
    params = config.get("params", {})
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if atype == "descriptive":
        params["columns"] = st.multiselect("Columns (empty = all numeric):", num_cols,
                                           default=params.get("columns", []),
                                           key=f"exp_an_dcols_{node_id}")
    elif atype in ("t_test", "mann_whitney"):
        params["column"] = st.selectbox("Numeric column:", num_cols,
                                        key=f"exp_an_col_{node_id}")
        params["group_column"] = st.selectbox("Group column:", cat_cols if cat_cols else df.columns.tolist(),
                                              key=f"exp_an_gcol_{node_id}")
        params["alpha"] = st.slider("Alpha:", 0.01, 0.10, value=params.get("alpha", 0.05),
                                    key=f"exp_an_alpha_{node_id}")
    elif atype == "paired_t_test":
        params["column1"] = st.selectbox("Column 1:", num_cols, key=f"exp_an_c1_{node_id}")
        params["column2"] = st.selectbox("Column 2:", num_cols,
                                         index=min(1, len(num_cols) - 1) if len(num_cols) > 1 else 0,
                                         key=f"exp_an_c2_{node_id}")
    elif atype == "anova":
        params["response"] = st.selectbox("Response:", num_cols, key=f"exp_an_resp_{node_id}")
        params["factor"] = st.selectbox("Factor:", cat_cols if cat_cols else df.columns.tolist(),
                                        key=f"exp_an_factor_{node_id}")
    elif atype == "correlation":
        params["columns"] = st.multiselect("Columns:", num_cols,
                                           default=params.get("columns", num_cols[:5]),
                                           key=f"exp_an_ccols_{node_id}")
        params["method"] = st.selectbox("Method:", ["pearson", "spearman", "kendall"],
                                        key=f"exp_an_cmethod_{node_id}")
    elif atype == "linear_regression":
        params["target"] = st.selectbox("Target:", num_cols, key=f"exp_an_target_{node_id}")
        remaining = [c for c in num_cols if c != params.get("target")]
        params["predictors"] = st.multiselect("Predictors:", remaining,
                                              default=params.get("predictors", []),
                                              key=f"exp_an_preds_{node_id}")
    elif atype == "chi_square":
        params["column1"] = st.selectbox("Column 1:", cat_cols if cat_cols else df.columns.tolist(),
                                         key=f"exp_an_chi1_{node_id}")
        params["column2"] = st.selectbox("Column 2:", cat_cols if cat_cols else df.columns.tolist(),
                                         index=min(1, len(cat_cols) - 1) if len(cat_cols) > 1 else 0,
                                         key=f"exp_an_chi2_{node_id}")
    elif atype == "normality_test":
        params["column"] = st.selectbox("Column:", num_cols, key=f"exp_an_normcol_{node_id}")

    config["params"] = params


def _config_visualization(node_id, config, df):
    """Configure Visualization node."""
    chart_types = ["scatter", "line", "bar", "histogram", "box", "violin", "heatmap"]
    config["chart_type"] = st.selectbox("Chart type:", chart_types,
                                        index=chart_types.index(config.get("chart_type", "scatter")),
                                        key=f"exp_viz_type_{node_id}")
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if config["chart_type"] != "heatmap":
        config["x"] = st.selectbox("X axis:", all_cols, key=f"exp_viz_x_{node_id}")
        if config["chart_type"] != "histogram":
            config["y"] = st.selectbox("Y axis:", all_cols,
                                       index=min(1, len(all_cols) - 1),
                                       key=f"exp_viz_y_{node_id}")
        config["color"] = st.selectbox("Color:", ["(none)"] + all_cols,
                                       key=f"exp_viz_color_{node_id}")
        if config["color"] == "(none)":
            config["color"] = None

    config["title"] = st.text_input("Title:", value=config.get("title", ""),
                                    key=f"exp_viz_title_{node_id}")


def _config_ai_prompt(node_id, config, df):
    """Configure AI Prompt node."""
    config["prompt"] = st.text_area("Prompt:", value=config.get("prompt", ""),
                                    height=120, key=f"exp_ai_prompt_{node_id}")
    config["include_data_summary"] = st.checkbox("Include data summary in context",
                                                 value=config.get("include_data_summary", True),
                                                 key=f"exp_ai_ctx_{node_id}")
    c1, c2 = st.columns(2)
    config["temperature"] = c1.slider("Temperature:", 0.0, 1.0,
                                      value=config.get("temperature", 0.3),
                                      key=f"exp_ai_temp_{node_id}")
    config["max_tokens"] = c2.slider("Max tokens:", 256, 4096,
                                     value=config.get("max_tokens", 1024),
                                     key=f"exp_ai_maxtok_{node_id}")

    if not st.session_state.get("exp_api_key"):
        st.caption("Set your API key in the Settings tab to enable AI nodes.")


def _config_validation(node_id, config, df):
    """Configure Validation node."""
    _ALL_CHECKS = [
        "normality", "missing_data", "outliers", "sample_size",
        "equal_variance", "multicollinearity", "independence", "homoscedasticity",
    ]
    _CHECK_LABELS = {
        "normality": "Normality (Shapiro-Wilk)",
        "missing_data": "Missing Data",
        "outliers": "Outliers (IQR)",
        "sample_size": "Sample Size",
        "equal_variance": "Equal Variance (Levene's)",
        "multicollinearity": "Multicollinearity (VIF)",
        "independence": "Independence (Durbin-Watson)",
        "homoscedasticity": "Homoscedasticity (Breusch-Pagan)",
    }

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    config["columns"] = st.multiselect(
        "Columns to validate (empty = all numeric):",
        num_cols, default=config.get("columns", []),
        key=f"exp_val_cols_{node_id}",
    )

    current_checks = config.get("checks", ["normality", "missing_data", "outliers"])
    config["checks"] = st.multiselect(
        "Checks to run:",
        _ALL_CHECKS,
        default=[c for c in current_checks if c in _ALL_CHECKS],
        format_func=lambda x: _CHECK_LABELS.get(x, x),
        key=f"exp_val_checks_{node_id}",
    )

    config["alpha"] = st.slider("Significance level (alpha):", 0.01, 0.10, config.get("alpha", 0.05),
                                0.01, key=f"exp_val_alpha_{node_id}")

    # Extra params for checks that need them
    if not config.get("params"):
        config["params"] = {}

    if "equal_variance" in config["checks"]:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            config["params"]["group_column"] = st.selectbox(
                "Group column (for equal variance):", cat_cols,
                key=f"exp_val_grp_{node_id}",
            )

    if "sample_size" in config["checks"]:
        config["params"]["test_type"] = st.selectbox(
            "Test type (for sample size check):",
            ["t-test", "paired-t", "anova", "chi-square", "correlation",
             "regression", "pca", "ml-classification", "survival"],
            key=f"exp_val_test_type_{node_id}",
        )

    help_tip("Validation Checks", """
- **Normality**: Shapiro-Wilk test per column
- **Missing Data**: Proportion of NaN values
- **Outliers**: IQR-based outlier detection per column
- **Sample Size**: Checks against recommended minimums for test type
- **Equal Variance**: Levene's test across groups (requires group column)
- **Multicollinearity**: VIF across selected numeric columns
- **Independence**: Durbin-Watson on residuals (first two columns)
- **Homoscedasticity**: Breusch-Pagan on residuals (first two columns)
""")


def _config_output(node_id, config, df):
    """Configure Output node."""
    config["format"] = st.selectbox("Output format:", ["table", "text", "download"],
                                    index=["table", "text", "download"].index(config.get("format", "table")),
                                    key=f"exp_out_fmt_{node_id}")


def _render_execution_results():
    """Display results from the last workflow run."""
    outputs = st.session_state.get("exp_node_outputs", {})
    statuses = st.session_state.get("exp_node_status", {})
    configs = st.session_state.get("exp_node_configs", {})

    if not outputs:
        return

    st.divider()
    section_header("Execution Results")

    for node_id, output in outputs.items():
        status = statuses.get(node_id, "pending")
        config = configs.get(node_id, {})
        ntype = config.get("type", "Unknown")
        info = NODE_TYPES.get(ntype, {})

        st.markdown(
            f'<div class="rdl-exp-result-card">'
            f'<div class="rdl-exp-result-header">'
            f'{info.get("icon", "")} {ntype} ({node_id}) '
            f'<span class="rdl-exp-status rdl-exp-status--{status}">{status}</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if status == "error":
            st.error(str(output))
        elif isinstance(output, pd.DataFrame):
            st.dataframe(output.head(500), use_container_width=True)
        elif isinstance(output, go.Figure):
            st.plotly_chart(output, use_container_width=True)
        elif isinstance(output, dict) and "validation_results" in output:
            # Validation node — render with validation_panel
            from modules.validation import ValidationCheck
            checks = [
                ValidationCheck(
                    name=r["check"], status=r["status"],
                    detail=r["detail"], suggestion=r.get("suggestion", ""),
                )
                for r in output["validation_results"]
            ]
            if checks:
                validation_panel(checks, title="Workflow Validation Results")
            summary = output.get("summary", {})
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Total Checks", summary.get("total", 0))
            sc2.metric("Passed", summary.get("pass", 0))
            sc3.metric("Warnings", summary.get("warn", 0))
            sc4.metric("Failed", summary.get("fail", 0))
        elif isinstance(output, dict):
            fmt = output.get("format")
            data = output.get("data")
            if fmt and data is not None:
                # Output node
                for item in (data if isinstance(data, list) else [data]):
                    if isinstance(item, pd.DataFrame):
                        if fmt == "download":
                            csv = item.to_csv(index=False)
                            st.download_button("Download CSV", csv, "output.csv", key=f"exp_dl_{node_id}")
                        else:
                            st.dataframe(item.head(500), use_container_width=True)
                    elif isinstance(item, go.Figure):
                        st.plotly_chart(item, use_container_width=True)
                    elif isinstance(item, str):
                        st.markdown(item)
                    elif isinstance(item, dict):
                        st.json(item)
            else:
                # Analysis result
                st.json(output)
        elif isinstance(output, str):
            st.markdown(output)


# ─── Workflow Save/Load ───────────────────────────────────────────────────

def _serialize_workflow():
    """Serialize current workflow to a JSON-compatible dict."""
    flow_state = st.session_state.get("exp_flow_state")
    if flow_state is None:
        return {"nodes": [], "edges": [], "configs": {}}

    nodes = []
    for n in flow_state.nodes:
        nodes.append({
            "id": n.id,
            "pos": list(n.pos) if hasattr(n.pos, "__iter__") else [0, 0],
            "data": n.data,
            "style": n.style if hasattr(n, "style") else {},
        })

    edges = []
    for e in flow_state.edges:
        edges.append({
            "id": e.id,
            "source": e.source,
            "target": e.target,
        })

    return {
        "name": st.session_state.get("exp_workflow_name", "Untitled"),
        "nodes": nodes,
        "edges": edges,
        "configs": st.session_state.get("exp_node_configs", {}),
        "counter": st.session_state.get("exp_node_counter", 0),
    }


def _deserialize_workflow(data):
    """Load a workflow from a serialized dict."""
    nodes = []
    for n in data.get("nodes", []):
        pos = tuple(n.get("pos", [0, 0]))
        nodes.append(StreamlitFlowNode(
            id=n["id"],
            pos=pos,
            data=n.get("data", {"content": "Node"}),
            node_type="default",
            source_position="right",
            target_position="left",
            style=n.get("style", {}),
        ))

    edges = []
    for e in data.get("edges", []):
        edges.append(StreamlitFlowEdge(
            id=e["id"],
            source=e["source"],
            target=e["target"],
            animated=True,
        ))

    st.session_state["exp_flow_state"] = StreamlitFlowState(nodes=nodes, edges=edges)
    st.session_state["exp_node_configs"] = data.get("configs", {})
    st.session_state["exp_node_counter"] = data.get("counter", 0)
    st.session_state["exp_workflow_name"] = data.get("name", "Untitled")
    st.session_state["exp_node_outputs"] = {}
    st.session_state["exp_node_status"] = {}
    st.session_state["exp_selected_node"] = None


def _save_workflow(name):
    """Save current workflow to session state list."""
    data = _serialize_workflow()
    data["name"] = name
    data["saved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.session_state["exp_workflow_name"] = name

    saved = st.session_state.get("exp_saved_workflows", [])
    # Replace if same name exists
    saved = [w for w in saved if w.get("name") != name]
    saved.append(data)
    st.session_state["exp_saved_workflows"] = saved


def _load_workflow(name):
    """Load a saved workflow by name."""
    saved = st.session_state.get("exp_saved_workflows", [])
    for w in saved:
        if w.get("name") == name:
            _deserialize_workflow(w)
            return


# ─── AI Assistant Tab UI ──────────────────────────────────────────────────

def _render_ai_tab(df):
    """Render the AI Assistant tab."""

    if not HAS_ANTHROPIC:
        empty_state(
            "anthropic SDK is not installed.",
            "Install with: pip install anthropic>=0.39.0",
        )
        return

    help_tip("AI Assistant", """
Chat with Claude about your data. The assistant can:
- Summarize datasets and describe distributions
- Suggest appropriate statistical tests
- Find patterns, anomalies, and relationships
- Write analysis reports
- Answer questions about your analysis results

Configure your API key and preferences in the Settings panel.
""")

    # Settings expander
    with st.expander("AI Settings", expanded=False):
        st.text_input("Anthropic API Key:", type="password",
                      value=st.session_state.get("exp_api_key", ""),
                      key="exp_api_key_input",
                      on_change=lambda: st.session_state.update({"exp_api_key": st.session_state["exp_api_key_input"]}))
        c1, c2 = st.columns(2)
        model = c1.selectbox("Model:", ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                             index=["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"].index(
                                 st.session_state.get("exp_model", "claude-opus-4-6")),
                             key="exp_model_sel")
        st.session_state["exp_model"] = model
        temp = c2.slider("Temperature:", 0.0, 1.0,
                         value=st.session_state.get("exp_temperature", 0.3),
                         key="exp_temp_sl")
        st.session_state["exp_temperature"] = temp
        max_tok = st.slider("Max tokens:", 256, 4096,
                            value=st.session_state.get("exp_max_tokens", 1024),
                            key="exp_maxtok_sl")
        st.session_state["exp_max_tokens"] = max_tok
        auto_ctx = st.checkbox("Auto-include data summary in context",
                               value=st.session_state.get("exp_auto_context", True),
                               key="exp_auto_ctx_cb")
        st.session_state["exp_auto_context"] = auto_ctx

    # Check for API key
    if not st.session_state.get("exp_api_key"):
        empty_state(
            "No API key configured.",
            "Enter your Anthropic API key in the AI Settings panel above to enable the assistant.",
        )
        return

    # Quick Actions
    section_header("Quick Actions")
    qa_cols = st.columns(len(_QUICK_ACTIONS))
    for i, action in enumerate(_QUICK_ACTIONS):
        if qa_cols[i].button(f"{action['icon']} {action['label']}",
                             key=f"exp_qa_{i}", use_container_width=True):
            st.session_state["exp_chat_history"].append({
                "role": "user", "content": action["prompt"]
            })
            st.rerun()

    st.divider()

    # Chat history
    for msg in st.session_state.get("exp_chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Check if the last message is from user and needs a response
    history = st.session_state.get("exp_chat_history", [])
    if history and history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            placeholder = st.empty()
            response = _stream_claude(history[-1]["content"], df, placeholder)
            if response:
                st.session_state["exp_chat_history"].append({
                    "role": "assistant", "content": response
                })

    # Chat input
    user_input = st.chat_input("Ask about your data...")
    if user_input:
        st.session_state["exp_chat_history"].append({
            "role": "user", "content": user_input
        })
        st.rerun()

    # Clear chat button
    if st.session_state.get("exp_chat_history"):
        if st.button("Clear Chat", key="exp_clear_chat"):
            st.session_state["exp_chat_history"] = []
            st.rerun()


# ─── Settings Tab UI ─────────────────────────────────────────────────────

def _render_settings_tab():
    """Render the Settings tab."""

    section_header("API Configuration")
    st.text_input("Anthropic API Key:", type="password",
                  value=st.session_state.get("exp_api_key", ""),
                  key="exp_settings_api_key",
                  on_change=lambda: st.session_state.update({"exp_api_key": st.session_state["exp_settings_api_key"]}))

    c1, c2 = st.columns(2)
    model = c1.selectbox("Default Model:", ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
                         index=["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"].index(
                             st.session_state.get("exp_model", "claude-opus-4-6")),
                         key="exp_settings_model")
    st.session_state["exp_model"] = model
    temp = c2.slider("Default Temperature:", 0.0, 1.0,
                     value=st.session_state.get("exp_temperature", 0.3),
                     key="exp_settings_temp")
    st.session_state["exp_temperature"] = temp

    max_tok = st.slider("Default Max Tokens:", 256, 4096,
                        value=st.session_state.get("exp_max_tokens", 1024),
                        key="exp_settings_maxtok")
    st.session_state["exp_max_tokens"] = max_tok

    st.text_area("System Prompt:", value=st.session_state.get("exp_system_prompt", _DEFAULT_SYSTEM_PROMPT),
                 height=150, key="exp_settings_sysprompt",
                 on_change=lambda: st.session_state.update({"exp_system_prompt": st.session_state["exp_settings_sysprompt"]}))

    st.divider()
    section_header("Workflow Defaults")
    st.session_state["exp_stop_on_error"] = st.checkbox(
        "Stop workflow on first error",
        value=st.session_state.get("exp_stop_on_error", True),
        key="exp_settings_stop_err",
    )
    st.session_state["exp_canvas_height"] = st.slider(
        "Canvas height (px):", 300, 800,
        value=st.session_state.get("exp_canvas_height", 500),
        key="exp_settings_height",
    )


# ─── Entry Point ──────────────────────────────────────────────────────────

def render_experimental(df):
    """Main render function for the Experimental module."""

    _init_session_state()
    _inject_css()

    if df is None or (hasattr(df, "empty") and df.empty):
        empty_state(
            "No data loaded.",
            "Upload a dataset or select a sample from the sidebar.",
        )
        return

    tabs = st.tabs(["Workflows", "AI Assistant", "Settings"])

    with tabs[0]:
        _render_workflow_tab(df)
    with tabs[1]:
        _render_ai_tab(df)
    with tabs[2]:
        _render_settings_tab()
