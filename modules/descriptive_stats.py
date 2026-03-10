"""
Descriptive Statistics Module - Summary statistics, distributions, outliers.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_descriptive_stats(df: pd.DataFrame):
    """Render descriptive statistics interface."""
    if df is None or df.empty:
        st.warning("No data loaded.")
        return

    tabs = st.tabs([
        "Summary Statistics", "Distribution Analysis",
        "Frequency Analysis", "Grouped Statistics", "Outlier Detection",
    ])

    with tabs[0]:
        _render_summary_stats(df)
    with tabs[1]:
        _render_distribution(df)
    with tabs[2]:
        _render_frequency(df)
    with tabs[3]:
        _render_grouped_stats(df)
    with tabs[4]:
        _render_outliers(df)


def _render_summary_stats(df: pd.DataFrame):
    """Comprehensive summary statistics table."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    selected = st.multiselect("Select columns:", num_cols, default=num_cols[:5], key="summ_cols")
    if not selected:
        return

    ci_level = st.selectbox("Confidence level:", [0.90, 0.95, 0.99], index=1, key="ci_level")

    records = []
    for col_name in selected:
        col = df[col_name].dropna()
        n = len(col)
        if n == 0:
            continue
        sem = col.std() / np.sqrt(n)
        ci = stats.t.interval(ci_level, df=n - 1, loc=col.mean(), scale=sem) if n > 1 else (np.nan, np.nan)
        mode_result = stats.mode(col, keepdims=True)

        records.append({
            "Column": col_name,
            "Count": n,
            "Missing": df[col_name].isnull().sum(),
            "Unique": df[col_name].nunique(),
            "Mean": col.mean(),
            "Median": col.median(),
            "Mode": mode_result.mode[0] if len(mode_result.mode) > 0 else np.nan,
            "Std Dev": col.std(),
            "Variance": col.var(),
            "SEM": sem,
            "Min": col.min(),
            "Max": col.max(),
            "Range": col.max() - col.min(),
            "Q1": col.quantile(0.25),
            "Q3": col.quantile(0.75),
            "IQR": col.quantile(0.75) - col.quantile(0.25),
            "Skewness": col.skew(),
            "Kurtosis": col.kurtosis(),
            "CV (%)": (col.std() / col.mean() * 100) if col.mean() != 0 else np.nan,
            f"CI Lower ({ci_level:.0%})": ci[0],
            f"CI Upper ({ci_level:.0%})": ci[1],
        })

    stats_df = pd.DataFrame(records)
    # Format numeric columns
    float_cols = stats_df.select_dtypes(include=[np.number]).columns
    stats_df[float_cols] = stats_df[float_cols].round(4)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with st.expander("Percentiles"):
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_records = []
        for col_name in selected:
            col = df[col_name].dropna()
            row = {"Column": col_name}
            for p in percentiles:
                row[f"P{p}"] = round(col.quantile(p / 100), 4)
            pct_records.append(row)
        st.dataframe(pd.DataFrame(pct_records), use_container_width=True, hide_index=True)


def _render_distribution(df: pd.DataFrame):
    """Distribution analysis with plots and normality tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    col_name = st.selectbox("Select column:", num_cols, key="dist_col")
    col_data = df[col_name].dropna()

    if len(col_data) < 3:
        st.warning("Need at least 3 data points.")
        return

    # Plot controls
    c1, c2, c3 = st.columns(3)
    n_bins = c1.slider("Bins:", 5, 100, 30, key="dist_bins")
    show_kde = c2.checkbox("Show KDE", value=True, key="dist_kde")
    show_normal = c3.checkbox("Normal overlay", value=False, key="dist_normal")

    # Histogram with KDE
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Histogram", "Box Plot", "QQ Plot", "ECDF"),
                        vertical_spacing=0.12, horizontal_spacing=0.1)

    # Histogram
    hist_vals, bin_edges = np.histogram(col_data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig.add_trace(go.Bar(x=bin_centers, y=hist_vals, name="Histogram",
                         marker_color="steelblue", opacity=0.7), row=1, col=1)

    if show_kde:
        kde_x = np.linspace(col_data.min(), col_data.max(), 200)
        try:
            kde = stats.gaussian_kde(col_data)
            kde_y = kde(kde_x) * len(col_data) * (bin_edges[1] - bin_edges[0])
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, name="KDE",
                                     line=dict(color="red", width=2)), row=1, col=1)
        except Exception:
            pass

    if show_normal:
        x_norm = np.linspace(col_data.min(), col_data.max(), 200)
        y_norm = stats.norm.pdf(x_norm, col_data.mean(), col_data.std()) * len(col_data) * (bin_edges[1] - bin_edges[0])
        fig.add_trace(go.Scatter(x=x_norm, y=y_norm, name="Normal",
                                 line=dict(color="green", width=2, dash="dash")), row=1, col=1)

    # Box plot
    fig.add_trace(go.Box(y=col_data, name=col_name, marker_color="steelblue",
                         boxpoints="outliers"), row=1, col=2)

    # QQ plot
    sorted_data = np.sort(col_data)
    n = len(sorted_data)
    theoretical_q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig.add_trace(go.Scatter(x=theoretical_q, y=sorted_data, mode="markers",
                             name="QQ Points", marker=dict(color="steelblue", size=4)),
                  row=2, col=1)
    # Reference line
    slope, intercept = np.polyfit(theoretical_q, sorted_data, 1)
    qq_line = slope * theoretical_q + intercept
    fig.add_trace(go.Scatter(x=theoretical_q, y=qq_line, mode="lines",
                             name="Reference", line=dict(color="red", dash="dash")),
                  row=2, col=1)

    # ECDF
    ecdf_x = np.sort(col_data)
    ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
    fig.add_trace(go.Scatter(x=ecdf_x, y=ecdf_y, mode="lines",
                             name="ECDF", line=dict(color="steelblue", width=2)),
                  row=2, col=2)
    # Theoretical normal CDF
    ecdf_norm_y = stats.norm.cdf(ecdf_x, col_data.mean(), col_data.std())
    fig.add_trace(go.Scatter(x=ecdf_x, y=ecdf_norm_y, mode="lines",
                             name="Normal CDF", line=dict(color="red", dash="dash")),
                  row=2, col=2)

    fig.update_layout(height=700, showlegend=True, title_text=f"Distribution Analysis: {col_name}")
    fig.update_xaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
    fig.update_xaxes(title_text="Value", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)
    st.plotly_chart(fig, use_container_width=True)

    # Violin plot
    with st.expander("Violin Plot"):
        fig_violin = go.Figure(go.Violin(y=col_data, name=col_name, box_visible=True,
                                          meanline_visible=True, fillcolor="lightblue"))
        fig_violin.update_layout(height=400, title=f"Violin Plot: {col_name}")
        st.plotly_chart(fig_violin, use_container_width=True)

    # Normality tests
    st.markdown("#### Normality Tests")
    test_results = []

    # Shapiro-Wilk (max 5000 samples)
    sample = col_data.values[:5000] if len(col_data) > 5000 else col_data.values
    try:
        sw_stat, sw_p = stats.shapiro(sample)
        test_results.append({"Test": "Shapiro-Wilk", "Statistic": sw_stat, "p-value": sw_p,
                             "Normal (α=0.05)?": "Yes" if sw_p > 0.05 else "No"})
    except Exception:
        pass

    # Anderson-Darling
    try:
        ad_result = stats.anderson(col_data, dist="norm")
        # Use 5% significance level (index 2)
        ad_crit = ad_result.critical_values[2]
        test_results.append({"Test": "Anderson-Darling", "Statistic": ad_result.statistic,
                             "p-value": np.nan,
                             "Normal (α=0.05)?": "Yes" if ad_result.statistic < ad_crit else "No"})
    except Exception:
        pass

    # D'Agostino-Pearson
    if len(col_data) >= 20:
        try:
            dag_stat, dag_p = stats.normaltest(col_data)
            test_results.append({"Test": "D'Agostino-Pearson", "Statistic": dag_stat,
                                 "p-value": dag_p,
                                 "Normal (α=0.05)?": "Yes" if dag_p > 0.05 else "No"})
        except Exception:
            pass

    # Kolmogorov-Smirnov
    try:
        ks_stat, ks_p = stats.kstest(col_data, "norm", args=(col_data.mean(), col_data.std()))
        test_results.append({"Test": "Kolmogorov-Smirnov", "Statistic": ks_stat,
                             "p-value": ks_p,
                             "Normal (α=0.05)?": "Yes" if ks_p > 0.05 else "No"})
    except Exception:
        pass

    # Jarque-Bera
    try:
        jb_stat, jb_p = stats.jarque_bera(col_data)
        test_results.append({"Test": "Jarque-Bera", "Statistic": jb_stat,
                             "p-value": jb_p,
                             "Normal (α=0.05)?": "Yes" if jb_p > 0.05 else "No"})
    except Exception:
        pass

    if test_results:
        results_df = pd.DataFrame(test_results)
        float_cols = results_df.select_dtypes(include=[np.number]).columns
        results_df[float_cols] = results_df[float_cols].round(6)
        st.dataframe(results_df, use_container_width=True, hide_index=True)


def _render_frequency(df: pd.DataFrame):
    """Frequency analysis for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Also include numeric columns with few unique values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 20:
            cat_cols.append(col)

    if not cat_cols:
        st.warning("No categorical columns (or numeric with ≤20 unique values) found.")
        return

    col_name = st.selectbox("Select column:", cat_cols, key="freq_col")
    freq = df[col_name].value_counts()
    freq_pct = df[col_name].value_counts(normalize=True) * 100

    freq_df = pd.DataFrame({
        "Value": freq.index,
        "Count": freq.values,
        "Percentage": freq_pct.values.round(2),
        "Cumulative %": freq_pct.cumsum().values.round(2),
    })
    st.dataframe(freq_df, use_container_width=True, hide_index=True)

    chart_type = st.radio("Chart type:", ["Bar", "Pareto", "Pie", "Donut"], horizontal=True, key="freq_chart")

    if chart_type == "Bar":
        sort_by = st.radio("Sort:", ["Frequency", "Alphabetical"], horizontal=True, key="freq_sort")
        if sort_by == "Alphabetical":
            freq_sorted = freq.sort_index()
        else:
            freq_sorted = freq
        fig = px.bar(x=freq_sorted.index.astype(str), y=freq_sorted.values,
                     labels={"x": col_name, "y": "Count"}, title=f"Frequency: {col_name}",
                     color=freq_sorted.values, color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pareto":
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=freq.index.astype(str), y=freq.values,
                             name="Count", marker_color="steelblue"), secondary_y=False)
        cum_pct = freq.cumsum() / freq.sum() * 100
        fig.add_trace(go.Scatter(x=freq.index.astype(str), y=cum_pct.values,
                                 name="Cumulative %", line=dict(color="red", width=2),
                                 mode="lines+markers"), secondary_y=True)
        fig.update_layout(title=f"Pareto Chart: {col_name}", height=500)
        fig.update_yaxes(title_text="Count", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0, 105])
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type in ("Pie", "Donut"):
        hole = 0.4 if chart_type == "Donut" else 0
        fig = px.pie(values=freq.values, names=freq.index.astype(str),
                     title=f"{chart_type} Chart: {col_name}", hole=hole)
        st.plotly_chart(fig, use_container_width=True)


def _render_grouped_stats(df: pd.DataFrame):
    """Statistics grouped by a categorical variable."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not cat_cols or not num_cols:
        st.warning("Need both categorical and numeric columns for grouped statistics.")
        return

    group_col = st.selectbox("Group by:", cat_cols, key="group_col")
    value_col = st.selectbox("Value column:", num_cols, key="group_val_col")

    grouped = df.groupby(group_col)[value_col]
    stats_df = grouped.agg(["count", "mean", "median", "std", "min", "max"]).round(4)
    stats_df.columns = ["Count", "Mean", "Median", "Std Dev", "Min", "Max"]
    stats_df["SEM"] = (stats_df["Std Dev"] / np.sqrt(stats_df["Count"])).round(4)
    stats_df["CV (%)"] = (stats_df["Std Dev"] / stats_df["Mean"] * 100).round(2)
    st.dataframe(stats_df, use_container_width=True)

    plot_type = st.radio("Plot:", ["Box Plot", "Violin Plot", "Histogram"], horizontal=True, key="group_plot")
    if plot_type == "Box Plot":
        fig = px.box(df, x=group_col, y=value_col, color=group_col,
                     title=f"{value_col} by {group_col}", points="outliers")
        st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Violin Plot":
        fig = px.violin(df, x=group_col, y=value_col, color=group_col,
                        title=f"{value_col} by {group_col}", box=True, points="all")
        st.plotly_chart(fig, use_container_width=True)
    elif plot_type == "Histogram":
        fig = px.histogram(df, x=value_col, color=group_col,
                           title=f"{value_col} by {group_col}",
                           barmode="overlay", opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)


def _render_outliers(df: pd.DataFrame):
    """Outlier detection methods."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    col_name = st.selectbox("Select column:", num_cols, key="outlier_col")
    col_data = df[col_name].dropna()

    method = st.selectbox("Detection method:",
                          ["IQR (1.5×)", "IQR (3×)", "Z-Score", "Modified Z-Score (MAD)"],
                          key="outlier_method")

    if method.startswith("IQR"):
        multiplier = 1.5 if "1.5" in method else 3.0
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        outliers = col_data[(col_data < lower) | (col_data > upper)]
        st.write(f"**Bounds:** [{lower:.4f}, {upper:.4f}]")

    elif method == "Z-Score":
        threshold = st.slider("Z-score threshold:", 1.0, 5.0, 3.0, 0.1, key="zscore_thresh")
        z_scores = np.abs(stats.zscore(col_data))
        outliers = col_data[z_scores > threshold]

    elif method == "Modified Z-Score (MAD)":
        threshold = st.slider("Threshold:", 1.0, 5.0, 3.5, 0.1, key="mad_thresh")
        median = col_data.median()
        mad = np.median(np.abs(col_data - median))
        if mad == 0:
            st.warning("MAD is 0. Cannot compute modified Z-scores.")
            return
        modified_z = 0.6745 * (col_data - median) / mad
        outliers = col_data[np.abs(modified_z) > threshold]

    c1, c2 = st.columns(2)
    c1.metric("Outliers Found", len(outliers))
    c2.metric("Outlier %", f"{len(outliers) / len(col_data) * 100:.1f}%")

    if len(outliers) > 0:
        with st.expander(f"Outlier Values ({len(outliers)})"):
            st.dataframe(outliers.to_frame(), use_container_width=True)

    # Box plot with outliers highlighted
    fig = go.Figure()
    fig.add_trace(go.Box(y=col_data, name=col_name, boxpoints="all",
                         jitter=0.3, pointpos=-1.5,
                         marker=dict(color="steelblue", size=4)))
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(y=outliers, x=[col_name] * len(outliers),
                                 mode="markers", name="Outliers",
                                 marker=dict(color="red", size=8, symbol="x")))
    fig.update_layout(title=f"Outlier Detection: {col_name}", height=500)
    st.plotly_chart(fig, use_container_width=True)
