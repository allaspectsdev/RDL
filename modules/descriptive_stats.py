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
from modules.ui_helpers import section_header, empty_state, validation_panel, interpretation_card, sample_size_indicator
from modules.validation import check_sample_size, interpret_effect_size


def render_descriptive_stats(df: pd.DataFrame):
    """Render descriptive statistics interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    tabs = st.tabs([
        "Summary Statistics", "Distribution Analysis",
        "Frequency Analysis", "Grouped Statistics", "Outlier Detection",
        "Distribution Fitting", "Distribution Platform",
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
    with tabs[5]:
        _render_distribution_fitting(df)
    with tabs[6]:
        _render_distribution_platform(df)


def _render_summary_stats(df: pd.DataFrame):
    """Comprehensive summary statistics table."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
        return

    selected = st.multiselect("Select columns:", num_cols, default=num_cols[:5], key="summ_cols")
    if not selected:
        return

    ci_level = st.selectbox("Confidence level:", [0.90, 0.95, 0.99], index=1, key="ci_level")

    # Sample size indicator
    try:
        total_n = min(len(df[c].dropna()) for c in selected)
        sample_size_indicator(total_n, 30)
    except Exception:
        pass

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
    float_cols = stats_df.select_dtypes(include=[np.number]).columns
    stats_df[float_cols] = stats_df[float_cols].round(4)

    # Sub-tabs for organized display
    sub_tabs = st.tabs(["Overview", "Spread & Shape", "Percentiles & CI", "Full Table"])

    with sub_tabs[0]:
        overview_cols = ["Column", "Count", "Missing", "Unique", "Mean", "Median", "Std Dev", "Min", "Max"]
        st.dataframe(stats_df[[c for c in overview_cols if c in stats_df.columns]],
                     use_container_width=True, hide_index=True)

    with sub_tabs[1]:
        spread_cols = ["Column", "Variance", "SEM", "Range", "IQR", "CV (%)", "Skewness", "Kurtosis"]
        st.dataframe(stats_df[[c for c in spread_cols if c in stats_df.columns]],
                     use_container_width=True, hide_index=True)

    with sub_tabs[2]:
        ci_cols = [c for c in stats_df.columns if c.startswith("CI") or c in ["Column", "Q1", "Q3"]]
        st.dataframe(stats_df[ci_cols], use_container_width=True, hide_index=True)

        # Percentiles table
        section_header("Percentiles")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_records = []
        for col_name in selected:
            col = df[col_name].dropna()
            row = {"Column": col_name}
            for p in percentiles:
                row[f"P{p}"] = round(col.quantile(p / 100), 4)
            pct_records.append(row)
        st.dataframe(pd.DataFrame(pct_records), use_container_width=True, hide_index=True)

    with sub_tabs[3]:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    # Tolerance Intervals
    with st.expander("Tolerance Intervals"):
        ti_col = st.selectbox("Column for tolerance interval:", selected, key="ti_col")
        ti_data = df[ti_col].dropna()
        n_ti = len(ti_data)

        if n_ti < 3:
            st.warning("Need at least 3 data points for tolerance intervals.")
        else:
            c_cov, c_conf = st.columns(2)
            coverage = c_cov.slider("Coverage probability (p):", 0.50, 0.999, 0.95, 0.005, key="ti_coverage")
            confidence = c_conf.slider("Confidence level (gamma):", 0.50, 0.999, 0.95, 0.005, key="ti_confidence")

            # Normal-based tolerance interval
            mean_ti = ti_data.mean()
            std_ti = ti_data.std(ddof=1)
            z_p = stats.norm.ppf((1 + coverage) / 2)
            chi2_val = stats.chi2.ppf(1 - confidence, df=n_ti - 1)
            # k-factor: k = z_p * sqrt((n-1) * (1 + 1/n) / chi2_val)
            if chi2_val > 0:
                k_factor = z_p * np.sqrt((n_ti - 1) * (1 + 1 / n_ti) / chi2_val)
            else:
                k_factor = np.nan
            normal_lower = mean_ti - k_factor * std_ti
            normal_upper = mean_ti + k_factor * std_ti

            # Distribution-free tolerance interval based on order statistics
            sorted_ti = np.sort(ti_data.values)
            # Find r,s such that P(coverage of order stats >= p) >= gamma
            # Using beta distribution: P(X_{(s)} - X_{(r)} covers p) = 1 - sum of beta CDF terms
            # For two-sided: find smallest interval [X_(r), X_(s)] with the required coverage
            df_free_lower = np.nan
            df_free_upper = np.nan
            df_free_r = None
            df_free_s = None
            for r_idx in range(n_ti):
                for s_idx in range(r_idx + 1, n_ti):
                    # Probability that at least coverage proportion lies between X_(r+1) and X_(s+1)
                    # This is P(Beta(s-r, n-s+r+1) >= coverage) but using order stats directly:
                    # P(proportion covered >= p) = 1 - betainc evaluated at p with a=s_idx-r_idx, b=n_ti-(s_idx-r_idx)
                    j = s_idx - r_idx  # number of observations in the interval (exclusive of endpoints)
                    prob = 1 - stats.beta.cdf(coverage, j, n_ti - j + 1)
                    if prob >= confidence:
                        df_free_lower = sorted_ti[r_idx]
                        df_free_upper = sorted_ti[s_idx]
                        df_free_r = r_idx + 1
                        df_free_s = s_idx + 1
                        break
                if df_free_r is not None:
                    break

            # Display results as table
            section_header("Tolerance Interval Results",
                           f"Coverage={coverage:.1%}, Confidence={confidence:.1%}")

            ti_records = []
            ti_records.append({
                "Method": "Normal-Based",
                "Lower Bound": round(normal_lower, 4),
                "Upper Bound": round(normal_upper, 4),
                "k-factor": round(k_factor, 4),
                "Note": f"Assumes normality (n={n_ti})",
            })
            if df_free_r is not None:
                ti_records.append({
                    "Method": "Distribution-Free",
                    "Lower Bound": round(df_free_lower, 4),
                    "Upper Bound": round(df_free_upper, 4),
                    "k-factor": np.nan,
                    "Note": f"Order statistics X({df_free_r}) to X({df_free_s})",
                })
            else:
                ti_records.append({
                    "Method": "Distribution-Free",
                    "Lower Bound": np.nan,
                    "Upper Bound": np.nan,
                    "k-factor": np.nan,
                    "Note": "Not enough data for the requested coverage/confidence",
                })
            st.dataframe(pd.DataFrame(ti_records), use_container_width=True, hide_index=True)


def _render_distribution(df: pd.DataFrame):
    """Distribution analysis with plots and normality tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
        return

    col_name = st.selectbox("Select column:", num_cols, key="dist_col")
    col_data = df[col_name].dropna()

    if len(col_data) < 3:
        empty_state("Need at least 3 data points.", "The selected column needs more non-null values for distribution analysis.")
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
                         opacity=0.7), row=1, col=1)

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
    fig.add_trace(go.Box(y=col_data, name=col_name,
                         boxpoints="outliers"), row=1, col=2)

    # QQ plot
    sorted_data = np.sort(col_data)
    n = len(sorted_data)
    theoretical_q = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    fig.add_trace(go.Scatter(x=theoretical_q, y=sorted_data, mode="markers",
                             name="QQ Points", marker=dict(size=4)),
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
                             name="ECDF", line=dict(width=2)),
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

    # P-P Plot
    with st.expander("P-P Plot"):
        pp_dist = st.selectbox("Reference distribution:", ["Normal", "Lognormal", "Exponential"],
                               key="pp_ref_dist")
        pp_data = np.sort(col_data.values)
        n_pp = len(pp_data)
        empirical_cdf = (np.arange(1, n_pp + 1) - 0.5) / n_pp

        if pp_dist == "Normal":
            theoretical_cdf = stats.norm.cdf(pp_data, loc=col_data.mean(), scale=col_data.std())
        elif pp_dist == "Lognormal":
            positive_data = pp_data[pp_data > 0]
            if len(positive_data) < 3:
                st.warning("Lognormal requires positive data values.")
                theoretical_cdf = np.full(n_pp, np.nan)
            else:
                try:
                    shape, loc, scale = stats.lognorm.fit(positive_data, floc=0)
                    theoretical_cdf = stats.lognorm.cdf(pp_data, shape, loc=loc, scale=scale)
                except Exception:
                    theoretical_cdf = np.full(n_pp, np.nan)
        else:  # Exponential
            positive_data = pp_data[pp_data > 0]
            if len(positive_data) < 3:
                st.warning("Exponential requires positive data values.")
                theoretical_cdf = np.full(n_pp, np.nan)
            else:
                try:
                    loc_e, scale_e = stats.expon.fit(positive_data)
                    theoretical_cdf = stats.expon.cdf(pp_data, loc=loc_e, scale=scale_e)
                except Exception:
                    theoretical_cdf = np.full(n_pp, np.nan)

        if not np.all(np.isnan(theoretical_cdf)):
            fig_pp = go.Figure()
            fig_pp.add_trace(go.Scatter(
                x=theoretical_cdf, y=empirical_cdf, mode="markers",
                name="P-P Points", marker=dict(size=4),
            ))
            fig_pp.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                name="Reference (45-degree)", line=dict(color="red", dash="dash"),
            ))
            fig_pp.update_layout(
                title=f"P-P Plot: {col_name} vs {pp_dist}",
                xaxis_title="Theoretical CDF",
                yaxis_title="Empirical CDF",
                height=450,
            )
            st.plotly_chart(fig_pp, use_container_width=True)

    # Normality tests
    section_header("Normality Tests")
    test_results = []

    # Shapiro-Wilk (max 5000 samples)
    sample = col_data.values[:5000] if len(col_data) > 5000 else col_data.values
    if len(col_data) > 5000:
        st.caption(f"Shapiro-Wilk computed on first 5,000 of {len(col_data):,} values.")
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

        # Aggregate normality assessment
        try:
            # Count rejections from the test table
            n_reject = sum(1 for tr in test_results if tr.get("Normal (\u03b1=0.05)?") == "No")
            n_total = len(test_results)
            if n_reject >= 3:
                interpretation_card({"title": "Normality Assessment", "body": f"{n_reject} of {n_total} tests reject normality \u2014 data is likely non-normal."})
            elif n_reject == 0:
                interpretation_card({"title": "Normality Assessment", "body": "All normality tests pass \u2014 data appears normally distributed."})
            else:
                interpretation_card({"title": "Normality Assessment", "body": f"{n_reject} of {n_total} tests reject normality \u2014 results are mixed."})
        except Exception:
            pass


def _render_frequency(df: pd.DataFrame):
    """Frequency analysis for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Also include numeric columns with few unique values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 20:
            cat_cols.append(col)

    if not cat_cols:
        empty_state("No categorical columns (or numeric with ≤20 unique values) found.")
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
                             name="Count"), secondary_y=False)
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
        empty_state("Need both categorical and numeric columns for grouped statistics.")
        return

    group_col = st.selectbox("Group by:", cat_cols, key="group_col")
    value_col = st.selectbox("Value column:", num_cols, key="group_val_col")

    grouped = df.groupby(group_col)[value_col]
    stats_df = grouped.agg(["count", "mean", "median", "std", "min", "max"]).round(4)
    stats_df.columns = ["Count", "Mean", "Median", "Std Dev", "Min", "Max"]
    stats_df["SEM"] = (stats_df["Std Dev"] / np.sqrt(stats_df["Count"])).round(4)
    stats_df["CV (%)"] = np.round(np.where(stats_df["Mean"] != 0, stats_df["Std Dev"] / stats_df["Mean"] * 100, np.nan), 2)
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
        empty_state("No numeric columns found.")
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
                         marker=dict(size=4)))
    if len(outliers) > 0:
        fig.add_trace(go.Scatter(y=outliers, x=[col_name] * len(outliers),
                                 mode="markers", name="Outliers",
                                 marker=dict(color="red", size=8, symbol="x")))
    fig.update_layout(title=f"Outlier Detection: {col_name}", height=500)
    st.plotly_chart(fig, use_container_width=True)


# ─── Distribution Fitting Helpers ─────────────────────────────────────────

_DISTRIBUTIONS = {
    "Normal": stats.norm,
    "Lognormal": stats.lognorm,
    "Exponential": stats.expon,
    "Weibull (min)": stats.weibull_min,
    "Weibull (max)": stats.weibull_max,
    "Gamma": stats.gamma,
    "Beta": stats.beta,
    "Uniform": stats.uniform,
    "Logistic": stats.logistic,
    "Pareto": stats.pareto,
    "t": stats.t,
    "Rayleigh": stats.rayleigh,
    "Gumbel (right)": stats.gumbel_r,
}


@st.cache_data(show_spinner=False)
def _fit_distributions(data_tuple):
    """Fit multiple distributions and return serializable results sorted by AIC."""
    data = np.array(data_tuple)
    n = len(data)
    results = []
    for name, dist in _DISTRIBUTIONS.items():
        try:
            params = dist.fit(data)
            log_lik = np.sum(dist.logpdf(data, *params))
            if not np.isfinite(log_lik):
                continue
            k = len(params)
            aic = 2 * k - 2 * log_lik
            bic = k * np.log(n) - 2 * log_lik
            results.append({
                "Distribution": name,
                "AIC": aic,
                "BIC": bic,
                "Log-Likelihood": log_lik,
                "Parameters": params,
                "k": k,
            })
        except Exception:
            continue
    results.sort(key=lambda r: r["AIC"])
    return results


def _render_distribution_fitting(df: pd.DataFrame):
    """Distribution fitting with AIC/BIC ranking, QQ/PP plots, and GOF tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
        return

    col_name = st.selectbox("Select column:", num_cols, key="fit_col")
    col_data = df[col_name].dropna()

    if len(col_data) < 10:
        empty_state("Need at least 10 data points for distribution fitting.",
                     "The selected column has too few non-null values.")
        return

    data_values = col_data.values

    with st.spinner("Fitting distributions..."):
        fit_results = _fit_distributions(tuple(data_values))

    if not fit_results:
        st.warning("No distributions could be fitted to this data.")
        return

    # Results table
    section_header("Fit Results (ranked by AIC)",
                   "Lower AIC/BIC indicates better fit. Parameters are in scipy order.")
    table_records = []
    for r in fit_results:
        param_str = ", ".join(f"{p:.4f}" for p in r["Parameters"])
        table_records.append({
            "Distribution": r["Distribution"],
            "AIC": round(r["AIC"], 2),
            "BIC": round(r["BIC"], 2),
            "Log-Likelihood": round(r["Log-Likelihood"], 2),
            "Parameters": param_str,
        })
    st.dataframe(pd.DataFrame(table_records), use_container_width=True, hide_index=True)

    best = fit_results[0]
    best_dist = _DISTRIBUTIONS[best["Distribution"]]
    st.success(f"Best fit: **{best['Distribution']}** (AIC = {best['AIC']:.2f})")

    # Histogram with best-fit PDF overlay
    section_header("Histogram with Best-Fit PDF")
    n_bins_fit = st.slider("Bins:", 10, 100, 40, key="fit_bins")
    hist_vals, bin_edges = np.histogram(data_values, bins=n_bins_fit, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Bar(
        x=bin_centers, y=hist_vals, width=bin_width * 0.9,
        name="Data (density)", opacity=0.7,
    ))
    x_pdf = np.linspace(data_values.min(), data_values.max(), 300)
    y_pdf = best_dist.pdf(x_pdf, *best["Parameters"])
    fig_hist.add_trace(go.Scatter(
        x=x_pdf, y=y_pdf, mode="lines",
        name=f"{best['Distribution']} PDF",
        line=dict(color="red", width=2),
    ))
    fig_hist.update_layout(title=f"Histogram with {best['Distribution']} Fit",
                           xaxis_title="Value", yaxis_title="Density", height=450)
    st.plotly_chart(fig_hist, use_container_width=True)

    # QQ and PP plots for a selected distribution
    dist_names = [r["Distribution"] for r in fit_results]
    selected_dist_name = st.selectbox("Distribution for QQ/PP plots and GOF tests:",
                                       dist_names, key="fit_qq_dist")
    sel = next(r for r in fit_results if r["Distribution"] == selected_dist_name)
    sel_dist = _DISTRIBUTIONS[selected_dist_name]
    sel_params = sel["Parameters"]

    col_qq, col_pp = st.columns(2)

    # QQ plot
    with col_qq:
        section_header("QQ Plot")
        sorted_data = np.sort(data_values)
        n_d = len(sorted_data)
        theoretical_q = sel_dist.ppf((np.arange(1, n_d + 1) - 0.5) / n_d, *sel_params)
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=theoretical_q, y=sorted_data, mode="markers",
                                     name="QQ Points", marker=dict(size=4)))
        valid = np.isfinite(theoretical_q) & np.isfinite(sorted_data)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(theoretical_q[valid], sorted_data[valid], 1)
            qq_line_x = theoretical_q[valid]
            fig_qq.add_trace(go.Scatter(x=qq_line_x, y=slope * qq_line_x + intercept,
                                         mode="lines", name="Reference",
                                         line=dict(color="red", dash="dash")))
        fig_qq.update_layout(title=f"QQ Plot: {selected_dist_name}",
                             xaxis_title="Theoretical Quantiles",
                             yaxis_title="Sample Quantiles", height=400)
        st.plotly_chart(fig_qq, use_container_width=True)

    # PP plot
    with col_pp:
        section_header("P-P Plot")
        empirical_cdf = (np.arange(1, n_d + 1) - 0.5) / n_d
        theoretical_cdf = sel_dist.cdf(sorted_data, *sel_params)
        fig_pp = go.Figure()
        fig_pp.add_trace(go.Scatter(x=theoretical_cdf, y=empirical_cdf, mode="markers",
                                     name="P-P Points", marker=dict(size=4)))
        fig_pp.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Reference (45-degree)",
                                     line=dict(color="red", dash="dash")))
        fig_pp.update_layout(title=f"P-P Plot: {selected_dist_name}",
                             xaxis_title="Theoretical CDF",
                             yaxis_title="Empirical CDF", height=400)
        st.plotly_chart(fig_pp, use_container_width=True)

    # Goodness-of-fit tests
    section_header("Goodness-of-Fit Tests")
    gof_results = []

    # KS test
    try:
        ks_stat, ks_p = stats.kstest(data_values, sel_dist.cdf, args=sel_params)
        gof_results.append({"Test": "Kolmogorov-Smirnov", "Statistic": round(ks_stat, 6),
                            "p-value": round(ks_p, 6),
                            "Result": "Pass" if ks_p > 0.05 else "Reject"})
    except Exception:
        pass

    # Anderson-Darling (only for supported distributions)
    ad_dist_map = {"Normal": "norm", "Exponential": "expon", "Logistic": "logistic",
                   "Gumbel (right)": "gumbel_r"}
    if selected_dist_name in ad_dist_map:
        try:
            ad_result = stats.anderson(data_values, dist=ad_dist_map[selected_dist_name])
            ad_crit = ad_result.critical_values[2]  # 5% level
            gof_results.append({"Test": "Anderson-Darling", "Statistic": round(ad_result.statistic, 6),
                                "p-value": np.nan,
                                "Result": "Pass" if ad_result.statistic < ad_crit else "Reject"})
        except Exception:
            pass

    # Chi-square GOF
    try:
        n_chi_bins = min(max(int(np.sqrt(len(data_values))), 5), 50)
        observed_freq, chi_bin_edges = np.histogram(data_values, bins=n_chi_bins)
        expected_freq = np.diff(sel_dist.cdf(chi_bin_edges, *sel_params)) * len(data_values)
        # Merge bins with expected < 5
        merged_obs, merged_exp = [], []
        cum_obs, cum_exp = 0, 0
        for o, e in zip(observed_freq, expected_freq):
            cum_obs += o
            cum_exp += e
            if cum_exp >= 5:
                merged_obs.append(cum_obs)
                merged_exp.append(cum_exp)
                cum_obs, cum_exp = 0, 0
        if cum_obs > 0 or cum_exp > 0:
            if merged_obs:
                merged_obs[-1] += cum_obs
                merged_exp[-1] += cum_exp
            else:
                merged_obs.append(cum_obs)
                merged_exp.append(cum_exp)
        if len(merged_obs) > sel["k"] + 1:
            chi2_stat, chi2_p = stats.chisquare(merged_obs, f_exp=merged_exp,
                                                 ddof=sel["k"])
            gof_results.append({"Test": "Chi-Square GOF", "Statistic": round(chi2_stat, 6),
                                "p-value": round(chi2_p, 6),
                                "Result": "Pass" if chi2_p > 0.05 else "Reject"})
    except Exception:
        pass

    if gof_results:
        st.dataframe(pd.DataFrame(gof_results), use_container_width=True, hide_index=True)
    else:
        st.info("No goodness-of-fit tests could be computed for this distribution.")

    # Bootstrap CIs on parameters
    with st.expander("Bootstrap Confidence Intervals on Parameters"):
        n_boot = st.number_input("Bootstrap samples:", 100, 5000, 1000, 100, key="fit_n_boot")
        if st.button("Run Bootstrap", key="fit_run_boot"):
            with st.spinner(f"Running {n_boot} bootstrap samples..."):
                boot_params = []
                for _ in range(int(n_boot)):
                    sample = np.random.choice(data_values, size=len(data_values), replace=True)
                    try:
                        bp = sel_dist.fit(sample)
                        boot_params.append(bp)
                    except Exception:
                        continue

            if len(boot_params) < 10:
                st.warning("Too few successful bootstrap fits.")
            else:
                boot_arr = np.array(boot_params)
                ci_records = []
                for i in range(boot_arr.shape[1]):
                    lo = np.percentile(boot_arr[:, i], 2.5)
                    hi = np.percentile(boot_arr[:, i], 97.5)
                    med = np.median(boot_arr[:, i])
                    ci_records.append({
                        "Parameter": f"Param {i + 1}",
                        "Estimate": round(sel_params[i], 4),
                        "Bootstrap Median": round(med, 4),
                        "95% CI Lower": round(lo, 4),
                        "95% CI Upper": round(hi, 4),
                    })
                st.dataframe(pd.DataFrame(ci_records), use_container_width=True, hide_index=True)
                st.caption(f"Based on {len(boot_params)} successful bootstrap fits out of {int(n_boot)} attempts.")


# ─── Distribution Platform (JMP-style unified view) ──────────────────────

def _render_distribution_platform(df: pd.DataFrame):
    """JMP-style unified distribution view: histogram+KDE, box+violin, QQ, plus stats."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
        return

    col_name = st.selectbox("Select column:", num_cols, key="dp_col")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    group_col = st.selectbox("Group By (optional):", ["None"] + cat_cols, key="dp_group")
    group_col = None if group_col == "None" else group_col

    groups = {}
    if group_col:
        for g_val in df[group_col].dropna().unique():
            g_data = df.loc[df[group_col] == g_val, col_name].dropna().values
            if len(g_data) >= 3:
                groups[str(g_val)] = g_data
        if not groups:
            empty_state("No groups with enough data (need >= 3 per group).")
            return
    else:
        col_data = df[col_name].dropna().values
        if len(col_data) < 3:
            empty_state("Need at least 3 data points.", "The selected column needs more non-null values.")
            return
        groups["All Data"] = col_data

    for g_label, g_data in groups.items():
        if group_col:
            st.markdown(f"---")
            section_header(f"{group_col} = {g_label}")

        # 2x2 subplot: Histogram+KDE, Box+Violin, QQ Plot, ECDF
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Histogram + KDE", "Box + Violin", "Normal QQ Plot", "ECDF"),
            vertical_spacing=0.14, horizontal_spacing=0.1,
        )

        # 1. Histogram + KDE
        n_bins = max(10, min(50, int(np.sqrt(len(g_data)))))
        hist_vals, bin_edges = np.histogram(g_data, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fig.add_trace(go.Bar(x=bin_centers, y=hist_vals, name="Histogram",
                             opacity=0.7, showlegend=False), row=1, col=1)
        try:
            kde = stats.gaussian_kde(g_data)
            kde_x = np.linspace(g_data.min(), g_data.max(), 200)
            kde_y = kde(kde_x) * len(g_data) * (bin_edges[1] - bin_edges[0])
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y, name="KDE",
                                     line=dict(color="#ef4444", width=2),
                                     showlegend=False), row=1, col=1)
        except Exception:
            pass

        # 2. Box + Violin
        fig.add_trace(go.Violin(y=g_data, name="Violin", side="positive",
                                line_color="#6366f1", fillcolor="rgba(99,102,241,0.15)",
                                showlegend=False), row=1, col=2)
        fig.add_trace(go.Box(y=g_data, name="Box", boxpoints="outliers",
                             marker=dict(size=3), showlegend=False), row=1, col=2)

        # 3. QQ Plot
        sorted_d = np.sort(g_data)
        n_d = len(sorted_d)
        theoretical_q = stats.norm.ppf((np.arange(1, n_d + 1) - 0.5) / n_d)
        fig.add_trace(go.Scatter(x=theoretical_q, y=sorted_d, mode="markers",
                                 marker=dict(size=3), name="QQ",
                                 showlegend=False), row=2, col=1)
        slope_qq, intercept_qq = np.polyfit(theoretical_q, sorted_d, 1)
        fig.add_trace(go.Scatter(x=theoretical_q,
                                 y=slope_qq * theoretical_q + intercept_qq,
                                 mode="lines", line=dict(color="red", dash="dash"),
                                 name="Ref", showlegend=False), row=2, col=1)

        # 4. ECDF
        ecdf_y = np.arange(1, n_d + 1) / n_d
        fig.add_trace(go.Scatter(x=sorted_d, y=ecdf_y, mode="lines",
                                 name="ECDF", line=dict(width=2),
                                 showlegend=False), row=2, col=2)
        norm_cdf_y = stats.norm.cdf(sorted_d, np.mean(g_data), np.std(g_data, ddof=1))
        fig.add_trace(go.Scatter(x=sorted_d, y=norm_cdf_y, mode="lines",
                                 name="Normal CDF",
                                 line=dict(color="red", dash="dash"),
                                 showlegend=False), row=2, col=2)

        fig.update_layout(height=650,
                          title_text=f"Distribution Platform: {col_name}" + (f" ({g_label})" if group_col else ""))
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Sample Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Value", row=2, col=2)
        fig.update_yaxes(title_text="Cumulative Probability", row=2, col=2)
        st.plotly_chart(fig, use_container_width=True)

        # Percentile table + Normality tests + Top-3 distribution fits
        c_left, c_right = st.columns(2)

        with c_left:
            section_header("Percentiles")
            pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            pct_vals = {f"P{p}": round(float(np.percentile(g_data, p)), 4) for p in pctiles}
            st.dataframe(pd.DataFrame([pct_vals]), use_container_width=True, hide_index=True)

        with c_right:
            section_header("Normality Tests")
            norm_records = []
            try:
                sw_s, sw_p = stats.shapiro(g_data[:5000] if len(g_data) > 5000 else g_data)
                norm_records.append({"Test": "Shapiro-Wilk", "Statistic": round(sw_s, 4),
                                     "p-value": round(sw_p, 4),
                                     "Normal?": "Yes" if sw_p > 0.05 else "No"})
            except Exception:
                pass
            try:
                ad_res = stats.anderson(g_data, dist="norm")
                ad_pass = "Yes" if ad_res.statistic < ad_res.critical_values[2] else "No"
                norm_records.append({"Test": "Anderson-Darling", "Statistic": round(ad_res.statistic, 4),
                                     "p-value": round(ad_res.critical_values[2], 4),
                                     "Normal?": ad_pass})
            except Exception:
                pass
            try:
                jb_s, jb_p = stats.jarque_bera(g_data)
                norm_records.append({"Test": "Jarque-Bera", "Statistic": round(jb_s, 4),
                                     "p-value": round(jb_p, 4),
                                     "Normal?": "Yes" if jb_p > 0.05 else "No"})
            except Exception:
                pass
            if norm_records:
                st.dataframe(pd.DataFrame(norm_records), use_container_width=True, hide_index=True)

        # Top-3 distribution fits
        section_header("Top Distribution Fits")
        try:
            fit_results = _fit_distributions(tuple(g_data))
            if fit_results:
                top3 = fit_results[:3]
                fit_table = []
                for r in top3:
                    param_str = ", ".join(f"{p:.4f}" for p in r["Parameters"])
                    fit_table.append({
                        "Distribution": r["Distribution"],
                        "AIC": round(r["AIC"], 2),
                        "BIC": round(r["BIC"], 2),
                        "Parameters": param_str,
                    })
                st.dataframe(pd.DataFrame(fit_table), use_container_width=True, hide_index=True)
        except Exception:
            st.caption("Could not fit distributions to this data.")
