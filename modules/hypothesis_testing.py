"""
Hypothesis Testing Module - t-tests, chi-square, power analysis, and more.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_hypothesis_testing(df: pd.DataFrame):
    """Render hypothesis testing interface."""
    if df is None or df.empty:
        st.warning("No data loaded.")
        return

    tabs = st.tabs([
        "One-Sample Tests", "Two-Sample Tests", "Chi-Square Tests",
        "Normality Tests", "Power Analysis", "Multiple Comparisons",
        "Bootstrap & Permutation",
    ])

    with tabs[0]:
        _render_one_sample(df)
    with tabs[1]:
        _render_two_sample(df)
    with tabs[2]:
        _render_chi_square(df)
    with tabs[3]:
        _render_normality(df)
    with tabs[4]:
        _render_power_analysis()
    with tabs[5]:
        _render_multiple_comparisons(df)
    with tabs[6]:
        _render_bootstrap_permutation(df)


def _interpret_p(p_value, alpha):
    if p_value < alpha:
        return f"**Reject H₀** (p = {p_value:.6f} < α = {alpha})"
    return f"**Fail to reject H₀** (p = {p_value:.6f} ≥ α = {alpha})"


def _cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def _effect_size_label(d):
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    return "large"


def _render_one_sample(df: pd.DataFrame):
    """One-sample hypothesis tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns available.")
        return

    test_type = st.selectbox("Test:", [
        "One-Sample t-test",
        "One-Sample Wilcoxon Signed-Rank",
        "One-Sample Proportion",
    ], key="one_sample_test")

    alpha = st.slider("Significance level (α):", 0.001, 0.10, 0.05, 0.001, key="one_alpha")

    if test_type == "One-Sample t-test":
        col_name = st.selectbox("Column:", num_cols, key="one_t_col")
        mu_0 = st.number_input("Hypothesized mean (μ₀):", value=0.0, key="one_t_mu")
        alt = st.selectbox("Alternative:", ["two-sided", "less", "greater"], key="one_t_alt")

        if st.button("Run Test", key="run_one_t"):
            data = df[col_name].dropna()
            n = len(data)
            stat, p = stats.ttest_1samp(data, mu_0, alternative=alt)
            se = data.std() / np.sqrt(n)
            d = (data.mean() - mu_0) / data.std() if data.std() != 0 else 0
            ci = stats.t.interval(1 - alpha, df=n - 1, loc=data.mean(), scale=se)

            st.markdown(f"**H₀:** μ = {mu_0}  |  **H₁:** μ {'≠' if alt == 'two-sided' else '<' if alt == 'less' else '>'} {mu_0}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("t-statistic", f"{stat:.4f}")
            c2.metric("p-value", f"{p:.6f}")
            c3.metric("Cohen's d", f"{d:.4f} ({_effect_size_label(d)})")
            c4.metric("Sample Mean", f"{data.mean():.4f}")
            st.write(f"**{1 - alpha:.0%} CI for mean:** [{ci[0]:.4f}, {ci[1]:.4f}]")
            st.markdown(_interpret_p(p, alpha))

            # Visualization
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Distribution", "t-Distribution"))
            fig.add_trace(go.Histogram(x=data, name="Data", opacity=0.7), row=1, col=1)
            fig.add_vline(x=mu_0, line_dash="dash", line_color="red", row=1, col=1,
                          annotation_text=f"μ₀={mu_0}")
            fig.add_vline(x=data.mean(), line_dash="dash", line_color="green", row=1, col=1,
                          annotation_text=f"x̄={data.mean():.3f}")

            # t-distribution with rejection region
            t_x = np.linspace(-4, 4, 200)
            t_y = stats.t.pdf(t_x, df=n - 1)
            fig.add_trace(go.Scatter(x=t_x, y=t_y, name="t-dist", line=dict(color="blue")), row=1, col=2)
            fig.add_vline(x=stat, line_dash="solid", line_color="red", row=1, col=2,
                          annotation_text=f"t={stat:.3f}")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif test_type == "One-Sample Wilcoxon Signed-Rank":
        col_name = st.selectbox("Column:", num_cols, key="one_wilc_col")
        mu_0 = st.number_input("Hypothesized median:", value=0.0, key="one_wilc_mu")
        alt = st.selectbox("Alternative:", ["two-sided", "less", "greater"], key="one_wilc_alt")

        if st.button("Run Test", key="run_one_wilc"):
            data = df[col_name].dropna()
            adjusted = data - mu_0
            # Remove zeros for Wilcoxon
            adjusted = adjusted[adjusted != 0]
            if len(adjusted) < 10:
                st.warning("Need at least 10 non-zero differences.")
                return
            stat, p = stats.wilcoxon(adjusted, alternative=alt)
            # Effect size r = Z / sqrt(N)
            n_w = len(adjusted)
            z_approx = (stat - n_w * (n_w + 1) / 4) / np.sqrt(n_w * (n_w + 1) * (2 * n_w + 1) / 24)
            r = abs(z_approx) / np.sqrt(n_w)

            st.markdown(f"**H₀:** median = {mu_0}")
            c1, c2, c3 = st.columns(3)
            c1.metric("W-statistic", f"{stat:.4f}")
            c2.metric("p-value", f"{p:.6f}")
            c3.metric("Effect size r", f"{r:.4f}")
            st.markdown(_interpret_p(p, alpha))

    elif test_type == "One-Sample Proportion":
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        bin_cols = [c for c in num_cols if df[c].nunique() == 2]
        all_cols = cat_cols + bin_cols
        if not all_cols:
            st.warning("No binary/categorical columns found.")
            return

        col_name = st.selectbox("Column:", all_cols, key="one_prop_col")
        vals = df[col_name].dropna().unique()
        success_val = st.selectbox("Success value:", vals, key="one_prop_success")
        p_0 = st.number_input("Hypothesized proportion (p₀):", 0.0, 1.0, 0.5, 0.01, key="one_prop_p0")

        if st.button("Run Test", key="run_one_prop"):
            data = df[col_name].dropna()
            n = len(data)
            x = (data == success_val).sum()
            p_hat = x / n
            # Z-test for proportion
            se = np.sqrt(p_0 * (1 - p_0) / n)
            z_stat = (p_hat - p_0) / se if se > 0 else 0
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            st.markdown(f"**H₀:** p = {p_0}  |  **H₁:** p ≠ {p_0}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Z-statistic", f"{z_stat:.4f}")
            c2.metric("p-value", f"{p_val:.6f}")
            c3.metric("Sample proportion", f"{p_hat:.4f}")
            # Wilson CI
            z_crit = stats.norm.ppf(1 - alpha / 2)
            denom = 1 + z_crit ** 2 / n
            center = (p_hat + z_crit ** 2 / (2 * n)) / denom
            margin = z_crit * np.sqrt((p_hat * (1 - p_hat) + z_crit ** 2 / (4 * n)) / n) / denom
            st.write(f"**{1 - alpha:.0%} Wilson CI:** [{center - margin:.4f}, {center + margin:.4f}]")
            st.markdown(_interpret_p(p_val, alpha))


def _render_two_sample(df: pd.DataFrame):
    """Two-sample hypothesis tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    test_type = st.selectbox("Test:", [
        "Independent t-test", "Welch's t-test", "Paired t-test",
        "Mann-Whitney U", "Wilcoxon Signed-Rank (paired)",
        "Two-Sample KS Test",
    ], key="two_sample_test")

    alpha = st.slider("Significance level (α):", 0.001, 0.10, 0.05, 0.001, key="two_alpha")

    if test_type in ("Independent t-test", "Welch's t-test", "Mann-Whitney U", "Two-Sample KS Test"):
        # Need a grouping variable
        if not cat_cols:
            st.warning("Need a categorical column to define groups.")
            return
        if not num_cols:
            st.warning("No numeric columns available.")
            return

        value_col = st.selectbox("Value column:", num_cols, key="two_val_col")
        group_col = st.selectbox("Group column:", cat_cols, key="two_grp_col")
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            st.warning("Need at least 2 groups.")
            return

        g1_name = st.selectbox("Group 1:", groups, index=0, key="two_g1")
        g2_name = st.selectbox("Group 2:", groups, index=min(1, len(groups) - 1), key="two_g2")

        if st.button("Run Test", key="run_two"):
            g1 = df[df[group_col] == g1_name][value_col].dropna()
            g2 = df[df[group_col] == g2_name][value_col].dropna()

            if test_type == "Independent t-test":
                stat, p = stats.ttest_ind(g1, g2, equal_var=True)
                test_label = "t"
            elif test_type == "Welch's t-test":
                stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                test_label = "t"
            elif test_type == "Mann-Whitney U":
                stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                test_label = "U"
            elif test_type == "Two-Sample KS Test":
                stat, p = stats.ks_2samp(g1, g2)
                test_label = "D"

            d = _cohens_d(g1, g2)

            st.markdown(f"**H₀:** The two groups have the same {'mean' if 'test' in test_type.lower() else 'distribution'}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{test_label}-statistic", f"{stat:.4f}")
            c2.metric("p-value", f"{p:.6f}")
            c3.metric("Cohen's d", f"{d:.4f} ({_effect_size_label(d)})")
            c4.metric("Diff of Means", f"{g1.mean() - g2.mean():.4f}")

            st.write(f"**Group 1 ({g1_name}):** n={len(g1)}, mean={g1.mean():.4f}, sd={g1.std():.4f}")
            st.write(f"**Group 2 ({g2_name}):** n={len(g2)}, mean={g2.mean():.4f}, sd={g2.std():.4f}")
            st.markdown(_interpret_p(p, alpha))

            # Visualization
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Distributions", "Box Plot"))
            fig.add_trace(go.Histogram(x=g1, name=str(g1_name), opacity=0.6), row=1, col=1)
            fig.add_trace(go.Histogram(x=g2, name=str(g2_name), opacity=0.6), row=1, col=1)
            fig.update_layout(barmode="overlay")
            fig.add_trace(go.Box(y=g1, name=str(g1_name)), row=1, col=2)
            fig.add_trace(go.Box(y=g2, name=str(g2_name)), row=1, col=2)
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif test_type in ("Paired t-test", "Wilcoxon Signed-Rank (paired)"):
        if len(num_cols) < 2:
            st.warning("Need at least 2 numeric columns for paired test.")
            return
        col1 = st.selectbox("Column 1 (before/condition A):", num_cols, key="pair_c1")
        col2 = st.selectbox("Column 2 (after/condition B):", num_cols, index=min(1, len(num_cols) - 1), key="pair_c2")

        if st.button("Run Test", key="run_paired"):
            data = df[[col1, col2]].dropna()
            g1, g2 = data[col1], data[col2]
            diff = g1 - g2

            if test_type == "Paired t-test":
                stat, p = stats.ttest_rel(g1, g2)
                d = diff.mean() / diff.std() if diff.std() != 0 else 0
                test_label = "t"
            else:
                stat, p = stats.wilcoxon(diff[diff != 0])
                d = diff.mean() / diff.std() if diff.std() != 0 else 0
                test_label = "W"

            st.markdown(f"**H₀:** No difference between paired measurements")
            c1, c2, c3 = st.columns(3)
            c1.metric(f"{test_label}-statistic", f"{stat:.4f}")
            c2.metric("p-value", f"{p:.6f}")
            c3.metric("Effect size d", f"{d:.4f} ({_effect_size_label(d)})")
            st.write(f"**Mean difference:** {diff.mean():.4f} ± {diff.std():.4f}")
            st.markdown(_interpret_p(p, alpha))

            fig = make_subplots(rows=1, cols=2, subplot_titles=("Paired Differences", "Before vs After"))
            fig.add_trace(go.Histogram(x=diff, name="Differences"), row=1, col=1)
            fig.add_vline(x=0, line_dash="dash", line_color="red", row=1, col=1)
            fig.add_trace(go.Scatter(x=g1, y=g2, mode="markers", name="Pairs"), row=1, col=2)
            min_val = min(g1.min(), g2.min())
            max_val = max(g1.max(), g2.max())
            fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode="lines", name="y=x", line=dict(dash="dash", color="red")),
                          row=1, col=2)
            fig.update_layout(height=400)
            fig.update_xaxes(title_text=col1, row=1, col=2)
            fig.update_yaxes(title_text=col2, row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)


def _render_chi_square(df: pd.DataFrame):
    """Chi-square tests."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # Include low-cardinality numeric
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].nunique() <= 15:
            cat_cols.append(c)

    if len(cat_cols) < 1:
        st.warning("Need categorical columns for chi-square tests.")
        return

    test_type = st.selectbox("Test:", [
        "Chi-Square Test of Independence",
        "Chi-Square Goodness of Fit",
    ], key="chi_test_type")

    alpha = st.slider("α:", 0.001, 0.10, 0.05, 0.001, key="chi_alpha")

    if test_type == "Chi-Square Test of Independence":
        if len(cat_cols) < 2:
            st.warning("Need at least 2 categorical columns.")
            return
        col1 = st.selectbox("Variable 1:", cat_cols, key="chi_col1")
        col2 = st.selectbox("Variable 2:", [c for c in cat_cols if c != col1], key="chi_col2")

        if st.button("Run Test", key="run_chi_ind"):
            ct = pd.crosstab(df[col1], df[col2])
            st.markdown("**Contingency Table (Observed):**")
            st.dataframe(ct, use_container_width=True)

            chi2, p, dof, expected = stats.chi2_contingency(ct)
            n = ct.values.sum()
            k = min(ct.shape)
            cramers_v = np.sqrt(chi2 / (n * (k - 1))) if (n * (k - 1)) > 0 else 0

            st.markdown("**Expected Frequencies:**")
            expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(2)
            st.dataframe(expected_df, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("χ²", f"{chi2:.4f}")
            c2.metric("p-value", f"{p:.6f}")
            c3.metric("df", str(dof))
            c4.metric("Cramér's V", f"{cramers_v:.4f}")
            st.markdown(_interpret_p(p, alpha))

            # Fisher's exact for 2x2
            if ct.shape == (2, 2):
                odds, fisher_p = stats.fisher_exact(ct)
                st.write(f"**Fisher's exact test:** odds ratio = {odds:.4f}, p = {fisher_p:.6f}")

            # Residuals heatmap
            with np.errstate(divide="ignore", invalid="ignore"):
                residuals = np.where(expected > 0, (ct.values - expected) / np.sqrt(expected), 0.0)
            fig = px.imshow(residuals, x=ct.columns.astype(str), y=ct.index.astype(str),
                            color_continuous_scale="RdBu_r", text_auto=".2f",
                            title="Standardized Residuals")
            st.plotly_chart(fig, use_container_width=True)

    elif test_type == "Chi-Square Goodness of Fit":
        col_name = st.selectbox("Variable:", cat_cols, key="chi_gof_col")
        dist_type = st.selectbox("Expected distribution:", ["Uniform", "Custom"], key="chi_gof_dist")

        if st.button("Run Test", key="run_chi_gof"):
            observed = df[col_name].value_counts().sort_index()
            n = observed.sum()

            if dist_type == "Uniform":
                k = len(observed)
                expected = np.full(k, n / k)
            else:
                expected = np.full(len(observed), n / len(observed))

            chi2, p = stats.chisquare(observed.values, f_exp=expected)

            st.markdown("**Observed vs Expected:**")
            comp_df = pd.DataFrame({
                "Category": observed.index,
                "Observed": observed.values,
                "Expected": expected.round(2),
            })
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("χ²", f"{chi2:.4f}")
            c2.metric("p-value", f"{p:.6f}")
            c3.metric("df", str(len(observed) - 1))
            st.markdown(_interpret_p(p, alpha))

            fig = go.Figure()
            fig.add_trace(go.Bar(x=observed.index.astype(str), y=observed.values, name="Observed"))
            fig.add_trace(go.Scatter(x=observed.index.astype(str), y=expected, name="Expected",
                                     mode="markers+lines", line=dict(color="red", width=2)))
            fig.update_layout(title="Observed vs Expected Frequencies", height=400)
            st.plotly_chart(fig, use_container_width=True)


def _render_normality(df: pd.DataFrame):
    """Comprehensive normality tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    selected = st.multiselect("Select columns:", num_cols, default=num_cols[:3], key="norm_cols")
    if not selected:
        return

    results = []
    for col_name in selected:
        data = df[col_name].dropna().values
        n = len(data)
        row = {"Column": col_name, "n": n}

        if n >= 3:
            sample = data[:5000] if n > 5000 else data
            try:
                sw_stat, sw_p = stats.shapiro(sample)
                row["Shapiro-Wilk W"] = sw_stat
                row["Shapiro-Wilk p"] = sw_p
            except Exception:
                row["Shapiro-Wilk W"] = np.nan
                row["Shapiro-Wilk p"] = np.nan

        if n >= 20:
            try:
                dag_stat, dag_p = stats.normaltest(data)
                row["D'Agostino K²"] = dag_stat
                row["D'Agostino p"] = dag_p
            except Exception:
                row["D'Agostino K²"] = np.nan
                row["D'Agostino p"] = np.nan

        try:
            jb_stat, jb_p = stats.jarque_bera(data)
            row["Jarque-Bera"] = jb_stat
            row["JB p"] = jb_p
        except Exception:
            pass

        try:
            ks_stat, ks_p = stats.kstest(data, "norm", args=(data.mean(), data.std()))
            row["KS D"] = ks_stat
            row["KS p"] = ks_p
        except Exception:
            pass

        results.append(row)

    results_df = pd.DataFrame(results)
    float_cols = results_df.select_dtypes(include=[np.number]).columns
    results_df[float_cols] = results_df[float_cols].round(6)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # QQ plots
    st.markdown("#### QQ Plots")
    n_plots = len(selected)
    cols_per_row = min(3, n_plots)
    rows = (n_plots + cols_per_row - 1) // cols_per_row
    fig = make_subplots(rows=rows, cols=cols_per_row,
                        subplot_titles=selected)
    for i, col_name in enumerate(selected):
        data = np.sort(df[col_name].dropna().values)
        n = len(data)
        theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        r, c = i // cols_per_row + 1, i % cols_per_row + 1
        fig.add_trace(go.Scatter(x=theoretical, y=data, mode="markers",
                                 marker=dict(size=3, color="steelblue"),
                                 showlegend=False), row=r, col=c)
        slope, intercept = np.polyfit(theoretical, data, 1)
        fig.add_trace(go.Scatter(x=theoretical, y=slope * theoretical + intercept,
                                 mode="lines", line=dict(color="red", dash="dash"),
                                 showlegend=False), row=r, col=c)
    fig.update_layout(height=350 * rows)
    st.plotly_chart(fig, use_container_width=True)


def _render_power_analysis():
    """Statistical power analysis."""
    analysis_type = st.selectbox("Analysis for:", [
        "One-Sample t-test", "Two-Sample t-test", "One-Way ANOVA",
    ], key="power_type")

    calc_mode = st.radio("Calculate:", ["Required Sample Size", "Achievable Power"], horizontal=True, key="power_mode")

    if analysis_type in ("One-Sample t-test", "Two-Sample t-test"):
        effect_size = st.slider("Cohen's d (effect size):", 0.1, 2.0, 0.5, 0.05, key="power_d")
        alpha = st.slider("α:", 0.001, 0.10, 0.05, 0.001, key="power_alpha")

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_target")
            if st.button("Calculate", key="calc_power_n"):
                from scipy.stats import norm
                # Using formula: n = ((z_alpha + z_beta) / d)^2
                z_alpha = norm.ppf(1 - alpha / 2)
                z_beta = norm.ppf(target_power)
                n = ((z_alpha + z_beta) / effect_size) ** 2
                n = int(np.ceil(n))
                if analysis_type == "Two-Sample t-test":
                    st.metric("Required n per group", n)
                    st.write(f"Total sample size: **{n * 2}**")
                else:
                    st.metric("Required n", n)

                # Power curve
                ns = np.arange(5, max(n * 2, 100))
                powers = []
                for ni in ns:
                    ncp = effect_size * np.sqrt(ni)
                    power_i = 1 - stats.t.cdf(stats.t.ppf(1 - alpha / 2, df=ni - 1), df=ni - 1, loc=ncp)
                    powers.append(power_i)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ns, y=powers, mode="lines", name="Power"))
                fig.add_hline(y=target_power, line_dash="dash", line_color="red",
                              annotation_text=f"Target={target_power}")
                fig.add_vline(x=n, line_dash="dash", line_color="green",
                              annotation_text=f"n={n}")
                fig.update_layout(title="Power Curve", xaxis_title="Sample Size", yaxis_title="Power",
                                  height=400)
                st.plotly_chart(fig, use_container_width=True)

        else:
            sample_n = st.number_input("Sample size (per group):", 5, 10000, 30, key="power_n")
            if st.button("Calculate", key="calc_power_p"):
                ncp = effect_size * np.sqrt(sample_n)
                crit = stats.t.ppf(1 - alpha / 2, df=sample_n - 1)
                power = 1 - stats.t.cdf(crit, df=sample_n - 1, loc=ncp)
                st.metric("Achievable Power", f"{power:.4f}")

                # Power for different effect sizes
                ds = [0.2, 0.5, 0.8, 1.0, 1.5]
                fig = go.Figure()
                ns = np.arange(5, 200)
                for d in ds:
                    powers = []
                    for ni in ns:
                        ncp_i = d * np.sqrt(ni)
                        power_i = 1 - stats.t.cdf(stats.t.ppf(1 - alpha / 2, df=ni - 1), df=ni - 1, loc=ncp_i)
                        powers.append(power_i)
                    fig.add_trace(go.Scatter(x=ns, y=powers, name=f"d={d}"))
                fig.add_hline(y=0.8, line_dash="dash", line_color="gray")
                fig.update_layout(title="Power vs Sample Size by Effect Size",
                                  xaxis_title="n", yaxis_title="Power", height=400)
                st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "One-Way ANOVA":
        effect_size = st.slider("Cohen's f (effect size):", 0.05, 1.0, 0.25, 0.05, key="power_f")
        k = st.number_input("Number of groups:", 2, 20, 3, key="power_k")
        alpha = st.slider("α:", 0.001, 0.10, 0.05, 0.001, key="power_anova_alpha")

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_anova_target")
            if st.button("Calculate", key="calc_power_anova_n"):
                # Iterative search
                for n in range(5, 5000):
                    dfn = k - 1
                    dfd = k * (n - 1)
                    ncp = k * n * effect_size ** 2
                    crit = stats.f.ppf(1 - alpha, dfn, dfd)
                    power = 1 - stats.f.cdf(crit, dfn, dfd, ncp)
                    if power >= target_power:
                        break
                st.metric("Required n per group", n)
                st.write(f"Total sample size: **{n * k}**")
        else:
            n_per_group = st.number_input("n per group:", 5, 5000, 30, key="power_anova_n")
            if st.button("Calculate", key="calc_power_anova_p"):
                dfn = k - 1
                dfd = k * (n_per_group - 1)
                ncp = k * n_per_group * effect_size ** 2
                crit = stats.f.ppf(1 - alpha, dfn, dfd)
                power = 1 - stats.f.cdf(crit, dfn, dfd, ncp)
                st.metric("Achievable Power", f"{power:.4f}")


def _render_multiple_comparisons(df: pd.DataFrame):
    """Multiple comparison corrections."""
    st.markdown("Enter p-values from multiple tests to apply corrections.")
    p_input = st.text_area("P-values (one per line or comma-separated):",
                           "0.01, 0.04, 0.03, 0.005, 0.08, 0.12, 0.001",
                           key="mc_pvals")
    alpha = st.slider("α:", 0.001, 0.10, 0.05, 0.001, key="mc_alpha")

    if st.button("Apply Corrections", key="apply_mc"):
        try:
            # Parse p-values
            p_vals = []
            for val in p_input.replace(",", " ").split():
                p_vals.append(float(val.strip()))
            p_vals = np.array(p_vals)
        except ValueError:
            st.error("Invalid p-value format.")
            return

        m = len(p_vals)

        # Bonferroni
        bonf = np.minimum(p_vals * m, 1.0)

        # Holm-Bonferroni
        sorted_idx = np.argsort(p_vals)
        holm = np.zeros(m)
        for i, idx in enumerate(sorted_idx):
            holm[idx] = min(p_vals[idx] * (m - i), 1.0)
        # Enforce monotonicity
        holm_sorted = holm[sorted_idx]
        for i in range(1, m):
            holm_sorted[i] = max(holm_sorted[i], holm_sorted[i - 1])
        holm[sorted_idx] = holm_sorted

        # Benjamini-Hochberg
        bh = np.zeros(m)
        sorted_p = p_vals[sorted_idx]
        for i in range(m):
            bh[i] = sorted_p[i] * m / (i + 1)
        # Enforce monotonicity from bottom
        for i in range(m - 2, -1, -1):
            bh[i] = min(bh[i], bh[i + 1])
        bh = np.minimum(bh, 1.0)
        bh_result = np.zeros(m)
        bh_result[sorted_idx] = bh

        results_df = pd.DataFrame({
            "Test #": np.arange(1, m + 1),
            "Original p": p_vals.round(6),
            "Bonferroni": bonf.round(6),
            "Holm": holm.round(6),
            "BH (FDR)": bh_result.round(6),
            f"Sig (α={alpha})": p_vals < alpha,
            "Sig (Bonf)": bonf < alpha,
            "Sig (Holm)": holm < alpha,
            "Sig (BH)": bh_result < alpha,
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Summary
        st.write(f"**Original:** {(p_vals < alpha).sum()}/{m} significant")
        st.write(f"**Bonferroni:** {(bonf < alpha).sum()}/{m} significant")
        st.write(f"**Holm:** {(holm < alpha).sum()}/{m} significant")
        st.write(f"**BH (FDR):** {(bh_result < alpha).sum()}/{m} significant")


def _render_bootstrap_permutation(df: pd.DataFrame):
    """Bootstrap confidence intervals and permutation tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        st.warning("No numeric columns available.")
        return

    analysis = st.selectbox("Analysis:", [
        "Bootstrap Confidence Interval",
        "Permutation Test (Two-Sample)",
    ], key="bp_analysis")

    if analysis == "Bootstrap Confidence Interval":
        col_name = st.selectbox("Column:", num_cols, key="bp_boot_col")
        statistic_name = st.selectbox("Statistic:", [
            "Mean", "Median", "Standard Deviation", "Variance",
            "Trimmed Mean (10%)", "IQR",
        ], key="bp_stat")
        ci_method = st.selectbox("CI Method:", ["percentile", "BCa"], key="bp_ci_method")
        n_resamples = st.slider("Number of bootstrap resamples:", 1000, 20000, 9999, 1000, key="bp_n_boot")
        alpha = st.slider("Confidence level:", 0.80, 0.99, 0.95, 0.01, key="bp_ci_level")

        if st.button("Run Bootstrap", key="run_bootstrap"):
            data = df[col_name].dropna().values
            n = len(data)

            stat_funcs = {
                "Mean": np.mean,
                "Median": np.median,
                "Standard Deviation": np.std,
                "Variance": np.var,
                "Trimmed Mean (10%)": lambda x, axis=None: stats.trim_mean(x, 0.1) if axis is None else np.apply_along_axis(lambda a: stats.trim_mean(a, 0.1), axis, x),
                "IQR": lambda x, axis=None: stats.iqr(x) if axis is None else np.apply_along_axis(stats.iqr, axis, x),
            }
            stat_func = stat_funcs[statistic_name]
            observed = stat_func(data)

            try:
                result = stats.bootstrap(
                    (data,), stat_func, n_resamples=n_resamples,
                    confidence_level=alpha, method=ci_method,
                    random_state=42,
                )
                ci_low, ci_high = result.confidence_interval.low, result.confidence_interval.high
                se = result.standard_error

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"Observed {statistic_name}", f"{observed:.4f}")
                c2.metric("Bootstrap SE", f"{se:.4f}")
                c3.metric(f"CI Lower ({alpha:.0%})", f"{ci_low:.4f}")
                c4.metric(f"CI Upper ({alpha:.0%})", f"{ci_high:.4f}")

                # Bootstrap distribution
                boot_dist = result.bootstrap_distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=boot_dist, nbinsx=60, name="Bootstrap Distribution",
                                           marker_color="steelblue", opacity=0.7))
                fig.add_vline(x=observed, line_dash="solid", line_color="red",
                              annotation_text=f"Observed={observed:.4f}")
                fig.add_vline(x=ci_low, line_dash="dash", line_color="green",
                              annotation_text=f"CI Low={ci_low:.4f}")
                fig.add_vline(x=ci_high, line_dash="dash", line_color="green",
                              annotation_text=f"CI High={ci_high:.4f}")
                fig.update_layout(title=f"Bootstrap Distribution of {statistic_name} (n={n_resamples})",
                                  xaxis_title=statistic_name, yaxis_title="Frequency", height=450)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Bootstrap failed: {e}")

    elif analysis == "Permutation Test (Two-Sample)":
        if not cat_cols:
            st.warning("Need a categorical grouping column.")
            return

        value_col = st.selectbox("Value column:", num_cols, key="bp_perm_val")
        group_col = st.selectbox("Group column:", cat_cols, key="bp_perm_grp")
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            st.warning("Need at least 2 groups.")
            return

        g1_name = st.selectbox("Group 1:", groups, index=0, key="bp_perm_g1")
        g2_name = st.selectbox("Group 2:", groups, index=min(1, len(groups) - 1), key="bp_perm_g2")

        test_stat = st.selectbox("Test statistic:", [
            "Difference of Means", "Difference of Medians",
        ], key="bp_perm_stat")
        n_resamples = st.slider("Number of permutations:", 1000, 20000, 9999, 1000, key="bp_n_perm")
        alt = st.selectbox("Alternative:", ["two-sided", "less", "greater"], key="bp_perm_alt")

        if st.button("Run Permutation Test", key="run_perm"):
            g1 = df[df[group_col] == g1_name][value_col].dropna().values
            g2 = df[df[group_col] == g2_name][value_col].dropna().values

            if test_stat == "Difference of Means":
                def stat_func(x, y, axis):
                    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
            else:
                def stat_func(x, y, axis):
                    return np.median(x, axis=axis) - np.median(y, axis=axis)

            try:
                result = stats.permutation_test(
                    (g1, g2), stat_func, n_resamples=n_resamples,
                    alternative=alt, random_state=42,
                )
                observed_stat = result.statistic
                p_val = result.pvalue

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Observed Statistic", f"{observed_stat:.4f}")
                c2.metric("p-value", f"{p_val:.6f}")
                c3.metric(f"Group 1 Mean ({g1_name})", f"{g1.mean():.4f}")
                c4.metric(f"Group 2 Mean ({g2_name})", f"{g2.mean():.4f}")

                st.markdown(_interpret_p(p_val, 0.05))

                # Null distribution
                null_dist = result.null_distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=null_dist, nbinsx=60,
                                           name="Null Distribution", marker_color="steelblue", opacity=0.7))
                fig.add_vline(x=observed_stat, line_dash="solid", line_color="red",
                              annotation_text=f"Observed={observed_stat:.4f}")
                fig.update_layout(
                    title=f"Permutation Null Distribution (n={n_resamples})",
                    xaxis_title=test_stat, yaxis_title="Frequency", height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Permutation test failed: {e}")
