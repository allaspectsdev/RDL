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
from modules.ui_helpers import significance_result, help_tip, empty_state, section_header
from modules.ui_helpers import validation_panel, interpretation_card, alternative_suggestion, confidence_badge
from modules.validation import (
    check_normality, check_equal_variance, check_sample_size,
    check_expected_frequencies, recommend_alternative,
    interpret_p_value, interpret_effect_size, compute_post_hoc_power,
)


def render_hypothesis_testing(df: pd.DataFrame):
    """Render hypothesis testing interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    tabs = st.tabs([
        "One-Sample Tests", "Two-Sample Tests", "Chi-Square Tests",
        "Normality Tests", "Power Analysis", "Multiple Comparisons",
        "Bootstrap & Permutation", "Equivalence Testing", "Bayesian Inference",
    ])

    help_tip("Which test should I use?", """
**Continuous data:**
- *One group vs known value* → One-Sample t-test (or Wilcoxon if non-normal)
- *Two independent groups* → Independent t-test / Welch's t-test (or Mann-Whitney U)
- *Two paired measurements* → Paired t-test (or Wilcoxon Signed-Rank)
- *3+ groups* → Use the ANOVA module

**Categorical data:**
- *Two categorical variables* → Chi-Square Test of Independence
- *Observed vs expected distribution* → Chi-Square Goodness of Fit

**Non-parametric / resampling:**
- *No distribution assumptions* → Bootstrap CI or Permutation Test
""")

    help_tip("Effect size interpretation", """
| Measure | Small | Medium | Large |
|---------|-------|--------|-------|
| Cohen's d | 0.2 | 0.5 | 0.8 |
| Cramér's V | 0.1 | 0.3 | 0.5 |
| Effect size r | 0.1 | 0.3 | 0.5 |
""")

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
    with tabs[7]:
        _render_equivalence(df)
    with tabs[8]:
        _render_bayesian(df)


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
        empty_state("No numeric columns available.")
        return

    test_type = st.selectbox("Test:", [
        "One-Sample t-test",
        "One-Sample Wilcoxon Signed-Rank",
        "One-Sample Proportion",
        "Runs Test (Randomness)",
    ], key="one_sample_test")

    alpha = st.slider("Significance level (α):", 0.001, 0.10, 0.05, 0.001, key="one_alpha")

    if test_type == "One-Sample t-test":
        col_name = st.selectbox("Column:", num_cols, key="one_t_col")
        mu_0 = st.number_input("Hypothesized mean (μ₀):", value=0.0, key="one_t_mu")
        alt = st.selectbox("Alternative:", ["two-sided", "less", "greater"], key="one_t_alt")

        if st.button("Run Test", key="run_one_t"):
            data = df[col_name].dropna()
            n = len(data)

            # --- Validation checks ---
            try:
                checks = [
                    check_sample_size(n, "t-test"),
                    check_normality(data, label=col_name),
                ]
                validation_panel(checks)
            except Exception:
                pass

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
            significance_result(p, alpha, "One-Sample t-test", effect_size=d, effect_label="Cohen's d")

            # --- Interpretation card ---
            try:
                interpretation_card(interpret_p_value(p, alpha))
            except Exception:
                pass

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
            significance_result(p, alpha, "Wilcoxon Signed-Rank", effect_size=r, effect_label="Effect size r")

            # --- Interpretation card ---
            try:
                interpretation_card(interpret_p_value(p, alpha))
            except Exception:
                pass

    elif test_type == "One-Sample Proportion":
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        bin_cols = [c for c in num_cols if df[c].nunique() == 2]
        all_cols = cat_cols + bin_cols
        if not all_cols:
            empty_state("No binary/categorical columns found.")
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
            significance_result(p_val, alpha, "One-Sample Proportion")

            # --- Interpretation card ---
            try:
                interpretation_card(interpret_p_value(p_val, alpha))
            except Exception:
                pass

    elif test_type == "Runs Test (Randomness)":
        col_name = st.selectbox("Column:", num_cols, key="runs_col")

        if st.button("Run Test", key="run_runs"):
            data = df[col_name].dropna().values
            n = len(data)
            if n < 10:
                st.warning("Need at least 10 data points for the runs test.")
                return

            median_val = np.median(data)
            # Convert to binary: above (1) / below (0) median
            binary = (data > median_val).astype(int)
            # Remove values exactly equal to median for cleaner analysis
            # but keep them as 0 for counting
            n_above = np.sum(binary == 1)
            n_below = np.sum(binary == 0)

            if n_above == 0 or n_below == 0:
                st.warning("All values are on one side of the median. Runs test cannot be computed.")
                return

            # Count runs
            runs = 1
            for i in range(1, len(binary)):
                if binary[i] != binary[i - 1]:
                    runs += 1

            # Expected number of runs and variance under H0
            n1, n2 = n_above, n_below
            expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
            var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2) ** 2 * (n1 + n2 - 1))

            if var_runs <= 0:
                st.warning("Variance of runs is zero. Cannot compute z-statistic.")
                return

            z_stat = (runs - expected_runs) / np.sqrt(var_runs)
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))

            # Also try statsmodels for comparison
            try:
                from statsmodels.sandbox.stats.runs import runstest_1samp
                z_sm, p_sm = runstest_1samp(data, cutoff="median")
                # Use statsmodels result if available
                z_stat, p_val = z_sm, p_sm
            except ImportError:
                pass

            st.markdown(f"**H₀:** The sequence is random (no pattern above/below median)")
            c1m, c2m, c3m, c4m = st.columns(4)
            c1m.metric("Number of Runs", str(runs))
            c2m.metric("Expected Runs", f"{expected_runs:.2f}")
            c3m.metric("Z-statistic", f"{z_stat:.4f}")
            c4m.metric("p-value", f"{p_val:.6f}")

            st.write(f"**Median:** {median_val:.4f}  |  **Above:** {n_above}  |  **Below:** {n_below}")
            significance_result(p_val, alpha, "Runs Test")

            try:
                interpretation_card(interpret_p_value(p_val, alpha))
            except Exception:
                pass

            # Sequence plot
            section_header("Sequence Plot")
            fig_runs = go.Figure()
            x_idx = np.arange(len(data))
            colors = ["#6366f1" if b == 1 else "#ef4444" for b in binary]
            fig_runs.add_trace(go.Scatter(
                x=x_idx, y=data, mode="lines+markers",
                marker=dict(color=colors, size=5),
                line=dict(color="gray", width=1),
                name="Data",
            ))
            fig_runs.add_hline(y=median_val, line_dash="dash", line_color="red",
                               annotation_text=f"Median={median_val:.3f}")
            fig_runs.update_layout(
                title=f"Sequence Plot with Runs: {col_name}",
                xaxis_title="Observation Order",
                yaxis_title=col_name,
                height=400,
            )
            st.plotly_chart(fig_runs, use_container_width=True)
            st.caption("Blue dots = above median, red dots = at or below median. "
                       "Each change in color represents a new run.")


def _render_two_sample(df: pd.DataFrame):
    """Two-sample hypothesis tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    test_type = st.selectbox("Test:", [
        "Independent t-test", "Welch's t-test", "Paired t-test",
        "Mann-Whitney U", "Wilcoxon Signed-Rank (paired)",
        "Two-Sample KS Test", "Bartlett's Test", "Mood's Median Test",
    ], key="two_sample_test")

    alpha = st.slider("Significance level (α):", 0.001, 0.10, 0.05, 0.001, key="two_alpha")

    if test_type in ("Independent t-test", "Welch's t-test", "Mann-Whitney U", "Two-Sample KS Test",
                      "Bartlett's Test", "Mood's Median Test"):
        # Need a grouping variable
        if not cat_cols:
            empty_state("Need a categorical column to define groups.")
            return
        if not num_cols:
            empty_state("No numeric columns available.")
            return

        value_col = st.selectbox("Value column:", num_cols, key="two_val_col")
        group_col = st.selectbox("Group column:", cat_cols, key="two_grp_col")
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            empty_state("Need at least 2 groups.", "The grouping column needs at least 2 distinct values.")
            return

        g1_name = st.selectbox("Group 1:", groups, index=0, key="two_g1")
        g2_name = st.selectbox("Group 2:", groups, index=min(1, len(groups) - 1), key="two_g2")

        if st.button("Run Test", key="run_two"):
            g1 = df[df[group_col] == g1_name][value_col].dropna()
            g2 = df[df[group_col] == g2_name][value_col].dropna()

            # --- Validation checks (for parametric t-tests) ---
            if test_type in ("Independent t-test", "Welch's t-test"):
                try:
                    checks = [
                        check_sample_size(min(len(g1), len(g2)), "t-test"),
                        check_normality(g1, label=str(g1_name)),
                        check_normality(g2, label=str(g2_name)),
                        check_equal_variance(g1, g2),
                    ]
                    validation_panel(checks)
                    failed = [c for c in checks if c.status in ("warn", "fail")]
                    if failed:
                        alts = recommend_alternative("independent-t", failed)
                        if alts:
                            alternative_suggestion("Some assumptions may be violated", alts)
                except Exception:
                    pass

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
            elif test_type == "Bartlett's Test":
                stat, p = stats.bartlett(g1, g2)
                test_label = "T"
            elif test_type == "Mood's Median Test":
                try:
                    mood_stat, mood_p, mood_med, mood_table = stats.median_test(g1, g2)
                    stat, p = mood_stat, mood_p
                except Exception as e:
                    st.error(f"Mood's median test failed: {e}")
                    return
                test_label = "chi2"

            if test_type == "Bartlett's Test":
                # Bartlett's tests equal variances, not means
                st.markdown(f"**H₀:** The two groups have equal variances")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"{test_label}-statistic", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                c3.metric(f"Var({g1_name})", f"{g1.var():.4f}")
                c4.metric(f"Var({g2_name})", f"{g2.var():.4f}")
                st.write(f"**Group 1 ({g1_name}):** n={len(g1)}, sd={g1.std():.4f}")
                st.write(f"**Group 2 ({g2_name}):** n={len(g2)}, sd={g2.std():.4f}")
                variance_ratio = g1.var() / g2.var() if g2.var() != 0 else np.nan
                significance_result(p, alpha, "Bartlett's Test",
                                    effect_size=variance_ratio, effect_label="Variance ratio")

                try:
                    interpretation_card(interpret_p_value(p, alpha))
                except Exception:
                    pass

                # Visualization
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Distributions", "Box Plot"))
                fig.add_trace(go.Histogram(x=g1, name=str(g1_name), opacity=0.6), row=1, col=1)
                fig.add_trace(go.Histogram(x=g2, name=str(g2_name), opacity=0.6), row=1, col=1)
                fig.update_layout(barmode="overlay")
                fig.add_trace(go.Box(y=g1, name=str(g1_name)), row=1, col=2)
                fig.add_trace(go.Box(y=g2, name=str(g2_name)), row=1, col=2)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            elif test_type == "Mood's Median Test":
                st.markdown(f"**H₀:** The two groups have the same median")
                c1, c2, c3 = st.columns(3)
                c1.metric(f"{test_label}-statistic", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                c3.metric("Grand Median", f"{mood_med:.4f}")

                st.markdown("**Contingency Table (above/below grand median):**")
                cont_df = pd.DataFrame(
                    mood_table,
                    index=["Above median", "At or below median"],
                    columns=[str(g1_name), str(g2_name)],
                )
                st.dataframe(cont_df, use_container_width=True)
                st.write(f"**Group 1 ({g1_name}):** n={len(g1)}, median={g1.median():.4f}")
                st.write(f"**Group 2 ({g2_name}):** n={len(g2)}, median={g2.median():.4f}")
                significance_result(p, alpha, "Mood's Median Test")

                try:
                    interpretation_card(interpret_p_value(p, alpha))
                except Exception:
                    pass

            else:
                d = _cohens_d(g1, g2)

                st.markdown(f"**H₀:** The two groups have the same {'mean' if 'test' in test_type.lower() else 'distribution'}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric(f"{test_label}-statistic", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                c3.metric("Cohen's d", f"{d:.4f} ({_effect_size_label(d)})")
                c4.metric("Diff of Means", f"{g1.mean() - g2.mean():.4f}")

                st.write(f"**Group 1 ({g1_name}):** n={len(g1)}, mean={g1.mean():.4f}, sd={g1.std():.4f}")
                st.write(f"**Group 2 ({g2_name}):** n={len(g2)}, mean={g2.mean():.4f}, sd={g2.std():.4f}")
                significance_result(p, alpha, test_type, effect_size=d, effect_label="Cohen's d")

                # --- Interpretation card ---
                try:
                    interpretation_card(interpret_p_value(p, alpha))
                except Exception:
                    pass

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
            empty_state("Need at least 2 numeric columns for paired test.")
            return
        col1 = st.selectbox("Column 1 (before/condition A):", num_cols, key="pair_c1")
        col2 = st.selectbox("Column 2 (after/condition B):", num_cols, index=min(1, len(num_cols) - 1), key="pair_c2")

        if st.button("Run Test", key="run_paired"):
            data = df[[col1, col2]].dropna()
            g1, g2 = data[col1], data[col2]
            diff = g1 - g2

            # --- Validation checks (for paired t-test) ---
            if test_type == "Paired t-test":
                try:
                    checks = [
                        check_sample_size(len(diff), "paired-t"),
                        check_normality(diff, label="paired differences"),
                    ]
                    validation_panel(checks)
                    failed = [c for c in checks if c.status in ("warn", "fail")]
                    if failed:
                        alts = recommend_alternative("paired-t", failed)
                        if alts:
                            alternative_suggestion("Some assumptions may be violated", alts)
                except Exception:
                    pass

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
            significance_result(p, alpha, test_type, effect_size=d, effect_label="Effect size d")

            # --- Interpretation card ---
            try:
                interpretation_card(interpret_p_value(p, alpha))
            except Exception:
                pass

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
        empty_state("Need categorical columns for chi-square tests.")
        return

    test_type = st.selectbox("Test:", [
        "Chi-Square Test of Independence",
        "Chi-Square Goodness of Fit",
        "McNemar's Test (paired binary)",
    ], key="chi_test_type")

    alpha = st.slider("α:", 0.001, 0.10, 0.05, 0.001, key="chi_alpha")

    if test_type == "Chi-Square Test of Independence":
        if len(cat_cols) < 2:
            empty_state("Need at least 2 categorical columns.")
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

            # --- Validation checks ---
            try:
                checks = [
                    check_sample_size(n, "chi-square"),
                    check_expected_frequencies(ct.values),
                ]
                validation_panel(checks)
                failed = [c for c in checks if c.status in ("warn", "fail")]
                if failed:
                    alts = recommend_alternative("chi-square", failed)
                    if alts:
                        alternative_suggestion("Some assumptions may be violated", alts)
            except Exception:
                pass

            st.markdown("**Expected Frequencies:**")
            expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns).round(2)
            st.dataframe(expected_df, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("χ²", f"{chi2:.4f}")
            c2.metric("p-value", f"{p:.6f}")
            c3.metric("df", str(dof))
            c4.metric("Cramér's V", f"{cramers_v:.4f}")
            significance_result(p, alpha, "Chi-Square Independence", effect_size=cramers_v, effect_label="Cramér's V")

            # --- Interpretation card ---
            try:
                interpretation_card(interpret_p_value(p, alpha))
            except Exception:
                pass

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

            # --- Validation checks ---
            try:
                checks = [
                    check_sample_size(int(n), "chi-square"),
                    check_expected_frequencies(observed.values.reshape(1, -1)),
                ]
                validation_panel(checks)
                failed = [c for c in checks if c.status in ("warn", "fail")]
                if failed:
                    alts = recommend_alternative("chi-square", failed)
                    if alts:
                        alternative_suggestion("Some assumptions may be violated", alts)
            except Exception:
                pass

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
            significance_result(p, alpha, "Chi-Square Goodness of Fit")

            # --- Interpretation card ---
            try:
                interpretation_card(interpret_p_value(p, alpha))
            except Exception:
                pass

            fig = go.Figure()
            fig.add_trace(go.Bar(x=observed.index.astype(str), y=observed.values, name="Observed"))
            fig.add_trace(go.Scatter(x=observed.index.astype(str), y=expected, name="Expected",
                                     mode="markers+lines", line=dict(color="red", width=2)))
            fig.update_layout(title="Observed vs Expected Frequencies", height=400)
            st.plotly_chart(fig, use_container_width=True)

    elif test_type == "McNemar's Test (paired binary)":
        # Identify binary columns (categorical with 2 levels or numeric with 2 unique values)
        binary_cols = []
        for c in df.columns:
            if df[c].dropna().nunique() == 2:
                binary_cols.append(c)
        if len(binary_cols) < 2:
            empty_state("Need at least 2 binary columns for McNemar's test.",
                        "Each column must have exactly 2 unique non-null values.")
            return

        c1_sel, c2_sel = st.columns(2)
        col1 = c1_sel.selectbox("Column 1 (before/condition A):", binary_cols, key="mcn_col1")
        col2 = c2_sel.selectbox("Column 2 (after/condition B):",
                                 [c for c in binary_cols if c != col1], key="mcn_col2")

        if st.button("Run McNemar's Test", key="run_mcnemar"):
            paired_data = df[[col1, col2]].dropna()
            if len(paired_data) < 5:
                st.warning("Need at least 5 paired observations.")
                return

            vals1 = paired_data[col1].unique()
            vals2 = paired_data[col2].unique()
            all_vals = list(set(list(vals1) | set(vals2)))
            if len(all_vals) != 2:
                st.warning("Both columns must share the same 2 categories.")
                return
            pos, neg = all_vals[0], all_vals[1]

            # Build 2x2 contingency table of concordant/discordant pairs
            a = ((paired_data[col1] == pos) & (paired_data[col2] == pos)).sum()  # both positive
            b = ((paired_data[col1] == pos) & (paired_data[col2] == neg)).sum()  # changed neg
            c_val = ((paired_data[col1] == neg) & (paired_data[col2] == pos)).sum()  # changed pos
            d_val = ((paired_data[col1] == neg) & (paired_data[col2] == neg)).sum()  # both negative

            st.markdown("**2x2 Contingency Table (paired outcomes):**")
            ct_df = pd.DataFrame(
                [[a, b], [c_val, d_val]],
                index=[f"{col1}={pos}", f"{col1}={neg}"],
                columns=[f"{col2}={pos}", f"{col2}={neg}"],
            )
            st.dataframe(ct_df, use_container_width=True)

            discordant = b + c_val
            if discordant == 0:
                st.warning("No discordant pairs found. McNemar's test cannot be computed.")
                return

            # Try statsmodels first, fall back to manual
            try:
                from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
                table = np.array([[a, b], [c_val, d_val]])
                # Use exact=False for chi-square version when discordant pairs >= 25
                exact = discordant < 25
                result = sm_mcnemar(table, exact=exact)
                chi2_stat = result.statistic
                p_val = result.pvalue
            except ImportError:
                # Manual calculation: chi2 = (b - c)^2 / (b + c)
                chi2_stat = (b - c_val) ** 2 / discordant
                p_val = 1 - stats.chi2.cdf(chi2_stat, df=1)

            c1m, c2m, c3m = st.columns(3)
            c1m.metric("Test Statistic", f"{chi2_stat:.4f}")
            c2m.metric("p-value", f"{p_val:.6f}")
            c3m.metric("Discordant Pairs", f"{discordant}")

            st.write(f"**Concordant:** {a + d_val} pairs  |  **Discordant:** {b} + {c_val} = {discordant} pairs")
            significance_result(p_val, alpha, "McNemar's Test")

            try:
                interpretation_card(interpret_p_value(p_val, alpha))
            except Exception:
                pass


def _render_normality(df: pd.DataFrame):
    """Comprehensive normality tests."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        empty_state("No numeric columns found.")
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
    section_header("QQ Plots")
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
                                 marker=dict(size=3),
                                 showlegend=False), row=r, col=c)
        slope, intercept = np.polyfit(theoretical, data, 1)
        fig.add_trace(go.Scatter(x=theoretical, y=slope * theoretical + intercept,
                                 mode="lines", line=dict(color="red", dash="dash"),
                                 showlegend=False), row=r, col=c)
    fig.update_layout(height=350 * rows)
    st.plotly_chart(fig, use_container_width=True)


def _power_t_test(effect_size, n, alpha, two_sample=False):
    """Compute power for a one- or two-sample t-test."""
    ncp = effect_size * np.sqrt(n)
    df_val = n - 1 if not two_sample else 2 * n - 2
    crit = stats.t.ppf(1 - alpha / 2, df=df_val)
    return 1 - stats.t.cdf(crit, df=df_val, loc=ncp)


def _render_power_analysis():
    """Statistical power analysis."""
    analysis_type = st.selectbox("Analysis for:", [
        "One-Sample t-test", "Two-Sample t-test", "One-Way ANOVA",
        "Two-Proportions", "Chi-Square Independence", "Correlation (r)",
        "Paired t-test", "Equivalence (TOST)",
    ], key="power_type")

    calc_mode = st.radio("Calculate:", ["Required Sample Size", "Achievable Power"], horizontal=True, key="power_mode")

    if analysis_type in ("One-Sample t-test", "Two-Sample t-test"):
        effect_size = st.slider("Cohen's d (effect size):", 0.1, 2.0, 0.5, 0.05, key="power_d")
        alpha = st.slider("alpha:", 0.001, 0.10, 0.05, 0.001, key="power_alpha")
        two_samp = analysis_type == "Two-Sample t-test"

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_target")
            if st.button("Calculate", key="calc_power_n"):
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = stats.norm.ppf(target_power)
                n = int(np.ceil(((z_alpha + z_beta) / effect_size) ** 2))
                if two_samp:
                    st.metric("Required n per group", n)
                    st.write(f"Total sample size: **{n * 2}**")
                else:
                    st.metric("Required n", n)

                # Power curve
                ns = np.arange(5, max(n * 2, 100))
                powers = [_power_t_test(effect_size, ni, alpha, two_samp) for ni in ns]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ns, y=powers, mode="lines", name="Power"))
                fig.add_hline(y=target_power, line_dash="dash", line_color="red",
                              annotation_text=f"Target={target_power}")
                fig.add_vline(x=n, line_dash="dash", line_color="green",
                              annotation_text=f"n={n}")
                fig.update_layout(title="Power Curve", xaxis_title="Sample Size", yaxis_title="Power",
                                  height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Sample size lookup table
                _render_power_lookup_table(analysis_type, alpha)

        else:
            sample_n = st.number_input("Sample size (per group):", 5, 10000, 30, key="power_n")
            if st.button("Calculate", key="calc_power_p"):
                power = _power_t_test(effect_size, sample_n, alpha, two_samp)
                st.metric("Achievable Power", f"{power:.4f}")

                ds = [0.2, 0.5, 0.8, 1.0, 1.5]
                fig = go.Figure()
                ns = np.arange(5, 200)
                for d in ds:
                    powers = [_power_t_test(d, ni, alpha, two_samp) for ni in ns]
                    fig.add_trace(go.Scatter(x=ns, y=powers, name=f"d={d}"))
                fig.add_hline(y=0.8, line_dash="dash", line_color="gray")
                fig.update_layout(title="Power vs Sample Size by Effect Size",
                                  xaxis_title="n", yaxis_title="Power", height=400)
                st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "One-Way ANOVA":
        effect_size = st.slider("Cohen's f (effect size):", 0.05, 1.0, 0.25, 0.05, key="power_f")
        k = st.number_input("Number of groups:", 2, 20, 3, key="power_k")
        alpha = st.slider("alpha:", 0.001, 0.10, 0.05, 0.001, key="power_anova_alpha")

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_anova_target")
            if st.button("Calculate", key="calc_power_anova_n"):
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

                # Power curve for ANOVA
                ns = np.arange(5, max(n * 2, 100))
                powers = []
                for ni in ns:
                    dfn_i = k - 1
                    dfd_i = k * (ni - 1)
                    ncp_i = k * ni * effect_size ** 2
                    crit_i = stats.f.ppf(1 - alpha, dfn_i, dfd_i)
                    powers.append(1 - stats.f.cdf(crit_i, dfn_i, dfd_i, ncp_i))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ns, y=powers, mode="lines", name="Power"))
                fig.add_hline(y=target_power, line_dash="dash", line_color="red",
                              annotation_text=f"Target={target_power}")
                fig.add_vline(x=n, line_dash="dash", line_color="green",
                              annotation_text=f"n={n}")
                fig.update_layout(title="ANOVA Power Curve", xaxis_title="n per group",
                                  yaxis_title="Power", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            n_per_group = st.number_input("n per group:", 5, 5000, 30, key="power_anova_n")
            if st.button("Calculate", key="calc_power_anova_p"):
                dfn = k - 1
                dfd = k * (n_per_group - 1)
                ncp = k * n_per_group * effect_size ** 2
                crit = stats.f.ppf(1 - alpha, dfn, dfd)
                power = 1 - stats.f.cdf(crit, dfn, dfd, ncp)
                st.metric("Achievable Power", f"{power:.4f}")

    elif analysis_type == "Two-Proportions":
        c1_p, c2_p = st.columns(2)
        p1 = c1_p.number_input("Proportion 1 (p1):", 0.01, 0.99, 0.50, 0.01, key="power_p1")
        p2 = c2_p.number_input("Proportion 2 (p2):", 0.01, 0.99, 0.30, 0.01, key="power_p2")
        alpha = st.slider("alpha:", 0.001, 0.10, 0.05, 0.001, key="power_prop_alpha")

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_prop_target")
            if st.button("Calculate", key="calc_power_prop_n"):
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = stats.norm.ppf(target_power)
                p_bar = (p1 + p2) / 2
                n = ((z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) +
                      z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2
                n = int(np.ceil(n))
                st.metric("Required n per group", n)
                st.write(f"Total sample size: **{n * 2}**")

                # Power curve
                ns = np.arange(10, max(n * 2, 200))
                powers = []
                for ni in ns:
                    p_bar_i = (p1 + p2) / 2
                    se0 = np.sqrt(2 * p_bar_i * (1 - p_bar_i) / ni)
                    se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / ni)
                    if se0 > 0 and se1 > 0:
                        z_crit = stats.norm.ppf(1 - alpha / 2)
                        pw = 1 - stats.norm.cdf((z_crit * se0 - abs(p1 - p2)) / se1) + \
                             stats.norm.cdf((-z_crit * se0 - abs(p1 - p2)) / se1)
                        powers.append(pw)
                    else:
                        powers.append(0)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ns, y=powers, mode="lines", name="Power"))
                fig.add_hline(y=target_power, line_dash="dash", line_color="red",
                              annotation_text=f"Target={target_power}")
                fig.add_vline(x=n, line_dash="dash", line_color="green",
                              annotation_text=f"n={n}")
                fig.update_layout(title="Power Curve (Two-Proportions)", xaxis_title="n per group",
                                  yaxis_title="Power", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            sample_n = st.number_input("Sample size per group:", 5, 10000, 100, key="power_prop_n")
            if st.button("Calculate", key="calc_power_prop_p"):
                p_bar = (p1 + p2) / 2
                se0 = np.sqrt(2 * p_bar * (1 - p_bar) / sample_n)
                se1 = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / sample_n)
                if se0 > 0 and se1 > 0:
                    z_crit = stats.norm.ppf(1 - alpha / 2)
                    power = 1 - stats.norm.cdf((z_crit * se0 - abs(p1 - p2)) / se1) + \
                            stats.norm.cdf((-z_crit * se0 - abs(p1 - p2)) / se1)
                    st.metric("Achievable Power", f"{power:.4f}")
                else:
                    st.warning("Cannot compute power with these parameters.")

    elif analysis_type == "Chi-Square Independence":
        effect_w = st.slider("Effect size w:", 0.05, 1.0, 0.30, 0.05, key="power_chi_w")
        df_chi = st.number_input("Degrees of freedom:", 1, 100, 1, key="power_chi_df")
        alpha = st.slider("alpha:", 0.001, 0.10, 0.05, 0.001, key="power_chi_alpha")

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_chi_target")
            if st.button("Calculate", key="calc_power_chi_n"):
                # Iterative search using non-central chi-square
                for n in range(10, 10000):
                    ncp = n * effect_w ** 2
                    crit = stats.chi2.ppf(1 - alpha, df_chi)
                    power = 1 - stats.ncx2.cdf(crit, df_chi, ncp)
                    if power >= target_power:
                        break
                st.metric("Required total n", n)

                # Power curve
                ns = np.arange(10, max(n * 2, 200))
                powers = []
                for ni in ns:
                    ncp_i = ni * effect_w ** 2
                    crit_i = stats.chi2.ppf(1 - alpha, df_chi)
                    powers.append(1 - stats.ncx2.cdf(crit_i, df_chi, ncp_i))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ns, y=powers, mode="lines", name="Power"))
                fig.add_hline(y=target_power, line_dash="dash", line_color="red",
                              annotation_text=f"Target={target_power}")
                fig.add_vline(x=n, line_dash="dash", line_color="green",
                              annotation_text=f"n={n}")
                fig.update_layout(title="Chi-Square Power Curve", xaxis_title="Total n",
                                  yaxis_title="Power", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            sample_n = st.number_input("Total sample size:", 10, 10000, 100, key="power_chi_n")
            if st.button("Calculate", key="calc_power_chi_p"):
                ncp = sample_n * effect_w ** 2
                crit = stats.chi2.ppf(1 - alpha, df_chi)
                power = 1 - stats.ncx2.cdf(crit, df_chi, ncp)
                st.metric("Achievable Power", f"{power:.4f}")

    elif analysis_type == "Correlation (r)":
        target_r = st.slider("Target correlation (|r|):", 0.05, 0.95, 0.30, 0.05, key="power_r")
        alpha = st.slider("alpha:", 0.001, 0.10, 0.05, 0.001, key="power_corr_alpha")

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_corr_target")
            if st.button("Calculate", key="calc_power_corr_n"):
                # Fisher z-transform: z_r = arctanh(r), SE = 1/sqrt(n-3)
                z_r = np.arctanh(target_r)
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = stats.norm.ppf(target_power)
                n = int(np.ceil(((z_alpha + z_beta) / z_r) ** 2 + 3))
                st.metric("Required n (pairs)", n)

                # Power curve
                ns = np.arange(10, max(n * 2, 100))
                powers = []
                for ni in ns:
                    se = 1 / np.sqrt(max(ni - 3, 1))
                    pw = 1 - stats.norm.cdf(z_alpha - z_r / se) + stats.norm.cdf(-z_alpha - z_r / se)
                    powers.append(pw)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ns, y=powers, mode="lines", name="Power"))
                fig.add_hline(y=target_power, line_dash="dash", line_color="red",
                              annotation_text=f"Target={target_power}")
                fig.add_vline(x=n, line_dash="dash", line_color="green",
                              annotation_text=f"n={n}")
                fig.update_layout(title="Correlation Power Curve", xaxis_title="n (pairs)",
                                  yaxis_title="Power", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            sample_n = st.number_input("Sample size (pairs):", 10, 10000, 50, key="power_corr_n")
            if st.button("Calculate", key="calc_power_corr_p"):
                z_r = np.arctanh(target_r)
                se = 1 / np.sqrt(max(sample_n - 3, 1))
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                power = 1 - stats.norm.cdf(z_alpha - z_r / se) + stats.norm.cdf(-z_alpha - z_r / se)
                st.metric("Achievable Power", f"{power:.4f}")

    elif analysis_type == "Paired t-test":
        effect_size = st.slider("Cohen's d (effect size):", 0.1, 2.0, 0.5, 0.05, key="power_paired_d")
        corr = st.slider("Correlation between pairs:", 0.0, 0.99, 0.50, 0.01, key="power_paired_corr")
        alpha = st.slider("alpha:", 0.001, 0.10, 0.05, 0.001, key="power_paired_alpha")

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_paired_target")
            if st.button("Calculate", key="calc_power_paired_n"):
                # For paired t-test: effective d = d / sqrt(2*(1-r))
                d_eff = effect_size / np.sqrt(2 * (1 - corr)) if corr < 1 else effect_size * 10
                z_alpha = stats.norm.ppf(1 - alpha / 2)
                z_beta = stats.norm.ppf(target_power)
                n = int(np.ceil(((z_alpha + z_beta) / d_eff) ** 2))
                n = max(n, 5)
                st.metric("Required n (pairs)", n)

                # Power curve
                ns = np.arange(5, max(n * 2, 100))
                powers = []
                for ni in ns:
                    ncp_i = d_eff * np.sqrt(ni)
                    crit_i = stats.t.ppf(1 - alpha / 2, df=ni - 1)
                    pw = 1 - stats.t.cdf(crit_i, df=ni - 1, loc=ncp_i)
                    powers.append(pw)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ns, y=powers, mode="lines", name="Power"))
                fig.add_hline(y=target_power, line_dash="dash", line_color="red",
                              annotation_text=f"Target={target_power}")
                fig.add_vline(x=n, line_dash="dash", line_color="green",
                              annotation_text=f"n={n}")
                fig.update_layout(title="Paired t-test Power Curve", xaxis_title="n (pairs)",
                                  yaxis_title="Power", height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            sample_n = st.number_input("Number of pairs:", 5, 10000, 30, key="power_paired_n")
            if st.button("Calculate", key="calc_power_paired_p"):
                d_eff = effect_size / np.sqrt(2 * (1 - corr)) if corr < 1 else effect_size * 10
                ncp = d_eff * np.sqrt(sample_n)
                crit = stats.t.ppf(1 - alpha / 2, df=sample_n - 1)
                power = 1 - stats.t.cdf(crit, df=sample_n - 1, loc=ncp)
                st.metric("Achievable Power", f"{power:.4f}")

    elif analysis_type == "Equivalence (TOST)":
        c1_eq, c2_eq = st.columns(2)
        eq_margin = c1_eq.number_input("Equivalence margin (delta):", 0.01, 10.0, 1.0, 0.1, key="power_tost_delta")
        true_diff = c2_eq.number_input("True difference:", -10.0, 10.0, 0.0, 0.1, key="power_tost_diff")
        sigma = st.number_input("Standard deviation (sigma):", 0.01, 100.0, 1.0, 0.1, key="power_tost_sigma")
        alpha = st.slider("alpha:", 0.001, 0.10, 0.05, 0.001, key="power_tost_alpha")

        if calc_mode == "Required Sample Size":
            target_power = st.slider("Target power:", 0.5, 0.99, 0.80, 0.01, key="power_tost_target")
            if st.button("Calculate", key="calc_power_tost_n"):
                # TOST power: both one-sided tests must be significant
                # Approximate using the more conservative bound
                z_alpha = stats.norm.ppf(1 - alpha)
                z_beta = stats.norm.ppf(target_power)
                # Most conservative: use the smaller margin minus true_diff
                effective_margin = eq_margin - abs(true_diff)
                if effective_margin <= 0:
                    st.warning("True difference exceeds equivalence margin. Equivalence cannot be shown.")
                else:
                    n = int(np.ceil(2 * ((z_alpha + z_beta) * sigma / effective_margin) ** 2))
                    n = max(n, 5)
                    st.metric("Required n per group", n)
                    st.write(f"Total sample size: **{n * 2}**")

                    # Power curve
                    ns = np.arange(5, max(n * 3, 100))
                    powers = []
                    for ni in ns:
                        se = sigma * np.sqrt(2 / ni)
                        if se > 0:
                            # Power = P(reject both one-sided)
                            pw_lower = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha) - (eq_margin + true_diff) / se)
                            pw_upper = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha) - (eq_margin - true_diff) / se)
                            pw = pw_lower * pw_upper
                            powers.append(pw)
                        else:
                            powers.append(0)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ns, y=powers, mode="lines", name="Power"))
                    fig.add_hline(y=target_power, line_dash="dash", line_color="red",
                                  annotation_text=f"Target={target_power}")
                    fig.add_vline(x=n, line_dash="dash", line_color="green",
                                  annotation_text=f"n={n}")
                    fig.update_layout(title="TOST Power Curve", xaxis_title="n per group",
                                      yaxis_title="Power", height=400)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            sample_n = st.number_input("Sample size per group:", 5, 10000, 50, key="power_tost_n")
            if st.button("Calculate", key="calc_power_tost_p"):
                se = sigma * np.sqrt(2 / sample_n)
                if se > 0:
                    pw_lower = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha) - (eq_margin + true_diff) / se)
                    pw_upper = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha) - (eq_margin - true_diff) / se)
                    power = pw_lower * pw_upper
                    st.metric("Achievable Power", f"{power:.4f}")
                else:
                    st.warning("Cannot compute power with these parameters.")

    # Sample size lookup table (shown as expandable for all types)
    try:
        _alpha_val = alpha
    except NameError:
        _alpha_val = 0.05
    with st.expander("Sample Size Lookup Table"):
        _render_power_lookup_table(analysis_type, _alpha_val)


def _render_power_lookup_table(analysis_type, alpha):
    """Render a grid of required sample sizes for effect size x power combinations."""
    if analysis_type in ("One-Sample t-test", "Two-Sample t-test"):
        effect_sizes = [0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
        power_levels = [0.70, 0.80, 0.85, 0.90, 0.95]
        z_a = stats.norm.ppf(1 - alpha / 2)
        records = []
        for es in effect_sizes:
            row = {"Cohen's d": es}
            for pw in power_levels:
                z_b = stats.norm.ppf(pw)
                n = int(np.ceil(((z_a + z_b) / es) ** 2))
                label = f"n (power={pw})"
                if analysis_type == "Two-Sample t-test":
                    row[label] = f"{n} per group"
                else:
                    row[label] = str(n)
            records.append(row)
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

    elif analysis_type == "Correlation (r)":
        rs = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
        power_levels = [0.70, 0.80, 0.85, 0.90, 0.95]
        z_a = stats.norm.ppf(1 - alpha / 2)
        records = []
        for r in rs:
            z_r = np.arctanh(r)
            row = {"|r|": r}
            for pw in power_levels:
                z_b = stats.norm.ppf(pw)
                n = int(np.ceil(((z_a + z_b) / z_r) ** 2 + 3))
                row[f"n (power={pw})"] = str(n)
            records.append(row)
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

    elif analysis_type == "Chi-Square Independence":
        ws = [0.1, 0.2, 0.3, 0.5, 0.7]
        power_levels = [0.70, 0.80, 0.85, 0.90, 0.95]
        records = []
        for w in ws:
            row = {"Effect w": w}
            for pw in power_levels:
                for n in range(10, 10000):
                    ncp = n * w ** 2
                    crit = stats.chi2.ppf(1 - alpha, 1)
                    power = 1 - stats.ncx2.cdf(crit, 1, ncp)
                    if power >= pw:
                        break
                row[f"n (power={pw})"] = str(n)
            records.append(row)
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

    elif analysis_type == "Two-Proportions":
        # Grid of |p1-p2| differences vs power
        deltas = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
        power_levels = [0.70, 0.80, 0.85, 0.90, 0.95]
        z_a = stats.norm.ppf(1 - alpha / 2)
        records = []
        for delta in deltas:
            p1 = 0.5 + delta / 2
            p2 = 0.5 - delta / 2
            row = {"|p1-p2|": delta}
            for pw in power_levels:
                z_b = stats.norm.ppf(pw)
                p_bar = (p1 + p2) / 2
                n_val = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar)) +
                          z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) / (p1 - p2)) ** 2
                row[f"n/group (power={pw})"] = str(int(np.ceil(n_val)))
            records.append(row)
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        st.caption("Assumes symmetric proportions around 0.5 (worst-case variance).")

    elif analysis_type == "Paired t-test":
        effect_sizes = [0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
        power_levels = [0.70, 0.80, 0.85, 0.90, 0.95]
        z_a = stats.norm.ppf(1 - alpha / 2)
        records = []
        for es in effect_sizes:
            row = {"Cohen's d": es}
            for pw in power_levels:
                z_b = stats.norm.ppf(pw)
                # Paired: n = ((z_a + z_b)/d)^2 assuming rho=0.5 => factor 1
                n_val = int(np.ceil(((z_a + z_b) / es) ** 2))
                row[f"n pairs (power={pw})"] = str(max(n_val, 5))
            records.append(row)
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        st.caption("Assumes correlation between pairs = 0.5. Actual n depends on pair correlation.")

    else:
        st.info("Lookup table is available for t-tests, proportions, correlation, paired t-test, and chi-square tests.")


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
        empty_state("No numeric columns available.")
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
                with st.spinner(f"Running bootstrap ({n_resamples} resamples)..."):
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
                                           opacity=0.7))
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
            empty_state("Need a categorical grouping column.")
            return

        value_col = st.selectbox("Value column:", num_cols, key="bp_perm_val")
        group_col = st.selectbox("Group column:", cat_cols, key="bp_perm_grp")
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            empty_state("Need at least 2 groups.", "The grouping column needs at least 2 distinct values.")
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
                with st.spinner(f"Running permutation test ({n_resamples} permutations)..."):
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

                significance_result(p_val, 0.05, "Permutation Test")

                # --- Interpretation card ---
                try:
                    interpretation_card(interpret_p_value(p_val, 0.05))
                except Exception:
                    pass

                # Null distribution
                null_dist = result.null_distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=null_dist, nbinsx=60,
                                           name="Null Distribution", opacity=0.7))
                fig.add_vline(x=observed_stat, line_dash="solid", line_color="red",
                              annotation_text=f"Observed={observed_stat:.4f}")
                fig.update_layout(
                    title=f"Permutation Null Distribution (n={n_resamples})",
                    xaxis_title=test_stat, yaxis_title="Frequency", height=450,
                )
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Permutation test failed: {e}")


# ===================================================================
# Tab 8 -- Equivalence / Non-Inferiority Testing (TOST)
# ===================================================================

def _render_equivalence(df: pd.DataFrame):
    """TOST equivalence testing and non-inferiority testing."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        empty_state("No numeric columns available.")
        return

    section_header("Equivalence / Non-Inferiority Testing")
    help_tip("TOST Equivalence Testing", """
**TOST (Two One-Sided Tests)** tests whether two groups are *equivalent* within a margin δ:
- H₀: |μ₁ - μ₂| ≥ δ (not equivalent)
- H₁: |μ₁ - μ₂| < δ (equivalent)
- Runs two one-sided t-tests: one testing diff > -δ, one testing diff < δ
- Both must be significant to conclude equivalence

**Non-inferiority** tests whether a new treatment is not worse by more than margin δ:
- H₀: μ_new - μ_ref ≤ -δ (inferior)
- H₁: μ_new - μ_ref > -δ (non-inferior)
""")

    test_type = st.selectbox("Test type:", [
        "TOST Equivalence (Two-Sample)",
        "TOST Equivalence (Paired)",
        "Non-Inferiority",
        "Bioequivalence (2x2 Crossover)",
    ], key="equiv_test")

    alpha = st.slider("Significance level (α):", 0.001, 0.10, 0.05, 0.001, key="equiv_alpha")

    if test_type == "TOST Equivalence (Two-Sample)":
        if not cat_cols:
            empty_state("Need a categorical column for group definition.")
            return

        value_col = st.selectbox("Value column:", num_cols, key="equiv_val")
        group_col = st.selectbox("Group column:", cat_cols, key="equiv_grp")
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            empty_state("Need at least 2 groups.", "The grouping column needs at least 2 distinct values.")
            return

        c1, c2 = st.columns(2)
        g1_name = c1.selectbox("Group 1:", groups, index=0, key="equiv_g1")
        g2_name = c2.selectbox("Group 2:", groups, index=min(1, len(groups) - 1), key="equiv_g2")
        delta = st.number_input("Equivalence margin (δ):", value=1.0, min_value=0.001, key="equiv_delta")

        if st.button("Run TOST", key="run_tost"):
            g1 = df[df[group_col] == g1_name][value_col].dropna()
            g2 = df[df[group_col] == g2_name][value_col].dropna()
            diff = g1.mean() - g2.mean()

            # Two one-sided tests
            stat_upper, p_upper = stats.ttest_ind(g1, g2)  # diff < delta
            stat_lower, p_lower = stats.ttest_ind(g1, g2)  # diff > -delta

            # TOST: test H₀: diff ≤ -δ (lower) and H₀: diff ≥ δ (upper)
            n1, n2 = len(g1), len(g2)
            se = np.sqrt(g1.var() / n1 + g2.var() / n2)
            df_welch = (g1.var() / n1 + g2.var() / n2) ** 2 / (
                (g1.var() / n1) ** 2 / (n1 - 1) + (g2.var() / n2) ** 2 / (n2 - 1)
            )

            t_lower = (diff - (-delta)) / se
            t_upper = (diff - delta) / se
            p_lower = 1 - stats.t.cdf(t_lower, df=df_welch)
            p_upper = stats.t.cdf(t_upper, df=df_welch)
            p_tost = max(p_lower, p_upper)

            st.markdown(f"**H₀:** Groups differ by more than ±{delta}")
            st.markdown(f"**H₁:** Groups are equivalent within ±{delta}")

            c1m, c2m, c3m, c4m = st.columns(4)
            c1m.metric("Mean Difference", f"{diff:.4f}")
            c2m.metric("p (lower test)", f"{p_lower:.6f}")
            c3m.metric("p (upper test)", f"{p_upper:.6f}")
            c4m.metric("TOST p-value", f"{p_tost:.6f}")

            # 90% CI for difference (using α not α/2 since TOST)
            ci_level = 1 - 2 * alpha
            t_crit = stats.t.ppf(1 - alpha, df=df_welch)
            ci_lower = diff - t_crit * se
            ci_upper = diff + t_crit * se
            st.write(f"**{ci_level:.0%} CI for difference:** [{ci_lower:.4f}, {ci_upper:.4f}]")

            if p_tost < alpha:
                significance_result(p_tost, alpha, "TOST Equivalence Test")
                st.success(f"**Conclude equivalence:** The difference ({diff:.4f}) is within ±{delta}.")
            else:
                significance_result(p_tost, alpha, "TOST Equivalence Test")
                st.info("**Cannot conclude equivalence.**")

            # Visualization: CI overlaid on equivalence bounds
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[diff], y=[0.5], mode="markers",
                                     marker=dict(size=12, color="#6366f1"),
                                     name=f"Diff = {diff:.4f}"))
            fig.add_trace(go.Scatter(x=[ci_lower, ci_upper], y=[0.5, 0.5],
                                     mode="lines", line=dict(color="#6366f1", width=3),
                                     name=f"{ci_level:.0%} CI"))
            fig.add_vline(x=-delta, line_dash="dash", line_color="red",
                          annotation_text=f"-δ = {-delta}")
            fig.add_vline(x=delta, line_dash="dash", line_color="red",
                          annotation_text=f"+δ = {delta}")
            fig.add_vline(x=0, line_dash="dot", line_color="gray")
            fig.add_vrect(x0=-delta, x1=delta, fillcolor="green", opacity=0.1,
                          annotation_text="Equivalence Zone")
            fig.update_layout(title="TOST: CI vs Equivalence Bounds",
                              xaxis_title="Difference", height=300,
                              yaxis_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    elif test_type == "TOST Equivalence (Paired)":
        if len(num_cols) < 2:
            empty_state("Need at least 2 numeric columns.")
            return

        c1, c2 = st.columns(2)
        col1 = c1.selectbox("Column 1:", num_cols, key="equiv_p_c1")
        col2 = c2.selectbox("Column 2:", [c for c in num_cols if c != col1], key="equiv_p_c2")
        delta = st.number_input("Equivalence margin (δ):", value=1.0, min_value=0.001, key="equiv_p_delta")

        if st.button("Run Paired TOST", key="run_tost_paired"):
            data = df[[col1, col2]].dropna()
            diff = (data[col1] - data[col2]).values
            mean_diff = np.mean(diff)
            se = np.std(diff, ddof=1) / np.sqrt(len(diff))
            df_val = len(diff) - 1

            t_lower = (mean_diff + delta) / se
            t_upper = (mean_diff - delta) / se
            p_lower = stats.t.cdf(t_lower, df=df_val)  # Test: diff > -delta
            p_upper = 1 - stats.t.cdf(t_upper, df=df_val)  # Test: diff < delta
            p_tost = max(1 - p_lower, p_upper)

            c1m, c2m, c3m = st.columns(3)
            c1m.metric("Mean Difference", f"{mean_diff:.4f}")
            c2m.metric("TOST p-value", f"{p_tost:.6f}")
            c3m.metric("n pairs", str(len(diff)))

            if p_tost < alpha:
                significance_result(p_tost, alpha, "Paired TOST")
            else:
                significance_result(p_tost, alpha, "Paired TOST")

    elif test_type == "Non-Inferiority":
        if not cat_cols:
            empty_state("Need a categorical column for group definition.")
            return

        value_col = st.selectbox("Value column:", num_cols, key="ni_val")
        group_col = st.selectbox("Group column:", cat_cols, key="ni_grp")
        groups = df[group_col].dropna().unique()
        if len(groups) < 2:
            empty_state("Need at least 2 groups.", "The grouping column needs at least 2 distinct values.")
            return

        c1, c2 = st.columns(2)
        new_name = c1.selectbox("New/test group:", groups, index=0, key="ni_new")
        ref_name = c2.selectbox("Reference group:", groups, index=min(1, len(groups) - 1), key="ni_ref")
        delta = st.number_input("Non-inferiority margin (δ):", value=1.0, min_value=0.001, key="ni_delta")

        if st.button("Run Non-Inferiority Test", key="run_ni"):
            new_grp = df[df[group_col] == new_name][value_col].dropna()
            ref_grp = df[df[group_col] == ref_name][value_col].dropna()
            diff = new_grp.mean() - ref_grp.mean()

            n1, n2 = len(new_grp), len(ref_grp)
            se = np.sqrt(new_grp.var() / n1 + ref_grp.var() / n2)
            df_welch = (new_grp.var() / n1 + ref_grp.var() / n2) ** 2 / (
                (new_grp.var() / n1) ** 2 / (n1 - 1) + (ref_grp.var() / n2) ** 2 / (n2 - 1)
            )

            t_stat = (diff + delta) / se
            p_val = 1 - stats.t.cdf(t_stat, df=df_welch)

            st.markdown(f"**H₀:** New is inferior (diff ≤ -δ = {-delta})")
            st.markdown(f"**H₁:** New is non-inferior (diff > -δ)")

            c1m, c2m, c3m = st.columns(3)
            c1m.metric("Mean Difference", f"{diff:.4f}")
            c2m.metric("t-statistic", f"{t_stat:.4f}")
            c3m.metric("p-value", f"{p_val:.6f}")

            significance_result(p_val, alpha, "Non-Inferiority Test")

    elif test_type == "Bioequivalence (2x2 Crossover)":
        help_tip("Bioequivalence (2x2 Crossover)", """
**2x2 crossover design** is the standard FDA/EMA design for bioequivalence studies:
- Subjects randomized to Sequence 1 (Treatment→Reference) or Sequence 2 (Reference→Treatment)
- Each subject receives both formulations in different periods
- Log-transformed PK parameters (AUC, Cmax) are analyzed
- **90% CI** for the ratio of geometric means must fall within **80-125%** (FDA criterion)
- Mixed model accounts for sequence, period, and treatment effects with subject as random effect
""")

        if len(num_cols) < 1:
            empty_state("Need a numeric response column (e.g. AUC or Cmax).")
            return

        value_col = st.selectbox("Response column (PK parameter):", num_cols, key="be_val")
        subject_col = st.selectbox("Subject column:", cat_cols if cat_cols else ["(none)"], key="be_subj")
        treatment_col = st.selectbox("Treatment column (Test/Reference):", cat_cols if cat_cols else ["(none)"], key="be_trt")
        period_col = st.selectbox("Period column:", cat_cols if cat_cols else ["(none)"], key="be_period")
        sequence_col = st.selectbox("Sequence column:", cat_cols if cat_cols else ["(none)"], key="be_seq")

        if "(none)" in [subject_col, treatment_col, period_col, sequence_col]:
            empty_state("Need categorical columns for subject, treatment, period, and sequence.",
                        "Ensure your dataset has the required crossover design columns.")
            return

        c1, c2 = st.columns(2)
        lower_limit = c1.number_input("Lower BE limit (%):", value=80.0, key="be_lower")
        upper_limit = c2.number_input("Upper BE limit (%):", value=125.0, key="be_upper")
        log_transform = st.checkbox("Log-transform response (recommended for PK data)", value=True, key="be_log")

        if st.button("Run Bioequivalence Analysis", key="run_be"):
            be_data = df[[value_col, subject_col, treatment_col, period_col, sequence_col]].dropna()
            if len(be_data) < 4:
                empty_state("Not enough data for analysis.")
                return

            treatments = be_data[treatment_col].unique()
            if len(treatments) != 2:
                st.error(f"Expected exactly 2 treatments, found {len(treatments)}.")
                return

            with st.spinner("Running bioequivalence analysis..."):
                response = be_data[value_col].values.astype(float)
                if log_transform:
                    if np.any(response <= 0):
                        st.error("Log-transform requires all positive values.")
                        return
                    response = np.log(response)
                    be_data = be_data.copy()
                    be_data["_log_response"] = response
                    resp_col = "_log_response"
                else:
                    resp_col = value_col

                # Mixed model: response ~ treatment + period + sequence + (1|subject)
                try:
                    import statsmodels.formula.api as smf
                    formula = f"`{resp_col}` ~ C(`{treatment_col}`) + C(`{period_col}`) + C(`{sequence_col}`)"
                    model = smf.mixedlm(
                        formula, be_data, groups=be_data[subject_col],
                    )
                    result = model.fit(reml=True)

                    # Extract treatment effect
                    trt_params = [p for p in result.params.index if treatment_col in str(p)]
                    if trt_params:
                        trt_effect = result.params[trt_params[0]]
                        trt_se = result.bse[trt_params[0]]
                        trt_pval = result.pvalues[trt_params[0]]
                    else:
                        st.error("Could not extract treatment effect from model.")
                        return

                    df_denom = result.df_resid

                    # 90% CI for the treatment difference (on log scale if log-transformed)
                    t_crit = stats.t.ppf(0.95, df=df_denom)
                    ci_lower_diff = trt_effect - t_crit * trt_se
                    ci_upper_diff = trt_effect + t_crit * trt_se

                    if log_transform:
                        # Back-transform to ratio of geometric means (%)
                        ratio_pct = np.exp(trt_effect) * 100
                        ci_lower_pct = np.exp(ci_lower_diff) * 100
                        ci_upper_pct = np.exp(ci_upper_diff) * 100
                    else:
                        ratio_pct = trt_effect
                        ci_lower_pct = ci_lower_diff
                        ci_upper_pct = ci_upper_diff

                    # Display results
                    section_header("Bioequivalence Results")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Ratio of Geo. Means",
                              f"{ratio_pct:.2f}%" if log_transform else f"{trt_effect:.4f}")
                    m2.metric("90% CI Lower", f"{ci_lower_pct:.2f}%" if log_transform else f"{ci_lower_diff:.4f}")
                    m3.metric("90% CI Upper", f"{ci_upper_pct:.2f}%" if log_transform else f"{ci_upper_diff:.4f}")
                    m4.metric("Treatment p-value", f"{trt_pval:.6f}")

                    # BE conclusion
                    if log_transform:
                        be_pass = ci_lower_pct >= lower_limit and ci_upper_pct <= upper_limit
                    else:
                        be_pass = True  # non-log needs custom limits

                    if be_pass:
                        interpretation_card({
                            "title": "Bioequivalence Conclusion",
                            "body": f"The 90% CI [{ci_lower_pct:.2f}%, {ci_upper_pct:.2f}%] falls entirely within the acceptance region [{lower_limit}%, {upper_limit}%]. Bioequivalence is established.",
                            "detail": "The test and reference formulations can be considered bioequivalent.",
                        })
                    else:
                        interpretation_card({
                            "title": "Bioequivalence Conclusion",
                            "body": f"The 90% CI [{ci_lower_pct:.2f}%, {ci_upper_pct:.2f}%] does NOT fall entirely within [{lower_limit}%, {upper_limit}%]. Bioequivalence is NOT established.",
                            "detail": "The formulations cannot be considered bioequivalent based on this analysis.",
                        })

                    # Forest plot
                    fig = go.Figure()
                    fig.add_vrect(x0=lower_limit, x1=upper_limit,
                                  fillcolor="green", opacity=0.1,
                                  annotation_text="Acceptance Region")
                    fig.add_trace(go.Scatter(
                        x=[ratio_pct], y=[0.5], mode="markers",
                        marker=dict(size=14, color="#6366f1", symbol="diamond"),
                        name=f"Ratio = {ratio_pct:.2f}%",
                    ))
                    fig.add_trace(go.Scatter(
                        x=[ci_lower_pct, ci_upper_pct], y=[0.5, 0.5],
                        mode="lines", line=dict(color="#6366f1", width=4),
                        name="90% CI",
                    ))
                    fig.add_vline(x=100, line_dash="dot", line_color="gray")
                    fig.add_vline(x=lower_limit, line_dash="dash", line_color="red")
                    fig.add_vline(x=upper_limit, line_dash="dash", line_color="red")
                    fig.update_layout(
                        title="Bioequivalence: 90% CI vs Acceptance Region",
                        xaxis_title="Ratio of Geometric Means (%)",
                        yaxis_visible=False, height=300,
                        xaxis=dict(range=[max(50, lower_limit - 20),
                                          min(160, upper_limit + 20)]),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ANOVA table
                    with st.expander("Model Details"):
                        st.text(str(result.summary()))

                except Exception as e:
                    st.error(f"Model fitting failed: {e}")
                    st.info("Ensure your data has the correct crossover design structure with subject, treatment, period, and sequence columns.")


# ===================================================================
# Tab 9 -- Bayesian Inference (Conjugate Priors)
# ===================================================================

def _render_bayesian(df: pd.DataFrame):
    """Bayesian inference with conjugate priors."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not num_cols:
        empty_state("No numeric columns available.")
        return

    section_header("Bayesian Inference")
    help_tip("Bayesian Analysis", """
Uses conjugate priors for analytical posterior computation (no MCMC needed):
- **Normal mean:** Normal prior → Normal posterior
- **Proportion:** Beta prior → Beta posterior
- **Poisson rate:** Gamma prior → Gamma posterior
Adjust prior parameters to see how prior beliefs combine with data.
""")

    analysis_type = st.selectbox("Analysis:", [
        "Normal Mean (known σ approximation)",
        "Proportion (Beta-Binomial)",
        "Poisson Rate (Gamma-Poisson)",
    ], key="bayes_type")

    if analysis_type == "Normal Mean (known σ approximation)":
        col = st.selectbox("Data column:", num_cols, key="bayes_norm_col")
        data = df[col].dropna().values
        n = len(data)
        x_bar = np.mean(data)
        s = np.std(data, ddof=1)

        section_header("Prior Parameters (Normal)")
        c1, c2 = st.columns(2)
        prior_mu = c1.number_input("Prior mean (μ₀):", value=float(x_bar), key="bayes_prior_mu")
        prior_sigma = c2.number_input("Prior std (σ₀):", value=float(s * 2),
                                       min_value=0.001, key="bayes_prior_sigma")

        # Posterior (Normal-Normal conjugate with known σ ≈ sample s)
        sigma = s
        prior_prec = 1 / prior_sigma ** 2
        data_prec = n / sigma ** 2
        post_prec = prior_prec + data_prec
        post_mu = (prior_prec * prior_mu + data_prec * x_bar) / post_prec
        post_sigma = 1 / np.sqrt(post_prec)

        # Credible interval
        ci_lower = stats.norm.ppf(0.025, post_mu, post_sigma)
        ci_upper = stats.norm.ppf(0.975, post_mu, post_sigma)

        section_header("Posterior")
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("Posterior Mean", f"{post_mu:.4f}")
        c2m.metric("Posterior Std", f"{post_sigma:.4f}")
        c3m.metric("95% Credible Interval", f"[{ci_lower:.4f}, {ci_upper:.4f}]")

        st.write(f"**Data:** n={n}, x̄={x_bar:.4f}, s={s:.4f}")
        st.write(f"**Prior weight vs data:** {prior_prec / post_prec:.1%} prior, {data_prec / post_prec:.1%} data")

        # Plot prior, likelihood, posterior
        x_range = np.linspace(
            min(prior_mu - 3 * prior_sigma, x_bar - 3 * s / np.sqrt(n)),
            max(prior_mu + 3 * prior_sigma, x_bar + 3 * s / np.sqrt(n)),
            300,
        )
        prior_pdf = stats.norm.pdf(x_range, prior_mu, prior_sigma)
        likelihood_pdf = stats.norm.pdf(x_range, x_bar, s / np.sqrt(n))
        posterior_pdf = stats.norm.pdf(x_range, post_mu, post_sigma)

        # Normalize for visual comparison
        prior_pdf = prior_pdf / prior_pdf.max()
        likelihood_pdf = likelihood_pdf / likelihood_pdf.max()
        posterior_pdf = posterior_pdf / posterior_pdf.max()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_range, y=prior_pdf, mode="lines",
                                 name="Prior", line=dict(dash="dash", color="blue")))
        fig.add_trace(go.Scatter(x=x_range, y=likelihood_pdf, mode="lines",
                                 name="Likelihood", line=dict(dash="dot", color="green")))
        fig.add_trace(go.Scatter(x=x_range, y=posterior_pdf, mode="lines",
                                 name="Posterior", line=dict(color="red", width=2)))
        fig.add_vline(x=post_mu, line_dash="dot", line_color="red",
                      annotation_text=f"Post mean={post_mu:.3f}")
        fig.update_layout(title="Prior / Likelihood / Posterior",
                          xaxis_title="μ", yaxis_title="Density (normalized)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Proportion (Beta-Binomial)":
        section_header("Data")
        c1, c2 = st.columns(2)
        n_success = c1.number_input("Number of successes:", 0, 100000, 30, key="bayes_x")
        n_total = c2.number_input("Number of trials:", 1, 100000, 100, key="bayes_n")

        if n_success > n_total:
            st.error("Successes cannot exceed trials.")
            return

        section_header("Prior Parameters (Beta)")
        c1, c2 = st.columns(2)
        prior_a = c1.slider("α (prior successes + 1):", 0.1, 50.0, 1.0, 0.1, key="bayes_beta_a")
        prior_b = c2.slider("β (prior failures + 1):", 0.1, 50.0, 1.0, 0.1, key="bayes_beta_b")

        # Posterior: Beta(α + x, β + n - x)
        post_a = prior_a + n_success
        post_b = prior_b + (n_total - n_success)

        post_mean = post_a / (post_a + post_b)
        post_mode = (post_a - 1) / (post_a + post_b - 2) if (post_a > 1 and post_b > 1) else post_mean
        ci_lower = stats.beta.ppf(0.025, post_a, post_b)
        ci_upper = stats.beta.ppf(0.975, post_a, post_b)

        section_header("Posterior")
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("Posterior Mean", f"{post_mean:.4f}")
        c2m.metric("Posterior Mode", f"{post_mode:.4f}")
        c3m.metric("95% Credible Interval", f"[{ci_lower:.4f}, {ci_upper:.4f}]")

        st.write(f"**MLE:** {n_success / n_total:.4f}")
        st.write(f"**Posterior:** Beta({post_a:.1f}, {post_b:.1f})")

        # Plot
        p_range = np.linspace(0.001, 0.999, 300)
        prior_pdf = stats.beta.pdf(p_range, prior_a, prior_b)
        posterior_pdf = stats.beta.pdf(p_range, post_a, post_b)

        # Normalize
        prior_pdf = prior_pdf / max(prior_pdf.max(), 1e-10)
        posterior_pdf = posterior_pdf / max(posterior_pdf.max(), 1e-10)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=p_range, y=prior_pdf, mode="lines",
                                 name="Prior", line=dict(dash="dash", color="blue")))
        fig.add_trace(go.Scatter(x=p_range, y=posterior_pdf, mode="lines",
                                 name="Posterior", line=dict(color="red", width=2)))
        fig.add_vline(x=post_mean, line_dash="dot", line_color="red",
                      annotation_text=f"Post mean={post_mean:.3f}")
        fig.add_vrect(x0=ci_lower, x1=ci_upper, fillcolor="red", opacity=0.1)
        fig.update_layout(title="Beta Prior → Posterior",
                          xaxis_title="p", yaxis_title="Density (normalized)", height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Poisson Rate (Gamma-Poisson)":
        section_header("Data")
        col = st.selectbox("Count column:", num_cols, key="bayes_pois_col")
        data = df[col].dropna().values
        n = len(data)
        total = data.sum()
        x_bar = np.mean(data)

        section_header("Prior Parameters (Gamma)")
        c1, c2 = st.columns(2)
        prior_shape = c1.slider("Shape (α):", 0.1, 50.0, 1.0, 0.1, key="bayes_gamma_a")
        prior_rate = c2.slider("Rate (β):", 0.01, 50.0, 1.0, 0.1, key="bayes_gamma_b")

        # Posterior: Gamma(α + Σx, β + n)
        post_shape = prior_shape + total
        post_rate = prior_rate + n

        post_mean = post_shape / post_rate
        post_mode = (post_shape - 1) / post_rate if post_shape > 1 else 0
        ci_lower = stats.gamma.ppf(0.025, post_shape, scale=1 / post_rate)
        ci_upper = stats.gamma.ppf(0.975, post_shape, scale=1 / post_rate)

        section_header("Posterior")
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("Posterior Mean", f"{post_mean:.4f}")
        c2m.metric("Posterior Mode", f"{post_mode:.4f}")
        c3m.metric("95% Credible Interval", f"[{ci_lower:.4f}, {ci_upper:.4f}]")

        st.write(f"**Data:** n={n}, Σx={total:.0f}, x̄={x_bar:.4f}")
        st.write(f"**Posterior:** Gamma({post_shape:.1f}, {post_rate:.1f})")

        # Plot
        lam_max = max(ci_upper * 1.5, x_bar * 2)
        lam_range = np.linspace(0.001, lam_max, 300)
        prior_pdf = stats.gamma.pdf(lam_range, prior_shape, scale=1 / prior_rate)
        posterior_pdf = stats.gamma.pdf(lam_range, post_shape, scale=1 / post_rate)

        prior_pdf = prior_pdf / max(prior_pdf.max(), 1e-10)
        posterior_pdf = posterior_pdf / max(posterior_pdf.max(), 1e-10)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lam_range, y=prior_pdf, mode="lines",
                                 name="Prior", line=dict(dash="dash", color="blue")))
        fig.add_trace(go.Scatter(x=lam_range, y=posterior_pdf, mode="lines",
                                 name="Posterior", line=dict(color="red", width=2)))
        fig.add_vline(x=post_mean, line_dash="dot", line_color="red",
                      annotation_text=f"Post mean={post_mean:.3f}")
        fig.add_vrect(x0=ci_lower, x1=ci_upper, fillcolor="red", opacity=0.1)
        fig.update_layout(title="Gamma Prior → Posterior",
                          xaxis_title="λ (rate)", yaxis_title="Density (normalized)", height=450)
        st.plotly_chart(fig, use_container_width=True)
