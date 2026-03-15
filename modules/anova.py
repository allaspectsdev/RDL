"""
ANOVA Module - One-way, Two-way, Repeated Measures, Kruskal-Wallis, ANCOVA.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from modules.ui_helpers import (
    significance_result, help_tip, section_header, empty_state,
    validation_panel, interpretation_card, alternative_suggestion, confidence_badge,
)
from modules.validation import (
    check_normality, check_equal_variance, check_sample_size,
    recommend_alternative, interpret_p_value, interpret_effect_size,
)

try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from statsmodels.stats.anova import anova_lm
    HAS_SM = True
except ImportError:
    HAS_SM = False

try:
    import pingouin as pg
    HAS_PG = True
except ImportError:
    HAS_PG = False


def render_anova(df: pd.DataFrame):
    """Render ANOVA interface."""
    if df is None or df.empty:
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    tabs = st.tabs([
        "One-Way ANOVA", "Two-Way ANOVA", "Repeated Measures",
        "Kruskal-Wallis", "Friedman Test", "ANCOVA", "MANOVA",
    ])

    with tabs[0]:
        _render_one_way(df)
    with tabs[1]:
        _render_two_way(df)
    with tabs[2]:
        _render_repeated_measures(df)
    with tabs[3]:
        _render_kruskal_wallis(df)
    with tabs[4]:
        _render_friedman(df)
    with tabs[5]:
        _render_ancova(df)
    with tabs[6]:
        _render_manova(df)


def _get_cat_cols(df):
    cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].nunique() <= 15:
            cols.append(c)
    return cols


def _render_one_way(df: pd.DataFrame):
    """One-way ANOVA."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = _get_cat_cols(df)

    if not num_cols or not cat_cols:
        empty_state("Need numeric and categorical columns.")
        return

    dep_var = st.selectbox("Dependent variable (numeric):", num_cols, key="ow_dep")
    factor = st.selectbox("Factor (categorical):", cat_cols, key="ow_factor")
    alpha = st.slider("α:", 0.001, 0.10, 0.05, 0.001, key="ow_alpha")

    if st.button("Run ANOVA", key="run_ow"):
        data = df[[dep_var, factor]].dropna()
        all_groups = [(name, group[dep_var].values) for name, group in data.groupby(factor)]
        small = [name for name, g in all_groups if len(g) < 2]
        if small:
            st.warning(f"Groups with <2 observations excluded: {', '.join(str(s) for s in small)}")
        valid = [(name, g) for name, g in all_groups if len(g) >= 2]
        group_names = [name for name, g in valid]
        groups = [g for name, g in valid]
        k = len(groups)

        if k < 2:
            st.error("Need at least 2 valid groups (each with ≥2 observations).")
            return

        # ── Assumption checks (rendered BEFORE results) ──
        try:
            checks = []
            # Levene's test for equal variance
            variance_check = check_equal_variance(*groups)
            checks.append(variance_check)

            # Shapiro-Wilk normality per group
            for name, g in zip(group_names, groups):
                checks.append(check_normality(g, label=str(name)))

            # Sample size per group
            min_group_n = min(len(g) for g in groups)
            checks.append(check_sample_size(min_group_n, "anova"))

            validation_panel(checks)

            # Alternative suggestion if Levene's fails
            if variance_check.status in ("warn", "fail"):
                alternative_suggestion(
                    "Unequal variances detected",
                    ["Welch's ANOVA", "Kruskal-Wallis test"],
                )
                # Show Welch's ANOVA when available
                if HAS_PG:
                    welch = pg.welch_anova(data=data, dv=dep_var, between=factor)
                    with st.expander("Welch's ANOVA Result"):
                        st.dataframe(welch.round(4), use_container_width=True, hide_index=True)

            # Alternative suggestion if normality fails for any group
            norm_failed = [c for c in checks if "Normality" in c.name and c.status in ("warn", "fail")]
            if norm_failed:
                alternative_suggestion(
                    "Normality assumption violated for one or more groups",
                    ["Kruskal-Wallis test"],
                )
        except Exception:
            pass  # validation is advisory — don't block analysis

        # One-way ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Effect sizes
        grand_mean = data[dep_var].mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
        ss_within = sum(np.sum((g - g.mean()) ** 2) for g in groups)
        ss_total = ss_between + ss_within
        df_between = k - 1
        df_within = len(data) - k
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        eta_sq = ss_between / ss_total
        omega_sq = (ss_between - df_between * ms_within) / (ss_total + ms_within)

        # ANOVA Table
        anova_table = pd.DataFrame({
            "Source": ["Between Groups", "Within Groups", "Total"],
            "SS": [ss_between, ss_within, ss_total],
            "df": [df_between, df_within, df_between + df_within],
            "MS": [ms_between, ms_within, np.nan],
            "F": [f_stat, np.nan, np.nan],
            "p-value": [p_value, np.nan, np.nan],
        }).round(4)
        section_header("ANOVA Table")
        st.dataframe(anova_table, use_container_width=True, hide_index=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("F-statistic", f"{f_stat:.4f}")
        c2.metric("η² (eta-squared)", f"{eta_sq:.4f}")
        c3.metric("ω² (omega-squared)", f"{omega_sq:.4f}")

        significance_result(p_value, alpha, "One-Way ANOVA", effect_size=eta_sq, effect_label="η²")

        help_tip("Effect size interpretation", """
- **η² (eta-squared):** Proportion of total variance explained. Small = 0.01, Medium = 0.06, Large = 0.14
- **ω² (omega-squared):** Less biased estimate. Same thresholds apply.
""")

        # Interpretation cards
        try:
            interpretation_card(interpret_effect_size(eta_sq, "eta-squared"))
            interpretation_card(interpret_p_value(p_value, alpha))
        except Exception:
            pass

        # Post-hoc tests
        if p_value < alpha:
            section_header("Post-Hoc Tests")
            posthoc_type = st.selectbox(
                "Method:",
                ["Tukey HSD", "Bonferroni (pairwise t-tests)", "Games-Howell",
                 "Dunnett's Test", "Scheffe's Test"],
                key="ow_posthoc",
            )

            if posthoc_type == "Tukey HSD":
                tukey = pairwise_tukeyhsd(data[dep_var], data[factor], alpha=alpha)
                tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                                        columns=tukey._results_table.data[0])
                st.dataframe(tukey_df, use_container_width=True, hide_index=True)

            elif posthoc_type == "Bonferroni (pairwise t-tests)":
                pairs = []
                from itertools import combinations
                for (n1, g1), (n2, g2) in combinations(zip(group_names, groups), 2):
                    t_stat, t_p = stats.ttest_ind(g1, g2)
                    n_comparisons = k * (k - 1) / 2
                    bonf_p = min(t_p * n_comparisons, 1.0)
                    pairs.append({"Group 1": str(n1), "Group 2": str(n2),
                                  "t": t_stat, "p (raw)": t_p, "p (Bonferroni)": bonf_p,
                                  "Significant?": "Yes" if bonf_p < alpha else "No"})
                st.dataframe(pd.DataFrame(pairs).round(4), use_container_width=True, hide_index=True)

            elif posthoc_type == "Games-Howell":
                from itertools import combinations
                pairs = []
                for (n1, g1), (n2, g2) in combinations(zip(group_names, groups), 2):
                    mean1, mean2 = g1.mean(), g2.mean()
                    n1_size, n2_size = len(g1), len(g2)
                    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
                    se = np.sqrt(var1 / n1_size + var2 / n2_size)
                    if se < 1e-12:
                        pairs.append({"Group 1": str(n1), "Group 2": str(n2),
                                      "Mean Diff": mean1 - mean2, "SE": 0,
                                      "t": np.nan, "df": np.nan, "p (Games-Howell)": 1.0,
                                      "CI Lower": np.nan, "CI Upper": np.nan,
                                      "Significant?": "No"})
                        continue
                    t_stat_gh = (mean1 - mean2) / se
                    # Welch-Satterthwaite df
                    num_df = (var1 / n1_size + var2 / n2_size) ** 2
                    denom_df = (var1 / n1_size) ** 2 / (n1_size - 1) + (var2 / n2_size) ** 2 / (n2_size - 1)
                    df_ws = num_df / denom_df if denom_df > 0 else 1

                    # p-value from studentized range distribution
                    try:
                        p_gh = stats.studentized_range.sf(np.abs(t_stat_gh) * np.sqrt(2), k, df_ws)
                    except Exception:
                        p_raw = 2 * (1 - stats.t.cdf(np.abs(t_stat_gh), df_ws))
                        n_comp = k * (k - 1) / 2
                        p_gh = min(p_raw * n_comp, 1.0)

                    # CI
                    try:
                        q_crit = stats.studentized_range.ppf(1 - alpha, k, df_ws) / np.sqrt(2)
                    except Exception:
                        q_crit = stats.t.ppf(1 - alpha / (2 * k * (k - 1) / 2), df_ws)
                    ci_lower = (mean1 - mean2) - q_crit * se
                    ci_upper = (mean1 - mean2) + q_crit * se

                    pairs.append({"Group 1": str(n1), "Group 2": str(n2),
                                  "Mean Diff": round(mean1 - mean2, 4), "SE": round(se, 4),
                                  "t": round(t_stat_gh, 4), "df": round(df_ws, 2),
                                  "p (Games-Howell)": round(p_gh, 6),
                                  "CI Lower": round(ci_lower, 4), "CI Upper": round(ci_upper, 4),
                                  "Significant?": "Yes" if p_gh < alpha else "No"})
                pairs_df = pd.DataFrame(pairs)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                help_tip("Games-Howell Test", "Recommended when group variances are unequal (Levene's test significant). Uses Welch's t-test with Satterthwaite df and studentized range distribution for p-values.")

            elif posthoc_type == "Dunnett's Test":
                control_group = st.selectbox("Control group:", group_names, key="ow_dunnett_control")
                control_idx = group_names.index(control_group)
                control_data = groups[control_idx]

                pairs = []
                try:
                    treatment_groups = [groups[i] for i in range(k) if i != control_idx]
                    treatment_names = [group_names[i] for i in range(k) if i != control_idx]
                    dunnett_result = stats.dunnett(*treatment_groups, control=control_data)
                    for i, name in enumerate(treatment_names):
                        mean_diff = treatment_groups[i].mean() - control_data.mean()
                        ci = dunnett_result.confidence_interval(confidence_level=1 - alpha)
                        pairs.append({
                            "Treatment": str(name), "Control": str(control_group),
                            "Mean Diff": round(mean_diff, 4),
                            "Statistic": round(dunnett_result.statistic[i], 4),
                            "p-value": round(dunnett_result.pvalue[i], 6),
                            "CI Lower": round(ci.low[i], 4),
                            "CI Upper": round(ci.high[i], 4),
                            "Significant?": "Yes" if dunnett_result.pvalue[i] < alpha else "No",
                        })
                except (AttributeError, TypeError):
                    st.info("Using Bonferroni-corrected t-tests (scipy.stats.dunnett not available).")
                    n_comp = k - 1
                    for i in range(k):
                        if i == control_idx:
                            continue
                        t_d, t_p = stats.ttest_ind(groups[i], control_data)
                        bonf_p = min(t_p * n_comp, 1.0)
                        mean_diff = groups[i].mean() - control_data.mean()
                        v1, v2 = groups[i].var(ddof=1), control_data.var(ddof=1)
                        n1_s, n2_s = len(groups[i]), len(control_data)
                        se_d = np.sqrt(v1 / n1_s + v2 / n2_s)
                        df_d = (v1/n1_s + v2/n2_s)**2 / ((v1/n1_s)**2/(n1_s-1) + (v2/n2_s)**2/(n2_s-1)) if (v1/n1_s + v2/n2_s) > 0 else 1
                        t_crit = stats.t.ppf(1 - alpha / (2 * n_comp), df_d)
                        ci_lower = mean_diff - t_crit * se_d
                        ci_upper = mean_diff + t_crit * se_d
                        pairs.append({
                            "Treatment": str(group_names[i]), "Control": str(control_group),
                            "Mean Diff": round(mean_diff, 4),
                            "t": round(t_d, 4),
                            "p (Bonferroni)": round(bonf_p, 6),
                            "CI Lower": round(ci_lower, 4),
                            "CI Upper": round(ci_upper, 4),
                            "Significant?": "Yes" if bonf_p < alpha else "No",
                        })
                pairs_df = pd.DataFrame(pairs)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                help_tip("Dunnett's Test", "Compares each treatment group against a single control group. More powerful than Bonferroni when only comparisons to a control are needed.")

            elif posthoc_type == "Scheffe's Test":
                from itertools import combinations
                n_total = len(data)
                f_crit = stats.f.ppf(1 - alpha, k - 1, n_total - k)
                scheffe_crit = (k - 1) * f_crit
                pairs = []

                for (n1, g1), (n2, g2) in combinations(zip(group_names, groups), 2):
                    mean_diff = g1.mean() - g2.mean()
                    f_scheffe = mean_diff ** 2 / (ms_within * (1 / len(g1) + 1 / len(g2)))
                    p_scheffe = 1 - stats.f.cdf(f_scheffe / (k - 1), k - 1, n_total - k)
                    se_s = np.sqrt(ms_within * (1 / len(g1) + 1 / len(g2)))
                    margin = np.sqrt(scheffe_crit) * se_s
                    ci_lower = mean_diff - margin
                    ci_upper = mean_diff + margin

                    pairs.append({
                        "Group 1": str(n1), "Group 2": str(n2),
                        "Mean Diff": round(mean_diff, 4),
                        "F (Scheffe)": round(f_scheffe, 4),
                        "p-value": round(p_scheffe, 6),
                        "CI Lower": round(ci_lower, 4),
                        "CI Upper": round(ci_upper, 4),
                        "Significant?": "Yes" if p_scheffe < alpha else "No",
                    })
                pairs_df = pd.DataFrame(pairs)
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
                help_tip("Scheffe's Test", "Most conservative post-hoc test. Controls family-wise error rate for all possible contrasts, not just pairwise comparisons.")

            # Forest plot for mean differences with CIs
            _render_posthoc_forest_plot(group_names, groups, ms_within, k, len(data), alpha, posthoc_type)

        # Visualizations
        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(data, x=factor, y=dep_var, color=factor,
                         title=f"{dep_var} by {factor}", points="outliers")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            means = data.groupby(factor)[dep_var].agg(["mean", "std", "count"]).reset_index()
            means["se"] = means["std"] / np.sqrt(means["count"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=means[factor].astype(str), y=means["mean"],
                error_y=dict(type="data", array=1.96 * means["se"].values),
                mode="markers+lines", marker=dict(size=10),
                line=dict(),
            ))
            fig.update_layout(title="Means Plot (±95% CI)", xaxis_title=factor,
                              yaxis_title=f"Mean {dep_var}", height=400)
            st.plotly_chart(fig, use_container_width=True)


def _render_two_way(df: pd.DataFrame):
    """Two-way ANOVA."""
    if not HAS_SM:
        empty_state("statsmodels required for two-way ANOVA.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = _get_cat_cols(df)

    if not num_cols or len(cat_cols) < 2:
        empty_state("Need numeric column and at least 2 categorical columns.")
        return

    dep_var = st.selectbox("Dependent variable:", num_cols, key="tw_dep")
    factor_a = st.selectbox("Factor A:", cat_cols, key="tw_fa")
    factor_b = st.selectbox("Factor B:", [c for c in cat_cols if c != factor_a], key="tw_fb")
    ss_type = st.selectbox("Sum of Squares type:", [2, 3], index=1, key="tw_ss")

    if st.button("Run Two-Way ANOVA", key="run_tw"):
        data = df[[dep_var, factor_a, factor_b]].dropna()
        # Ensure factors are strings for formula
        data[factor_a] = data[factor_a].astype(str)
        data[factor_b] = data[factor_b].astype(str)

        # ── Assumption checks (rendered BEFORE results) ──
        try:
            checks = []
            # Levene's test across all cells
            cell_groups = [
                g[dep_var].values
                for _, g in data.groupby([factor_a, factor_b])
                if len(g) >= 2
            ]
            if len(cell_groups) >= 2:
                checks.append(check_equal_variance(*cell_groups))
            # Normality of residuals (run on overall DV as proxy before model fit)
            checks.append(check_normality(data[dep_var].values, label="overall"))
            # Sample size
            min_cell_n = min(len(g) for _, g in data.groupby([factor_a, factor_b]))
            checks.append(check_sample_size(min_cell_n, "anova"))
            validation_panel(checks)
        except Exception:
            pass

        formula = f"Q('{dep_var}') ~ C(Q('{factor_a}')) * C(Q('{factor_b}'))"
        try:
            model = ols(formula, data=data).fit()
            anova_table = anova_lm(model, typ=ss_type)
            anova_table = anova_table.round(4)
            section_header("Two-Way ANOVA Table")
            st.dataframe(anova_table, use_container_width=True)

            # Effect sizes
            ss_total = anova_table["sum_sq"].sum()
            for idx in anova_table.index[:-1]:  # Skip Residual
                eta_sq = anova_table.loc[idx, "sum_sq"] / ss_total
                st.write(f"**{idx}:** η² = {eta_sq:.4f}")

            # Interpretation cards for each effect
            try:
                for idx in anova_table.index[:-1]:
                    eta_sq_val = anova_table.loc[idx, "sum_sq"] / ss_total
                    p_val = anova_table.loc[idx, "PR(>F)"] if "PR(>F)" in anova_table.columns else None
                    if p_val is not None and not np.isnan(p_val):
                        interpretation_card(interpret_p_value(p_val, 0.05))
                interpretation_card(interpret_effect_size(
                    anova_table.loc[anova_table.index[0], "sum_sq"] / ss_total, "eta-squared"
                ))
            except Exception:
                pass

            # Interaction plot
            means = data.groupby([factor_a, factor_b])[dep_var].mean().reset_index()
            fig = px.line(means, x=factor_a, y=dep_var, color=factor_b,
                          markers=True, title="Interaction Plot")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error fitting model: {e}")


def _render_repeated_measures(df: pd.DataFrame):
    """Repeated measures ANOVA."""
    if not HAS_PG:
        empty_state("pingouin required for repeated measures ANOVA.", "Install with: pip install pingouin")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    st.markdown("Data should be in long format with columns for: subject ID, within-subject factor, and dependent variable.")
    subject_col = st.selectbox("Subject ID column:", all_cols, key="rm_subj")
    within_col = st.selectbox("Within-subject factor:", [c for c in all_cols if c != subject_col], key="rm_within")
    dep_var = st.selectbox("Dependent variable:", [c for c in num_cols if c != subject_col], key="rm_dep")

    if st.button("Run Repeated Measures ANOVA", key="run_rm"):
        data = df[[subject_col, within_col, dep_var]].dropna()
        try:
            rm_anova = pg.rm_anova(data=data, dv=dep_var, within=within_col, subject=subject_col)
            section_header("Repeated Measures ANOVA")
            st.dataframe(rm_anova.round(4), use_container_width=True, hide_index=True)

            # Check sphericity
            if "W-spher" in rm_anova.columns:
                st.write(f"**Mauchly's W:** {rm_anova['W-spher'].values[0]:.4f}, "
                         f"p = {rm_anova['p-spher'].values[0]:.4f}")
                if rm_anova['p-spher'].values[0] < 0.05:
                    st.warning("Sphericity violated. Use Greenhouse-Geisser or Huynh-Feldt correction.")
                    if "eps" in rm_anova.columns:
                        st.write(f"**Greenhouse-Geisser ε:** {rm_anova['eps'].values[0]:.4f}")

            # Post-hoc paired comparisons
            if rm_anova['p-unc'].values[0] < 0.05:
                section_header("Post-Hoc Paired Comparisons")
                posthoc = pg.pairwise_tests(data=data, dv=dep_var, within=within_col,
                                            subject=subject_col, padjust="bonf")
                st.dataframe(posthoc.round(4), use_container_width=True, hide_index=True)

            # Visualization
            fig = px.box(data, x=within_col, y=dep_var, color=within_col,
                         title=f"{dep_var} across {within_col}", points="all")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")


def _render_kruskal_wallis(df: pd.DataFrame):
    """Kruskal-Wallis H test (nonparametric one-way ANOVA)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = _get_cat_cols(df)

    if not num_cols or not cat_cols:
        empty_state("Need numeric and categorical columns.")
        return

    dep_var = st.selectbox("Dependent variable:", num_cols, key="kw_dep")
    factor = st.selectbox("Factor:", cat_cols, key="kw_factor")

    if st.button("Run Kruskal-Wallis", key="run_kw"):
        data = df[[dep_var, factor]].dropna()
        groups = [g[dep_var].values for _, g in data.groupby(factor)]
        group_names = list(data[factor].unique())

        if len(groups) < 2:
            st.error("Need at least 2 groups.")
            return

        h_stat, p_value = stats.kruskal(*groups)
        n = len(data)
        k = len(groups)
        # Effect size: eta-squared for Kruskal-Wallis
        eta_sq_h = max(0, (h_stat - k + 1) / (n - k))

        c1, c2, c3 = st.columns(3)
        c1.metric("H-statistic", f"{h_stat:.4f}")
        c2.metric("p-value", f"{p_value:.6f}")
        c3.metric("η²_H", f"{eta_sq_h:.4f}")

        if p_value < 0.05:
            st.success(f"**Significant** (p = {p_value:.6f})")
        else:
            st.info(f"**Not significant** (p = {p_value:.6f})")

        # Interpretation cards
        try:
            interpretation_card(interpret_p_value(p_value, 0.05))
            interpretation_card(interpret_effect_size(eta_sq_h, "eta-squared"))
        except Exception:
            pass

        # Dunn's post-hoc test
        if p_value < 0.05 and HAS_PG:
            section_header("Dunn's Post-Hoc Test")
            try:
                dunn = pg.pairwise_tests(data=data, dv=dep_var, between=factor,
                                         parametric=False, padjust="bonf")
                st.dataframe(dunn.round(4), use_container_width=True, hide_index=True)
            except Exception as e:
                st.warning(f"Post-hoc test error: {e}")

        fig = px.box(data, x=factor, y=dep_var, color=factor,
                     title=f"Kruskal-Wallis: {dep_var} by {factor}", points="all")
        st.plotly_chart(fig, use_container_width=True)


def _render_friedman(df: pd.DataFrame):
    """Friedman test (nonparametric repeated measures)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 3:
        empty_state("Need at least 3 numeric columns for Friedman test.")
        return

    st.markdown("Select columns representing repeated measurements on the same subjects.")
    selected = st.multiselect("Measurement columns:", num_cols, key="friedman_cols")

    if len(selected) < 3:
        st.info("Select at least 3 columns.")
        return

    if st.button("Run Friedman Test", key="run_friedman"):
        data = df[selected].dropna()
        arrays = [data[c].values for c in selected]

        chi2, p_value = stats.friedmanchisquare(*arrays)
        n = len(data)
        k = len(selected)
        # Kendall's W
        w = chi2 / (n * (k - 1))

        c1, c2, c3 = st.columns(3)
        c1.metric("χ²", f"{chi2:.4f}")
        c2.metric("p-value", f"{p_value:.6f}")
        c3.metric("Kendall's W", f"{w:.4f}")

        if p_value < 0.05:
            st.success(f"**Significant** (p = {p_value:.6f})")
        else:
            st.info(f"**Not significant** (p = {p_value:.6f})")

        # Interpretation cards
        try:
            interpretation_card(interpret_p_value(p_value, 0.05))
        except Exception:
            pass

        # Visualization
        melt_df = data.melt(var_name="Condition", value_name="Value")
        fig = px.box(melt_df, x="Condition", y="Value", color="Condition",
                     title="Friedman Test: Distributions", points="all")
        st.plotly_chart(fig, use_container_width=True)


def _render_ancova(df: pd.DataFrame):
    """Analysis of Covariance."""
    if not HAS_PG:
        empty_state("pingouin required for ANCOVA.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = _get_cat_cols(df)

    if not num_cols or not cat_cols:
        empty_state("Need numeric and categorical columns.")
        return

    dep_var = st.selectbox("Dependent variable:", num_cols, key="anc_dep")
    factor = st.selectbox("Factor:", cat_cols, key="anc_factor")
    covariates = st.multiselect("Covariate(s):", [c for c in num_cols if c != dep_var], key="anc_cov")

    if not covariates:
        st.info("Select at least one covariate.")
        return

    if st.button("Run ANCOVA", key="run_anc"):
        data = df[[dep_var, factor] + covariates].dropna()
        data[factor] = data[factor].astype(str)

        try:
            ancova_result = pg.ancova(data=data, dv=dep_var, between=factor, covar=covariates)
            section_header("ANCOVA Table")
            st.dataframe(ancova_result.round(4), use_container_width=True, hide_index=True)

            # Homogeneity of regression slopes check
            section_header("Homogeneity of Regression Slopes")
            for cov in covariates:
                fig = px.scatter(data, x=cov, y=dep_var, color=factor,
                                 trendline="ols", title=f"{dep_var} vs {cov} by {factor}")
                st.plotly_chart(fig, use_container_width=True)

            # Adjusted means
            section_header("Group Means")
            means = data.groupby(factor)[dep_var].agg(["mean", "std", "count"]).round(4)
            st.dataframe(means, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")


# ─── Forest Plot for Post-Hoc Comparisons ──────────────────────────────

def _render_posthoc_forest_plot(group_names, groups, ms_within, k, n_total, alpha, method_name):
    """Render a forest-style plot showing mean differences with CIs for pairwise comparisons."""
    from itertools import combinations

    labels = []
    mean_diffs = []
    ci_lowers = []
    ci_uppers = []

    for (n1, g1), (n2, g2) in combinations(zip(group_names, groups), 2):
        mean_diff = g1.mean() - g2.mean()
        se = np.sqrt(ms_within * (1 / len(g1) + 1 / len(g2)))

        # Use appropriate CI method
        if method_name == "Scheffe's Test":
            f_crit = stats.f.ppf(1 - alpha, k - 1, n_total - k)
            margin = np.sqrt((k - 1) * f_crit) * se
        elif method_name == "Games-Howell":
            var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
            se_gh = np.sqrt(var1 / len(g1) + var2 / len(g2))
            num_df = (var1 / len(g1) + var2 / len(g2)) ** 2
            denom_df = (var1 / len(g1)) ** 2 / (len(g1) - 1) + (var2 / len(g2)) ** 2 / (len(g2) - 1)
            df_ws = num_df / denom_df if denom_df > 0 else 1
            try:
                q_crit = stats.studentized_range.ppf(1 - alpha, k, df_ws) / np.sqrt(2)
            except Exception:
                q_crit = stats.t.ppf(1 - alpha / (2 * k * (k - 1) / 2), df_ws)
            se = se_gh
            margin = q_crit * se
        elif method_name == "Dunnett's Test":
            # For Dunnett's, skip the pairwise forest — it's handled separately
            return
        else:
            # Tukey/Bonferroni: use t-distribution with Bonferroni correction
            n_comp = k * (k - 1) / 2
            t_crit = stats.t.ppf(1 - alpha / (2 * n_comp), n_total - k)
            margin = t_crit * se

        labels.append(f"{n1} - {n2}")
        mean_diffs.append(mean_diff)
        ci_lowers.append(mean_diff - margin)
        ci_uppers.append(mean_diff + margin)

    if not labels:
        return

    section_header("Forest Plot: Mean Differences")
    fig = go.Figure()
    colors = ["#ef4444" if (cl > 0 or cu < 0) else "#6366f1"
              for cl, cu in zip(ci_lowers, ci_uppers)]

    fig.add_trace(go.Scatter(
        x=mean_diffs, y=labels, mode="markers",
        marker=dict(size=10, color=colors),
        error_x=dict(
            type="data", symmetric=False,
            array=[u - m for u, m in zip(ci_uppers, mean_diffs)],
            arrayminus=[m - l for m, l in zip(mean_diffs, ci_lowers)],
            color="gray", thickness=1.5, width=6,
        ),
        showlegend=False,
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title=f"Pairwise Mean Differences ({method_name})",
        xaxis_title="Mean Difference",
        yaxis_title="Comparison",
        height=max(300, len(labels) * 40 + 100),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── MANOVA ─────────────────────────────────────────────────────────────

def _render_manova(df: pd.DataFrame):
    """Multivariate Analysis of Variance."""
    if not HAS_SM:
        empty_state("statsmodels required for MANOVA.", "Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = _get_cat_cols(df)

    if len(num_cols) < 2:
        empty_state("Need at least 2 numeric columns for MANOVA.", "MANOVA requires multiple dependent variables.")
        return
    if not cat_cols:
        empty_state("Need at least 1 categorical column for MANOVA.")
        return

    section_header("MANOVA", "Test whether group means differ across multiple dependent variables simultaneously.")
    help_tip("MANOVA Overview", """
**Multivariate Analysis of Variance** tests whether the means of multiple dependent variables
differ simultaneously across groups defined by one or more factors.

Test statistics:
- **Wilks' Lambda:** Most commonly used. Values near 0 indicate group differences.
- **Pillai's Trace:** Most robust, especially with unequal cell sizes or non-normality.
- **Hotelling-Lawley Trace:** Most powerful when there is one large eigenvalue.
- **Roy's Greatest Root:** Most powerful but most sensitive to violations of assumptions.
""")

    dep_vars = st.multiselect("Dependent variables (numeric, at least 2):", num_cols, key="manova_dvs")
    if len(dep_vars) < 2:
        st.info("Select at least 2 dependent variables.")
        return

    n_factors = st.radio("Number of factors:", [1, 2], horizontal=True, key="manova_n_factors")
    factors = []
    if n_factors >= 1:
        factor1 = st.selectbox("Factor 1:", cat_cols, key="manova_f1")
        factors.append(factor1)
    if n_factors == 2:
        remaining_cats = [c for c in cat_cols if c != factor1]
        if not remaining_cats:
            st.warning("No additional categorical columns available for a second factor.")
        else:
            factor2 = st.selectbox("Factor 2:", remaining_cats, key="manova_f2")
            factors.append(factor2)

    if st.button("Run MANOVA", key="run_manova"):
        cols_needed = dep_vars + factors
        data = df[cols_needed].dropna()

        # Ensure factors are strings
        for f in factors:
            data[f] = data[f].astype(str)

        n_obs = len(data)
        if n_obs < len(dep_vars) + len(factors) + 2:
            st.error(f"Not enough observations ({n_obs}). Need more data points than variables.")
            return

        # Check minimum group sizes
        try:
            if len(factors) == 1:
                min_grp = data.groupby(factors[0]).size().min()
            else:
                min_grp = data.groupby(factors).size().min()
            if min_grp < len(dep_vars) + 1:
                st.warning(f"Smallest group has {min_grp} observations. MANOVA requires group size > number of dependent variables ({len(dep_vars)}).")
        except Exception:
            pass

        # Build formula
        dv_formula = " + ".join(dep_vars)
        factor_formula = " + ".join([f"C({f})" for f in factors])
        formula = f"{dv_formula} ~ {factor_formula}"

        try:
            with st.spinner("Fitting MANOVA..."):
                from statsmodels.multivariate.manova import MANOVA
                manova = MANOVA.from_formula(formula, data=data)
                manova_result = manova.mv_test()

            section_header("MANOVA Results")

            # Display test statistics for each factor
            for factor_name in factors:
                st.markdown(f"**Factor: {factor_name}**")
                try:
                    factor_key = f"C({factor_name})"
                    result_table = manova_result.results[factor_key]
                    stat_table = result_table["stat"]
                    st.dataframe(pd.DataFrame(stat_table).round(6), use_container_width=True)
                except Exception:
                    pass

            # Show full summary
            with st.expander("Full MANOVA Summary", expanded=False):
                summary_str = str(manova_result)
                st.text(summary_str)

            # Follow-up: univariate ANOVAs for each DV
            section_header("Follow-Up Univariate ANOVAs")
            univariate_rows = []
            for dv in dep_vars:
                for f in factors:
                    try:
                        factor_groups = [g[dv].values for _, g in data.groupby(f) if len(g) >= 2]
                        if len(factor_groups) >= 2:
                            f_stat_uv, p_val_uv = stats.f_oneway(*factor_groups)
                            grand_mean = data[dv].mean()
                            ss_b = sum(len(fg) * (fg.mean() - grand_mean) ** 2 for fg in factor_groups)
                            ss_t = np.sum((data[dv].values - grand_mean) ** 2)
                            eta_sq_uv = ss_b / ss_t if ss_t > 0 else 0

                            univariate_rows.append({
                                "DV": dv, "Factor": f,
                                "F": round(f_stat_uv, 4),
                                "p-value": round(p_val_uv, 6),
                                "eta-sq": round(eta_sq_uv, 4),
                                "Significant?": "Yes" if p_val_uv < 0.05 else "No",
                            })
                    except Exception:
                        pass

            if univariate_rows:
                uv_df = pd.DataFrame(univariate_rows)
                st.dataframe(uv_df, use_container_width=True, hide_index=True)
            else:
                st.warning("Could not compute univariate follow-up ANOVAs.")

            # Visualization: means plot for each DV by factor
            section_header("Means Plots")
            for f in factors:
                n_dvs = len(dep_vars)
                cols_per_row = min(3, n_dvs)
                rows_plot = (n_dvs + cols_per_row - 1) // cols_per_row

                fig = make_subplots(
                    rows=rows_plot, cols=cols_per_row,
                    subplot_titles=dep_vars,
                )

                for idx, dv in enumerate(dep_vars):
                    row = idx // cols_per_row + 1
                    col = idx % cols_per_row + 1

                    means_data = data.groupby(f)[dv].agg(["mean", "std", "count"]).reset_index()
                    means_data["se"] = means_data["std"] / np.sqrt(means_data["count"])

                    fig.add_trace(go.Scatter(
                        x=means_data[f].astype(str),
                        y=means_data["mean"],
                        error_y=dict(type="data", array=1.96 * means_data["se"].values),
                        mode="markers+lines",
                        marker=dict(size=10),
                        showlegend=False,
                    ), row=row, col=col)

                    fig.update_xaxes(title_text=f, row=row, col=col)
                    if col == 1:
                        fig.update_yaxes(title_text="Mean", row=row, col=col)

                fig.update_layout(
                    title_text=f"Group Means by {f} (with 95% CI)",
                    height=350 * rows_plot,
                )
                st.plotly_chart(fig, use_container_width=True)

        except np.linalg.LinAlgError:
            st.error("MANOVA failed: singular matrix. This can happen when there are too few observations per group or highly correlated dependent variables.")
        except Exception as e:
            st.error(f"MANOVA failed: {e}")
