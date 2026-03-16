"""
Monte Carlo Simulation Module - Distribution simulation, process simulation, risk analysis,
tolerance analysis, measurement uncertainty, and what-if scenarios.
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.ui_helpers import section_header, empty_state, help_tip, rdl_plotly_chart, log_analysis


_DISTRIBUTIONS = {
    "Normal": {"params": ["Mean", "Std Dev"], "defaults": [0.0, 1.0],
               "fn": lambda n, p: np.random.normal(p[0], p[1], n)},
    "Uniform": {"params": ["Low", "High"], "defaults": [0.0, 1.0],
                "fn": lambda n, p: np.random.uniform(p[0], p[1], n)},
    "Triangular": {"params": ["Low", "Mode", "High"], "defaults": [0.0, 0.5, 1.0],
                   "fn": lambda n, p: np.random.triangular(p[0], p[1], p[2], n)},
    "Beta": {"params": ["Alpha", "Beta"], "defaults": [2.0, 5.0],
             "fn": lambda n, p: np.random.beta(p[0], p[1], n)},
    "Lognormal": {"params": ["Mean (log)", "Sigma (log)"], "defaults": [0.0, 0.5],
                  "fn": lambda n, p: np.random.lognormal(p[0], p[1], n)},
    "Weibull": {"params": ["Shape (k)", "Scale (lambda)"], "defaults": [2.0, 1.0],
                "fn": lambda n, p: p[1] * np.random.weibull(p[0], n)},
    "Exponential": {"params": ["Rate (lambda)"], "defaults": [1.0],
                    "fn": lambda n, p: np.random.exponential(1 / p[0], n)},
    "Poisson": {"params": ["Lambda"], "defaults": [5.0],
                "fn": lambda n, p: np.random.poisson(p[0], n)},
    "Binomial": {"params": ["n (trials)", "p (probability)"], "defaults": [10.0, 0.5],
                 "fn": lambda n, p: np.random.binomial(int(p[0]), p[1], n)},
}


def render_monte_carlo(df: pd.DataFrame):
    """Main entry point for Monte Carlo Simulation module."""
    tabs = st.tabs([
        "Distribution Simulator", "Process Simulation", "Risk Analysis",
        "Tolerance Analysis", "Measurement Uncertainty", "What-If Scenarios",
    ])

    with tabs[0]:
        _render_distribution_simulator()
    with tabs[1]:
        _render_process_simulation()
    with tabs[2]:
        _render_risk_analysis()
    with tabs[3]:
        _render_tolerance_analysis()
    with tabs[4]:
        _render_measurement_uncertainty()
    with tabs[5]:
        _render_what_if_scenarios()


# ─── Distribution Simulator ──────────────────────────────────────────────────

def _render_distribution_simulator():
    section_header("Distribution Simulator")
    help_tip("Distribution Simulator",
             "Choose a distribution, set parameters, and run N simulations to visualize the output.")

    dist_name = st.selectbox("Distribution:", list(_DISTRIBUTIONS.keys()), key="mc_dist")
    dist = _DISTRIBUTIONS[dist_name]

    cols = st.columns(len(dist["params"]))
    param_values = []
    for i, (name, default) in enumerate(zip(dist["params"], dist["defaults"])):
        val = cols[i].number_input(name, value=default, key=f"mc_p_{i}")
        param_values.append(val)

    n_sim = st.number_input("Number of simulations:", min_value=100, max_value=1_000_000,
                            value=10000, step=1000, key="mc_nsim")

    if st.button("Run Simulation", key="mc_run"):
        with st.spinner("Simulating..."):
            np.random.seed(None)
            samples = dist["fn"](int(n_sim), param_values)

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean", f"{np.mean(samples):.4f}")
        c2.metric("Std Dev", f"{np.std(samples):.4f}")
        c3.metric("Median", f"{np.median(samples):.4f}")
        c4.metric("Min", f"{np.min(samples):.4f}")
        c5.metric("Max", f"{np.max(samples):.4f}")

        # Histogram
        fig = px.histogram(x=samples, nbins=50, title=f"{dist_name} Distribution (n={n_sim:,})")
        fig.update_layout(xaxis_title="Value", yaxis_title="Frequency", height=400)
        rdl_plotly_chart(fig)

        # Percentiles table
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(samples, percentiles)
        pct_df = pd.DataFrame({
            "Percentile": [f"{p}%" for p in percentiles],
            "Value": [f"{v:.4f}" for v in pct_values],
        })
        st.dataframe(pct_df, use_container_width=True, hide_index=True)

        # CDF
        sorted_samples = np.sort(samples)
        cdf = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
        step = max(1, len(sorted_samples) // 2000)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=sorted_samples[::step], y=cdf[::step],
                                  mode="lines", name="CDF"))
        fig2.update_layout(title="Cumulative Distribution Function",
                           xaxis_title="Value", yaxis_title="Cumulative Probability",
                           height=350)
        rdl_plotly_chart(fig2)


# ─── Process Simulation ──────────────────────────────────────────────────────

def _render_process_simulation():
    section_header("Process Simulation")
    help_tip("Process Simulation",
             "Define input variables with distributions and a formula. "
             "Monte Carlo propagation produces the output distribution and sensitivity analysis.")

    st.markdown("Define input variables and a formula to propagate uncertainty.")

    n_vars = st.number_input("Number of input variables:", min_value=1, max_value=10,
                             value=3, key="ps_nvars")

    variables = {}
    for i in range(int(n_vars)):
        with st.expander(f"Variable {i + 1}", expanded=(i < 3)):
            name = st.text_input("Name:", value=f"X{i + 1}", key=f"ps_name_{i}")
            dist_name = st.selectbox("Distribution:", list(_DISTRIBUTIONS.keys()),
                                     key=f"ps_dist_{i}")
            dist = _DISTRIBUTIONS[dist_name]
            param_values = []
            pcols = st.columns(len(dist["params"]))
            for j, (pname, default) in enumerate(zip(dist["params"], dist["defaults"])):
                val = pcols[j].number_input(pname, value=default, key=f"ps_p_{i}_{j}")
                param_values.append(val)
            variables[name] = {"dist": dist_name, "params": param_values}

    formula = st.text_input(
        "Output formula (use variable names):",
        value=" + ".join(variables.keys()) if variables else "X1 + X2 + X3",
        key="ps_formula",
    )
    st.caption("Example: `X1 * X2 + X3 ** 2`, `(X1 - X2) / X3`")

    n_sim = st.number_input("Simulations:", min_value=1000, max_value=1_000_000,
                            value=10000, step=1000, key="ps_nsim")

    # --- Correlated inputs option ---
    correlate_inputs = st.checkbox(
        "Correlate inputs",
        key="ps_correlate",
        help="Apply a correlation structure between input variables using Cholesky decomposition.",
    )

    corr_matrix = None
    var_names = list(variables.keys())
    n_v = len(var_names)
    if correlate_inputs and n_v >= 2:
        st.markdown("**Input Correlation Matrix** (enter off-diagonal correlations, -1 to 1):")
        if f"ps_corr_matrix_{n_v}" not in st.session_state:
            st.session_state[f"ps_corr_matrix_{n_v}"] = np.eye(n_v).tolist()
        corr_vals = st.session_state[f"ps_corr_matrix_{n_v}"]

        # Header row
        header_cols = st.columns(n_v + 1)
        header_cols[0].markdown("**Var**")
        for j in range(n_v):
            header_cols[j + 1].markdown(f"**{var_names[j]}**")

        corr_matrix = np.eye(n_v)
        for i_row in range(n_v):
            row_cols = st.columns(n_v + 1)
            row_cols[0].markdown(f"**{var_names[i_row]}**")
            for j_col in range(n_v):
                if j_col == i_row:
                    row_cols[j_col + 1].text("1.00")
                elif j_col > i_row:
                    default_val = 0.0
                    if i_row < len(corr_vals) and j_col < len(corr_vals[0]):
                        default_val = float(corr_vals[i_row][j_col])
                    val = row_cols[j_col + 1].number_input(
                        f"r({var_names[i_row]},{var_names[j_col]})",
                        min_value=-1.0, max_value=1.0,
                        value=default_val, step=0.1,
                        key=f"ps_corr_{i_row}_{j_col}",
                        label_visibility="collapsed",
                    )
                    corr_matrix[i_row, j_col] = val
                    corr_matrix[j_col, i_row] = val
                else:
                    row_cols[j_col + 1].text(f"{corr_matrix[i_row, j_col]:.2f}")

        st.session_state[f"ps_corr_matrix_{n_v}"] = corr_matrix.tolist()

    if st.button("Run Process Simulation", key="ps_run"):
        with st.spinner("Simulating..."):
            np.random.seed(None)
            n = int(n_sim)
            samples = {}

            if correlate_inputs and corr_matrix is not None and n_v >= 2:
                # Cholesky decomposition for correlated samples
                try:
                    L = np.linalg.cholesky(corr_matrix)
                except np.linalg.LinAlgError:
                    st.error("Correlation matrix is not positive definite. Adjust correlations.")
                    return

                # Generate independent standard normals, then correlate
                z = np.random.normal(0, 1, size=(n, n_v))
                correlated_z = z @ L.T

                # Probability integral transform: correlated normal -> uniform -> target
                correlated_u = stats.norm.cdf(correlated_z)

                for idx, (vname, cfg) in enumerate(variables.items()):
                    dname = cfg["dist"]
                    p = cfg["params"]
                    u = correlated_u[:, idx]
                    if dname == "Normal":
                        samples[vname] = stats.norm.ppf(u, loc=p[0], scale=p[1])
                    elif dname == "Uniform":
                        samples[vname] = stats.uniform.ppf(u, loc=p[0], scale=p[1] - p[0])
                    elif dname == "Triangular":
                        c_param = (p[1] - p[0]) / (p[2] - p[0]) if p[2] != p[0] else 0.5
                        samples[vname] = stats.triang.ppf(u, c_param, loc=p[0], scale=p[2] - p[0])
                    elif dname == "Beta":
                        samples[vname] = stats.beta.ppf(u, p[0], p[1])
                    elif dname == "Lognormal":
                        samples[vname] = stats.lognorm.ppf(u, s=p[1], scale=np.exp(p[0]))
                    elif dname == "Weibull":
                        samples[vname] = stats.weibull_min.ppf(u, p[0], scale=p[1])
                    elif dname == "Exponential":
                        samples[vname] = stats.expon.ppf(u, scale=1 / p[0])
                    elif dname == "Poisson":
                        samples[vname] = stats.poisson.ppf(
                            np.clip(u, 0, 0.9999999), p[0]
                        ).astype(float)
                    elif dname == "Binomial":
                        samples[vname] = stats.binom.ppf(
                            np.clip(u, 0, 0.9999999), int(p[0]), p[1]
                        ).astype(float)
                    else:
                        dist_obj = _DISTRIBUTIONS[dname]
                        samples[vname] = dist_obj["fn"](n, p)
            else:
                for vname, cfg in variables.items():
                    d = _DISTRIBUTIONS[cfg["dist"]]
                    samples[vname] = d["fn"](n, cfg["params"])

            # Evaluate formula safely
            allowed = {vname: samples[vname] for vname in variables}
            allowed.update({
                "np": np, "sqrt": np.sqrt, "log": np.log,
                "exp": np.exp, "abs": np.abs, "sin": np.sin,
                "cos": np.cos, "pi": np.pi,
            })
            try:
                output = eval(formula, {"__builtins__": {}}, allowed)
            except Exception as e:
                st.error(f"Formula error: {e}")
                return

        # Store results for What-If Scenarios tab
        st.session_state["_ps_last_output"] = output
        st.session_state["_ps_last_variables"] = variables
        st.session_state["_ps_last_formula"] = formula
        st.session_state["_ps_last_n_sim"] = n_sim

        # Output distribution
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Output", f"{np.mean(output):.4f}")
        c2.metric("Std Dev", f"{np.std(output):.4f}")
        c3.metric("CV%", f"{np.std(output) / abs(np.mean(output)) * 100:.2f}%"
                  if np.mean(output) != 0 else "N/A")

        fig = px.histogram(x=output, nbins=50, title="Output Distribution")
        fig.update_layout(xaxis_title="Output Value", yaxis_title="Frequency", height=400)
        rdl_plotly_chart(fig)

        # Sensitivity tornado chart
        section_header("Sensitivity Analysis (Tornado Chart)")
        correlations = {}
        for vname in variables:
            correlations[vname] = np.corrcoef(samples[vname], output)[0, 1]

        sens_df = pd.DataFrame({
            "Variable": list(correlations.keys()),
            "Correlation": list(correlations.values()),
        }).sort_values("Correlation", key=abs, ascending=True)

        fig2 = px.bar(sens_df, x="Correlation", y="Variable", orientation="h",
                      title="Sensitivity: Input-Output Correlation",
                      color="Correlation",
                      color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
        fig2.update_layout(height=max(300, len(variables) * 50))
        rdl_plotly_chart(fig2)

        # --- Convergence Plot ---
        section_header("Convergence Analysis")
        help_tip("Convergence",
                 "Shows how the running mean and 95% CI stabilize as iteration count increases.")

        n_total = len(output)
        n_points = 200
        indices = np.unique(np.linspace(1, n_total, n_points, dtype=int))
        running_means = np.array([np.mean(output[:k]) for k in indices])
        running_stds = np.array([np.std(output[:k]) for k in indices])
        running_ci_low = running_means - 1.96 * running_stds / np.sqrt(indices)
        running_ci_high = running_means + 1.96 * running_stds / np.sqrt(indices)

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=indices, y=running_ci_high, mode="lines",
            line=dict(width=0), showlegend=False, name="Upper CI",
        ))
        fig_conv.add_trace(go.Scatter(
            x=indices, y=running_ci_low, mode="lines",
            fill="tonexty", fillcolor="rgba(99,102,241,0.15)",
            line=dict(width=0), name="95% CI",
        ))
        fig_conv.add_trace(go.Scatter(
            x=indices, y=running_means, mode="lines",
            name="Running Mean", line=dict(color="#6366f1", width=2),
        ))
        fig_conv.add_hline(
            y=np.mean(output), line_dash="dash", line_color="#94a3b8",
            annotation_text=f"Final Mean = {np.mean(output):.4f}",
        )
        fig_conv.update_layout(
            title="Running Mean and 95% CI vs. Iteration Count",
            xaxis_title="Iteration", yaxis_title="Running Mean",
            height=380,
        )
        rdl_plotly_chart(fig_conv)


# ─── Risk Analysis ───────────────────────────────────────────────────────────

def _render_risk_analysis():
    section_header("Risk Analysis")
    help_tip("Risk Analysis",
             "Compute probability of exceeding thresholds and confidence intervals for quantiles.")

    st.markdown("Set up a distribution and analyze threshold exceedance and quantile uncertainty.")

    dist_name = st.selectbox("Distribution:", list(_DISTRIBUTIONS.keys()), key="ra_dist")
    dist = _DISTRIBUTIONS[dist_name]

    cols = st.columns(len(dist["params"]))
    param_values = []
    for i, (name, default) in enumerate(zip(dist["params"], dist["defaults"])):
        val = cols[i].number_input(name, value=default, key=f"ra_p_{i}")
        param_values.append(val)

    n_sim = st.number_input("Simulations:", min_value=1000, max_value=1_000_000,
                            value=50000, step=5000, key="ra_nsim")

    threshold = st.number_input("Threshold value:", value=0.0, key="ra_thresh")
    direction = st.radio("Exceedance direction:", ["P(X > threshold)", "P(X < threshold)"],
                         horizontal=True, key="ra_dir")

    if st.button("Run Risk Analysis", key="ra_run"):
        with st.spinner("Simulating..."):
            np.random.seed(None)
            samples = dist["fn"](int(n_sim), param_values)

        # Threshold probability
        if ">" in direction:
            p_exceed = np.mean(samples > threshold)
        else:
            p_exceed = np.mean(samples < threshold)

        c1, c2, c3 = st.columns(3)
        c1.metric("Exceedance Prob", f"{p_exceed:.4f}")
        c2.metric("Exceedance %", f"{p_exceed * 100:.2f}%")
        # Bootstrap CI for exceedance probability
        n_boot = 1000
        boot_probs = []
        for _ in range(n_boot):
            boot = np.random.choice(samples, size=len(samples), replace=True)
            if ">" in direction:
                boot_probs.append(np.mean(boot > threshold))
            else:
                boot_probs.append(np.mean(boot < threshold))
        ci_low, ci_high = np.percentile(boot_probs, [2.5, 97.5])
        c3.metric("95% CI", f"[{ci_low:.4f}, {ci_high:.4f}]")

        # Histogram with threshold line
        fig = px.histogram(x=samples, nbins=50, title="Distribution with Threshold")
        fig.add_vline(x=threshold, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Threshold = {threshold}")
        fig.update_layout(height=400)
        rdl_plotly_chart(fig)

        # Quantile CI table
        section_header("Quantile Confidence Intervals")
        quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        q_data = []
        for q in quantiles:
            point_est = np.quantile(samples, q)
            boot_qs = [np.quantile(np.random.choice(samples, size=len(samples), replace=True), q)
                       for _ in range(n_boot)]
            ci_l, ci_h = np.percentile(boot_qs, [2.5, 97.5])
            q_data.append({
                "Quantile": f"{q:.0%}",
                "Estimate": f"{point_est:.4f}",
                "95% CI Lower": f"{ci_l:.4f}",
                "95% CI Upper": f"{ci_h:.4f}",
            })
        st.dataframe(pd.DataFrame(q_data), use_container_width=True, hide_index=True)


# ─── Tolerance Analysis ──────────────────────────────────────────────────────

def _render_tolerance_analysis():
    section_header("Tolerance Analysis")
    help_tip("Tolerance Analysis",
             "Analyze mechanical/manufacturing assembly tolerances using worst-case, "
             "RSS, or Monte Carlo stackup methods.")

    st.markdown("Define components with nominal dimensions and tolerances to predict assembly variation.")

    n_comp = st.number_input("Number of components:", min_value=2, max_value=10,
                             value=3, key="ta_ncomp")

    components = []
    for i in range(int(n_comp)):
        with st.expander(f"Component {i + 1}", expanded=(i < 3)):
            c1, c2 = st.columns(2)
            comp_name = c1.text_input("Name:", value=f"Part {i + 1}", key=f"ta_name_{i}")
            comp_dist = c2.selectbox("Distribution:",
                                     ["Normal", "Uniform", "Triangular"],
                                     key=f"ta_dist_{i}")
            c3, c4 = st.columns(2)
            nominal = c3.number_input("Nominal dimension:", value=10.0,
                                      key=f"ta_nom_{i}", format="%.4f")
            tolerance = c4.number_input("Tolerance (+/-):", value=0.1, min_value=0.0001,
                                        key=f"ta_tol_{i}", format="%.4f")
            components.append({
                "name": comp_name,
                "nominal": nominal,
                "tolerance": tolerance,
                "distribution": comp_dist,
            })

    st.divider()

    method = st.radio("Assembly method:",
                      ["Arithmetic (Worst Case)", "RSS (Root Sum of Squares)", "Monte Carlo"],
                      horizontal=True, key="ta_method")

    c_spec1, c_spec2 = st.columns(2)
    assembly_nominal = sum(c["nominal"] for c in components)
    lsl = c_spec1.number_input("Lower Spec Limit (LSL):", value=assembly_nominal - 1.0,
                               key="ta_lsl", format="%.4f")
    usl = c_spec2.number_input("Upper Spec Limit (USL):", value=assembly_nominal + 1.0,
                               key="ta_usl", format="%.4f")

    n_sim = 50000
    if method == "Monte Carlo":
        n_sim = st.number_input("Simulations:", min_value=1000, max_value=1_000_000,
                                value=50000, step=5000, key="ta_nsim")

    if st.button("Run Tolerance Analysis", key="ta_run"):
        with st.spinner("Analyzing tolerances..."):
            np.random.seed(None)
            n = int(n_sim)
            component_samples = []

            if method == "Arithmetic (Worst Case)":
                assembly_mean = sum(c["nominal"] for c in components)
                assembly_tol = sum(c["tolerance"] for c in components)
                assembly_std = assembly_tol / 3.0  # Assume 3-sigma = tolerance
                # Illustrative distribution
                assembly_samples = np.random.normal(assembly_mean, assembly_std, n)
                for c in components:
                    component_samples.append(
                        np.random.normal(c["nominal"], c["tolerance"] / 3.0, n)
                    )

            elif method == "RSS (Root Sum of Squares)":
                assembly_mean = sum(c["nominal"] for c in components)
                assembly_tol = np.sqrt(sum(c["tolerance"] ** 2 for c in components))
                assembly_std = assembly_tol / 3.0
                assembly_samples = np.random.normal(assembly_mean, assembly_std, n)
                for c in components:
                    component_samples.append(
                        np.random.normal(c["nominal"], c["tolerance"] / 3.0, n)
                    )

            else:  # Monte Carlo
                for c in components:
                    nom = c["nominal"]
                    tol = c["tolerance"]
                    if c["distribution"] == "Normal":
                        s = np.random.normal(nom, tol / 3.0, n)
                    elif c["distribution"] == "Uniform":
                        s = np.random.uniform(nom - tol, nom + tol, n)
                    elif c["distribution"] == "Triangular":
                        s = np.random.triangular(nom - tol, nom, nom + tol, n)
                    else:
                        s = np.random.normal(nom, tol / 3.0, n)
                    component_samples.append(s)

                assembly_samples = np.sum(component_samples, axis=0)
                assembly_mean = np.mean(assembly_samples)
                assembly_std = np.std(assembly_samples)
                assembly_tol = 3.0 * assembly_std

        # Cp and Cpk
        if assembly_std > 0:
            cp = (usl - lsl) / (6 * assembly_std)
            cpu = (usl - assembly_mean) / (3 * assembly_std)
            cpl = (assembly_mean - lsl) / (3 * assembly_std)
            cpk = min(cpu, cpl)
        else:
            cp = cpk = float("inf")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Assembly Mean", f"{assembly_mean:.4f}")
        m2.metric("Assembly Std Dev", f"{assembly_std:.4f}")
        m3.metric("Cp", f"{cp:.3f}")
        m4.metric("Cpk", f"{cpk:.3f}")

        m5, m6 = st.columns(2)
        m5.metric("Assembly Tolerance (3s)", f"+/- {assembly_tol:.4f}")
        pct_out = np.mean((assembly_samples < lsl) | (assembly_samples > usl)) * 100
        m6.metric("% Out of Spec", f"{pct_out:.3f}%")

        # Assembly distribution histogram
        fig = px.histogram(x=assembly_samples, nbins=60,
                           title=f"Assembly Distribution ({method})")
        fig.add_vline(x=lsl, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"LSL = {lsl:.4f}")
        fig.add_vline(x=usl, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"USL = {usl:.4f}")
        fig.add_vline(x=assembly_mean, line_dash="dot", line_color="#6366f1",
                      annotation_text=f"Mean = {assembly_mean:.4f}")
        fig.update_layout(xaxis_title="Assembly Dimension", yaxis_title="Frequency", height=420)
        rdl_plotly_chart(fig)

        # Variance contribution pie chart
        section_header("Component Variance Contribution")

        if len(component_samples) > 0:
            variances = [np.var(s) for s in component_samples]
        else:
            variances = [(c["tolerance"] / 3.0) ** 2 for c in components]

        total_var = sum(variances)
        contrib_pct = [(v / total_var * 100) if total_var > 0 else 0 for v in variances]
        comp_names = [c["name"] for c in components]

        fig_pie = px.pie(
            names=comp_names, values=contrib_pct,
            title="Component Contribution to Total Assembly Variance",
        )
        fig_pie.update_traces(textinfo="label+percent", textposition="inside")
        fig_pie.update_layout(height=400)
        rdl_plotly_chart(fig_pie)

        # Contribution table
        contrib_df = pd.DataFrame({
            "Component": comp_names,
            "Nominal": [f"{c['nominal']:.4f}" for c in components],
            "Tolerance (+/-)": [f"{c['tolerance']:.4f}" for c in components],
            "Distribution": [c["distribution"] for c in components],
            "Variance": [f"{v:.6f}" for v in variances],
            "Contribution %": [f"{p:.1f}%" for p in contrib_pct],
        })
        st.dataframe(contrib_df, use_container_width=True, hide_index=True)


# ─── Measurement Uncertainty ─────────────────────────────────────────────────

def _render_measurement_uncertainty():
    section_header("Measurement Uncertainty (GUM)")
    help_tip("Measurement Uncertainty",
             "Evaluate measurement uncertainty per the GUM (Guide to the Expression of "
             "Uncertainty in Measurement). Define uncertainty sources, compute combined "
             "and expanded uncertainty.")

    st.markdown("Define uncertainty sources and compute the combined/expanded measurement uncertainty.")

    measured_value = st.number_input("Measured value (best estimate):", value=100.0,
                                     key="mu_measured", format="%.4f")

    n_sources = st.number_input("Number of uncertainty sources:", min_value=1, max_value=15,
                                value=3, key="mu_nsources")

    sources = []
    for i in range(int(n_sources)):
        with st.expander(f"Source {i + 1}", expanded=(i < 3)):
            sc1, sc2 = st.columns(2)
            src_name = sc1.text_input("Source name:", value=f"Source {i + 1}",
                                      key=f"mu_name_{i}")
            src_type = sc2.selectbox("Type:", ["Type A (Statistical)", "Type B (Other)"],
                                     key=f"mu_type_{i}")

            sc3, sc4, sc5 = st.columns(3)
            src_dist = sc3.selectbox(
                "Distribution:",
                ["Normal", "Rectangular (Uniform)", "Triangular", "U-shaped"],
                key=f"mu_dist_{i}",
            )
            src_value = sc4.number_input("Value (half-width or std dev):", value=1.0,
                                         min_value=0.0001, key=f"mu_val_{i}", format="%.4f")
            sensitivity = sc5.number_input("Sensitivity coefficient (ci):", value=1.0,
                                           key=f"mu_sens_{i}", format="%.4f")

            # Convert value to standard uncertainty based on distribution
            if src_dist == "Normal":
                std_unc = src_value
                divisor = 1.0
            elif src_dist == "Rectangular (Uniform)":
                std_unc = src_value / np.sqrt(3)
                divisor = np.sqrt(3)
            elif src_dist == "Triangular":
                std_unc = src_value / np.sqrt(6)
                divisor = np.sqrt(6)
            elif src_dist == "U-shaped":
                std_unc = src_value / np.sqrt(2)
                divisor = np.sqrt(2)
            else:
                std_unc = src_value
                divisor = 1.0

            sources.append({
                "name": src_name,
                "type": "A" if "A" in src_type else "B",
                "distribution": src_dist,
                "value": src_value,
                "divisor": divisor,
                "std_uncertainty": std_unc,
                "sensitivity": sensitivity,
            })

    st.divider()

    coverage_options = {
        "k = 1 (68.27% confidence)": 1.0,
        "k = 2 (95.45% confidence)": 2.0,
        "k = 3 (99.73% confidence)": 3.0,
    }
    coverage_sel = st.selectbox("Coverage factor:", list(coverage_options.keys()),
                                index=1, key="mu_coverage")
    k = coverage_options[coverage_sel]

    if st.button("Calculate Uncertainty", key="mu_run"):
        # Combined standard uncertainty
        contributions = []
        for src in sources:
            ci = src["sensitivity"]
            ui = src["std_uncertainty"]
            contributions.append((ci * ui) ** 2)

        uc = np.sqrt(sum(contributions))
        U = k * uc

        # Metrics
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Combined Std Uncertainty (uc)", f"{uc:.4f}")
        mc2.metric(f"Expanded Uncertainty (U, k={k:.0f})", f"{U:.4f}")
        mc3.metric("Result", f"{measured_value:.4f} +/- {U:.4f}")

        st.success(
            f"**Measurement result:** {measured_value:.4f} +/- {U:.4f} "
            f"(k = {k:.0f}, {coverage_sel.split('(')[1]}"
        )

        # Uncertainty budget table
        section_header("Uncertainty Budget")

        total_variance = sum(contributions)
        budget_data = []
        for src, contrib in zip(sources, contributions):
            pct = (contrib / total_variance * 100) if total_variance > 0 else 0
            budget_data.append({
                "Source": src["name"],
                "Type": src["type"],
                "Distribution": src["distribution"],
                "Value": f"{src['value']:.4f}",
                "Divisor": f"{src['divisor']:.4f}",
                "Std Uncertainty (ui)": f"{src['std_uncertainty']:.4f}",
                "Sensitivity (ci)": f"{src['sensitivity']:.4f}",
                "ci*ui": f"{src['sensitivity'] * src['std_uncertainty']:.4f}",
                "(ci*ui)^2": f"{contrib:.6f}",
                "Contribution %": f"{pct:.1f}%",
            })
        st.dataframe(pd.DataFrame(budget_data), use_container_width=True, hide_index=True)

        # Contribution bar chart
        section_header("Uncertainty Contributions")

        contrib_names = [src["name"] for src in sources]
        contrib_pcts = [
            (c / total_variance * 100) if total_variance > 0 else 0
            for c in contributions
        ]
        contrib_types = [src["type"] for src in sources]

        fig_bar = px.bar(
            x=contrib_names, y=contrib_pcts, color=contrib_types,
            title="Contribution to Combined Uncertainty",
            labels={"x": "Source", "y": "Contribution %", "color": "Type"},
            color_discrete_map={"A": "#6366f1", "B": "#f59e0b"},
        )
        fig_bar.update_layout(height=400, xaxis_title="Uncertainty Source",
                              yaxis_title="Contribution %")
        rdl_plotly_chart(fig_bar)


# ─── What-If Scenarios ───────────────────────────────────────────────────────

def _render_what_if_scenarios():
    section_header("What-If Scenarios")
    help_tip("What-If Scenarios",
             "Save Process Simulation results and compare multiple scenarios side by side. "
             "Run different configurations in the Process Simulation tab, save each one, "
             "then compare here.")

    # Initialize scenario storage
    if "mc_scenarios" not in st.session_state:
        st.session_state["mc_scenarios"] = []

    scenarios = st.session_state["mc_scenarios"]

    # Save current scenario
    st.markdown("**Save a scenario from the last Process Simulation run:**")
    sc1, sc2 = st.columns([2, 1])
    scenario_name = sc1.text_input("Scenario name:",
                                   value=f"Scenario {len(scenarios) + 1}",
                                   key="wi_name")

    last_output = st.session_state.get("_ps_last_output")
    last_vars = st.session_state.get("_ps_last_variables")
    last_formula = st.session_state.get("_ps_last_formula")
    last_n_sim = st.session_state.get("_ps_last_n_sim")

    if sc2.button("Save Current", key="wi_save"):
        if last_output is None:
            st.warning("No Process Simulation results to save. Run a Process Simulation first.")
        else:
            out_mean = float(np.mean(last_output))
            scenario = {
                "name": scenario_name,
                "output": last_output.copy(),
                "variables": last_vars.copy() if last_vars else {},
                "formula": last_formula or "",
                "n_sim": last_n_sim or 0,
                "mean": out_mean,
                "std": float(np.std(last_output)),
                "cv_pct": (float(np.std(last_output) / abs(out_mean) * 100)
                           if out_mean != 0 else 0.0),
            }
            scenarios.append(scenario)
            st.session_state["mc_scenarios"] = scenarios
            st.success(f"Saved scenario: **{scenario_name}**")
            st.rerun()

    if len(scenarios) == 0:
        empty_state("No scenarios saved yet.",
                    "Run a Process Simulation and click 'Save Current' to begin comparing scenarios.")
        return

    # Clear scenarios
    if st.button("Clear All Scenarios", key="wi_clear"):
        st.session_state["mc_scenarios"] = []
        st.rerun()

    # Comparison table
    section_header("Scenario Comparison")

    threshold = st.number_input("Threshold for P(exceed):", value=0.0, key="wi_thresh")

    comp_data = []
    for sc in scenarios:
        output = sc["output"]
        p_exceed = float(np.mean(output > threshold))
        comp_data.append({
            "Scenario": sc["name"],
            "Formula": sc.get("formula", ""),
            "N Sims": f"{sc.get('n_sim', len(output)):,}",
            "Mean": f"{sc['mean']:.4f}",
            "Std Dev": f"{sc['std']:.4f}",
            "CV%": f"{sc['cv_pct']:.2f}%",
            f"P(X > {threshold})": f"{p_exceed:.4f}",
        })
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

    # Side-by-side histogram overlay (up to 4 scenarios)
    section_header("Distribution Overlay")

    display_scenarios = scenarios[:4]
    if len(scenarios) > 4:
        st.info("Showing first 4 scenarios. Clear older scenarios to compare newer ones.")

    fig_overlay = go.Figure()
    for sc in display_scenarios:
        fig_overlay.add_trace(go.Histogram(
            x=sc["output"], name=sc["name"],
            opacity=0.55, nbinsx=50,
        ))

    fig_overlay.update_layout(
        barmode="overlay",
        title="Scenario Distribution Comparison",
        xaxis_title="Output Value",
        yaxis_title="Frequency",
        height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    if threshold != 0.0:
        fig_overlay.add_vline(x=threshold, line_dash="dash", line_color="#ef4444",
                              annotation_text=f"Threshold = {threshold}")
    rdl_plotly_chart(fig_overlay)

    # Per-scenario detail
    section_header("Individual Scenario Details")
    for idx, sc in enumerate(scenarios):
        with st.expander(f"{sc['name']} - details", expanded=False):
            output = sc["output"]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Mean", f"{sc['mean']:.4f}")
            d2.metric("Std Dev", f"{sc['std']:.4f}")
            d3.metric("Min", f"{np.min(output):.4f}")
            d4.metric("Max", f"{np.max(output):.4f}")
            if sc.get("formula"):
                st.caption(f"Formula: `{sc['formula']}`")
            if sc.get("variables"):
                var_info = []
                for vname, vcfg in sc["variables"].items():
                    var_info.append(
                        f"{vname}: {vcfg['dist']}({', '.join(str(p) for p in vcfg['params'])})"
                    )
                st.caption("Inputs: " + " | ".join(var_info))
