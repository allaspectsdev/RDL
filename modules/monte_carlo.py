"""
Monte Carlo Simulation Module - Distribution simulation, process simulation, and risk analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats

from modules.ui_helpers import section_header, empty_state, help_tip


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
    tabs = st.tabs(["Distribution Simulator", "Process Simulation", "Risk Analysis"])

    with tabs[0]:
        _render_distribution_simulator()
    with tabs[1]:
        _render_process_simulation()
    with tabs[2]:
        _render_risk_analysis()


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

        import plotly.express as px
        import plotly.graph_objects as go

        # Summary metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Mean", f"{np.mean(samples):.4f}")
        c2.metric("Std Dev", f"{np.std(samples):.4f}")
        c3.metric("Median", f"{np.median(samples):.4f}")
        c4.metric("Min", f"{np.min(samples):.4f}")
        c5.metric("Max", f"{np.max(samples):.4f}")

        # Histogram
        fig = px.histogram(x=samples, nbins=50, title=f"{dist_name} Distribution (n={n_sim:,})",
                           color_discrete_sequence=["#6366f1"])
        fig.update_layout(xaxis_title="Value", yaxis_title="Frequency", height=400)
        st.plotly_chart(fig, use_container_width=True)

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
                                  mode="lines", name="CDF",
                                  line=dict(color="#6366f1")))
        fig2.update_layout(title="Cumulative Distribution Function",
                           xaxis_title="Value", yaxis_title="Cumulative Probability",
                           height=350)
        st.plotly_chart(fig2, use_container_width=True)


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

    if st.button("Run Process Simulation", key="ps_run"):
        with st.spinner("Simulating..."):
            np.random.seed(None)
            samples = {}
            for name, cfg in variables.items():
                dist = _DISTRIBUTIONS[cfg["dist"]]
                samples[name] = dist["fn"](int(n_sim), cfg["params"])

            # Evaluate formula safely
            allowed = {name: samples[name] for name in variables}
            allowed.update({"np": np, "sqrt": np.sqrt, "log": np.log,
                            "exp": np.exp, "abs": np.abs, "sin": np.sin,
                            "cos": np.cos, "pi": np.pi})
            try:
                output = eval(formula, {"__builtins__": {}}, allowed)
            except Exception as e:
                st.error(f"Formula error: {e}")
                return

        import plotly.express as px

        # Output distribution
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean Output", f"{np.mean(output):.4f}")
        c2.metric("Std Dev", f"{np.std(output):.4f}")
        c3.metric("CV%", f"{np.std(output) / abs(np.mean(output)) * 100:.2f}%"
                  if np.mean(output) != 0 else "N/A")

        fig = px.histogram(x=output, nbins=50, title="Output Distribution",
                           color_discrete_sequence=["#6366f1"])
        fig.update_layout(xaxis_title="Output Value", yaxis_title="Frequency", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Sensitivity tornado chart
        section_header("Sensitivity Analysis (Tornado Chart)")
        correlations = {}
        for name in variables:
            correlations[name] = np.corrcoef(samples[name], output)[0, 1]

        sens_df = pd.DataFrame({
            "Variable": list(correlations.keys()),
            "Correlation": list(correlations.values()),
        }).sort_values("Correlation", key=abs, ascending=True)

        fig2 = px.bar(sens_df, x="Correlation", y="Variable", orientation="h",
                      title="Sensitivity: Input-Output Correlation",
                      color="Correlation",
                      color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
        fig2.update_layout(height=max(300, len(variables) * 50))
        st.plotly_chart(fig2, use_container_width=True)


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

        import plotly.express as px
        import plotly.graph_objects as go

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
        fig = px.histogram(x=samples, nbins=50, title="Distribution with Threshold",
                           color_discrete_sequence=["#6366f1"])
        fig.add_vline(x=threshold, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Threshold = {threshold}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

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
