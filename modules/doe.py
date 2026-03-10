"""
Design of Experiments (DOE) Module - Design Generation, Analysis, Response Surface.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from itertools import combinations

try:
    import pyDOE2
    HAS_PYDOE = True
except ImportError:
    HAS_PYDOE = False

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_SM = True
except ImportError:
    HAS_SM = False


def render_doe(df: pd.DataFrame):
    """Render design of experiments interface."""
    if df is None or df.empty:
        st.warning("No data loaded.")
        return

    tabs = st.tabs([
        "Design Generation", "Design Analysis", "Response Surface",
    ])

    with tabs[0]:
        _render_design_generation(df)
    with tabs[1]:
        _render_design_analysis(df)
    with tabs[2]:
        _render_response_surface(df)


# ---------------------------------------------------------------------------
# Tab 1: Design Generation
# ---------------------------------------------------------------------------

def _render_design_generation(df: pd.DataFrame):
    """Generate experimental designs."""
    if not HAS_PYDOE:
        st.warning("pyDOE2 required for design generation. Install with: pip install pyDOE2")
        return

    st.markdown("#### Design Parameters")

    n_factors = st.slider("Number of factors:", 2, 8, 3, key="doe_n_factors")

    design_type = st.selectbox("Design type:", [
        "Full Factorial",
        "Fractional Factorial (Resolution III)",
        "Fractional Factorial (Resolution IV)",
        "Fractional Factorial (Resolution V)",
        "Plackett-Burman",
        "Central Composite (CCD)",
        "Box-Behnken",
    ], key="doe_design_type")

    # Factor definitions
    st.markdown("#### Factor Definitions")
    factor_names = []
    factor_lows = []
    factor_highs = []

    for i in range(n_factors):
        c1, c2, c3 = st.columns([2, 1, 1])
        name = c1.text_input(f"Factor {i + 1} name:", value=f"X{i + 1}",
                             key=f"doe_fname_{i}")
        low = c2.number_input(f"Low ({name}):", value=-1.0,
                              key=f"doe_flow_{i}")
        high = c3.number_input(f"High ({name}):", value=1.0,
                               key=f"doe_fhigh_{i}")
        factor_names.append(name)
        factor_lows.append(low)
        factor_highs.append(high)

    if st.button("Generate Design", key="doe_generate"):
        try:
            coded_matrix = _generate_design(design_type, n_factors)
        except Exception as e:
            st.error(f"Design generation failed: {e}")
            return

        if coded_matrix is None:
            return

        n_runs = coded_matrix.shape[0]
        st.success(f"Design generated: **{n_runs} runs** for {n_factors} factors.")

        # Build coded dataframe
        coded_df = pd.DataFrame(coded_matrix, columns=factor_names)
        coded_df.index = range(1, n_runs + 1)
        coded_df.index.name = "Run"

        # Build actual values dataframe
        actual_df = coded_df.copy()
        for i, name in enumerate(factor_names):
            low, high = factor_lows[i], factor_highs[i]
            mid = (low + high) / 2.0
            half_range = (high - low) / 2.0
            actual_df[name] = coded_df[name] * half_range + mid

        # Display coded design
        st.markdown("#### Coded Design Matrix (-1 / +1)")
        st.dataframe(coded_df.round(4), use_container_width=True)

        # Display actual values
        st.markdown("#### Actual Values Design Matrix")
        st.dataframe(actual_df.round(4), use_container_width=True)

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Number of Runs", n_runs)
        c2.metric("Number of Factors", n_factors)
        c3.metric("Design Type", design_type.split("(")[0].strip())

        # Download buttons
        c1, c2 = st.columns(2)
        with c1:
            csv_coded = coded_df.to_csv()
            st.download_button(
                "Download Coded Design (CSV)",
                csv_coded,
                file_name="doe_coded_design.csv",
                mime="text/csv",
                key="doe_dl_coded",
            )
        with c2:
            csv_actual = actual_df.to_csv()
            st.download_button(
                "Download Actual Design (CSV)",
                csv_actual,
                file_name="doe_actual_design.csv",
                mime="text/csv",
                key="doe_dl_actual",
            )

        # Store in session state for Tab 2
        st.session_state["doe_coded_design"] = coded_df
        st.session_state["doe_actual_design"] = actual_df
        st.session_state["doe_factor_names"] = factor_names


def _generate_design(design_type: str, n_factors: int) -> np.ndarray:
    """Generate a design matrix using pyDOE2."""
    if design_type == "Full Factorial":
        return pyDOE2.ff2n(n_factors)

    elif design_type.startswith("Fractional Factorial"):
        # Build generator string based on resolution
        if "Resolution III" in design_type:
            gen = _get_fracfact_generator(n_factors, resolution=3)
        elif "Resolution IV" in design_type:
            gen = _get_fracfact_generator(n_factors, resolution=4)
        elif "Resolution V" in design_type:
            gen = _get_fracfact_generator(n_factors, resolution=5)
        else:
            gen = _get_fracfact_generator(n_factors, resolution=3)

        if gen is None:
            st.warning(
                f"Cannot build a fractional factorial at this resolution for "
                f"{n_factors} factors. Falling back to full factorial."
            )
            return pyDOE2.ff2n(n_factors)
        return pyDOE2.fracfact(gen)

    elif design_type == "Plackett-Burman":
        return pyDOE2.pbdesign(n_factors)

    elif design_type == "Central Composite (CCD)":
        return pyDOE2.ccdesign(n_factors, center=(1, 1))

    elif design_type == "Box-Behnken":
        if n_factors < 3:
            st.warning("Box-Behnken requires at least 3 factors. Using CCD instead.")
            return pyDOE2.ccdesign(n_factors, center=(1, 1))
        return pyDOE2.bbdesign(n_factors, center=1)

    return None


def _get_fracfact_generator(n_factors: int, resolution: int) -> str:
    """Return a fracfact generator string for the given number of factors and resolution.

    Letters a-h map to factors 1-8.  Independent factors are listed first,
    then dependent (generated) columns are expressed as products of
    independent columns.
    """
    # Standard fractional factorial generators
    # Keys: (n_factors, resolution)
    generators = {
        # Resolution III designs
        (3, 3): "a b ab",
        (4, 3): "a b ab ac",
        (5, 3): "a b ab ac bc",
        (6, 3): "a b ab ac bc abc",
        (7, 3): "a b c ab ac bc abc",
        (8, 3): "a b c ab ac bc abc abcd",
        # Resolution IV designs
        (4, 4): "a b c abc",
        (5, 4): "a b c ab abc",
        (6, 4): "a b c ab ac abc",
        (7, 4): "a b c d abc abd acd",
        (8, 4): "a b c d abc abd acd bcd",
        # Resolution V designs
        (5, 5): "a b c d abcd",
        (6, 5): "a b c d abcd abce",
        (7, 5): "a b c d e abcd abce",
        (8, 5): "a b c d e abcd abce abde",
    }

    gen = generators.get((n_factors, resolution))
    if gen is not None:
        return gen

    # For 2 factors, fractional factorial is not meaningful
    if n_factors <= 2:
        return None

    # Fallback: try the closest available resolution
    for r in [resolution, resolution - 1, resolution + 1]:
        gen = generators.get((n_factors, r))
        if gen is not None:
            return gen

    return None


# ---------------------------------------------------------------------------
# Tab 2: Design Analysis
# ---------------------------------------------------------------------------

def _render_design_analysis(df: pd.DataFrame):
    """Analyze a completed DOE dataset."""
    if not HAS_SM:
        st.warning("statsmodels required for design analysis. Install with: pip install statsmodels")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 3:
        st.warning("Need at least 3 numeric columns (2+ factors and 1 response).")
        return

    # Check if a design was generated in Tab 1
    has_generated = "doe_factor_names" in st.session_state
    if has_generated:
        st.info("A design was generated in the Design Generation tab. "
                "You can use your loaded dataset or the generated design.")

    factor_cols = st.multiselect(
        "Factor columns:", num_cols,
        default=st.session_state.get("doe_factor_names", [])[:len(num_cols)],
        key="doe_analysis_factors",
    )
    remaining = [c for c in num_cols if c not in factor_cols]
    if not remaining:
        st.warning("Need at least one column for the response variable.")
        return
    response_col = st.selectbox("Response column:", remaining, key="doe_analysis_response")

    if len(factor_cols) < 2:
        st.info("Select at least 2 factor columns.")
        return

    if st.button("Analyze Design", key="doe_run_analysis"):
        data = df[factor_cols + [response_col]].dropna()

        if len(data) < len(factor_cols) + 2:
            st.error("Not enough data points for analysis.")
            return

        # Build formula: response ~ main effects + 2-factor interactions
        main_terms = [f"Q('{c}')" for c in factor_cols]
        interaction_terms = [f"Q('{a}'):Q('{b}')" for a, b in combinations(factor_cols, 2)]
        all_terms = main_terms + interaction_terms
        formula = f"Q('{response_col}') ~ " + " + ".join(all_terms)

        try:
            model = smf.ols(formula, data=data).fit()
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            return

        # Model fit metrics
        st.markdown("#### Model Fit")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R\u00b2", f"{model.rsquared:.4f}")
        c2.metric("Adj R\u00b2", f"{model.rsquared_adj:.4f}")
        c3.metric("F-statistic", f"{model.fvalue:.4f}")
        c4.metric("p (F-test)", f"{model.f_pvalue:.6f}")

        # ANOVA table
        st.markdown("#### ANOVA Table")
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_table = anova_table.rename(columns={
            "sum_sq": "SS", "df": "df", "F": "F", "PR(>F)": "p-value",
        })
        anova_table["MS"] = anova_table["SS"] / anova_table["df"]
        anova_table = anova_table[["SS", "df", "MS", "F", "p-value"]]
        st.dataframe(anova_table.round(4), use_container_width=True)

        # Coefficients table
        with st.expander("Regression Coefficients"):
            coef_df = pd.DataFrame({
                "Term": model.params.index,
                "Coefficient": model.params.values,
                "Std Error": model.bse.values,
                "t-value": model.tvalues.values,
                "p-value": model.pvalues.values,
            }).round(6)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

        # Pareto chart of effects
        st.markdown("#### Pareto Chart of Effects")
        _plot_pareto_chart(model)

        # Main effects plot
        st.markdown("#### Main Effects Plot")
        _plot_main_effects(data, factor_cols, response_col)

        # Interaction plots
        if len(factor_cols) >= 2:
            st.markdown("#### Interaction Plots")
            _plot_interactions(data, factor_cols, response_col)

        # Half-normal plot
        st.markdown("#### Half-Normal Plot of Effects")
        _plot_half_normal(model)


def _plot_pareto_chart(model):
    """Pareto chart of standardized effects (absolute t-values)."""
    terms = model.params.index[1:]  # Skip intercept
    t_vals = np.abs(model.tvalues.values[1:])
    p_vals = model.pvalues.values[1:]

    # Sort by absolute t-value
    order = np.argsort(t_vals)
    terms_sorted = [terms[i] for i in order]
    t_sorted = t_vals[order]
    p_sorted = p_vals[order]

    # Clean up term names for display
    display_names = []
    for t in terms_sorted:
        name = t.replace("Q('", "").replace("')", "").replace(":", " x ")
        display_names.append(name)

    colors = ["#EF553B" if p < 0.05 else "#636EFA" for p in p_sorted]

    # Significance threshold (t-value for alpha=0.05)
    dof = model.df_resid
    t_crit = stats.t.ppf(0.975, dof) if dof > 0 else 2.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=display_names, x=t_sorted, orientation="h",
        marker_color=colors,
        name="Effects",
    ))
    fig.add_vline(x=t_crit, line_dash="dash", line_color="red",
                  annotation_text=f"t-critical = {t_crit:.2f}")
    fig.update_layout(
        title="Pareto Chart of Standardized Effects",
        xaxis_title="|t-value|",
        yaxis_title="Term",
        height=max(350, len(terms) * 35),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _plot_main_effects(data: pd.DataFrame, factor_cols: list, response_col: str):
    """Main effects plot: mean response at each level of each factor."""
    n_factors = len(factor_cols)
    cols_per_row = min(n_factors, 4)
    rows = (n_factors + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=[f"Main Effect: {c}" for c in factor_cols],
    )

    grand_mean = data[response_col].mean()

    for idx, col in enumerate(factor_cols):
        row = idx // cols_per_row + 1
        c = idx % cols_per_row + 1

        # Bin continuous factors into low/high using median split
        values = data[col].values
        unique_vals = np.unique(values)

        if len(unique_vals) <= 5:
            # Use actual levels
            levels = sorted(unique_vals)
            means = [data.loc[data[col] == lv, response_col].mean() for lv in levels]
        else:
            # Bin into low/mid/high by tertiles
            q33 = np.percentile(values, 33)
            q66 = np.percentile(values, 66)
            low_mean = data.loc[values <= q33, response_col].mean()
            mid_mean = data.loc[(values > q33) & (values <= q66), response_col].mean()
            high_mean = data.loc[values > q66, response_col].mean()
            levels = ["Low", "Mid", "High"]
            means = [low_mean, mid_mean, high_mean]

        fig.add_trace(go.Scatter(
            x=[str(lv) for lv in levels], y=means,
            mode="lines+markers",
            marker=dict(size=10, color="steelblue"),
            line=dict(color="steelblue", width=2),
            showlegend=False,
        ), row=row, col=c)

        # Grand mean reference line
        fig.add_hline(y=grand_mean, line_dash="dash", line_color="gray",
                      row=row, col=c)

    fig.update_layout(height=300 * rows, title_text="Main Effects Plot")
    st.plotly_chart(fig, use_container_width=True)


def _plot_interactions(data: pd.DataFrame, factor_cols: list, response_col: str):
    """Interaction plots for all pairs of factors."""
    pairs = list(combinations(factor_cols, 2))
    n_pairs = len(pairs)
    cols_per_row = min(n_pairs, 2)
    rows = (n_pairs + cols_per_row - 1) // cols_per_row

    fig = make_subplots(
        rows=rows, cols=cols_per_row,
        subplot_titles=[f"{a} x {b}" for a, b in pairs],
    )

    for idx, (fa, fb) in enumerate(pairs):
        row = idx // cols_per_row + 1
        c = idx % cols_per_row + 1

        # Discretize factors for interaction plot
        a_vals = data[fa].values
        b_vals = data[fb].values
        a_med = np.median(a_vals)
        b_med = np.median(b_vals)

        a_levels = np.where(a_vals <= a_med, "Low", "High")
        b_levels = np.where(b_vals <= b_med, "Low", "High")

        temp = pd.DataFrame({
            "A": a_levels, "B": b_levels, "Y": data[response_col].values
        })
        means = temp.groupby(["A", "B"])["Y"].mean().reset_index()

        for b_label, color in [("Low", "steelblue"), ("High", "#EF553B")]:
            sub = means[means["B"] == b_label]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["A"].values, y=sub["Y"].values,
                    mode="lines+markers",
                    marker=dict(size=8, color=color),
                    line=dict(color=color, width=2),
                    name=f"{fb}={b_label}",
                    showlegend=(idx == 0),
                ), row=row, col=c)

        fig.update_xaxes(title_text=fa, row=row, col=c)
        fig.update_yaxes(title_text=response_col, row=row, col=c)

    fig.update_layout(height=350 * rows, title_text="Interaction Plots")
    st.plotly_chart(fig, use_container_width=True)


def _plot_half_normal(model):
    """Half-normal probability plot of effects."""
    effects = np.abs(model.params.values[1:])  # Skip intercept
    terms = model.params.index[1:]

    # Sort effects
    order = np.argsort(effects)
    effects_sorted = effects[order]
    terms_sorted = [terms[i] for i in order]

    # Clean up term names
    display_names = []
    for t in terms_sorted:
        name = t.replace("Q('", "").replace("')", "").replace(":", " x ")
        display_names.append(name)

    n = len(effects_sorted)
    # Half-normal quantiles
    p_i = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.halfnorm.ppf(p_i)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=theoretical, y=effects_sorted,
        mode="markers+text",
        text=display_names,
        textposition="top right",
        textfont=dict(size=9),
        marker=dict(size=8, color="steelblue"),
        name="Effects",
    ))

    # Reference line through origin
    if n > 1 and theoretical[-1] > 0:
        # Fit line through the lower-half points (likely inactive effects)
        mid = max(n // 2, 2)
        slope, intercept, _, _, _ = stats.linregress(
            theoretical[:mid], effects_sorted[:mid]
        )
        x_line = np.array([0, theoretical[-1] * 1.1])
        fig.add_trace(go.Scatter(
            x=x_line, y=intercept + slope * x_line,
            mode="lines", line=dict(color="red", dash="dash"),
            name="Reference Line",
        ))

    fig.update_layout(
        title="Half-Normal Plot of Effects",
        xaxis_title="Half-Normal Quantiles",
        yaxis_title="|Effect|",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Response Surface
# ---------------------------------------------------------------------------

def _render_response_surface(df: pd.DataFrame):
    """Response surface methodology with 3D surface and contour plots."""
    if not HAS_SM:
        st.warning("statsmodels required for response surface analysis.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 3:
        st.warning("Need at least 3 numeric columns (2 factors + 1 response).")
        return

    c1, c2 = st.columns(2)
    x1_col = c1.selectbox("X1 (factor 1):", num_cols, key="doe_rsm_x1")
    x2_col = c2.selectbox("X2 (factor 2):",
                           [c for c in num_cols if c != x1_col],
                           key="doe_rsm_x2")
    remaining = [c for c in num_cols if c not in [x1_col, x2_col]]
    response_col = st.selectbox("Response:", remaining, key="doe_rsm_response")

    # Additional factors to hold constant
    other_factors = [c for c in num_cols if c not in [x1_col, x2_col, response_col]]
    hold_values = {}
    if other_factors:
        with st.expander("Hold other factors constant"):
            st.markdown("Set values for factors not included in the surface plot.")
            for col in other_factors:
                val = st.number_input(
                    f"{col}:", value=float(df[col].median()),
                    key=f"doe_rsm_hold_{col}",
                )
                hold_values[col] = val

    optimize_goal = st.selectbox("Optimization goal:",
                                  ["Maximize Response", "Minimize Response"],
                                  key="doe_rsm_goal")

    if st.button("Fit Response Surface", key="doe_rsm_fit"):
        data = df[num_cols].dropna()
        if len(data) < 6:
            st.error("Need at least 6 data points for quadratic model.")
            return

        x1 = data[x1_col].values
        x2 = data[x2_col].values
        y = data[response_col].values

        # Build quadratic model: y = b0 + b1*x1 + b2*x2 + b11*x1^2 + b22*x2^2 + b12*x1*x2
        X_design = np.column_stack([
            np.ones(len(x1)),
            x1, x2,
            x1 ** 2, x2 ** 2,
            x1 * x2,
        ])

        try:
            model = sm.OLS(y, X_design).fit()
        except Exception as e:
            st.error(f"Model fitting failed: {e}")
            return

        b = model.params
        st.markdown("#### Quadratic Model Fit")
        c1m, c2m, c3m = st.columns(3)
        c1m.metric("R\u00b2", f"{model.rsquared:.4f}")
        c2m.metric("Adj R\u00b2", f"{model.rsquared_adj:.4f}")
        c3m.metric("RMSE", f"{np.sqrt(model.mse_resid):.4f}")

        # Coefficient table
        coef_names = ["Intercept", x1_col, x2_col,
                      f"{x1_col}\u00b2", f"{x2_col}\u00b2",
                      f"{x1_col}:{x2_col}"]
        coef_df = pd.DataFrame({
            "Term": coef_names,
            "Coefficient": model.params,
            "Std Error": model.bse,
            "t-value": model.tvalues,
            "p-value": model.pvalues,
        }).round(6)
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        st.markdown(
            f"**Equation:** y = {b[0]:.4f} + {b[1]:.4f}\u00b7{x1_col} + "
            f"{b[2]:.4f}\u00b7{x2_col} + {b[3]:.4f}\u00b7{x1_col}\u00b2 + "
            f"{b[4]:.4f}\u00b7{x2_col}\u00b2 + {b[5]:.4f}\u00b7{x1_col}\u00b7{x2_col}"
        )

        # Generate surface grid
        x1_range = np.linspace(x1.min(), x1.max(), 50)
        x2_range = np.linspace(x2.min(), x2.max(), 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        z_grid = (b[0] + b[1] * x1_grid + b[2] * x2_grid +
                  b[3] * x1_grid ** 2 + b[4] * x2_grid ** 2 +
                  b[5] * x1_grid * x2_grid)

        # 3D Surface plot
        st.markdown("#### 3D Response Surface")
        fig_3d = go.Figure()
        fig_3d.add_trace(go.Surface(
            x=x1_range, y=x2_range, z=z_grid,
            colorscale="Viridis", opacity=0.85,
            name="Surface",
        ))
        fig_3d.add_trace(go.Scatter3d(
            x=x1, y=x2, z=y,
            mode="markers",
            marker=dict(size=4, color="red", opacity=0.8),
            name="Data Points",
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title=x1_col,
                yaxis_title=x2_col,
                zaxis_title=response_col,
            ),
            title="Response Surface",
            height=600,
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        # 2D Contour plot
        st.markdown("#### Contour Plot")
        fig_contour = go.Figure()
        fig_contour.add_trace(go.Contour(
            x=x1_range, y=x2_range, z=z_grid,
            colorscale="Viridis",
            contours=dict(showlabels=True),
            name="Response",
        ))
        fig_contour.add_trace(go.Scatter(
            x=x1, y=x2, mode="markers",
            marker=dict(size=6, color="red", line=dict(width=1, color="white")),
            name="Data Points",
        ))
        fig_contour.update_layout(
            xaxis_title=x1_col,
            yaxis_title=x2_col,
            title=f"Contour Plot: {response_col}",
            height=500,
        )
        st.plotly_chart(fig_contour, use_container_width=True)

        # Find optimal factor settings
        st.markdown("#### Optimal Factor Settings")
        _find_optimum(b, x1_col, x2_col, response_col, x1, x2,
                      optimize_goal, hold_values, other_factors)


def _find_optimum(b, x1_col, x2_col, response_col, x1, x2,
                  optimize_goal, hold_values, other_factors):
    """Find the optimal factor settings from the quadratic model."""
    # Grid search for the optimum within the data range
    x1_range = np.linspace(x1.min(), x1.max(), 200)
    x2_range = np.linspace(x2.min(), x2.max(), 200)
    x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
    z_grid = (b[0] + b[1] * x1_grid + b[2] * x2_grid +
              b[3] * x1_grid ** 2 + b[4] * x2_grid ** 2 +
              b[5] * x1_grid * x2_grid)

    if optimize_goal == "Maximize Response":
        opt_idx = np.unravel_index(np.argmax(z_grid), z_grid.shape)
    else:
        opt_idx = np.unravel_index(np.argmin(z_grid), z_grid.shape)

    opt_x1 = x1_grid[opt_idx]
    opt_x2 = x2_grid[opt_idx]
    opt_y = z_grid[opt_idx]

    # Also try analytical solution (stationary point)
    # For y = b0 + b1*x1 + b2*x2 + b11*x1^2 + b22*x2^2 + b12*x1*x2
    # dy/dx1 = b1 + 2*b11*x1 + b12*x2 = 0
    # dy/dx2 = b2 + 2*b22*x2 + b12*x1 = 0
    # Solve: [[2*b11, b12], [b12, 2*b22]] @ [x1, x2] = [-b1, -b2]
    try:
        H = np.array([[2 * b[3], b[5]], [b[5], 2 * b[4]]])
        grad = np.array([-b[1], -b[2]])
        stationary = np.linalg.solve(H, grad)
        s_x1, s_x2 = stationary

        # Check if within data range
        if (x1.min() <= s_x1 <= x1.max() and x2.min() <= s_x2 <= x2.max()):
            s_y = (b[0] + b[1] * s_x1 + b[2] * s_x2 +
                   b[3] * s_x1 ** 2 + b[4] * s_x2 ** 2 + b[5] * s_x1 * s_x2)

            # Determine nature of stationary point
            eigvals = np.linalg.eigvals(H)
            if np.all(eigvals > 0):
                nature = "Minimum"
            elif np.all(eigvals < 0):
                nature = "Maximum"
            else:
                nature = "Saddle Point"

            with st.expander("Stationary Point (Analytical)"):
                st.write(f"**Nature:** {nature}")
                st.write(f"**{x1_col}:** {s_x1:.4f}")
                st.write(f"**{x2_col}:** {s_x2:.4f}")
                st.write(f"**Predicted {response_col}:** {s_y:.4f}")

            # Use stationary point if it matches the goal
            if (optimize_goal == "Maximize Response" and nature == "Maximum"):
                opt_x1, opt_x2, opt_y = s_x1, s_x2, s_y
            elif (optimize_goal == "Minimize Response" and nature == "Minimum"):
                opt_x1, opt_x2, opt_y = s_x1, s_x2, s_y
    except np.linalg.LinAlgError:
        pass  # Singular matrix, use grid search result

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Optimal {x1_col}", f"{opt_x1:.4f}")
    c2.metric(f"Optimal {x2_col}", f"{opt_x2:.4f}")
    c3.metric(f"Predicted {response_col}", f"{opt_y:.4f}")

    if hold_values:
        st.markdown("**Other factors held at:**")
        for col, val in hold_values.items():
            st.write(f"- {col} = {val:.4f}")
