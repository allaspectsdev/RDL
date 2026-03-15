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
from modules.ui_helpers import section_header, empty_state, help_tip, validation_panel, interpretation_card
from modules.validation import check_residual_normality, check_homoscedasticity, interpret_r_squared

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
        empty_state("No data loaded.", "Upload a dataset from the sidebar to begin.")
        return

    tabs = st.tabs([
        "Design Generation", "Design Analysis", "Response Surface",
        "Augment Design", "Desirability",
    ])

    with tabs[0]:
        _render_design_generation(df)
    with tabs[1]:
        _render_design_analysis(df)
    with tabs[2]:
        _render_response_surface(df)
    with tabs[3]:
        _render_augment_design(df)
    with tabs[4]:
        _render_desirability(df)


# ---------------------------------------------------------------------------
# Tab 1: Design Generation
# ---------------------------------------------------------------------------

def _render_design_generation(df: pd.DataFrame):
    """Generate experimental designs."""
    if not HAS_PYDOE:
        st.warning("pyDOE2 required for design generation. Install with: pip install pyDOE2")
        return

    section_header("Design Parameters")

    n_factors = st.slider("Number of factors:", 2, 8, 3, key="doe_n_factors")

    design_type = st.selectbox("Design type:", [
        "Full Factorial",
        "Fractional Factorial (Resolution III)",
        "Fractional Factorial (Resolution IV)",
        "Fractional Factorial (Resolution V)",
        "Plackett-Burman",
        "Central Composite (CCD)",
        "Box-Behnken",
        "Definitive Screening (DSD)",
        "D-optimal",
        "I-optimal",
        "Mixture Design",
    ], key="doe_design_type")

    # Factor definitions
    section_header("Factor Definitions")
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

    if design_type == "Mixture Design":
        st.info("For mixture designs, factors represent component proportions that sum to 1. "
                "Low/High values define the feasible range for each component.")
        mixture_type = st.selectbox("Mixture design type:", [
            "Simplex Lattice", "Simplex Centroid", "Extreme Vertices",
        ], key="doe_mixture_type")
        st.session_state["doe_mixture_type"] = mixture_type

    if design_type in ("D-optimal", "I-optimal"):
        c1, c2 = st.columns(2)
        n_runs = c1.number_input("Number of runs:", n_factors + 1, 100,
                                  2 * n_factors + 1, key="doe_d_opt_runs_input")
        model_type = c2.selectbox("Model type:", [
            "main", "main+interactions", "quadratic",
        ], key="doe_d_opt_model_input")
        st.session_state["doe_d_opt_runs"] = n_runs
        st.session_state["doe_d_opt_model"] = model_type

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
        is_mixture = st.session_state.pop("doe_is_mixture", False)
        if is_mixture:
            actual_df = coded_df.copy()
            st.info("Mixture design: values represent component proportions (sum to 1).")
        else:
            actual_df = coded_df.copy()
            for i, name in enumerate(factor_names):
                low, high = factor_lows[i], factor_highs[i]
                mid = (low + high) / 2.0
                half_range = (high - low) / 2.0
                actual_df[name] = coded_df[name] * half_range + mid

        # Display coded design
        section_header("Coded Design Matrix (-1 / +1)")
        st.dataframe(coded_df.round(4), use_container_width=True)

        # Display actual values
        section_header("Actual Values Design Matrix")
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

        # Ternary plot for 3-component mixtures
        if is_mixture and n_factors == 3:
            section_header("Ternary Plot")
            fig_tern = go.Figure(go.Scatterternary(
                a=actual_df[factor_names[0]],
                b=actual_df[factor_names[1]],
                c=actual_df[factor_names[2]],
                mode="markers",
                marker=dict(size=10, color="#6366f1"),
                text=[f"Run {i+1}" for i in range(n_runs)],
            ))
            fig_tern.update_layout(
                title="Mixture Design Points",
                ternary=dict(
                    aaxis=dict(title=factor_names[0]),
                    baxis=dict(title=factor_names[1]),
                    caxis=dict(title=factor_names[2]),
                ),
                height=500,
            )
            st.plotly_chart(fig_tern, use_container_width=True)

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

    elif design_type == "Definitive Screening (DSD)":
        return _generate_dsd(n_factors)

    elif design_type == "D-optimal":
        return _generate_d_optimal(n_factors)

    elif design_type == "I-optimal":
        return _generate_i_optimal(n_factors)

    elif design_type == "Mixture Design":
        return _generate_mixture_design(n_factors)

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


def _generate_dsd(n_factors):
    """Definitive Screening Design (Jones & Nachtsheim 2011).

    Generates 2k+1 runs via conference matrix construction.
    """
    k = n_factors
    n_runs = 2 * k + 1

    # Build conference matrix for k factors
    # Start with identity-like pattern
    design = np.zeros((n_runs, k))

    # First k rows: fold-over pairs
    for i in range(k):
        # Row 2i: factor i at +1, others follow a specific pattern
        # Row 2i+1: factor i at -1, others follow the negated pattern
        design[2 * i, i] = 1
        design[2 * i + 1, i] = -1
        for j in range(k):
            if j != i:
                # Balanced assignment
                if (i + j) % 2 == 0:
                    design[2 * i, j] = 1
                    design[2 * i + 1, j] = -1
                else:
                    design[2 * i, j] = -1
                    design[2 * i + 1, j] = 1

    # Last row: center point (all zeros)
    design[n_runs - 1, :] = 0

    return design


def _generate_d_optimal(n_factors):
    """D-optimal design via coordinate exchange algorithm."""
    import streamlit as st

    n_runs = st.session_state.get("doe_d_opt_runs", 2 * n_factors + 1)
    model_type = st.session_state.get("doe_d_opt_model", "main+interactions")
    n_restarts = 10

    # Build candidate model matrix function
    def model_matrix(X):
        n, k = X.shape
        cols = [np.ones(n)]  # Intercept
        for j in range(k):
            cols.append(X[:, j])  # Main effects
        if model_type in ("main+interactions", "quadratic"):
            for j1 in range(k):
                for j2 in range(j1 + 1, k):
                    cols.append(X[:, j1] * X[:, j2])
        if model_type == "quadratic":
            for j in range(k):
                cols.append(X[:, j] ** 2)
        return np.column_stack(cols)

    best_det = -np.inf
    best_design = None

    for _ in range(n_restarts):
        # Random starting design
        X = np.random.choice([-1, 0, 1], size=(n_runs, n_factors))

        # Coordinate exchange
        improved = True
        while improved:
            improved = False
            for i in range(n_runs):
                for j in range(n_factors):
                    current_val = X[i, j]
                    best_val = current_val
                    M = model_matrix(X)
                    try:
                        current_det = np.linalg.det(M.T @ M)
                    except Exception:
                        current_det = 0

                    for candidate in [-1, 0, 1]:
                        if candidate == current_val:
                            continue
                        X[i, j] = candidate
                        M = model_matrix(X)
                        try:
                            new_det = np.linalg.det(M.T @ M)
                        except Exception:
                            new_det = 0
                        if new_det > current_det:
                            current_det = new_det
                            best_val = candidate
                            improved = True
                    X[i, j] = best_val

        M = model_matrix(X)
        try:
            det_val = np.linalg.det(M.T @ M)
        except Exception:
            det_val = 0

        if det_val > best_det:
            best_det = det_val
            best_design = X.copy()

    if best_design is not None:
        # Show D-efficiency
        M = model_matrix(best_design)
        p = M.shape[1]
        d_eff = (np.linalg.det(M.T @ M) ** (1.0 / p)) / n_runs * 100 if best_det > 0 else 0
        st.write(f"**D-efficiency:** {d_eff:.1f}%")
        return best_design

    # Fallback to random design
    return np.random.choice([-1, 0, 1], size=(n_runs, n_factors))


def _generate_i_optimal(n_factors):
    """I-optimal design: minimize average prediction variance."""
    import streamlit as st

    n_runs = st.session_state.get("doe_d_opt_runs", 2 * n_factors + 1)
    n_restarts = 10

    def model_matrix(X):
        n, k = X.shape
        cols = [np.ones(n)]
        for j in range(k):
            cols.append(X[:, j])
        for j1 in range(k):
            for j2 in range(j1 + 1, k):
                cols.append(X[:, j1] * X[:, j2])
        return np.column_stack(cols)

    # Generate candidate points for evaluating prediction variance
    n_eval = 100
    eval_points = np.random.uniform(-1, 1, (n_eval, n_factors))
    F_eval = model_matrix(eval_points)

    best_apv = np.inf
    best_design = None

    for _ in range(n_restarts):
        X = np.random.choice([-1, 0, 1], size=(n_runs, n_factors))

        improved = True
        while improved:
            improved = False
            for i in range(n_runs):
                for j in range(n_factors):
                    current_val = X[i, j]
                    best_val = current_val
                    M = model_matrix(X)
                    try:
                        MtM_inv = np.linalg.inv(M.T @ M)
                        current_apv = np.mean(np.sum((F_eval @ MtM_inv) * F_eval, axis=1))
                    except Exception:
                        current_apv = np.inf

                    for candidate in [-1, 0, 1]:
                        if candidate == current_val:
                            continue
                        X[i, j] = candidate
                        M = model_matrix(X)
                        try:
                            MtM_inv = np.linalg.inv(M.T @ M)
                            new_apv = np.mean(np.sum((F_eval @ MtM_inv) * F_eval, axis=1))
                        except Exception:
                            new_apv = np.inf
                        if new_apv < current_apv:
                            current_apv = new_apv
                            best_val = candidate
                            improved = True
                    X[i, j] = best_val

        M = model_matrix(X)
        try:
            MtM_inv = np.linalg.inv(M.T @ M)
            apv = np.mean(np.sum((F_eval @ MtM_inv) * F_eval, axis=1))
        except Exception:
            apv = np.inf

        if apv < best_apv:
            best_apv = apv
            best_design = X.copy()

    if best_design is not None:
        st.write(f"**Average prediction variance:** {best_apv:.4f}")
        return best_design

    return np.random.choice([-1, 0, 1], size=(n_runs, n_factors))


def _generate_mixture_design(n_factors):
    """Generate mixture design (components sum to 1)."""
    import streamlit as st
    from itertools import combinations_with_replacement

    mixture_type = st.session_state.get("doe_mixture_type", "Simplex Lattice")
    k = n_factors

    if mixture_type == "Simplex Lattice":
        # Degree 2 simplex lattice: all permutations of (1,0,...,0), (0.5,0.5,0,...,0), etc.
        degree = 2
        points = set()
        # Generate all compositions of degree into k parts
        def compositions(n, k):
            if k == 1:
                yield (n,)
            else:
                for i in range(n + 1):
                    for c in compositions(n - i, k - 1):
                        yield (i,) + c

        for comp in compositions(degree, k):
            points.add(tuple(c / degree for c in comp))

        design = np.array(list(points))

    elif mixture_type == "Simplex Centroid":
        # All subsets of components at equal proportions
        points = []
        for r in range(1, k + 1):
            for combo in combinations(range(k), r):
                point = np.zeros(k)
                for idx in combo:
                    point[idx] = 1.0 / r
                points.append(point)
        design = np.array(points)

    elif mixture_type == "Extreme Vertices":
        # For constrained mixtures, generate vertices of the feasible region
        # Simple case: evenly spaced grid on simplex
        n_grid = 3  # Points per edge
        points = set()

        def simplex_grid(k, n_grid):
            if k == 1:
                yield (1.0,)
            else:
                for i in range(n_grid + 1):
                    for rest in simplex_grid(k - 1, n_grid - i):
                        yield (i / n_grid,) + rest

        for pt in simplex_grid(k, n_grid):
            if abs(sum(pt) - 1.0) < 1e-10:
                points.add(pt)

        design = np.array(list(points))

    else:
        design = np.eye(k)

    # For mixture designs, the values are already proportions (0-1), not coded -1/+1
    # Store a flag so the actual-values conversion handles it correctly
    st.session_state["doe_is_mixture"] = True

    return design


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
        empty_state("Need at least 3 numeric columns (2+ factors and 1 response).")
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
        empty_state("Need at least one column for the response variable.")
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

        with st.spinner("Analyzing design..."):
            try:
                model = smf.ols(formula, data=data).fit()
            except Exception as e:
                st.error(f"Model fitting failed: {e}")
                return

        # Model fit metrics
        section_header("Model Fit")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R\u00b2", f"{model.rsquared:.4f}")
        c2.metric("Adj R\u00b2", f"{model.rsquared_adj:.4f}")
        c3.metric("F-statistic", f"{model.fvalue:.4f}")
        c4.metric("p (F-test)", f"{model.f_pvalue:.6f}")

        # R-squared interpretation card
        try:
            interpretation_card(interpret_r_squared(model.rsquared, model.rsquared_adj))
        except Exception:
            pass

        # Model diagnostics: residual checks
        try:
            residuals = model.resid.values
            X_vals = model.model.exog
            diag_checks = [check_residual_normality(residuals)]
            try:
                diag_checks.append(check_homoscedasticity(residuals, X_vals))
            except Exception:
                pass
            validation_panel(diag_checks, title="Model Diagnostics")
        except Exception:
            pass

        # ANOVA table
        section_header("ANOVA Table")
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
        section_header("Pareto Chart of Effects")
        _plot_pareto_chart(model)

        # Main effects plot
        section_header("Main Effects Plot")
        _plot_main_effects(data, factor_cols, response_col)

        # Interaction plots
        if len(factor_cols) >= 2:
            section_header("Interaction Plots")
            _plot_interactions(data, factor_cols, response_col)

        # Half-normal plot
        section_header("Half-Normal Plot of Effects")
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
            marker=dict(size=10),
            line=dict(width=2),
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

        for b_label, color in [("Low", "#6366f1"), ("High", "#EF553B")]:
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
        marker=dict(size=8),
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
        empty_state("Need at least 3 numeric columns (2 factors + 1 response).")
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
        section_header("Quadratic Model Fit")
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
        section_header("3D Response Surface")
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
        section_header("Contour Plot")
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
        section_header("Optimal Factor Settings")
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


# ---------------------------------------------------------------------------
# Tab 4: Augment Design
# ---------------------------------------------------------------------------

def _render_augment_design(df: pd.DataFrame):
    """Augment an existing experimental design."""
    if not HAS_PYDOE:
        st.warning("pyDOE2 required for design augmentation.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        empty_state("Need a design loaded with at least 2 numeric factor columns.")
        return

    section_header("Augment Existing Design")
    help_tip("Design Augmentation", """
Extend an existing experiment by adding runs:
- **Center Points:** Add runs at the midpoint of all factors (tests for curvature)
- **Axial Points:** Add star points to convert factorial to CCD
- **Fold-over:** Add the mirror image of the design (resolves aliasing)
- **Replicate:** Duplicate existing runs (estimates pure error)
""")

    factor_cols = st.multiselect("Factor columns (from existing design):", num_cols,
                                  key="aug_factors")
    if len(factor_cols) < 2:
        st.info("Select at least 2 factor columns.")
        return

    aug_type = st.selectbox("Augmentation type:", [
        "Add Center Points",
        "Add Axial Points (Star Points)",
        "Add Fold-over",
        "Add Replicates",
    ], key="aug_type")

    if aug_type == "Add Center Points":
        n_center = st.number_input("Number of center points:", 1, 10, 3, key="aug_n_center")
    elif aug_type == "Add Axial Points (Star Points)":
        alpha_star = st.number_input("Axial distance (\u03b1):", 0.5, 3.0, 1.414, 0.001,
                                      format="%.3f", key="aug_alpha_star")
    elif aug_type == "Add Replicates":
        n_reps = st.number_input("Number of replicates:", 1, 5, 1, key="aug_n_reps")

    if st.button("Augment Design", key="aug_run"):
        existing = df[factor_cols].dropna()
        k = len(factor_cols)

        if aug_type == "Add Center Points":
            center = np.zeros((n_center, k))
            # Use column means as center
            for j, col in enumerate(factor_cols):
                center[:, j] = existing[col].mean()
            new_runs = pd.DataFrame(center, columns=factor_cols)

        elif aug_type == "Add Axial Points (Star Points)":
            axial = np.zeros((2 * k, k))
            for j, col in enumerate(factor_cols):
                col_mean = existing[col].mean()
                col_range = (existing[col].max() - existing[col].min()) / 2
                axial[2 * j, j] = col_mean + alpha_star * col_range
                axial[2 * j + 1, j] = col_mean - alpha_star * col_range
                # Other columns at center
                for j2 in range(k):
                    if j2 != j:
                        axial[2 * j, j2] = existing[factor_cols[j2]].mean()
                        axial[2 * j + 1, j2] = existing[factor_cols[j2]].mean()
            new_runs = pd.DataFrame(axial, columns=factor_cols)

        elif aug_type == "Add Fold-over":
            # Mirror the design around the center
            center = existing.mean()
            folded = 2 * center - existing
            new_runs = folded.reset_index(drop=True)

        elif aug_type == "Add Replicates":
            new_runs = pd.concat([existing] * n_reps, ignore_index=True)

        augmented = pd.concat([existing.reset_index(drop=True), new_runs.reset_index(drop=True)],
                              ignore_index=True)
        augmented.index = range(1, len(augmented) + 1)
        augmented.index.name = "Run"

        section_header("Augmented Design")
        c1, c2, c3 = st.columns(3)
        c1.metric("Original Runs", len(existing))
        c2.metric("New Runs", len(new_runs))
        c3.metric("Total Runs", len(augmented))

        st.dataframe(augmented.round(4), use_container_width=True)

        csv_data = augmented.to_csv()
        st.download_button("Download Augmented Design (CSV)", csv_data,
                           file_name="augmented_design.csv", mime="text/csv",
                           key="aug_dl")

        st.session_state["doe_coded_design"] = augmented
        st.session_state["doe_actual_design"] = augmented
        st.session_state["doe_factor_names"] = factor_cols


# ---------------------------------------------------------------------------
# Tab 5: Desirability Functions
# ---------------------------------------------------------------------------

def _render_desirability(df: pd.DataFrame):
    """Multi-response optimization via desirability functions."""
    if not HAS_SM:
        st.warning("statsmodels required for desirability optimization.")
        return

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 3:
        empty_state("Need at least 3 numeric columns (factors + responses).")
        return

    section_header("Multi-Response Optimization")
    help_tip("Desirability Functions", """
Simultaneously optimize multiple response variables:
- Each response gets an individual desirability score (0=unacceptable, 1=ideal)
- **Maximize:** d = ((y-L)/(T-L))^s when L \u2264 y \u2264 T, else 0 or 1
- **Minimize:** d = ((U-y)/(U-T))^s when T \u2264 y \u2264 U, else 0 or 1
- **Target:** d = ((y-L)/(T-L))^s when L \u2264 y \u2264 T, ((U-y)/(U-T))^s when T \u2264 y \u2264 U
- Overall desirability D = (\u220f d\u1d62^w\u1d62)^(1/\u03a3w\u1d62)
""")

    factor_cols = st.multiselect("Factor columns:", num_cols, key="des_factors")
    remaining = [c for c in num_cols if c not in factor_cols]
    response_cols = st.multiselect("Response columns:", remaining, key="des_responses")

    if len(factor_cols) < 1 or len(response_cols) < 1:
        st.info("Select at least 1 factor and 1 response column.")
        return

    # Response settings
    section_header("Response Settings")
    response_settings = []
    for resp in response_cols:
        with st.expander(f"Settings for: {resp}"):
            data_vals = df[resp].dropna()
            c1, c2 = st.columns(2)
            goal = c1.selectbox("Goal:", ["Maximize", "Minimize", "Target"],
                                key=f"des_goal_{resp}")
            weight = c2.slider("Weight:", 0.1, 10.0, 1.0, 0.1, key=f"des_weight_{resp}")

            c1, c2, c3 = st.columns(3)
            lower = c1.number_input("Lower bound:", value=float(data_vals.min()),
                                     key=f"des_lower_{resp}")
            target_val = c2.number_input("Target:", value=float(data_vals.mean()),
                                          key=f"des_target_{resp}")
            upper = c3.number_input("Upper bound:", value=float(data_vals.max()),
                                     key=f"des_upper_{resp}")
            shape = st.slider("Shape (s):", 0.1, 10.0, 1.0, 0.1, key=f"des_shape_{resp}",
                              help="s=1: linear, s<1: less sensitive near bounds, s>1: more sensitive near bounds")

            response_settings.append({
                "name": resp, "goal": goal, "weight": weight,
                "lower": lower, "target": target_val, "upper": upper, "shape": shape,
            })

    if st.button("Optimize", key="des_optimize"):
        data = df[factor_cols + response_cols].dropna()
        if len(data) < len(factor_cols) + 2:
            st.error("Not enough data points.")
            return

        # Fit response surface models for each response
        from scipy.optimize import minimize

        models = {}
        for resp in response_cols:
            X_factors = data[factor_cols].values
            y = data[resp].values
            # Quadratic model
            X_design = [np.ones(len(y))]
            for j in range(len(factor_cols)):
                X_design.append(X_factors[:, j])
            for j in range(len(factor_cols)):
                X_design.append(X_factors[:, j] ** 2)
            for j1 in range(len(factor_cols)):
                for j2 in range(j1 + 1, len(factor_cols)):
                    X_design.append(X_factors[:, j1] * X_factors[:, j2])
            X_design = np.column_stack(X_design)

            try:
                model = sm.OLS(y, X_design).fit()
                models[resp] = (model, X_design.shape[1])
            except Exception as e:
                st.warning(f"Could not fit model for {resp}: {e}")
                return

        def predict_response(x, resp_name):
            model, n_params = models[resp_name]
            X_new = [1.0]
            for j in range(len(factor_cols)):
                X_new.append(x[j])
            for j in range(len(factor_cols)):
                X_new.append(x[j] ** 2)
            for j1 in range(len(factor_cols)):
                for j2 in range(j1 + 1, len(factor_cols)):
                    X_new.append(x[j1] * x[j2])
            return model.predict(np.array([X_new]))[0]

        def individual_desirability(y, settings):
            L, T, U, s = settings["lower"], settings["target"], settings["upper"], settings["shape"]
            goal = settings["goal"]

            if goal == "Maximize":
                if y <= L:
                    return 0.0
                elif y >= T:
                    return 1.0
                else:
                    return ((y - L) / (T - L)) ** s
            elif goal == "Minimize":
                if y >= U:
                    return 0.0
                elif y <= T:
                    return 1.0
                else:
                    return ((U - y) / (U - T)) ** s
            else:  # Target
                if y < L or y > U:
                    return 0.0
                elif y <= T:
                    return ((y - L) / (T - L)) ** s if T > L else 1.0
                else:
                    return ((U - y) / (U - T)) ** s if U > T else 1.0

        def overall_desirability(x):
            d_values = []
            weights = []
            for settings in response_settings:
                y_pred = predict_response(x, settings["name"])
                d = individual_desirability(y_pred, settings)
                d_values.append(d)
                weights.append(settings["weight"])

            # Weighted geometric mean
            d_values = np.array(d_values)
            weights = np.array(weights)
            if np.any(d_values == 0):
                return 0.0
            log_d = np.sum(weights * np.log(d_values)) / np.sum(weights)
            return np.exp(log_d)

        def neg_desirability(x):
            return -overall_desirability(x)

        # Optimize
        bounds = [(data[col].min(), data[col].max()) for col in factor_cols]
        x0 = data[factor_cols].mean().values

        with st.spinner("Optimizing desirability..."):
            best_result = None
            best_D = -np.inf
            # Multi-start optimization
            for _ in range(20):
                x_start = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
                try:
                    result = minimize(neg_desirability, x_start, bounds=bounds,
                                      method="L-BFGS-B")
                    if -result.fun > best_D:
                        best_D = -result.fun
                        best_result = result
                except Exception:
                    pass

            # Also try from data mean
            try:
                result = minimize(neg_desirability, x0, bounds=bounds, method="L-BFGS-B")
                if -result.fun > best_D:
                    best_D = -result.fun
                    best_result = result
            except Exception:
                pass

        if best_result is not None:
            section_header("Optimal Settings")
            opt_x = best_result.x

            cols = st.columns(len(factor_cols) + 1)
            for i, col in enumerate(factor_cols):
                cols[i].metric(f"Optimal {col}", f"{opt_x[i]:.4f}")
            cols[-1].metric("Overall Desirability", f"{best_D:.4f}")

            # Predicted responses at optimum
            section_header("Predicted Responses at Optimum")
            for settings in response_settings:
                y_pred = predict_response(opt_x, settings["name"])
                d_i = individual_desirability(y_pred, settings)
                st.write(f"**{settings['name']}:** {y_pred:.4f} (d = {d_i:.4f}, goal: {settings['goal']})")

            # Desirability profile plots
            section_header("Desirability Profiles")
            n_factors_plot = len(factor_cols)
            fig = make_subplots(rows=len(response_cols) + 1, cols=n_factors_plot,
                                subplot_titles=[f"{resp} vs {fac}"
                                                for resp in response_cols + ["Overall D"]
                                                for fac in factor_cols],
                                vertical_spacing=0.05,
                                horizontal_spacing=0.05)

            for fi, fac in enumerate(factor_cols):
                x_range = np.linspace(bounds[fi][0], bounds[fi][1], 50)
                for ri, settings in enumerate(response_settings):
                    y_vals = []
                    d_vals = []
                    for x_val in x_range:
                        x_test = opt_x.copy()
                        x_test[fi] = x_val
                        y_pred = predict_response(x_test, settings["name"])
                        y_vals.append(y_pred)
                        d_vals.append(individual_desirability(y_pred, settings))

                    fig.add_trace(go.Scatter(x=x_range, y=y_vals, mode="lines",
                                             showlegend=False, line=dict(width=2)),
                                  row=ri + 1, col=fi + 1)
                    fig.add_vline(x=opt_x[fi], line_dash="dot", line_color="red",
                                  row=ri + 1, col=fi + 1)

                # Overall desirability
                d_overall = []
                for x_val in x_range:
                    x_test = opt_x.copy()
                    x_test[fi] = x_val
                    d_overall.append(overall_desirability(x_test))
                fig.add_trace(go.Scatter(x=x_range, y=d_overall, mode="lines",
                                         showlegend=False, line=dict(width=2, color="red")),
                              row=len(response_cols) + 1, col=fi + 1)
                fig.add_vline(x=opt_x[fi], line_dash="dot", line_color="red",
                              row=len(response_cols) + 1, col=fi + 1)

            fig.update_layout(height=250 * (len(response_cols) + 1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Optimization did not converge. Try different settings.")
