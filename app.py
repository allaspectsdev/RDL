"""
DataLens - Visual Data Analysis Tool
A comprehensive, interactive data analysis platform modeled after
JMP, MATLAB, and other leading statistical analysis tools.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="DataLens - Visual Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for cleaner appearance
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-top: 0;
        padding-top: 0;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px;
    }
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Import modules
from modules.data_manager import render_upload, render_data_manager
from modules.descriptive_stats import render_descriptive_stats
from modules.hypothesis_testing import render_hypothesis_testing
from modules.regression import render_regression
from modules.anova import render_anova
from modules.correlation import render_correlation
from modules.visualization import render_visualization
from modules.time_series import render_time_series
from modules.machine_learning import render_machine_learning


def load_sample_dataset(name: str) -> pd.DataFrame:
    """Load a built-in sample dataset for demonstration."""
    if name == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["species"] = pd.Categorical([data.target_names[t] for t in data.target])
        return df
    elif name == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["wine_class"] = pd.Categorical([str(t) for t in data.target])
        return df
    elif name == "Boston-style Housing":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["MedianValue"] = data.target
        return df
    elif name == "Diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["progression"] = data.target
        return df
    elif name == "Tips":
        import plotly.express as px
        return px.data.tips()
    elif name == "Gapminder":
        import plotly.express as px
        return px.data.gapminder()
    elif name == "Stocks":
        import plotly.express as px
        return px.data.stocks()
    return None


def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown('<p class="main-header">DataLens</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Visual Data Analysis Tool</p>', unsafe_allow_html=True)
        st.divider()

        # Data source section
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose data source:",
            ["Upload File", "Sample Dataset"],
            label_visibility="collapsed",
        )

        if data_source == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload your dataset",
                type=["csv", "xlsx", "xls", "tsv", "json"],
                help="Supported formats: CSV, Excel, TSV, JSON",
            )
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith(".csv"):
                        st.session_state["df"] = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith((".xlsx", ".xls")):
                        st.session_state["df"] = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith(".tsv"):
                        st.session_state["df"] = pd.read_csv(uploaded_file, sep="\t")
                    elif uploaded_file.name.endswith(".json"):
                        st.session_state["df"] = pd.read_json(uploaded_file)
                    st.session_state["data_name"] = uploaded_file.name
                    st.success(f"Loaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        else:
            sample = st.selectbox(
                "Select dataset:",
                ["Iris", "Wine", "Boston-style Housing", "Diabetes", "Tips", "Gapminder", "Stocks"],
            )
            if st.button("Load Dataset", use_container_width=True):
                st.session_state["df"] = load_sample_dataset(sample)
                st.session_state["data_name"] = sample
                st.success(f"Loaded: {sample}")

        st.divider()

        # Analysis module selection
        st.subheader("Analysis Module")
        module = st.radio(
            "Select module:",
            [
                "Data Manager",
                "Descriptive Statistics",
                "Visualization Builder",
                "Hypothesis Testing",
                "Correlation & Multivariate",
                "Regression Analysis",
                "ANOVA",
                "Time Series Analysis",
                "Machine Learning",
            ],
            label_visibility="collapsed",
        )

        # Dataset info in sidebar
        if "df" in st.session_state and st.session_state["df"] is not None:
            st.divider()
            st.subheader("Dataset Info")
            df = st.session_state["df"]
            st.caption(f"**Name:** {st.session_state.get('data_name', 'Unknown')}")
            st.caption(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
            st.caption(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            n_numeric = len(df.select_dtypes(include=[np.number]).columns)
            n_categorical = len(df.select_dtypes(include=["object", "category"]).columns)
            n_datetime = len(df.select_dtypes(include=["datetime64"]).columns)
            st.caption(f"**Types:** {n_numeric} numeric, {n_categorical} categorical, {n_datetime} datetime")
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
            st.caption(f"**Missing:** {missing_pct:.1f}%")

    # Main content area
    if "df" not in st.session_state or st.session_state["df"] is None:
        # Landing page when no data is loaded
        st.markdown('<p class="main-header">Welcome to DataLens</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">A comprehensive visual data analysis platform</p>', unsafe_allow_html=True)
        st.markdown("---")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Upload & Explore")
            st.markdown(
                "Upload CSV, Excel, TSV, or JSON files. "
                "Preview, clean, transform, and prepare your data."
            )
        with col2:
            st.markdown("### Analyze")
            st.markdown(
                "Descriptive statistics, hypothesis testing, regression, ANOVA, "
                "correlation analysis, time series, and machine learning."
            )
        with col3:
            st.markdown("### Visualize")
            st.markdown(
                "20+ interactive chart types with full customization. "
                "Scatter, bar, histogram, heatmap, 3D, parallel coordinates, and more."
            )

        st.info("Upload a dataset or select a sample dataset from the sidebar to get started.")

        # Feature overview
        st.markdown("---")
        st.markdown("### Feature Overview")
        features = {
            "Data Management": "Upload, preview, clean, transform, filter, sort, encode, and sample your data.",
            "Descriptive Statistics": "Summary stats, distributions, normality tests, outlier detection, grouped analysis.",
            "Visualization Builder": "22 chart types including scatter, bar, histogram, heatmap, 3D, treemap, radar, and more.",
            "Hypothesis Testing": "t-tests, chi-square, Mann-Whitney, power analysis, multiple comparison corrections.",
            "Correlation & Multivariate": "Correlation matrices, PCA, t-SNE, factor analysis, scatter matrices.",
            "Regression Analysis": "Linear, multiple, polynomial, logistic regression, curve fitting, full diagnostics.",
            "ANOVA": "One-way, two-way, repeated measures, ANCOVA, Kruskal-Wallis, post-hoc tests.",
            "Time Series": "Decomposition, ACF/PACF, ARIMA/SARIMA, smoothing, forecasting.",
            "Machine Learning": "Clustering, classification, regression, dimensionality reduction, model comparison.",
        }
        cols = st.columns(3)
        for i, (name, desc) in enumerate(features.items()):
            with cols[i % 3]:
                st.markdown(f"**{name}**")
                st.caption(desc)
        return

    # Route to selected module
    df = st.session_state["df"]

    if module == "Data Manager":
        st.markdown("## Data Manager")
        result = render_data_manager(df)
        if result is not None:
            st.session_state["df"] = result
    elif module == "Descriptive Statistics":
        st.markdown("## Descriptive Statistics")
        render_descriptive_stats(df)
    elif module == "Visualization Builder":
        st.markdown("## Visualization Builder")
        render_visualization(df)
    elif module == "Hypothesis Testing":
        st.markdown("## Hypothesis Testing")
        render_hypothesis_testing(df)
    elif module == "Correlation & Multivariate":
        st.markdown("## Correlation & Multivariate Analysis")
        render_correlation(df)
    elif module == "Regression Analysis":
        st.markdown("## Regression Analysis")
        render_regression(df)
    elif module == "ANOVA":
        st.markdown("## ANOVA")
        render_anova(df)
    elif module == "Time Series Analysis":
        st.markdown("## Time Series Analysis")
        render_time_series(df)
    elif module == "Machine Learning":
        st.markdown("## Machine Learning")
        render_machine_learning(df)


if __name__ == "__main__":
    main()
