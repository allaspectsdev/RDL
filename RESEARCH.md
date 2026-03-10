# Visual Data Analysis Tool - Competitive Research & Feature Specification

> **Date:** March 2026
> **Purpose:** Comprehensive competitive analysis of leading visual data analysis tools and feature specification for building a web-based visual data analysis platform.

---

## Table of Contents

1. [Competitive Landscape Summary](#1-competitive-landscape-summary)
2. [Tool-by-Tool Feature Analysis](#2-tool-by-tool-feature-analysis)
3. [Complete Statistical & Analytical Feature List](#3-complete-statistical--analytical-feature-list)
4. [Complete Visualization Type List](#4-complete-visualization-type-list)
5. [Recommended Tech Stack](#5-recommended-tech-stack)
6. [Proposed Architecture](#6-proposed-architecture)
7. [Feature Priority List (MVP vs Advanced)](#7-feature-priority-list-mvp-vs-advanced)

---

## 1. Competitive Landscape Summary

The visual data analysis tool market spans a spectrum from code-heavy programming environments to fully GUI-driven desktop applications. Each tool occupies a distinct niche:

| Tool | Primary Strength | Interface | Interactivity | Price Model | Target User |
|------|-----------------|-----------|---------------|-------------|-------------|
| **JMP (SAS)** | Exploratory data analysis, DOE | GUI-first, scripting optional | High (linked graphs, profilers) | Commercial license (~$1,800+/yr) | Scientists, engineers, quality professionals |
| **MATLAB** | Numerical computing, signal processing | Code-first, apps available | Medium (App Designer, Live Editor) | Commercial license (~$900+/yr + toolboxes) | Engineers, researchers, academics |
| **Tableau** | Business intelligence dashboards | GUI drag-and-drop | Very High (drill-down, filters, dashboards) | Commercial license (~$75/user/mo) | Business analysts, data teams |
| **R / RStudio** | Statistical computing, extensibility | Code-first (RStudio IDE) | Low-Medium (Shiny for interactivity) | Free / Open Source | Statisticians, academics, data scientists |
| **Python Ecosystem** | General-purpose data science | Code-first | Low-High (depends on library) | Free / Open Source | Data scientists, developers, ML engineers |
| **Minitab** | Quality improvement, Six Sigma | GUI-first | Medium | Commercial license (~$1,500+/yr) | Quality engineers, manufacturing |
| **SPSS (IBM)** | Social science statistics | GUI-first, syntax optional | Low-Medium | Commercial license (~$100+/mo) | Social scientists, researchers, students |

### Key Market Gaps & Opportunities

1. **No open-source web-based tool** combines JMP-level interactive exploration with Python-ecosystem analytical power.
2. **JMP's interactivity** (linked graphs, real-time profilers, dynamic brushing) is unmatched but locked behind expensive desktop licenses.
3. **Tableau excels at dashboards** but lacks serious statistical analysis (no hypothesis testing, no DOE, no regression diagnostics).
4. **Python/R are powerful but fragmented** -- users must assemble dozens of libraries and write substantial code.
5. **Minitab/SPSS are aging** desktop applications with limited visualization and no real-time interactivity.
6. **A web-based tool** combining interactive visualization, serious statistics, and modern UX would fill a significant gap.

---

## 2. Tool-by-Tool Feature Analysis

### 2.1 JMP (SAS)

**Overview:** JMP is the gold standard for interactive exploratory data analysis. Originally created by John Sall (co-founder of SAS) in 1989, it pioneered linked dynamic graphics and the prediction profiler. JMP 19.0 (September 2025) is the latest release.

**Core Statistical Methods:**
- Descriptive statistics (mean, median, mode, std dev, quantiles, skewness, kurtosis)
- One-sample, two-sample, and paired t-tests
- One-way, two-way, multi-way ANOVA and ANOCOVA
- Simple and multiple linear regression
- Stepwise regression and best subsets
- Logistic regression (binary, ordinal, nominal)
- Nonlinear regression and curve fitting
- Generalized Linear Models (GLM)
- Mixed models (random effects, repeated measures)
- Nonparametric tests (Wilcoxon, Kruskal-Wallis, Friedman)
- Equivalence tests
- Power and sample size calculations
- Bootstrapping and permutation tests
- Time series analysis (ARIMA, smoothing, decomposition)
- Survival / reliability analysis (Kaplan-Meier, Cox proportional hazards, Weibull, lognormal)
- Measurement system analysis (Gage R&R)
- Process capability analysis (Cp, Cpk, Pp, Ppk)

**Multivariate Methods:**
- Principal Component Analysis (PCA)
- Factor analysis (multiple extraction methods, axis rotations)
- Discriminant analysis
- Cluster analysis (hierarchical, k-means)
- Partial Least Squares (PLS)
- Multidimensional Scaling (MDS)
- Correspondence analysis
- Item analysis

**Design of Experiments (DOE):**
- Screening designs (Plackett-Burman, definitive screening)
- Full factorial designs
- Fractional factorial designs
- Response surface designs (Central Composite, Box-Behnken)
- Mixture designs (simplex lattice, simplex centroid, extreme vertices)
- Custom designs (D-optimal, I-optimal, A-optimal)
- Split-plot designs
- Augmentation of existing designs
- Taguchi designs
- Space-filling designs (Latin Hypercube, sphere packing)

**Machine Learning (JMP Pro):**
- Neural networks
- Decision trees (partition)
- Bootstrap forest (random forest)
- Boosted trees (gradient boosting)
- Support Vector Machines (SVM)
- Naive Bayes
- K-nearest neighbors
- Model comparison and validation
- Cross-validation
- Text mining / text exploration

**Visualization Types:**
- Histograms with fitted distributions
- Box plots and outlier box plots
- Scatter plots (2D and 3D rotating)
- Bubble plots (static and animated)
- Line plots and overlay plots
- Bar charts and pie charts
- Mosaic plots
- Contour plots
- Surface plots (3D)
- Heat maps and cell plots
- Treemaps
- Parallel coordinate plots
- Ternary plots
- Scatter plot matrices (SPLOM)
- Control charts (Xbar-R, Xbar-S, I-MR, P, NP, C, U, CUSUM, EWMA)
- Pareto charts
- Normal probability / quantile plots
- Variability / gauge charts
- Profilers (prediction, contour, surface, mixture, custom)
- Graph Builder (drag-and-drop graph construction)

**Standout / Unique Features:**
- **Dynamic linking**: Selecting data in one graph highlights corresponding points across all open graphs and the data table simultaneously.
- **Prediction Profiler**: Interactive, real-time visualization of response surfaces with desirability functions.
- **Contour Profiler**: 2D contour overlays with constraint shading for multi-response optimization.
- **Graph Builder**: Drag variables onto zones to build visualizations interactively -- no coding required.
- **Column Switcher**: Rapidly cycle through variables in any analysis without rebuilding.
- **Local Data Filter**: Apply filters to any analysis window in real time.
- **Distribution platform**: One-click descriptive statistics with fitted distributions and goodness-of-fit tests.
- **Formula editor**: In-column formula definitions for derived variables.

---

### 2.2 MATLAB

**Overview:** MATLAB (MathWorks) is the premier numerical computing environment. Its strength lies in matrix operations, signal processing, and engineering applications. The Statistics and Machine Learning Toolbox extends it into statistical analysis.

**Core Statistical Methods:**
- Descriptive statistics (all standard measures)
- Probability distributions (100+ continuous and discrete distributions)
- Hypothesis tests (t-test, z-test, chi-square, F-test, Kolmogorov-Smirnov, Anderson-Darling, Lilliefors)
- One-way, two-way, multi-way, multivariate ANOVA
- Analysis of covariance (ANOCOVA)
- Repeated measures ANOVA (RANOVA)
- Linear, multiple, and polynomial regression
- Stepwise regression
- Nonlinear regression
- Generalized Linear Models
- Mixed-effects models (linear and nonlinear)
- Robust regression
- Ridge and Lasso regression
- Bayesian optimization
- Monte Carlo simulation
- Bootstrapping

**Signal Processing & Specialized:**
- FFT and spectral analysis
- Digital and analog filter design
- Wavelet analysis
- Time-frequency analysis
- Curve fitting (parametric and nonparametric)
- Interpolation and extrapolation
- Optimization (linear, nonlinear, integer programming)
- Control system design
- Image processing and computer vision

**Machine Learning:**
- Classification: SVM, decision trees, discriminant analysis, Naive Bayes, k-NN, ensemble methods, neural networks
- Regression: linear, regression trees, SVM regression, Gaussian process regression, ensemble methods
- Clustering: k-means, k-medoids, hierarchical, Gaussian mixture models, DBSCAN
- Dimensionality reduction: PCA, factor analysis, t-SNE, UMAP
- Feature selection and engineering
- AutoML (Classification Learner, Regression Learner apps)
- Deep learning (via Deep Learning Toolbox)
- Transfer learning
- Tall arrays for out-of-memory data

**Visualization Types:**
- Line plots and scatter plots
- Bar charts (grouped, stacked, horizontal)
- Histograms and histogram2
- Box plots
- Pie charts
- Area plots
- Stem plots (2D and 3D)
- Stair step plots
- Error bar plots
- Polar plots (line, scatter, histogram)
- Contour plots (2D and 3D)
- Surface plots (surf, surfc, surfl)
- Mesh plots (mesh, meshc, meshz)
- Waterfall plots
- Ribbon plots
- Quiver plots (vector fields)
- Stream plots
- Heatmaps / imagesc
- Comet plots (animated trajectory)
- Voronoi diagrams
- Geographic plots (geoplot, geoscatter, geobubble)

**Standout / Unique Features:**
- **Live Editor**: Interactive notebooks combining code, output, and narrative.
- **App Designer**: Build interactive GUI applications with MATLAB backends.
- **Toolbox ecosystem**: 90+ specialized toolboxes for domain-specific analysis.
- **Matrix-first computing**: Everything is a matrix operation, enabling fast vectorized computation.
- **Simulink integration**: Model-based design for dynamic systems.
- **Code generation**: Generate C/C++ and CUDA code from MATLAB.

---

### 2.3 Tableau

**Overview:** Tableau (Salesforce) is the market leader in business intelligence and interactive dashboarding. It emphasizes visual exploration over statistical depth. Tableau 2026.1 is the current release.

**Analytical Capabilities:**
- Descriptive statistics (aggregations: SUM, AVG, MEDIAN, COUNT, MIN, MAX, STDEV, VAR, PERCENTILE)
- Table calculations (running totals, moving averages, percent of total, rank, percentile)
- Trend lines (linear, logarithmic, exponential, polynomial, power)
- Forecasting (exponential smoothing)
- Clustering (k-means)
- Reference lines, bands, and distributions
- Statistical summary cards
- LOD (Level of Detail) expressions
- Parameter controls
- Set actions and set analytics
- AI-powered "Explain Data" and "Ask Data" (natural language queries)

**Visualization Types:**
- Bar charts (horizontal, vertical, stacked, grouped, diverging)
- Line charts (continuous, discrete, step)
- Area charts (standard, stacked)
- Scatter plots
- Bubble charts and packed bubbles
- Pie charts and donut charts
- Treemaps
- Heat maps and highlight tables
- Box-and-whisker plots
- Histograms
- Bullet charts
- Gantt charts
- Waterfall charts
- Lollipop charts
- Bump charts (rank changes over time)
- Slope charts
- Sparklines
- Word clouds
- Maps (symbol, filled/choropleth, flow, density/heat, dual-axis, mixed geometry)
- Dual-axis and combination charts
- Reference lines and bands
- Dashboard layouts with interactivity

**Data Connectivity:**
- 75+ native data connectors (databases, cloud services, files)
- Live connections and data extracts
- Cross-database joins
- Data blending
- Custom SQL
- Spatial data support (shapefiles, GeoJSON)

**Standout / Unique Features:**
- **Drag-and-drop**: The most intuitive visual exploration interface in the market.
- **Show Me**: Automatic chart type recommendation based on selected fields.
- **Dashboard Actions**: Click on one chart to filter/highlight all others.
- **Stories**: Guided analytical narratives combining multiple dashboards.
- **Tableau Prep**: Visual data preparation and cleaning pipeline.
- **Tableau Server / Cloud**: Enterprise sharing, governance, and collaboration.
- **Mixed Geometry Support**: Points, lines, and polygons from a single column (2025+).
- **Explain Data**: AI-driven automatic explanation of outliers and patterns.
- **VizQL**: Proprietary visual query language that translates drag-and-drop actions into database queries.

---

### 2.4 R / RStudio

**Overview:** R is the lingua franca of academic statistics. With 20,000+ CRAN packages, it has the broadest statistical method coverage of any platform. RStudio (now Posit) provides the primary IDE, and Shiny enables web applications.

**Core Statistical Methods:**
- Comprehensive descriptive statistics
- Every standard hypothesis test (t, z, chi-square, F, Fisher's exact, McNemar, Cochran's Q)
- Linear and generalized linear models (Gaussian, binomial, Poisson, gamma, inverse Gaussian, negative binomial, quasi families)
- Mixed-effects models (lme4, nlme, glmmTMB, GLMMadaptive)
- Nonlinear regression (nls, nlstools)
- Robust regression (MASS::rlm, robustbase)
- Ridge, Lasso, Elastic Net (glmnet)
- ANOVA (one-way, two-way, multi-way, repeated measures, MANOVA, ANOCOVA)
- Nonparametric tests (Wilcoxon, Mann-Whitney, Kruskal-Wallis, Friedman, sign test, runs test)
- Survival analysis (survival, survminer -- Kaplan-Meier, Cox PH, parametric models, competing risks)
- Time series (ARIMA, GARCH, state-space models, prophet, forecast package)
- Bayesian statistics (brms, rstanarm, MCMCpack, rjags, INLA)
- Structural equation modeling (lavaan)
- Item Response Theory (mirt, ltm)
- Meta-analysis (metafor)
- Spatial statistics (sp, sf, spdep, gstat)
- Functional data analysis (fda)
- Network analysis (igraph, statnet)
- Text mining and NLP (tm, tidytext, quanteda)
- Causal inference (MatchIt, CausalImpact, DAGitty)

**Visualization (ggplot2 ecosystem):**
- Core geoms: point, line, bar, histogram, boxplot, violin, density, area, tile, raster, polygon, path, step, ribbon, errorbar, crossbar, pointrange, linerange, rug, jitter, hex, contour, density_2d, smooth, text, label, segment, curve, spoke, rect, map, sf
- Extensions: ggridges (ridgeline plots), ggbeeswarm (beeswarm plots), ggrepel (non-overlapping labels), ggforce (arcs, circles, Voronoi), ggalluvial (Sankey/alluvial diagrams), gganimate (animations), ggraph (network graphs), ggcorrplot (correlation matrices), ggmosaic (mosaic plots), patchwork (multi-plot composition), GGally (generalized pairs plots)
- plotly::ggplotly() for converting any ggplot to interactive
- lattice graphics (trellis plots, panel functions)
- Base R graphics (versatile, low-level control)

**Shiny (Web Applications):**
- Reactive programming model for interactive web apps
- Input widgets (sliders, dropdowns, text fields, file upload, date pickers)
- Output renderers (plots, tables, text, UI elements)
- Shiny dashboards (shinydashboard, bs4Dash)
- Real-time data updates
- Deployment via shinyapps.io or Shiny Server

**Standout / Unique Features:**
- **Unmatched statistical breadth**: If a statistical method exists, there is an R package for it.
- **ggplot2 Grammar of Graphics**: Compositional, layered approach to building any visualization.
- **Tidyverse**: Coherent ecosystem for data manipulation (dplyr, tidyr, purrr, stringr, lubridate).
- **R Markdown / Quarto**: Reproducible reports combining code, results, and narrative.
- **CRAN Task Views**: Curated lists of packages by statistical domain.
- **Bioconductor**: 2,000+ packages for genomics and bioinformatics.

---

### 2.5 Python Ecosystem

**Overview:** Python has become the dominant language for data science, machine learning, and AI. Its ecosystem is vast and modular -- users assemble libraries to build custom workflows.

**Core Libraries & Capabilities:**

#### Data Manipulation
- **pandas**: DataFrames, Series, groupby, merge/join, pivot tables, reshaping, time series, rolling windows, categorical data, multi-indexing, query syntax
- **polars**: Rust-powered alternative to pandas with lazy evaluation, multi-threaded execution, and Apache Arrow backend -- dramatically faster on large datasets
- **NumPy**: N-dimensional arrays, linear algebra, random number generation, broadcasting, universal functions, FFT
- **DuckDB**: In-process SQL OLAP database, queries pandas/polars/Arrow DataFrames directly, out-of-core processing, Parquet/CSV/JSON ingestion

#### Statistical Analysis
- **scipy.stats**: 100+ probability distributions (continuous and discrete), hypothesis tests (t-test, chi-square, ANOVA, Kruskal-Wallis, Kolmogorov-Smirnov, Anderson-Darling, Shapiro-Wilk, Levene, Bartlett, Mann-Whitney U, Wilcoxon signed-rank, Spearman, Pearson, Kendall tau, Fisher exact, binomial), kernel density estimation, statistical distance metrics
- **statsmodels**: OLS, GLS, WLS, GLM (all families), mixed-effects models, robust regression, quantile regression, ANOVA (Type I, II, III), time series (ARIMA, SARIMAX, VAR, VECM, exponential smoothing, Holt-Winters), state-space models, survival analysis, nonparametric methods, diagnostic tests (heteroscedasticity, autocorrelation, normality), multiple comparison corrections (Bonferroni, Holm, FDR)
- **pingouin**: Simplified statistical testing (ANOVA, t-tests, correlations, effect sizes, Bayesian tests, circular statistics)
- **lifelines**: Survival analysis (Kaplan-Meier, Nelson-Aalen, Cox PH, parametric models, competing risks)

#### Machine Learning
- **scikit-learn**: Classification (SVM, random forest, gradient boosting, k-NN, logistic regression, Naive Bayes, decision trees, AdaBoost, extra trees), Regression (linear, Ridge, Lasso, Elastic Net, SVR, tree-based, Gaussian process), Clustering (k-means, DBSCAN, hierarchical/agglomerative, spectral, Gaussian mixture, mean shift, BIRCH, OPTICS), Dimensionality reduction (PCA, t-SNE, UMAP via umap-learn, LDA, kernel PCA, NMF, truncated SVD), Feature selection, preprocessing, pipelines, cross-validation, grid search, model evaluation metrics
- **XGBoost**: Gradient boosting (classification, regression, ranking)
- **LightGBM**: Fast gradient boosting with categorical feature support
- **CatBoost**: Gradient boosting with native categorical handling

#### Visualization
- **matplotlib**: The foundational plotting library. Line, scatter, bar, histogram, pie, box, violin, stem, step, fill_between, contour, contourf, imshow, pcolormesh, quiver, streamplot, polar, 3D plots (surface, wireframe, scatter3d, bar3d, contour3d), subplots, twin axes, annotations, custom styles
- **seaborn**: Statistical visualization built on matplotlib. relplot (scatter, line), displot (histogram, KDE, ECDF, rug), catplot (strip, swarm, box, violin, boxen, point, bar, count), heatmap, clustermap, pairplot, jointplot, regression plots (regplot, residplot), FacetGrid, PairGrid, JointGrid
- **plotly**: Interactive charts. Scatter, line, bar, pie, sunburst, treemap, icicle, funnel, waterfall, sankey, parallel coordinates, parallel categories, scatter_matrix, density heatmap, density contour, box, violin, strip, histogram, ECDF, scatter_3d, line_3d, surface, mesh3d, cone, streamtube, isosurface, choropleth, scatter_geo, scatter_mapbox, density_mapbox, line_mapbox, polar (scatter, line, bar), ternary, radar, candlestick, OHLC, indicator/gauge, table, animation frames
- **Altair**: Declarative visualization based on Vega-Lite grammar of graphics
- **Bokeh**: Interactive visualization library for browsers with streaming/real-time capabilities

#### Web Frameworks
- **Streamlit**: Rapid prototyping of data apps with Python scripts. Widgets, caching, session state, columns/tabs layout, file upload, chat interface. Best for quick demos and internal tools.
- **Dash (Plotly)**: Production-grade analytical web apps. Callback-driven architecture, full layout control with Dash HTML/Bootstrap components, integration with Plotly charts, AG Grid for data tables, deployment via Gunicorn/Docker. Better for complex multi-page applications.
- **Panel (HoloViz)**: Flexible dashboarding that works with multiple plotting libraries (matplotlib, plotly, bokeh, altair).
- **Gradio**: Rapid ML model demo interfaces.
- **NiceGUI**: Modern web-based GUI framework for Python with Tailwind CSS integration.

**Standout / Unique Features:**
- **Ecosystem breadth**: No other language matches Python's combined strength across data science, ML, web development, and DevOps.
- **Jupyter notebooks**: The de facto standard for interactive data exploration and sharing.
- **Apache Arrow / Parquet**: Columnar data format enabling zero-copy interop between libraries and efficient I/O.
- **Free and open source**: No licensing costs for any core library.
- **Community**: Largest data science community, most StackOverflow answers, most tutorials.

---

### 2.6 Minitab

**Overview:** Minitab is the standard tool for quality improvement and Six Sigma (DMAIC process). It is heavily used in manufacturing, pharmaceuticals, and process engineering.

**Core Statistical Methods:**
- Descriptive statistics (all standard measures, graphical summaries)
- Hypothesis tests (1-sample/2-sample t, paired t, 1/2 proportion, 1/2 variance, chi-square goodness of fit)
- One-way, two-way, multi-way, balanced ANOVA, General Linear Model (GLM)
- Simple, multiple, polynomial, stepwise, best subsets regression
- Binary, ordinal, nominal logistic regression
- Nonlinear regression
- Orthogonal regression (Deming)
- Poisson regression
- Stability studies
- Equivalence tests
- Power and sample size calculations for all common tests
- Nonparametric tests (Mann-Whitney, Kruskal-Wallis, Mood's median, Friedman, 1-sample sign, 1-sample Wilcoxon, runs test)
- Correlation (Pearson, Spearman)
- Cross-tabulation and chi-square tests
- Time series analysis (trend, decomposition, moving average, exponential smoothing, ARIMA)
- Multivariate analysis (PCA, factor analysis, cluster analysis, discriminant analysis, correspondence analysis)

**Quality & Process Tools:**
- Control charts: Xbar-R, Xbar-S, I-MR, I-MR-R/S, P, NP, C, U, Laney P', Laney U', CUSUM, EWMA, Zone, time-weighted
- Process capability: normal and non-normal capability analysis (Cp, Cpk, Pp, Ppk, Cpm, Z-bench)
- Capability Sixpack
- Tolerance intervals (normal and non-normal)
- Measurement System Analysis: Gage R&R (crossed and nested), attribute agreement analysis, attribute gage study, bias study, linearity study, type 1 gage study
- Acceptance sampling (by attributes, by variables)
- Multi-vari charts

**Design of Experiments:**
- Factorial designs (2-level full factorial, 2-level fractional factorial, Plackett-Burman, general full factorial)
- Response surface designs (Central Composite, Box-Behnken)
- Mixture designs (simplex centroid, simplex lattice, extreme vertices)
- Taguchi designs
- Split-plot designs
- Optimal designs (D-optimal)
- Response optimization

**Reliability / Survival:**
- Distribution analysis (right-censored, arbitrarily censored, with or without covariates)
- Parametric and nonparametric distribution analysis
- Warranty analysis
- Accelerated life testing
- Test plans (estimation, demonstration)
- Probit analysis
- Weibayes analysis
- Repairable systems analysis

**Machine Learning:**
- CART (Classification and Regression Trees)
- TreeNet (stochastic gradient boosting)
- Random Forests
- Support vector machines

**Visualization Types:**
- Scatterplots and matrix plots
- Histograms and dotplots
- Boxplots and individual value plots
- Interval plots and bar charts
- Pie charts
- Time series plots
- Contour and surface plots
- Probability plots (normal, Weibull, etc.)
- Marginal plots
- Bubble plots
- Multi-vari charts
- Pareto charts
- Cause-and-effect (fishbone) diagrams

**Standout / Unique Features:**
- **Assistant**: Guided analysis workflow that helps non-statisticians choose the right test and interpret results.
- **Six Sigma integration**: Built specifically to support DMAIC methodology.
- **Minitab Connect**: IoT and real-time data connectivity.
- **Real-Time SPC**: Cloud-based statistical process control monitoring.
- **Simul8**: Integrated process simulation.

---

### 2.7 SPSS (IBM)

**Overview:** SPSS (Statistical Package for the Social Sciences) has been the dominant tool in social science, education, and health research for decades. IBM acquired it in 2009. Recent versions integrate AI via watsonx.

**Core Statistical Methods:**
- Descriptive statistics (frequencies, descriptives, explore, crosstabs, ratio statistics)
- Compare means (one-sample t, independent-samples t, paired-samples t, one-way ANOVA)
- General Linear Model (univariate, multivariate, repeated measures)
- Generalized Linear Models (GLM, GEE)
- Mixed models (linear, generalized linear)
- Correlations (bivariate: Pearson, Spearman, Kendall; partial correlations)
- Linear regression (enter, stepwise, backward, forward)
- Binary logistic regression
- Multinomial logistic regression
- Ordinal regression (PLUM)
- Probit analysis
- Nonlinear regression
- Weighted least squares
- Two-stage least squares
- Nonparametric tests (chi-square, binomial, runs, 1-sample K-S, 2 independent samples, k independent samples, 2 related samples, k related samples, Jonckheere-Terpstra, Cochran's Q)
- Reliability analysis (Cronbach's alpha, split-half, Guttman, parallel, strict parallel)
- Scale analysis (multidimensional scaling, ALSCAL, PROXSCAL)

**Dimension Reduction & Classification:**
- Factor analysis (principal components, principal axis factoring, maximum likelihood, unweighted least squares, generalized least squares, alpha factoring, image factoring; rotations: Varimax, Quartimax, Equamax, Direct Oblimin, Promax)
- Cluster analysis (hierarchical, k-means, two-step)
- Discriminant analysis
- Nearest neighbor analysis
- Optimal scaling (CATPCA, correspondence analysis, multiple correspondence analysis, multidimensional unfolding)

**Advanced Analytics (add-on modules):**
- SPSS Advanced Statistics: GLM multivariate, variance components, linear mixed models, GEE, survival analysis (life tables, Kaplan-Meier, Cox regression)
- SPSS Regression: binary/multinomial/ordinal logistic, probit, nonlinear, WLS, 2SLS
- SPSS Categories: optimal scaling, correspondence analysis, categorical PCA
- SPSS Exact Tests: exact p-values for small samples
- SPSS Forecasting: time series analysis (ARIMA, exponential smoothing, seasonal decomposition, spectral analysis, Expert Modeler)
- SPSS Decision Trees: CHAID, exhaustive CHAID, CRT, QUEST
- SPSS Neural Networks: multilayer perceptron, radial basis function
- SPSS Bootstrapping: bootstrap confidence intervals for any statistic
- SPSS Direct Marketing: RFM analysis, cluster analysis, propensity scoring
- SPSS Complex Samples: analysis of data from complex survey designs

**Visualization Types:**
- Bar charts (simple, clustered, stacked)
- Line charts
- Area charts
- Pie charts
- Scatterplots (simple, matrix, overlay, 3D)
- Histograms
- Box plots
- Error bar charts
- Population pyramids
- High-low charts
- Pareto charts
- Control charts
- Sequence charts
- Q-Q plots and P-P plots
- ROC curves

**Standout / Unique Features:**
- **SPSS Syntax**: Reproducible analysis scripting language.
- **Variable View**: Rich metadata (labels, value labels, missing values, measurement level) attached to every variable.
- **Output Viewer**: Organized hierarchical output navigator.
- **AI Output Assistant**: watsonx.ai-powered interpretation of results (2025+).
- **Complex Samples module**: Specialized support for stratified, cluster, and multi-stage survey designs.
- **Direct Marketing module**: Built-in RFM and propensity scoring.

---

## 3. Complete Statistical & Analytical Feature List

All statistical and analytical features that should be implemented, organized by category. Priority markers: **[MVP]** = minimum viable product, **[ADV]** = advanced feature.

### 3.1 Descriptive Statistics
| Feature | Priority |
|---------|----------|
| Mean, median, mode | **[MVP]** |
| Standard deviation, variance | **[MVP]** |
| Min, max, range | **[MVP]** |
| Quartiles and percentiles (Q1, Q3, IQR, arbitrary) | **[MVP]** |
| Skewness and kurtosis | **[MVP]** |
| Count, missing count, valid count | **[MVP]** |
| Sum | **[MVP]** |
| Coefficient of variation | **[MVP]** |
| Trimmed mean | **[ADV]** |
| Winsorized mean | **[ADV]** |
| Standard error of the mean | **[MVP]** |
| Confidence intervals for the mean | **[MVP]** |
| Frequency tables and cross-tabulation | **[MVP]** |
| Grouped statistics (group-by aggregations) | **[MVP]** |

### 3.2 Hypothesis Testing
| Feature | Priority |
|---------|----------|
| One-sample t-test | **[MVP]** |
| Independent two-sample t-test | **[MVP]** |
| Paired t-test | **[MVP]** |
| Welch's t-test | **[MVP]** |
| One-sample z-test | **[MVP]** |
| Two-proportion z-test | **[ADV]** |
| Chi-square test of independence | **[MVP]** |
| Chi-square goodness-of-fit test | **[MVP]** |
| Fisher's exact test | **[ADV]** |
| McNemar's test | **[ADV]** |
| Shapiro-Wilk normality test | **[MVP]** |
| Kolmogorov-Smirnov test | **[ADV]** |
| Anderson-Darling test | **[ADV]** |
| Lilliefors test | **[ADV]** |
| Levene's test for equal variances | **[MVP]** |
| Bartlett's test for equal variances | **[ADV]** |
| F-test for equal variances | **[ADV]** |
| Runs test for randomness | **[ADV]** |
| Binomial test | **[ADV]** |

### 3.3 Correlation & Association
| Feature | Priority |
|---------|----------|
| Pearson correlation coefficient | **[MVP]** |
| Spearman rank correlation | **[MVP]** |
| Kendall's tau | **[ADV]** |
| Point-biserial correlation | **[ADV]** |
| Partial correlation | **[ADV]** |
| Correlation matrix with p-values | **[MVP]** |
| Autocorrelation function (ACF) | **[ADV]** |
| Partial autocorrelation function (PACF) | **[ADV]** |
| Cramers V (categorical association) | **[ADV]** |
| Phi coefficient | **[ADV]** |
| Mutual information | **[ADV]** |

### 3.4 Analysis of Variance (ANOVA)
| Feature | Priority |
|---------|----------|
| One-way ANOVA | **[MVP]** |
| Two-way ANOVA | **[MVP]** |
| Multi-way (N-way) ANOVA | **[ADV]** |
| Repeated measures ANOVA | **[ADV]** |
| Mixed ANOVA (between + within) | **[ADV]** |
| MANOVA (multivariate) | **[ADV]** |
| ANOCOVA (analysis of covariance) | **[ADV]** |
| Welch's ANOVA (unequal variances) | **[MVP]** |
| Kruskal-Wallis test (nonparametric one-way) | **[MVP]** |
| Friedman test (nonparametric repeated measures) | **[ADV]** |
| Post-hoc tests: Tukey HSD | **[MVP]** |
| Post-hoc tests: Bonferroni | **[MVP]** |
| Post-hoc tests: Scheffe | **[ADV]** |
| Post-hoc tests: Dunnett | **[ADV]** |
| Post-hoc tests: Games-Howell | **[ADV]** |
| Post-hoc tests: Dunn's test (nonparametric) | **[ADV]** |
| Effect sizes (eta-squared, omega-squared, Cohen's d/f) | **[MVP]** |

### 3.5 Regression Analysis
| Feature | Priority |
|---------|----------|
| Simple linear regression | **[MVP]** |
| Multiple linear regression | **[MVP]** |
| Polynomial regression | **[MVP]** |
| Stepwise regression (forward, backward, both) | **[ADV]** |
| Best subsets regression | **[ADV]** |
| Ridge regression (L2) | **[ADV]** |
| Lasso regression (L1) | **[ADV]** |
| Elastic net regression | **[ADV]** |
| Logistic regression (binary) | **[MVP]** |
| Multinomial logistic regression | **[ADV]** |
| Ordinal logistic regression | **[ADV]** |
| Poisson regression | **[ADV]** |
| Negative binomial regression | **[ADV]** |
| Generalized Linear Models (GLM) | **[ADV]** |
| Robust regression (M-estimators, Huber, bisquare) | **[ADV]** |
| Quantile regression | **[ADV]** |
| Nonlinear regression (user-defined models) | **[ADV]** |
| Weighted least squares (WLS) | **[ADV]** |
| Regression diagnostics: residual plots | **[MVP]** |
| Regression diagnostics: leverage / hat values | **[MVP]** |
| Regression diagnostics: Cook's distance | **[MVP]** |
| Regression diagnostics: VIF / multicollinearity | **[MVP]** |
| Regression diagnostics: Durbin-Watson | **[ADV]** |
| Regression diagnostics: Breusch-Pagan heteroscedasticity | **[ADV]** |
| Confidence and prediction intervals | **[MVP]** |
| R-squared, adjusted R-squared, AIC, BIC | **[MVP]** |

### 3.6 Nonparametric Methods
| Feature | Priority |
|---------|----------|
| Mann-Whitney U test | **[MVP]** |
| Wilcoxon signed-rank test | **[MVP]** |
| Kruskal-Wallis test | **[MVP]** |
| Friedman test | **[ADV]** |
| Mood's median test | **[ADV]** |
| Sign test | **[ADV]** |
| Runs test | **[ADV]** |
| Kolmogorov-Smirnov two-sample test | **[ADV]** |
| Spearman rank correlation | **[MVP]** |
| Kernel density estimation | **[MVP]** |
| Bootstrap confidence intervals | **[ADV]** |
| Permutation tests | **[ADV]** |

### 3.7 Multivariate Analysis
| Feature | Priority |
|---------|----------|
| Principal Component Analysis (PCA) | **[MVP]** |
| Factor analysis (exploratory) | **[ADV]** |
| Confirmatory factor analysis | **[ADV]** |
| Linear Discriminant Analysis (LDA) | **[ADV]** |
| Cluster analysis: k-means | **[MVP]** |
| Cluster analysis: hierarchical (agglomerative) | **[ADV]** |
| Cluster analysis: DBSCAN | **[ADV]** |
| Cluster analysis: Gaussian mixture models | **[ADV]** |
| Multidimensional Scaling (MDS) | **[ADV]** |
| t-SNE dimensionality reduction | **[ADV]** |
| UMAP dimensionality reduction | **[ADV]** |
| Correspondence analysis | **[ADV]** |
| Canonical correlation analysis | **[ADV]** |
| Partial Least Squares (PLS) | **[ADV]** |

### 3.8 Time Series Analysis
| Feature | Priority |
|---------|----------|
| Trend analysis (linear, quadratic) | **[MVP]** |
| Moving averages (simple, weighted, exponential) | **[MVP]** |
| Seasonal decomposition (additive, multiplicative) | **[ADV]** |
| Exponential smoothing (single, double, Holt-Winters) | **[ADV]** |
| ARIMA modeling | **[ADV]** |
| SARIMA (seasonal ARIMA) | **[ADV]** |
| Auto-ARIMA (automatic order selection) | **[ADV]** |
| ACF and PACF plots | **[ADV]** |
| Stationarity tests (ADF, KPSS, Phillips-Perron) | **[ADV]** |
| Forecasting with confidence intervals | **[ADV]** |
| Change point detection | **[ADV]** |

### 3.9 Survival / Reliability Analysis
| Feature | Priority |
|---------|----------|
| Kaplan-Meier estimator and curves | **[ADV]** |
| Log-rank test | **[ADV]** |
| Cox proportional hazards regression | **[ADV]** |
| Parametric survival models (Weibull, lognormal, exponential) | **[ADV]** |
| Nelson-Aalen estimator | **[ADV]** |
| Life tables | **[ADV]** |
| Reliability distribution fitting | **[ADV]** |
| Warranty analysis | **[ADV]** |

### 3.10 Quality & Process Analysis
| Feature | Priority |
|---------|----------|
| Control charts (Xbar-R, Xbar-S, I-MR) | **[ADV]** |
| Control charts (P, NP, C, U) | **[ADV]** |
| Control charts (CUSUM, EWMA) | **[ADV]** |
| Process capability (Cp, Cpk, Pp, Ppk) | **[ADV]** |
| Measurement System Analysis (Gage R&R) | **[ADV]** |
| Pareto analysis | **[ADV]** |
| Cause-and-effect diagrams | **[ADV]** |
| Multi-vari analysis | **[ADV]** |
| Acceptance sampling | **[ADV]** |

### 3.11 Design of Experiments (DOE)
| Feature | Priority |
|---------|----------|
| Full factorial designs | **[ADV]** |
| Fractional factorial designs | **[ADV]** |
| Screening designs (Plackett-Burman, definitive screening) | **[ADV]** |
| Response surface designs (CCD, Box-Behnken) | **[ADV]** |
| Mixture designs | **[ADV]** |
| Custom/optimal designs (D-optimal, I-optimal) | **[ADV]** |
| Taguchi designs | **[ADV]** |
| Design augmentation | **[ADV]** |
| Response optimization (desirability functions) | **[ADV]** |

### 3.12 Machine Learning
| Feature | Priority |
|---------|----------|
| Linear regression | **[MVP]** |
| Logistic regression | **[MVP]** |
| Decision trees (classification and regression) | **[MVP]** |
| Random forest | **[MVP]** |
| Gradient boosting (XGBoost / LightGBM) | **[ADV]** |
| k-Nearest Neighbors | **[MVP]** |
| Support Vector Machines (SVM) | **[ADV]** |
| Naive Bayes | **[ADV]** |
| k-Means clustering | **[MVP]** |
| Hierarchical clustering | **[ADV]** |
| DBSCAN | **[ADV]** |
| Principal Component Analysis (PCA) | **[MVP]** |
| Cross-validation (k-fold, stratified, LOOCV) | **[MVP]** |
| Train/test split | **[MVP]** |
| Model evaluation metrics (accuracy, precision, recall, F1, AUC-ROC, RMSE, MAE, R-squared) | **[MVP]** |
| Confusion matrix | **[MVP]** |
| ROC curve | **[MVP]** |
| Feature importance (permutation, Gini, SHAP) | **[ADV]** |
| Hyperparameter tuning (grid search, random search) | **[ADV]** |
| AutoML (automatic model selection) | **[ADV]** |
| Neural networks (basic MLP) | **[ADV]** |

### 3.13 Data Manipulation & Preparation
| Feature | Priority |
|---------|----------|
| File import (CSV, Excel, JSON, Parquet, SQL databases) | **[MVP]** |
| Data type detection and conversion | **[MVP]** |
| Column renaming and reordering | **[MVP]** |
| Filtering and sorting | **[MVP]** |
| Group-by aggregation | **[MVP]** |
| Pivot / unpivot (wide to long, long to wide) | **[MVP]** |
| Merge / join (inner, outer, left, right) | **[MVP]** |
| Concatenation (append rows/columns) | **[MVP]** |
| Handling missing values (drop, fill, interpolate) | **[MVP]** |
| Outlier detection (IQR, z-score, isolation forest) | **[MVP]** |
| Data normalization / standardization | **[MVP]** |
| Binning / discretization | **[MVP]** |
| String operations (split, extract, replace, regex) | **[MVP]** |
| Date/time parsing and extraction | **[MVP]** |
| Formula columns (computed/derived columns) | **[MVP]** |
| Sampling (random, stratified) | **[MVP]** |
| Encoding (one-hot, label, ordinal, target) | **[MVP]** |
| Log, sqrt, Box-Cox transformations | **[MVP]** |
| Duplicate detection and removal | **[MVP]** |
| Column statistics summary (data profiling) | **[MVP]** |

### 3.14 Distribution Fitting
| Feature | Priority |
|---------|----------|
| Normal distribution fitting and testing | **[MVP]** |
| Log-normal distribution | **[MVP]** |
| Exponential distribution | **[ADV]** |
| Weibull distribution | **[ADV]** |
| Gamma distribution | **[ADV]** |
| Beta distribution | **[ADV]** |
| Poisson distribution | **[ADV]** |
| Binomial distribution | **[ADV]** |
| Uniform distribution | **[ADV]** |
| Chi-square distribution | **[ADV]** |
| Student's t distribution | **[ADV]** |
| Automatic best-fit distribution selection | **[ADV]** |
| Q-Q plots for distribution assessment | **[MVP]** |
| P-P plots | **[ADV]** |
| Goodness-of-fit tests (KS, AD, chi-square) | **[ADV]** |

---

## 4. Complete Visualization Type List

All visualization types that should be supported, organized by category.

### 4.1 Basic Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Bar chart (vertical, horizontal, stacked, grouped) | **[MVP]** | Tooltips, click-to-filter, sort |
| Line chart (continuous, stepped) | **[MVP]** | Tooltips, zoom, pan, hover crosshair |
| Area chart (standard, stacked, normalized) | **[MVP]** | Tooltips, hover |
| Pie chart / donut chart | **[MVP]** | Tooltips, click-to-explode |
| Scatter plot | **[MVP]** | Tooltips, brush selection, zoom, color/size encoding |
| Bubble chart | **[MVP]** | Tooltips, size encoding, color encoding |
| Table / data grid | **[MVP]** | Sort, filter, pagination, column resize, conditional formatting |

### 4.2 Distribution Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Histogram (with bin control) | **[MVP]** | Adjustable bins, overlay distribution curve |
| Density plot (KDE) | **[MVP]** | Bandwidth adjustment |
| Box plot / box-and-whisker | **[MVP]** | Outlier identification, hover stats |
| Violin plot | **[MVP]** | Hover quartile lines |
| Strip plot / jitter plot | **[MVP]** | Point selection |
| Swarm / beeswarm plot | **[ADV]** | Point selection |
| Ridgeline / joy plot | **[ADV]** | Scroll through groups |
| ECDF (empirical CDF) plot | **[ADV]** | Hover percentile |
| Q-Q plot (quantile-quantile) | **[MVP]** | Reference line, hover point identity |
| P-P plot (probability-probability) | **[ADV]** | Reference line |
| Dot plot | **[ADV]** | Hover values |
| Rug plot (as overlay) | **[ADV]** | -- |

### 4.3 Comparison Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Grouped bar chart | **[MVP]** | Toggle groups |
| Stacked bar chart (absolute and %) | **[MVP]** | Toggle categories |
| Diverging bar chart | **[ADV]** | Hover values |
| Lollipop chart | **[ADV]** | Hover values |
| Bullet chart | **[ADV]** | Target line overlay |
| Dumbbell / Cleveland dot plot | **[ADV]** | Hover start/end values |
| Slope chart | **[ADV]** | Highlight individual lines |
| Bump chart (rank over time) | **[ADV]** | Highlight series |
| Radar / spider chart | **[ADV]** | Hover axis values |
| Parallel coordinates plot | **[ADV]** | Brush selection on axes, reorder axes |

### 4.4 Correlation & Relationship Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Scatter plot with trend line | **[MVP]** | Toggle regression types, confidence band |
| Scatter plot matrix (SPLOM) | **[MVP]** | Brush across all panels |
| Correlation heatmap / matrix | **[MVP]** | Click cell to scatter plot, hover r-value |
| Bubble plot (3-variable scatter) | **[MVP]** | Size and color legend |
| Hexbin plot | **[ADV]** | Color by count |
| 2D density contour plot | **[ADV]** | Contour level adjustment |
| Joint plot (scatter + marginal distributions) | **[ADV]** | Toggle marginal type |
| Pair plot / generalized pairs plot | **[MVP]** | Brush selection |
| Residual plot | **[MVP]** | Hover point identity |

### 4.5 Composition & Part-of-Whole Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Pie chart | **[MVP]** | Explode segments, hover percentages |
| Donut chart | **[MVP]** | Inner label, hover |
| Treemap | **[MVP]** | Drill-down, hover |
| Sunburst chart | **[ADV]** | Drill-down through levels |
| Icicle chart | **[ADV]** | Drill-down |
| Stacked area chart (100%) | **[MVP]** | Hover composition |
| Mosaic plot / Marimekko chart | **[ADV]** | Click cells |
| Waffle chart | **[ADV]** | Hover count |
| Waterfall chart | **[ADV]** | Hover cumulative total |
| Funnel chart | **[ADV]** | Hover conversion rates |
| Sankey / alluvial diagram | **[ADV]** | Hover flow values |

### 4.6 Time Series Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Time series line chart | **[MVP]** | Zoom, pan, range selector |
| Multi-series time series | **[MVP]** | Toggle series, shared tooltip |
| Area chart with time axis | **[MVP]** | Range brush |
| Candlestick / OHLC chart | **[ADV]** | Zoom, crosshair |
| Sparklines (inline mini charts) | **[ADV]** | Hover value |
| Calendar heatmap | **[ADV]** | Hover date |
| Gantt chart | **[ADV]** | Scroll, zoom |
| Step chart | **[ADV]** | Hover transitions |
| Event timeline | **[ADV]** | Zoom, hover details |

### 4.7 Statistical Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Error bar chart (mean +/- SE/SD/CI) | **[MVP]** | Toggle error type |
| Forest plot (meta-analysis) | **[ADV]** | Hover study details |
| Bland-Altman plot | **[ADV]** | Hover point identity |
| ROC curve | **[MVP]** | Hover threshold, AUC display |
| Precision-Recall curve | **[ADV]** | Hover threshold |
| Lift / gain chart | **[ADV]** | Hover percentile |
| Calibration plot | **[ADV]** | Reference line |
| Normal probability plot | **[MVP]** | Reference line |
| Control chart (I-MR, Xbar-R, P, C) | **[ADV]** | Control limit highlights, out-of-control signals |
| Pareto chart | **[ADV]** | Cumulative line overlay |
| Multi-vari chart | **[ADV]** | Hover values |
| Capability histogram (with spec limits) | **[ADV]** | Cp/Cpk overlay |

### 4.8 Geospatial Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Choropleth map (filled regions) | **[ADV]** | Hover region values, zoom |
| Symbol / bubble map | **[ADV]** | Hover, size/color legend |
| Point map (scatter on map) | **[ADV]** | Cluster, zoom |
| Heatmap on map (density) | **[ADV]** | Intensity adjustment |
| Flow map (origin-destination) | **[ADV]** | Hover flow values |

### 4.9 3D Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| 3D scatter plot | **[ADV]** | Rotate, zoom, hover |
| 3D surface plot | **[ADV]** | Rotate, zoom, hover Z-value |
| 3D wireframe / mesh plot | **[ADV]** | Rotate, transparency |
| 3D bar chart | **[ADV]** | Rotate, hover |
| Contour plot (2D projection of 3D) | **[MVP]** | Level adjustment, hover Z-value |

### 4.10 Heatmaps & Matrices
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Heatmap (generic) | **[MVP]** | Hover value, color scale adjustment |
| Correlation matrix heatmap | **[MVP]** | Click to scatter, hover r + p-value |
| Cluster heatmap (with dendrograms) | **[ADV]** | Reorder, zoom |
| Confusion matrix | **[MVP]** | Hover counts and percentages |
| Highlight table (text + color) | **[MVP]** | Sort, hover |
| Calendar heatmap | **[ADV]** | Hover date, value |

### 4.11 Profilers & Interactive Analysis (JMP-Inspired)
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Prediction profiler (one-at-a-time factor traces) | **[ADV]** | Drag factor values, real-time response update |
| Contour profiler (2D response surface) | **[ADV]** | Drag axes, constraint overlay |
| Surface profiler (3D response surface) | **[ADV]** | Rotate, select factors |
| Desirability profiler | **[ADV]** | Set desirability functions |
| Interaction profiler | **[ADV]** | Select factor pairs |
| Distribution platform (histogram + stats + fitted curve) | **[MVP]** | Toggle fits, adjust bins |

### 4.12 Network & Hierarchical Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Network graph (force-directed) | **[ADV]** | Drag nodes, zoom, hover |
| Dendrogram (hierarchical clustering) | **[ADV]** | Cut at height, hover |
| Tree diagram | **[ADV]** | Expand/collapse, hover |
| Chord diagram | **[ADV]** | Hover connections |

### 4.13 Word & Text Charts
| Visualization | Priority | Interactive Features |
|--------------|----------|---------------------|
| Word cloud | **[ADV]** | Hover frequency, click to filter |
| Bar chart of term frequency | **[ADV]** | Sort, hover |

---

## 5. Recommended Tech Stack

### 5.1 Backend: Python

| Component | Library | Rationale |
|-----------|---------|-----------|
| **Web Framework** | **FastAPI** | Async-first, high performance (ASGI), automatic OpenAPI docs, type hints with Pydantic, 38% adoption among Python devs in 2025. Best choice for data-intensive APIs with real-time capabilities. |
| **Data Manipulation** | **pandas** + **polars** | pandas for compatibility and ecosystem breadth; polars for large dataset performance (lazy evaluation, multi-threaded Rust engine). Use polars for heavy computation, pandas for statsmodels/scipy compatibility. |
| **In-Process SQL** | **DuckDB** | Query DataFrames and Parquet files with SQL. Out-of-core processing for datasets larger than RAM. Zero-copy Arrow integration. |
| **Numerical Computing** | **NumPy** | Foundation for all numerical operations. Array broadcasting, linear algebra, random number generation. |
| **Statistical Analysis** | **scipy.stats** + **statsmodels** + **pingouin** | scipy.stats for hypothesis tests and distributions; statsmodels for regression, GLM, time series, ANOVA; pingouin for simplified statistical testing with effect sizes. |
| **Machine Learning** | **scikit-learn** + **XGBoost** + **LightGBM** | scikit-learn for core ML (classification, regression, clustering, dimensionality reduction, pipelines, evaluation). XGBoost/LightGBM for gradient boosting. |
| **Survival Analysis** | **lifelines** | Kaplan-Meier, Cox PH, parametric models, competing risks. |
| **DOE** | **pyDOE2** + **formulaic** | Design generation (factorial, response surface, Latin hypercube). formulaic for Wilkinson formula notation (like R's formula syntax). |
| **Data Serialization** | **Apache Arrow** + **Parquet** | Zero-copy data interchange between libraries. Parquet for efficient columnar storage and fast I/O. |
| **Task Queue** | **Celery** + **Redis** | Background processing for long-running analyses. Redis as message broker and caching layer. |
| **Database** | **PostgreSQL** | Project metadata, user sessions, saved analyses. Robust, proven, free. |
| **File Storage** | **MinIO** (S3-compatible) or local filesystem | Store uploaded datasets and exported files. |
| **WebSocket** | **FastAPI WebSocket** or **Socket.IO** | Real-time updates for long-running analyses and collaborative features. |
| **Authentication** | **FastAPI-Users** or **Auth0** | User management, OAuth2 / JWT tokens. |

### 5.2 Frontend: React + TypeScript

| Component | Library | Rationale |
|-----------|---------|-----------|
| **Framework** | **React 19** + **TypeScript** | Dominant frontend framework, strong typing, vast ecosystem. |
| **Build Tool** | **Vite** | Fast development server, optimized production builds. |
| **Charting (primary)** | **Plotly.js** (via react-plotly.js) | 50+ chart types, 3D support, built-in interactivity (zoom, pan, hover, selection, animation), statistical charts, WebGL for large datasets. Direct correspondence with Python Plotly for server-side rendering. |
| **Charting (supplemental)** | **D3.js** | Custom visualizations not covered by Plotly (network graphs, Sankey, custom profilers). |
| **Charting (supplemental)** | **Observable Plot** or **Vega-Lite** | Grammar-of-graphics declarative charts for rapid prototyping. |
| **Data Grid** | **AG Grid** (Community) or **TanStack Table** | AG Grid for enterprise-grade data table with sorting, filtering, grouping, pivoting, infinite scroll, cell editing, CSV export. TanStack Table as headless alternative for custom UI. |
| **State Management** | **Zustand** or **Jotai** | Lightweight, minimal boilerplate, works well with React concurrent features. |
| **Data Fetching** | **TanStack Query** (React Query) | Caching, background refetching, pagination, optimistic updates. Structural sharing prevents unnecessary re-renders. |
| **UI Components** | **shadcn/ui** + **Tailwind CSS** | Accessible, composable components with utility-first CSS. Not a heavy component library -- copy-paste components you own. |
| **Drag-and-Drop** | **dnd-kit** | Modern DnD library for drag-and-drop graph building (JMP Graph Builder inspired). |
| **Layout** | **React Grid Layout** or **react-mosaic** | Resizable, draggable panel layout for multi-view analysis workspaces. |
| **Code Editor** | **Monaco Editor** (VS Code editor) | For formula editing, custom SQL queries, Python scripting within the app. |
| **Rich Text** | **Tiptap** or **Lexical** | For report/annotation editing. |
| **Routing** | **React Router 7** or **TanStack Router** | Type-safe routing for multi-page app structure. |

### 5.3 Infrastructure & DevOps

| Component | Tool | Rationale |
|-----------|------|-----------|
| **Containerization** | **Docker** + **Docker Compose** | Reproducible development and deployment environments. |
| **Orchestration** | **Kubernetes** (production) | Horizontal scaling for compute-intensive analyses. |
| **CI/CD** | **GitHub Actions** | Automated testing, linting, building, deployment. |
| **Monitoring** | **Prometheus** + **Grafana** | Performance monitoring and alerting. |
| **Logging** | **structlog** (Python) | Structured logging for debugging and audit. |
| **Testing (Backend)** | **pytest** + **httpx** | Unit and integration testing. |
| **Testing (Frontend)** | **Vitest** + **Playwright** | Unit testing and end-to-end testing. |
| **API Documentation** | **FastAPI auto-docs** (Swagger/ReDoc) | Automatically generated from type hints. |

---

## 6. Proposed Architecture

### 6.1 High-Level Architecture

```
+------------------------------------------------------------------+
|                         CLIENT (Browser)                          |
|                                                                   |
|  +------------------+  +------------------+  +-----------------+  |
|  |   Graph Builder  |  |  Analysis Panel  |  |   Data Table    |  |
|  |  (Drag & Drop)   |  | (Stats/ML Forms) |  |   (AG Grid)     |  |
|  +--------+---------+  +--------+---------+  +--------+--------+  |
|           |                      |                     |          |
|  +--------v---------+  +--------v---------+            |          |
|  |  Plotly.js / D3   |  |  Result Viewer   |            |          |
|  |  (Visualization)  |  |  (Tables/Stats)  |            |          |
|  +-------------------+  +------------------+            |          |
|                                                         |          |
|  +------------------------------------------------------v-------+ |
|  |              React State (Zustand / Jotai)                    | |
|  |  - Active dataset reference   - Current analysis config      | |
|  |  - View layout                - Selection/brush state        | |
|  +----------------------------+----------------------------------+ |
|                               |                                   |
+-------------------------------|-----------------------------------+
                                | REST API + WebSocket
                                |
+-------------------------------|-----------------------------------+
|                         API GATEWAY                               |
|                        (FastAPI / Uvicorn)                        |
+-------------------------------|-----------------------------------+
                                |
        +-----------------------+-----------------------+
        |                       |                       |
+-------v-------+    +---------v--------+    +---------v--------+
|  Data Service |    | Analysis Service |    |  ML Service      |
|               |    |                  |    |                  |
| - File upload |    | - scipy.stats    |    | - scikit-learn   |
| - Data parse  |    | - statsmodels    |    | - XGBoost        |
| - DuckDB SQL  |    | - pingouin       |    | - LightGBM       |
| - Transform   |    | - lifelines      |    | - Model mgmt     |
| - Profile     |    | - pyDOE2         |    | - Cross-val      |
| - pandas/     |    | - Hypothesis     |    | - Feature eng    |
|   polars      |    |   tests          |    |                  |
+-------+-------+    +---------+--------+    +---------+--------+
        |                       |                       |
        +-----------------------+-----------------------+
                                |
                  +-------------v--------------+
                  |     Data Layer             |
                  |                            |
                  |  +----------+ +----------+ |
                  |  |PostgreSQL| |  Redis    | |
                  |  |(metadata)| | (cache/   | |
                  |  |          | |  sessions)| |
                  |  +----------+ +----------+ |
                  |                            |
                  |  +----------+ +----------+ |
                  |  |  MinIO   | |  Parquet  | |
                  |  |(files)   | | (datasets)| |
                  |  +----------+ +----------+ |
                  +----------------------------+
```

### 6.2 Core Design Principles

1. **Interactive-First**: Every visualization supports hover, click, brush selection, and zoom. Selections in one view propagate to all linked views (cross-filtering), inspired by JMP's dynamic linking.

2. **Analysis as Configuration**: Statistical analyses are defined as JSON configuration objects (dataset, method, parameters, output format). This enables reproducibility, undo/redo, and sharing.

3. **Lazy Computation**: Data transformations and analyses are computed on-demand. The frontend sends analysis specifications; the backend computes and returns results. Large datasets stay on the server.

4. **Columnar Data Pipeline**: Data flows through the system in Apache Arrow columnar format. DuckDB handles SQL queries. pandas/polars handle transformations. Results are serialized as Arrow IPC or JSON for the frontend.

5. **Progressive Disclosure**: The UI presents simple options by default (like Minitab's Assistant), with advanced options available on expansion. Non-statisticians can run analyses; experts can customize every parameter.

### 6.3 Key Application Modules

#### Module 1: Data Workspace
- **Data Table View**: Spreadsheet-like interface with column types, sorting, filtering, and inline editing.
- **Data Profiler**: Automatic summary of each column (distribution histogram, missing %, unique count, descriptive stats).
- **Data Import**: CSV, Excel, JSON, Parquet, SQL connection, URL fetch, paste from clipboard.
- **Transform Pipeline**: Visual pipeline of transformations (filter, mutate, join, pivot, aggregate) with preview.
- **Formula Editor**: Define computed columns using a formula language (similar to JMP or Excel formulas).

#### Module 2: Visualization Builder (Graph Builder)
- **Drag-and-Drop**: Drag columns onto X, Y, Color, Size, Shape, Facet zones to build charts (inspired by JMP Graph Builder and Tableau).
- **Smart Chart Recommendations**: Based on data types of dragged variables, suggest appropriate chart types (like Tableau Show Me).
- **Chart Type Selector**: Gallery of all available chart types with previews.
- **Customization Panel**: Titles, axis labels, colors, scales (linear/log), legends, annotations, reference lines.
- **Multi-View Layout**: Arrange multiple charts in a resizable grid. Charts are linked for cross-filtering.
- **Export**: PNG, SVG, PDF, interactive HTML.

#### Module 3: Analysis Console
- **Guided Analysis**: Select analysis type from a categorized menu. Fill in a form (response variable, factors, options). Run and view results.
- **Result Viewer**: Structured output with tables, charts, diagnostic plots, and interpretation hints.
- **Analysis History**: Every analysis is logged and can be re-run, modified, or shared.
- **Comparison View**: Compare results of multiple analyses side by side.

#### Module 4: Dashboard & Reporting
- **Dashboard Builder**: Compose charts, tables, and text into interactive dashboards.
- **Filters and Parameters**: Add global filters (dropdowns, sliders, date ranges) that control all dashboard elements.
- **Export**: PDF report, standalone HTML, embedded iframe.
- **Sharing**: Generate shareable links with view-only access.

#### Module 5: Scripting & Extensibility
- **Built-in Python Console**: Execute custom Python code against the active dataset.
- **Custom Analysis Plugins**: Users can write Python functions that appear as custom analyses in the UI.
- **API Access**: REST API for programmatic access to all features.
- **Notebook Integration**: Export analysis to Jupyter notebook format.

### 6.4 Data Flow for a Typical Analysis

```
User uploads CSV
       |
       v
[Data Service] parses CSV -> Arrow Table -> stores as Parquet
       |
       v
[Data Profiler] generates column summaries -> cached in Redis
       |
       v
User drags columns to Graph Builder
       |
       v
[Frontend] sends viz spec: { dataset_id, x: "age", y: "income", color: "region", type: "scatter" }
       |
       v
[API] receives spec -> [Data Service] queries DuckDB -> returns Arrow IPC
       |
       v
[Frontend] renders Plotly chart with interactivity
       |
       v
User selects "Linear Regression" from Analysis Console
       |
       v
[Frontend] sends analysis spec: { dataset_id, method: "ols", y: "income", x: ["age", "education", "region"] }
       |
       v
[Analysis Service] runs statsmodels OLS -> returns {coefficients, p_values, r_squared, residual_plot_data, diagnostic_plots_data}
       |
       v
[Frontend] renders results: coefficient table + 4 diagnostic plots (residuals vs fitted, Q-Q, scale-location, leverage)
```

### 6.5 Cross-Filtering / Dynamic Linking Architecture

Inspired by JMP's linked graphs:

```
User brushes points in Scatter Plot A
       |
       v
[Frontend Selection Manager] captures selected row indices: [12, 45, 67, 89, ...]
       |
       v
[Broadcast to all linked views via Zustand store]
       |
       +--> Scatter Plot B: highlights corresponding points
       +--> Histogram C: highlights corresponding bars / overlays selection distribution
       +--> Data Table: scrolls to and highlights selected rows
       +--> Box Plot D: highlights corresponding points
       |
       v
User can filter to selection (keep only selected) or exclude selection
```

This is a client-side operation -- no server round-trips needed for selection propagation. The shared state store maintains a `selectedIndices: Set<number>` that all visualization components subscribe to.

---

## 7. Feature Priority List (MVP vs Advanced)

### Phase 1: MVP (Months 1-4)
**Goal:** A functional web-based data exploration tool that can import data, show summaries, create basic visualizations, and run fundamental statistical analyses.

#### Data
- [ ] CSV and Excel file upload (drag-and-drop, up to 500MB)
- [ ] Automatic data type detection (numeric, categorical, datetime, text)
- [ ] Data table view with sorting, filtering, and pagination
- [ ] Column profiler (automatic histogram, missing %, unique count, descriptive stats for every column)
- [ ] Basic data transformations (filter rows, select columns, rename, sort, add computed column)
- [ ] Handle missing values (drop rows, fill with mean/median/mode/constant)
- [ ] Data normalization and standardization

#### Visualizations
- [ ] Scatter plot (with optional trend line, color/size encoding)
- [ ] Line chart (single and multi-series)
- [ ] Bar chart (vertical, horizontal, stacked, grouped)
- [ ] Histogram (adjustable bins, optional distribution overlay)
- [ ] Box plot (single and grouped)
- [ ] Violin plot
- [ ] Density plot (KDE)
- [ ] Pie / donut chart
- [ ] Heatmap (generic and correlation matrix)
- [ ] Scatter plot matrix (SPLOM)
- [ ] Q-Q plot
- [ ] Area chart
- [ ] Error bar chart
- [ ] Confusion matrix
- [ ] ROC curve
- [ ] Contour plot (2D)
- [ ] Residual plots (4-panel diagnostic)
- [ ] Distribution platform (histogram + stats + fitted curve, JMP-inspired)

#### Interactivity
- [ ] Hover tooltips on all charts
- [ ] Zoom and pan on all charts
- [ ] Brush selection on scatter plots
- [ ] Cross-filtering between linked views (select in one chart, highlight in others)
- [ ] Click-to-drill on treemaps and hierarchical charts
- [ ] Chart export (PNG, SVG)

#### Statistical Analysis
- [ ] Descriptive statistics (all standard measures with confidence intervals)
- [ ] Frequency tables and cross-tabulation
- [ ] One-sample, two-sample, and paired t-tests
- [ ] Chi-square test of independence and goodness-of-fit
- [ ] Shapiro-Wilk normality test
- [ ] Levene's test for equal variances
- [ ] One-way and two-way ANOVA with Tukey HSD and Bonferroni post-hoc
- [ ] Pearson and Spearman correlation
- [ ] Correlation matrix with p-values
- [ ] Simple and multiple linear regression (with diagnostics)
- [ ] Logistic regression (binary)
- [ ] Polynomial regression
- [ ] Mann-Whitney U test
- [ ] Wilcoxon signed-rank test
- [ ] Kruskal-Wallis test
- [ ] Normal distribution fitting
- [ ] Effect sizes (Cohen's d, eta-squared)
- [ ] R-squared, adjusted R-squared, AIC, BIC

#### Machine Learning
- [ ] Train/test split
- [ ] k-Fold cross-validation
- [ ] Linear and logistic regression
- [ ] Decision tree (classification and regression)
- [ ] Random forest
- [ ] k-Nearest Neighbors
- [ ] k-Means clustering
- [ ] PCA
- [ ] Model evaluation metrics (accuracy, precision, recall, F1, RMSE, MAE, AUC-ROC)
- [ ] Confusion matrix visualization
- [ ] ROC curve visualization
- [ ] Feature importance bar chart

#### UI/UX
- [ ] Multi-panel workspace layout (data table, chart, analysis results)
- [ ] Graph Builder with drag-and-drop column assignment
- [ ] Analysis menu with categorized statistical tests
- [ ] Result viewer with structured tables and embedded charts
- [ ] Dark mode and light mode
- [ ] Responsive layout for different screen sizes

---

### Phase 2: Advanced Analytics (Months 5-8)
**Goal:** Add sophisticated statistical methods, more visualization types, and productivity features.

#### Data
- [ ] JSON and Parquet file import
- [ ] SQL database connections (PostgreSQL, MySQL, SQLite)
- [ ] Data merge/join (inner, outer, left, right)
- [ ] Pivot and unpivot
- [ ] Group-by aggregation builder
- [ ] String operations (split, extract, replace, regex)
- [ ] Date/time extraction (year, month, day, hour, day of week)
- [ ] Outlier detection (IQR method, z-score, isolation forest)
- [ ] Sampling (random, stratified)
- [ ] Encoding (one-hot, label, ordinal)
- [ ] Log, sqrt, Box-Cox transformations
- [ ] Duplicate detection and removal
- [ ] DuckDB SQL query editor

#### Visualizations
- [ ] Treemap
- [ ] Sunburst chart
- [ ] Waterfall chart
- [ ] Bubble chart (animated over time dimension)
- [ ] Parallel coordinates plot
- [ ] Pair plot with configurable diagonal
- [ ] Strip/jitter plot
- [ ] Ridgeline / joy plot
- [ ] ECDF plot
- [ ] Lollipop chart
- [ ] Bullet chart
- [ ] Diverging bar chart
- [ ] Slope chart
- [ ] Bump chart
- [ ] Radar / spider chart
- [ ] 3D scatter plot
- [ ] 3D surface plot
- [ ] Mosaic plot
- [ ] Gantt chart
- [ ] Calendar heatmap
- [ ] Control charts (I-MR, Xbar-R, P, C)
- [ ] Pareto chart
- [ ] Cluster heatmap with dendrograms
- [ ] Sankey diagram
- [ ] Choropleth map
- [ ] Sparklines
- [ ] Multi-vari chart

#### Statistical Analysis
- [ ] Multi-way ANOVA
- [ ] Repeated measures ANOVA
- [ ] MANOVA
- [ ] ANOCOVA
- [ ] Welch's ANOVA
- [ ] Additional post-hoc tests (Scheffe, Dunnett, Games-Howell, Dunn)
- [ ] Stepwise regression (forward, backward, both)
- [ ] Ridge, Lasso, Elastic Net regression
- [ ] Polynomial and nonlinear regression
- [ ] GLM (all families: Gaussian, binomial, Poisson, gamma)
- [ ] Robust regression
- [ ] Quantile regression
- [ ] Fisher's exact test
- [ ] McNemar's test
- [ ] Kolmogorov-Smirnov test
- [ ] Anderson-Darling test
- [ ] Bartlett's test
- [ ] Kendall's tau
- [ ] Partial correlation
- [ ] Autocorrelation (ACF) and partial autocorrelation (PACF)
- [ ] Time series decomposition
- [ ] Exponential smoothing
- [ ] ARIMA / SARIMA modeling
- [ ] Bootstrap confidence intervals
- [ ] Permutation tests
- [ ] Distribution fitting (log-normal, Weibull, gamma, exponential, Poisson, binomial)
- [ ] Automatic best-fit distribution
- [ ] Goodness-of-fit tests
- [ ] Power and sample size calculations

#### Machine Learning
- [ ] Gradient boosting (XGBoost, LightGBM)
- [ ] Support Vector Machines
- [ ] Naive Bayes
- [ ] Hierarchical clustering
- [ ] DBSCAN
- [ ] Gaussian Mixture Models
- [ ] t-SNE and UMAP
- [ ] Factor analysis
- [ ] Feature importance (SHAP values)
- [ ] Hyperparameter tuning (grid search, random search)
- [ ] Learning curves
- [ ] Precision-Recall curve
- [ ] Lift / gain chart

#### Multivariate Analysis
- [ ] Factor analysis (exploratory)
- [ ] Linear Discriminant Analysis
- [ ] Multidimensional Scaling
- [ ] Correspondence analysis
- [ ] Canonical correlation

#### UI/UX
- [ ] Dashboard builder (compose charts + filters)
- [ ] Shareable dashboard links
- [ ] Analysis history and undo/redo
- [ ] Saved analysis templates
- [ ] PDF/HTML report export
- [ ] Custom color palettes and themes
- [ ] Annotation tools (add text, arrows, shapes to charts)

---

### Phase 3: Specialized & Enterprise Features (Months 9-14)
**Goal:** Add domain-specific capabilities, collaboration, and enterprise features that match the depth of JMP, Minitab, and SPSS.

#### Quality & Process
- [ ] Full control chart suite (CUSUM, EWMA, Zone, Laney)
- [ ] Process capability analysis (Cp, Cpk, Pp, Ppk, Cpm, Z-bench)
- [ ] Capability Sixpack
- [ ] Measurement System Analysis (Gage R&R crossed/nested)
- [ ] Acceptance sampling
- [ ] Tolerance intervals
- [ ] Cause-and-effect (fishbone) diagrams
- [ ] Multi-vari analysis

#### Design of Experiments
- [ ] Full factorial design generation
- [ ] Fractional factorial designs
- [ ] Screening designs (Plackett-Burman, definitive screening)
- [ ] Response surface designs (CCD, Box-Behnken)
- [ ] Mixture designs
- [ ] Custom/D-optimal designs
- [ ] Taguchi designs
- [ ] Design augmentation
- [ ] Response optimization with desirability functions
- [ ] Prediction profiler (interactive, JMP-inspired)
- [ ] Contour profiler
- [ ] Surface profiler

#### Survival / Reliability
- [ ] Kaplan-Meier estimator and survival curves
- [ ] Log-rank test
- [ ] Cox proportional hazards regression
- [ ] Parametric survival models (Weibull, lognormal, exponential)
- [ ] Reliability distribution fitting
- [ ] Warranty analysis
- [ ] Life tables

#### Advanced Modeling
- [ ] Mixed-effects models (linear and generalized)
- [ ] Generalized Estimating Equations (GEE)
- [ ] Structural equation modeling
- [ ] Bayesian regression (basic)
- [ ] Neural networks (MLP)
- [ ] AutoML (automated model selection and tuning)

#### Geospatial
- [ ] Choropleth maps with GeoJSON/Shapefile support
- [ ] Point and bubble maps
- [ ] Heatmaps on maps
- [ ] Flow maps
- [ ] Geocoding integration

#### Collaboration & Enterprise
- [ ] User accounts and authentication (OAuth2, SSO)
- [ ] Team workspaces with shared datasets and analyses
- [ ] Role-based access control (viewer, analyst, admin)
- [ ] Audit log of all analyses
- [ ] Scheduled reports (email delivery)
- [ ] API keys for programmatic access
- [ ] Embedded analytics (iframe embed with tokens)
- [ ] Version control for analyses
- [ ] Comments and annotations on shared dashboards
- [ ] Real-time collaborative editing

#### Extensibility
- [ ] Built-in Python scripting console
- [ ] Custom analysis plugin system (Python functions registered as UI-accessible analyses)
- [ ] Custom visualization plugin system
- [ ] Jupyter notebook export
- [ ] R script export
- [ ] Webhook integrations (Slack, Teams, email on analysis completion)
- [ ] REST API for all features

---

### Implementation Priority Summary

| Phase | Timeline | Key Deliverable | Competitive Position |
|-------|----------|----------------|---------------------|
| **Phase 1 (MVP)** | Months 1-4 | Functional data exploration tool with core stats, basic ML, interactive visualizations, and cross-filtering | Matches basic Tableau exploration + exceeds its statistical depth. Simplified alternative to writing Python/R code. |
| **Phase 2 (Advanced)** | Months 5-8 | Comprehensive statistical platform with advanced regression, time series, multivariate analysis, quality charts, dashboards | Approaches SPSS/Minitab statistical depth with superior interactivity and web delivery. |
| **Phase 3 (Specialized)** | Months 9-14 | Full DOE, quality engineering, survival analysis, profilers, collaboration, enterprise features | Matches JMP's interactive profilers and Minitab's quality tools in a web-based platform. |

---

### Success Metrics

1. **Time-to-first-insight**: A new user should go from file upload to first meaningful chart in under 60 seconds.
2. **Statistical breadth**: Phase 2 should cover 90%+ of tests available in an introductory statistics textbook.
3. **Interactivity**: Cross-filtering latency should be under 100ms for datasets up to 1M rows.
4. **Performance**: Handle datasets up to 10M rows with acceptable responsiveness (< 2s for aggregations).
5. **Visualization coverage**: Phase 2 should support 40+ chart types, covering all common analytical use cases.

---

*This document should be treated as a living specification. Update it as competitive landscape changes, user feedback is gathered, and technical constraints are discovered during implementation.*
