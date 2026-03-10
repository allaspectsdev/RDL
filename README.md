# DataLens — Visual Data Analysis Tool

A comprehensive, interactive data analysis platform built in Python. Modeled after the capabilities of JMP, MATLAB, Tableau, and other leading statistical analysis tools — accessible from any web browser.

Upload a dataset and graphically explore your data with 22+ chart types, full statistical analysis, machine learning, and time series forecasting.

---

## Features

### Data Management
- Upload CSV, Excel (.xlsx/.xls), TSV, and JSON files
- 7 built-in sample datasets (Iris, Wine, Housing, Diabetes, Tips, Gapminder, Stocks)
- Data preview, cleaning, type conversion, missing value handling
- Column transforms (log, z-score, normalize, binning), encoding (one-hot, label), computed columns

### Descriptive Statistics
- Summary statistics (mean, median, mode, std, variance, SEM, skewness, kurtosis, CV, confidence intervals)
- Distribution analysis with histogram, KDE, QQ plot, box plot, violin plot, ECDF
- 5 normality tests (Shapiro-Wilk, Anderson-Darling, D'Agostino-Pearson, Kolmogorov-Smirnov, Jarque-Bera)
- Outlier detection (IQR, Z-score, Modified Z-score/MAD)
- Grouped statistics with comparative visualizations

### Visualization Builder
22 interactive chart types powered by Plotly:

| Category | Charts |
|----------|--------|
| Basic | Scatter, Line, Bar, Histogram, Area |
| Distribution | Box Plot, Violin, Strip Plot, Joint Plot |
| Composition | Pie/Donut, Treemap, Sunburst, Waterfall, Funnel |
| Correlation | Heatmap, Bubble, Contour, Parallel Coordinates |
| 3D | 3D Scatter, 3D Surface |
| Specialized | Radar/Spider, Candlestick/OHLC |

All charts include customizable titles, color palettes, opacity, axis controls, and Plotly's built-in export toolbar.

### Hypothesis Testing
- **One-sample:** t-test, Wilcoxon signed-rank, proportion test
- **Two-sample:** Independent/Welch's/paired t-test, Mann-Whitney U, Wilcoxon paired, KS test
- **Chi-square:** Independence, goodness of fit, Fisher's exact (2×2)
- **Normality:** 5 tests with QQ plots
- **Power analysis:** Sample size calculation, power curves for t-tests and ANOVA
- **Multiple comparisons:** Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (FDR)

Every test reports: test statistic, p-value, effect size (Cohen's d, Cramér's V), confidence intervals, and plain-language interpretation.

### Correlation & Multivariate Analysis
- Correlation matrices (Pearson, Spearman, Kendall) with p-value heatmaps
- Scatter plot matrix (SPLOM) with color grouping
- Pairwise scatter with marginal distributions, trendlines, LOESS
- PCA with scree plot, explained variance, biplots, Kaiser criterion
- t-SNE (2D/3D) with perplexity control
- Factor analysis with varimax rotation
- Partial correlation controlling for confounders

### Regression Analysis
- **Simple linear:** OLS with confidence/prediction bands, full coefficient table
- **Multiple linear:** VIF for multicollinearity, coefficient plot with CIs
- **Polynomial:** Degree 2–6, model comparison across degrees
- **Logistic:** Odds ratios, ROC/AUC, confusion matrix, classification report
- **Curve fitting:** 7 models (exponential, logarithmic, power, sigmoid, Gaussian, etc.) — similar to MATLAB's cftool
- **Diagnostics:** Residuals vs fitted, QQ, scale-location, Cook's distance, Durbin-Watson, Breusch-Pagan, Ramsey RESET

### ANOVA
- One-way ANOVA with effect sizes (η², ω²), Levene's test, Welch's ANOVA
- Post-hoc: Tukey HSD, Bonferroni pairwise
- Two-way ANOVA (Type II/III SS) with interaction plots
- Repeated measures ANOVA (Mauchly's sphericity, GG/HF corrections)
- Kruskal-Wallis (nonparametric) with Dunn's post-hoc
- Friedman test with Kendall's W
- ANCOVA with homogeneity of slopes check

### Time Series Analysis
- Interactive exploration with rolling statistics and lag plots
- Decomposition (classical additive/multiplicative, STL)
- Stationarity testing (ADF, KPSS) with differencing
- ACF/PACF analysis with interpretation guide and Ljung-Box test
- Smoothing (SMA, EMA, Holt's linear, Holt-Winters)
- ARIMA/SARIMA modeling with diagnostic plots and forecasting
- Auto-ARIMA grid search
- Multi-model forecast comparison (MAE, RMSE, MAPE)

### Machine Learning
- **Clustering:** K-Means (with elbow/silhouette analysis), DBSCAN, Agglomerative — cluster profiles, 2D/3D scatter
- **Classification:** Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes, Decision Tree — confusion matrix, ROC, precision-recall, feature importance
- **Regression:** Linear, Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, SVR, KNN — predicted vs actual, residuals, feature importance
- **Dimensionality Reduction:** PCA, t-SNE (2D/3D)
- **Model Comparison Dashboard:** Run all algorithms, compare in table + radar chart, auto-highlight best model

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI Framework | [Streamlit](https://streamlit.io) |
| Visualization | [Plotly](https://plotly.com/python/) (interactive, web-native) |
| Data | pandas, NumPy |
| Statistics | SciPy, statsmodels, pingouin |
| Machine Learning | scikit-learn |
| Time Series | statsmodels (ARIMA, SARIMAX, Holt-Winters) |
| Survival | lifelines |
| Web Server | Nginx (reverse proxy with WebSocket support) |
| Containerization | Docker + Docker Compose |
| SSL | Let's Encrypt (Certbot) |

---

## Quick Start (Local)

```bash
# Clone
git clone https://github.com/allaspectsdev/RDL.git
cd RDL

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Production Deployment (Linux Server)

### Option A: Docker (recommended)

```bash
# On your server:
git clone https://github.com/allaspectsdev/RDL.git
cd RDL

# Deploy (installs Docker if needed, builds, starts nginx)
./scripts/deploy.sh

# After DNS is configured, add SSL:
# First edit nginx/datalens.conf and replace YOUR_DOMAIN.com
./scripts/deploy.sh --ssl yourdomain.com
```

### Option B: Systemd (no Docker)

```bash
./scripts/deploy.sh --no-docker
./scripts/deploy.sh --no-docker --ssl yourdomain.com
```

### Updating

```bash
./scripts/update.sh
```

### Management

```bash
# Docker
docker compose logs -f datalens
docker compose restart

# Systemd
sudo systemctl status datalens
sudo journalctl -u datalens -f
```

---

## Project Structure

```
RDL/
├── app.py                        # Main application entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container image
├── docker-compose.yml            # App + Nginx + Certbot stack
├── .streamlit/config.toml        # Streamlit production config
├── modules/
│   ├── data_manager.py           # Upload, preview, clean, transform
│   ├── descriptive_stats.py      # Summary stats, distributions, outliers
│   ├── hypothesis_testing.py     # t-tests, chi-square, power analysis
│   ├── regression.py             # Linear, polynomial, logistic, curve fitting
│   ├── anova.py                  # One-way, two-way, repeated measures, ANCOVA
│   ├── correlation.py            # Correlation matrices, PCA, t-SNE, factor analysis
│   ├── visualization.py          # 22 interactive chart types
│   ├── time_series.py            # Decomposition, ARIMA, forecasting
│   └── machine_learning.py       # Clustering, classification, regression, comparison
├── nginx/
│   ├── nginx.conf                # Main nginx config
│   └── datalens.conf             # Reverse proxy + WebSocket config
└── scripts/
    ├── deploy.sh                 # One-command deployment
    └── update.sh                 # Pull & restart
```

---

## License

MIT
