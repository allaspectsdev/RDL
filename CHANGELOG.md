# Changelog

## [1.1.0] — 2026-03-10

### Robustness Improvements

Comprehensive hardening pass: crash prevention, user-facing warnings for silent failures, and strategic caching. No new features or architectural changes — all surgical 2-10 line edits across 10 files.

#### Crash Prevention
- **Error boundaries** — Every module render is wrapped in try/except; one module crashing no longer kills the whole app (`app.py`)
- **Division by zero** — Grouped statistics CV (`descriptive_stats.py`), chi-square residuals (`hypothesis_testing.py`)
- **Empty mode** — `mode().iloc[0]` crash when all values are unique or NaN (`data_manager.py`)
- **Wilcoxon effect size** — Replaced incorrect `norm.ppf(p/2)` formula with proper Z-approximation from the W-statistic (`hypothesis_testing.py`)
- **Singular matrix** — VIF calculation wrapped in try/except, returns `inf` on failure (`regression.py`)
- **Constant target / too-few points** — Polynomial regression guards `ss_tot == 0` and `n <= degree + 1` (`regression.py`)
- **Small groups** — ANOVA filters out groups with <2 observations and warns user (`anova.py`)
- **Negative eta-squared** — Kruskal-Wallis eta-squared clamped to `max(0, ...)` (`anova.py`)
- **Perfect correlation** — `arctanh(1.0)` inf guard in pairwise scatter CI (`correlation.py`)
- **Factor analysis** — Slider max capped at `n_features - 1` (`correlation.py`)
- **Empty test set** — ARIMA train/test split checks for minimum sizes (`time_series.py`)
- **Imbalanced stratification** — Classification falls back to non-stratified split with warning (`machine_learning.py`)
- **Missing feature importance** — Logistic Regression now shows `|coefficients|` as feature importance (`machine_learning.py`)

#### Silent Failure Fixes
- **Type conversion NaNs** — Warns when values cannot be converted to numeric/datetime (`data_manager.py`)
- **Shapiro-Wilk subsample** — Shows caption when test uses first 5,000 of N values (`descriptive_stats.py`)
- **Heatmap truncation** — Shows caption when raw data heatmap shows only first 50 rows (`visualization.py`)
- **Pie chart overload** — Warns and limits to top 20 categories when >30 unique values (`visualization.py`)
- **sklearn fallback** — Shows warning when logistic regression falls back from statsmodels to sklearn (`regression.py`)
- **MAPE undefined** — Shows "N/A" with explanation when test data contains zeros (`time_series.py`)
- **Frequency inference** — Shows caption when datetime frequency cannot be inferred (`time_series.py`)

#### Performance (Strategic Caching)
- **Sample datasets** — `@st.cache_data` on `load_sample_dataset()` (`app.py`)
- **Sidebar info** — Cached `_dataset_info()` avoids recalculating `memory_usage(deep=True)` etc. on every rerender (`app.py`)
- **Correlation matrix** — Cached `_compute_corr_and_pvals()` for the O(n^2) p-value loop (`correlation.py`)
- **Model comparison** — Cached `_compare_classifiers()` and `_compare_regressors()` (7 models x 5-fold CV = 35 fits) (`machine_learning.py`)
- **Auto-ARIMA** — Cached `_auto_arima_search()` grid search (48 ARIMA model fits) (`time_series.py`)

### Files Changed
| File | Lines | Changes |
|------|------:|---------|
| `app.py` | +69 | Error boundaries, cached dataset loading + sidebar info |
| `modules/machine_learning.py` | +122 | Cached model comparison, stratified split fallback, coef_ importance |
| `modules/time_series.py` | +64 | Cached auto-ARIMA, train/test guards, MAPE/frequency notices |
| `modules/correlation.py` | +62 | Cached p-value matrix, arctanh guard, factor analysis cap |
| `modules/regression.py` | +19 | VIF try/except, polynomial R^2 guards, sklearn fallback warning |
| `modules/visualization.py` | +15 | Heatmap truncation notice, pie chart category limit |
| `modules/anova.py` | +13 | Small-group filter, eta-squared clamp |
| `modules/data_manager.py` | +11 | Mode guard, conversion NaN warning |
| `modules/hypothesis_testing.py` | +8 | Chi-square residual guard, Wilcoxon Z-approx fix |
| `modules/descriptive_stats.py` | +4 | CV zero-division guard, Shapiro subsample notice |
