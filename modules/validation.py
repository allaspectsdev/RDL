"""
RDL Validation Engine — Pure-logic assumption checks, recommendation engine,
and interpretation templates.  Returns dataclass objects consumed by ui_helpers.
No Streamlit UI code lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class ValidationCheck:
    """Single assumption / quality check result."""
    name: str
    status: str          # "pass", "warn", "fail"
    detail: str          # e.g. "p = 0.342 > 0.05"
    suggestion: str = "" # e.g. "Consider Mann-Whitney U test"


@dataclass
class Interpretation:
    """Plain-language interpretation of a statistical result."""
    title: str
    body: str
    detail: str = ""


# ─── Sample-size thresholds by test type ─────────────────────────────────────

_MIN_N: dict[str, int] = {
    "t-test": 20,
    "paired-t": 20,
    "anova": 20,       # per group
    "chi-square": 30,
    "correlation": 30,
    "regression": 30,
    "pca": 50,
    "ml-classification": 50,
    "ml-regression": 50,
    "survival": 30,
    "arima": 50,
}


# ─── Core Check Functions ────────────────────────────────────────────────────

def check_normality(data: np.ndarray, label: str = "") -> ValidationCheck:
    """Shapiro-Wilk normality test.  Subsamples if n > 5000."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 8:
        return ValidationCheck(
            name=f"Normality{f' ({label})' if label else ''}",
            status="warn",
            detail=f"n = {n} is too small for a reliable normality test",
            suggestion="Collect more data or use a non-parametric test",
        )
    sample = arr if n <= 5000 else np.random.default_rng(0).choice(arr, 5000, replace=False)
    stat, p = stats.shapiro(sample)
    if p >= 0.05:
        return ValidationCheck(
            name=f"Normality{f' ({label})' if label else ''}",
            status="pass",
            detail=f"Shapiro-Wilk p = {p:.4f} (data appears normally distributed)",
        )
    return ValidationCheck(
        name=f"Normality{f' ({label})' if label else ''}",
        status="warn" if p >= 0.01 else "fail",
        detail=f"Shapiro-Wilk p = {p:.4f} (non-normal distribution detected)",
        suggestion="Consider a non-parametric alternative",
    )


def check_equal_variance(*groups: np.ndarray) -> ValidationCheck:
    """Levene's test for equality of variances across groups."""
    clean = [np.asarray(g, dtype=float)[np.isfinite(np.asarray(g, dtype=float))] for g in groups]
    clean = [g for g in clean if len(g) >= 2]
    if len(clean) < 2:
        return ValidationCheck(
            name="Equal Variance",
            status="warn",
            detail="Not enough valid groups to test (need >= 2 groups with n >= 2)",
        )
    stat, p = stats.levene(*clean)
    if p >= 0.05:
        return ValidationCheck(
            name="Equal Variance (Levene's)",
            status="pass",
            detail=f"Levene's p = {p:.4f} (variances appear equal)",
        )
    return ValidationCheck(
        name="Equal Variance (Levene's)",
        status="warn" if p >= 0.01 else "fail",
        detail=f"Levene's p = {p:.4f} (unequal variances detected)",
        suggestion="Consider Welch's correction or a non-parametric test",
    )


def check_sample_size(n: int, test_type: str) -> ValidationCheck:
    """Check whether sample size meets minimum recommendations."""
    min_n = _MIN_N.get(test_type, 20)
    if n >= min_n:
        return ValidationCheck(
            name="Sample Size",
            status="pass",
            detail=f"n = {n} meets the recommended minimum of {min_n} for {test_type}",
        )
    if n >= min_n * 0.5:
        return ValidationCheck(
            name="Sample Size",
            status="warn",
            detail=f"n = {n} is below the recommended {min_n} for {test_type}",
            suggestion="Results may have low statistical power — interpret with caution",
        )
    return ValidationCheck(
        name="Sample Size",
        status="fail",
        detail=f"n = {n} is well below the recommended {min_n} for {test_type}",
        suggestion="Results may be unreliable — consider collecting more data",
    )


def check_multicollinearity(
    X: np.ndarray | pd.DataFrame,
    col_names: list[str] | None = None,
    threshold: float = 10.0,
) -> ValidationCheck:
    """VIF-based multicollinearity check."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    if isinstance(X, pd.DataFrame):
        col_names = col_names or list(X.columns)
        X_arr = X.values.astype(float)
    else:
        X_arr = np.asarray(X, dtype=float)
        col_names = col_names or [f"X{i}" for i in range(X_arr.shape[1])]

    # Drop rows with NaN
    mask = np.all(np.isfinite(X_arr), axis=1)
    X_clean = X_arr[mask]
    if X_clean.shape[0] < X_clean.shape[1] + 1:
        return ValidationCheck(
            name="Multicollinearity (VIF)",
            status="warn",
            detail="Not enough observations to compute VIF",
        )

    try:
        vifs = [variance_inflation_factor(X_clean, i) for i in range(X_clean.shape[1])]
        max_vif = max(vifs)
        worst = col_names[vifs.index(max_vif)]
        if max_vif < threshold:
            return ValidationCheck(
                name="Multicollinearity (VIF)",
                status="pass",
                detail=f"Max VIF = {max_vif:.1f} ({worst}) — below threshold of {threshold}",
            )
        return ValidationCheck(
            name="Multicollinearity (VIF)",
            status="warn" if max_vif < 20 else "fail",
            detail=f"Max VIF = {max_vif:.1f} ({worst}) — exceeds threshold of {threshold}",
            suggestion="Consider removing or combining highly correlated predictors",
        )
    except Exception:
        return ValidationCheck(
            name="Multicollinearity (VIF)",
            status="warn",
            detail="Could not compute VIF (possible singular matrix)",
        )


def check_class_balance(y: np.ndarray | pd.Series, threshold: float = 0.1) -> ValidationCheck:
    """Check minority class proportion for classification tasks."""
    counts = pd.Series(y).value_counts(normalize=True)
    minority = counts.min()
    minority_class = counts.idxmin()
    if minority >= threshold:
        return ValidationCheck(
            name="Class Balance",
            status="pass",
            detail=f"Minority class '{minority_class}' = {minority:.1%} of samples",
        )
    return ValidationCheck(
        name="Class Balance",
        status="warn" if minority >= 0.05 else "fail",
        detail=f"Minority class '{minority_class}' = {minority:.1%} of samples (imbalanced)",
        suggestion="Consider SMOTE, class weights, or stratified sampling",
    )


def check_missing_data(data: pd.DataFrame, threshold: float = 0.1) -> ValidationCheck:
    """Check proportion of missing values in the dataset."""
    total = data.size
    missing = data.isna().sum().sum()
    pct = missing / total if total > 0 else 0
    if pct == 0:
        return ValidationCheck(name="Missing Data", status="pass", detail="No missing values")
    if pct < threshold:
        return ValidationCheck(
            name="Missing Data",
            status="pass",
            detail=f"{missing} missing values ({pct:.1%}) — within acceptable range",
        )
    return ValidationCheck(
        name="Missing Data",
        status="warn",
        detail=f"{missing} missing values ({pct:.1%}) — may affect results",
        suggestion="Consider imputation or excluding incomplete cases",
    )


def check_outlier_proportion(data: np.ndarray, threshold: float = 0.05) -> ValidationCheck:
    """IQR-based outlier proportion check."""
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 4:
        return ValidationCheck(
            name="Outliers", status="warn",
            detail="Too few data points to assess outliers",
        )
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int(np.sum((arr < lower) | (arr > upper)))
    pct = n_outliers / len(arr)
    if pct <= threshold:
        return ValidationCheck(
            name="Outliers",
            status="pass",
            detail=f"{n_outliers} outlier(s) ({pct:.1%}) via IQR method",
        )
    return ValidationCheck(
        name="Outliers",
        status="warn",
        detail=f"{n_outliers} outlier(s) ({pct:.1%}) detected — may influence results",
        suggestion="Review outliers; consider robust methods or trimming",
    )


def check_independence(residuals: np.ndarray) -> ValidationCheck:
    """Durbin-Watson test for autocorrelation of residuals."""
    from statsmodels.stats.stattools import durbin_watson

    arr = np.asarray(residuals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return ValidationCheck(
            name="Independence (Durbin-Watson)", status="warn",
            detail="Not enough residuals to test",
        )
    dw = durbin_watson(arr)
    if 1.5 <= dw <= 2.5:
        return ValidationCheck(
            name="Independence (Durbin-Watson)",
            status="pass",
            detail=f"DW = {dw:.3f} (no significant autocorrelation)",
        )
    return ValidationCheck(
        name="Independence (Durbin-Watson)",
        status="warn",
        detail=f"DW = {dw:.3f} ({'positive' if dw < 1.5 else 'negative'} autocorrelation suspected)",
        suggestion="Consider adding lagged terms or using time-series methods",
    )


def check_homoscedasticity(residuals: np.ndarray, X: np.ndarray) -> ValidationCheck:
    """Breusch-Pagan test for heteroscedasticity."""
    from statsmodels.stats.diagnostic import het_breuschpagan
    import statsmodels.api as sm

    resid = np.asarray(residuals, dtype=float)
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    mask = np.all(np.isfinite(np.column_stack([resid, X_arr])), axis=1)
    resid, X_arr = resid[mask], X_arr[mask]
    if len(resid) < X_arr.shape[1] + 2:
        return ValidationCheck(
            name="Homoscedasticity (Breusch-Pagan)", status="warn",
            detail="Not enough observations to test",
        )
    try:
        X_with_const = sm.add_constant(X_arr)
        _, p, _, _ = het_breuschpagan(resid, X_with_const)
        if p >= 0.05:
            return ValidationCheck(
                name="Homoscedasticity (Breusch-Pagan)",
                status="pass",
                detail=f"BP p = {p:.4f} (constant variance assumption met)",
            )
        return ValidationCheck(
            name="Homoscedasticity (Breusch-Pagan)",
            status="warn" if p >= 0.01 else "fail",
            detail=f"BP p = {p:.4f} (heteroscedasticity detected)",
            suggestion="Consider robust standard errors or weighted least squares",
        )
    except Exception:
        return ValidationCheck(
            name="Homoscedasticity (Breusch-Pagan)", status="warn",
            detail="Could not run Breusch-Pagan test",
        )


def check_residual_normality(residuals: np.ndarray) -> ValidationCheck:
    """Shapiro-Wilk on model residuals."""
    return check_normality(residuals, label="residuals")


def check_stationarity(series: np.ndarray) -> ValidationCheck:
    """Augmented Dickey-Fuller test for stationarity."""
    from statsmodels.tsa.stattools import adfuller

    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return ValidationCheck(
            name="Stationarity (ADF)",
            status="warn",
            detail=f"n = {len(arr)} is too short for reliable stationarity testing",
        )
    try:
        result = adfuller(arr, autolag="AIC")
        stat, p = result[0], result[1]
        if p < 0.05:
            return ValidationCheck(
                name="Stationarity (ADF)",
                status="pass",
                detail=f"ADF stat = {stat:.3f}, p = {p:.4f} (series appears stationary)",
            )
        return ValidationCheck(
            name="Stationarity (ADF)",
            status="warn" if p < 0.10 else "fail",
            detail=f"ADF stat = {stat:.3f}, p = {p:.4f} (series is non-stationary)",
            suggestion="Apply differencing before fitting ARIMA",
        )
    except Exception:
        return ValidationCheck(
            name="Stationarity (ADF)", status="warn",
            detail="Could not run ADF test",
        )


def check_kmo_bartlett(X: np.ndarray | pd.DataFrame) -> ValidationCheck:
    """KMO measure of sampling adequacy and Bartlett's test of sphericity."""
    if isinstance(X, pd.DataFrame):
        X_arr = X.values.astype(float)
    else:
        X_arr = np.asarray(X, dtype=float)

    mask = np.all(np.isfinite(X_arr), axis=1)
    X_clean = X_arr[mask]
    n, p = X_clean.shape
    if n < p + 1:
        return ValidationCheck(
            name="Sampling Adequacy (KMO)",
            status="warn",
            detail="Not enough observations relative to variables",
        )

    try:
        corr = np.corrcoef(X_clean.T)
        # Bartlett's test
        det = np.linalg.det(corr)
        if det <= 0:
            det = 1e-300
        chi2 = -((n - 1) - (2 * p + 5) / 6) * np.log(det)
        dof = p * (p - 1) / 2
        bartlett_p = 1 - stats.chi2.cdf(chi2, dof)

        # KMO approximation
        inv_corr = np.linalg.pinv(corr)
        partial = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
        np.fill_diagonal(partial, 0)
        np.fill_diagonal(corr, 0)
        sum_r2 = np.sum(corr ** 2)
        sum_p2 = np.sum(partial ** 2)
        kmo = sum_r2 / (sum_r2 + sum_p2) if (sum_r2 + sum_p2) > 0 else 0

        if kmo >= 0.6 and bartlett_p < 0.05:
            label = "meritorious" if kmo >= 0.8 else "mediocre" if kmo >= 0.7 else "acceptable"
            return ValidationCheck(
                name="Sampling Adequacy (KMO & Bartlett's)",
                status="pass",
                detail=f"KMO = {kmo:.3f} ({label}), Bartlett's p = {bartlett_p:.4f}",
            )
        issues = []
        if kmo < 0.6:
            issues.append(f"KMO = {kmo:.3f} (below 0.6 threshold)")
        if bartlett_p >= 0.05:
            issues.append(f"Bartlett's p = {bartlett_p:.4f} (correlations may be too weak)")
        return ValidationCheck(
            name="Sampling Adequacy (KMO & Bartlett's)",
            status="warn" if kmo >= 0.5 else "fail",
            detail="; ".join(issues),
            suggestion="Data may not be suitable for factor analysis / PCA",
        )
    except Exception:
        return ValidationCheck(
            name="Sampling Adequacy (KMO & Bartlett's)",
            status="warn",
            detail="Could not compute KMO/Bartlett's (possible singular correlation matrix)",
        )


def check_expected_frequencies(observed: np.ndarray, min_expected: float = 5.0) -> ValidationCheck:
    """Check if chi-square expected cell frequencies are adequate."""
    arr = np.asarray(observed, dtype=float)
    total = arr.sum()
    if total == 0:
        return ValidationCheck(
            name="Expected Frequencies",
            status="fail",
            detail="No observations in contingency table",
        )
    n_rows, n_cols = arr.shape if arr.ndim == 2 else (1, len(arr))
    if arr.ndim == 2:
        row_sums = arr.sum(axis=1, keepdims=True)
        col_sums = arr.sum(axis=0, keepdims=True)
        expected = row_sums * col_sums / total
    else:
        expected = np.full_like(arr, total / len(arr))
    n_low = int(np.sum(expected < min_expected))
    pct_low = n_low / expected.size
    if n_low == 0:
        return ValidationCheck(
            name="Expected Frequencies",
            status="pass",
            detail=f"All expected cell counts >= {min_expected}",
        )
    if pct_low <= 0.2:
        return ValidationCheck(
            name="Expected Frequencies",
            status="warn",
            detail=f"{n_low} cell(s) have expected count < {min_expected} ({pct_low:.0%})",
            suggestion="Consider combining sparse categories or using Fisher's exact test",
        )
    return ValidationCheck(
        name="Expected Frequencies",
        status="fail",
        detail=f"{n_low} cell(s) have expected count < {min_expected} ({pct_low:.0%})",
        suggestion="Too many sparse cells — use Fisher's exact test or combine categories",
    )


# ─── Recommendation Engine ───────────────────────────────────────────────────

_ALTERNATIVES: dict[str, dict[str, list[str]]] = {
    "independent-t": {
        "normality": ["Mann-Whitney U test"],
        "equal-variance": ["Welch's t-test"],
    },
    "paired-t": {
        "normality": ["Wilcoxon Signed-Rank test"],
    },
    "one-way-anova": {
        "normality": ["Kruskal-Wallis test"],
        "equal-variance": ["Welch's ANOVA"],
        "sample-size": ["Collect more data", "Kruskal-Wallis test", "Permutation test"],
    },
    "pearson-correlation": {
        "normality": ["Spearman rank correlation"],
    },
    "chi-square": {
        "expected-frequencies": ["Fisher's exact test"],
    },
    "linear-regression": {
        "homoscedasticity": ["Robust regression (HC3)", "Weighted least squares"],
        "residual-normality": ["Bootstrap confidence intervals", "GLM"],
        "multicollinearity": ["Ridge regression", "Remove correlated predictors", "PCA dimensionality reduction"],
        "independence": ["GLS", "Newey-West standard errors", "Add lagged variables"],
    },
    "arima": {
        "stationarity": ["Difference the series first"],
    },
}


def recommend_alternative(
    test_name: str, failed_checks: list[ValidationCheck]
) -> list[str]:
    """Given a test name and its failed checks, suggest alternatives."""
    alts: dict[str, list[str]] = _ALTERNATIVES.get(test_name, {})
    suggestions: list[str] = []
    for check in failed_checks:
        if check.status in ("warn", "fail"):
            name_lower = check.name.lower()
            for key, options in alts.items():
                if key in name_lower:
                    suggestions.extend(options)
    return list(dict.fromkeys(suggestions))  # deduplicate preserving order


# ─── Interpretation Templates ────────────────────────────────────────────────

def interpret_p_value(p: float, alpha: float = 0.05) -> Interpretation:
    """Plain-language interpretation of a p-value."""
    if p < 0.001:
        strength = "very strong"
    elif p < 0.01:
        strength = "strong"
    elif p < alpha:
        strength = "moderate"
    elif p < 0.10:
        strength = "weak (marginally significant)"
    else:
        strength = "insufficient"

    if p < alpha:
        body = (
            f"The p-value of {p:.4f} is below the significance level of {alpha}. "
            f"There is {strength} evidence against the null hypothesis."
        )
    else:
        body = (
            f"The p-value of {p:.4f} exceeds the significance level of {alpha}. "
            f"There is {strength} evidence against the null hypothesis — "
            f"the observed result is consistent with random chance."
        )
    return Interpretation(title="Interpretation", body=body)


def interpret_effect_size(d: float, test_type: str = "cohen-d") -> Interpretation:
    """Interpret effect size using standard thresholds."""
    abs_d = abs(d)
    thresholds = {
        "cohen-d": [(0.2, "small"), (0.5, "medium"), (0.8, "large")],
        "eta-squared": [(0.01, "small"), (0.06, "medium"), (0.14, "large")],
        "r-squared": [(0.02, "small"), (0.13, "medium"), (0.26, "large")],
        "cramers-v": [(0.1, "small"), (0.3, "medium"), (0.5, "large")],
    }
    levels = thresholds.get(test_type, thresholds["cohen-d"])
    label = "negligible"
    for cutoff, name in levels:
        if abs_d >= cutoff:
            label = name
    pct = f"{abs_d:.1%}" if test_type in ("eta-squared", "r-squared") else f"{abs_d:.3f}"

    if test_type == "eta-squared":
        body = f"An eta-squared of {abs_d:.4f} means the factor explains about {abs_d:.1%} of the total variance, which is considered a {label} effect."
    elif test_type == "r-squared":
        body = f"An R-squared of {abs_d:.4f} means the model explains about {abs_d:.1%} of the variance in the outcome, which is considered a {label} effect."
    elif test_type == "cramers-v":
        body = f"A Cramer's V of {abs_d:.4f} indicates a {label} association between the variables."
    else:
        body = f"A Cohen's d of {abs_d:.4f} indicates a {label} effect size."

    return Interpretation(title="Effect Size", body=body)


def interpret_r_squared(r2: float, adj_r2: float | None = None) -> Interpretation:
    """Interpret R-squared (and optionally adjusted R-squared)."""
    interp = interpret_effect_size(r2, "r-squared")
    if adj_r2 is not None and (r2 - adj_r2) > 0.05:
        interp.detail = (
            f"The gap between R² ({r2:.4f}) and Adjusted R² ({adj_r2:.4f}) suggests "
            f"some predictors may not be contributing meaningfully."
        )
    return interp


def interpret_capability(cpk: float) -> Interpretation:
    """Interpret process capability index Cpk."""
    if cpk >= 2.0:
        label, ppm = "world-class", "<3.4"
    elif cpk >= 1.67:
        label, ppm = "excellent", "~0.6"
    elif cpk >= 1.33:
        label, ppm = "capable", "~66"
    elif cpk >= 1.0:
        label, ppm = "marginally capable", "~2,700"
    elif cpk >= 0.67:
        label, ppm = "poor", "~22,750"
    else:
        label, ppm = "incapable", ">66,800"
    body = (
        f"A Cpk of {cpk:.3f} indicates the process is {label}, "
        f"with an estimated defect rate of approximately {ppm} ppm."
    )
    return Interpretation(title="Capability", body=body)


def interpret_correlation(r: float, p: float, method: str = "Pearson") -> Interpretation:
    """Interpret a correlation coefficient."""
    abs_r = abs(r)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "negligible"
    direction = "positive" if r > 0 else "negative"
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    body = (
        f"The {method} correlation of {r:.4f} indicates a {strength} {direction} "
        f"relationship (p = {p:.4f}, {sig})."
    )
    return Interpretation(title="Correlation", body=body)


def interpret_silhouette(score: float) -> Interpretation:
    """Interpret a silhouette score for clustering."""
    if score >= 0.7:
        label = "strong structure — clusters are well-separated"
    elif score >= 0.5:
        label = "reasonable structure — clusters are distinguishable"
    elif score >= 0.25:
        label = "weak structure — clusters overlap substantially"
    else:
        label = "no meaningful structure — data may not have natural clusters"
    body = f"A silhouette score of {score:.3f} indicates {label}."
    return Interpretation(title="Cluster Quality", body=body)


def interpret_stationarity(adf_p: float) -> Interpretation:
    """Interpret ADF stationarity test result."""
    if adf_p < 0.05:
        body = (
            f"The ADF test (p = {adf_p:.4f}) indicates the series is stationary — "
            f"suitable for ARIMA modeling without differencing."
        )
    else:
        body = (
            f"The ADF test (p = {adf_p:.4f}) indicates the series is non-stationary. "
            f"Consider applying differencing (d >= 1) before fitting ARIMA."
        )
    return Interpretation(title="Stationarity", body=body)


def compute_post_hoc_power(
    effect_size: float, n: int, alpha: float = 0.05, test_type: str = "t-test"
) -> float:
    """Approximate post-hoc power for common tests using normal approximation."""
    from scipy.stats import norm

    if test_type in ("t-test", "paired-t"):
        # Two-sided t-test power approximation
        se = 1.0 / np.sqrt(n)
        z_alpha = norm.ppf(1 - alpha / 2)
        ncp = abs(effect_size) / se
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)
    elif test_type == "anova":
        # F-test power approximation using Cohen's f
        f_effect = np.sqrt(effect_size / (1 - effect_size)) if effect_size < 1 else effect_size
        ncp = n * f_effect ** 2
        z_alpha = norm.ppf(1 - alpha)
        power = 1 - norm.cdf(z_alpha - np.sqrt(ncp))
    else:
        # Generic approximation
        z_alpha = norm.ppf(1 - alpha / 2)
        ncp = abs(effect_size) * np.sqrt(n)
        power = 1 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp)

    return float(np.clip(power, 0, 1))


# ─── Phase 1: Data Readiness, Diagnostics, and Additional Checks ────────────

@dataclass
class DataReadiness:
    """Overall data-readiness scorecard computed from a list of ValidationChecks."""
    score: float        # 0-100
    grade: str          # A-F
    summary: str
    checks: list


def compute_data_readiness(checks: list) -> DataReadiness:
    """Compute an aggregate readiness score from a list of ValidationCheck results.

    Each check contributes equally: pass=1.0, warn=0.5, fail=0.0.
    The weighted average is scaled to 0-100.
    """
    if not checks:
        return DataReadiness(score=0.0, grade="F", summary="No checks performed", checks=[])

    score_map = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
    values = [score_map.get(c.status, 0.0) for c in checks]
    score = (sum(values) / len(values)) * 100

    n_pass = sum(1 for c in checks if c.status == "pass")
    n_warn = sum(1 for c in checks if c.status == "warn")
    n_fail = sum(1 for c in checks if c.status == "fail")

    if score >= 90:
        grade = "A"
    elif score >= 75:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    summary = f"{n_pass} passed, {n_warn} warning{'s' if n_warn != 1 else ''}, {n_fail} failed"
    return DataReadiness(score=round(score, 1), grade=grade, summary=summary, checks=checks)


def check_duplicates(
    data: pd.DataFrame, subset: list[str] | None = None
) -> ValidationCheck:
    """Check for exact duplicate rows (or duplicates on a column subset)."""
    n_total = len(data)
    if n_total == 0:
        return ValidationCheck(
            name="Duplicate Rows",
            status="pass",
            detail="No data to check",
        )
    n_dups = int(data.duplicated(subset=subset).sum())
    pct = n_dups / n_total
    detail = f"{n_dups} duplicate row(s) ({pct:.1%} of {n_total})"
    if subset:
        detail += f" based on columns {subset}"
    if pct > 0.15:
        return ValidationCheck(
            name="Duplicate Rows",
            status="fail",
            detail=detail,
            suggestion="Review and deduplicate before analysis",
        )
    if pct > 0.05:
        return ValidationCheck(
            name="Duplicate Rows",
            status="warn",
            detail=detail,
            suggestion="Consider whether duplicates are intentional",
        )
    return ValidationCheck(name="Duplicate Rows", status="pass", detail=detail)


def check_range_validity(
    data: pd.DataFrame, column: str, lower: float | None = None, upper: float | None = None
) -> ValidationCheck:
    """Check values outside [lower, upper] bounds for a given column."""
    arr = pd.to_numeric(data[column], errors="coerce")
    arr = arr.dropna()
    n = len(arr)
    if n == 0:
        return ValidationCheck(
            name=f"Range Validity ({column})",
            status="warn",
            detail="No valid numeric values to check",
        )
    out_of_range = pd.Series([False] * n, index=arr.index)
    if lower is not None:
        out_of_range = out_of_range | (arr < lower)
    if upper is not None:
        out_of_range = out_of_range | (arr > upper)
    n_bad = int(out_of_range.sum())
    pct = n_bad / n
    bounds_str = f"[{lower}, {upper}]"
    detail = f"{n_bad} value(s) ({pct:.1%}) outside {bounds_str} in '{column}'"
    if pct > 0.10:
        return ValidationCheck(
            name=f"Range Validity ({column})",
            status="fail",
            detail=detail,
            suggestion="Investigate values outside the expected range",
        )
    if n_bad > 0:
        return ValidationCheck(
            name=f"Range Validity ({column})",
            status="warn",
            detail=detail,
            suggestion="Review out-of-range values for data entry errors",
        )
    return ValidationCheck(
        name=f"Range Validity ({column})",
        status="pass",
        detail=f"All values within {bounds_str} in '{column}'",
    )


def generate_diagnostic_plots(
    data: pd.DataFrame, columns: list[str]
) -> "go.Figure":
    """Generate a 2x2 diagnostic subplot: Q-Q, histogram+KDE, box, missing bar chart.

    Uses the first column in *columns* for the single-variable plots.
    The missing-value chart covers all specified columns.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    col = columns[0]
    arr = pd.to_numeric(data[col], errors="coerce").dropna().values

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Q-Q Plot", "Histogram + KDE", "Box Plot", "Missing Values"],
    )

    # --- Top-left: Q-Q plot ---
    sorted_vals = np.sort(arr)
    theoretical = stats.norm.ppf(
        (np.arange(1, len(sorted_vals) + 1) - 0.5) / len(sorted_vals)
    )
    fig.add_trace(
        go.Scatter(x=theoretical, y=sorted_vals, mode="markers", name="Q-Q",
                   marker=dict(size=4)),
        row=1, col=1,
    )
    # Reference line
    mn, mx = theoretical.min(), theoretical.max()
    fig.add_trace(
        go.Scatter(x=[mn, mx], y=[arr.mean() + arr.std() * mn, arr.mean() + arr.std() * mx],
                   mode="lines", name="Reference", line=dict(dash="dash")),
        row=1, col=1,
    )

    # --- Top-right: Histogram + KDE ---
    fig.add_trace(
        go.Histogram(x=arr, nbinsx=30, name="Histogram", opacity=0.7,
                     histnorm="probability density"),
        row=1, col=2,
    )
    if len(arr) >= 2:
        try:
            kde = stats.gaussian_kde(arr)
            x_grid = np.linspace(arr.min(), arr.max(), 200)
            fig.add_trace(
                go.Scatter(x=x_grid, y=kde(x_grid), mode="lines", name="KDE"),
                row=1, col=2,
            )
        except Exception:
            pass  # KDE can fail on degenerate data

    # --- Bottom-left: Box plot ---
    fig.add_trace(
        go.Box(y=arr, name=col, boxmean="sd"),
        row=2, col=1,
    )

    # --- Bottom-right: Missing value bar chart ---
    missing_counts = data[columns].isna().sum()
    fig.add_trace(
        go.Bar(x=missing_counts.index.tolist(), y=missing_counts.values.tolist(),
               name="Missing"),
        row=2, col=2,
    )

    fig.update_layout(
        template="plotly+rdl",
        height=700,
        showlegend=False,
        title_text=f"Diagnostic Plots — {col}",
    )
    return fig


def interpret_durbin_watson(dw: float) -> Interpretation:
    """Plain-language interpretation of a Durbin-Watson statistic."""
    if dw < 1.5:
        body = (
            f"A Durbin-Watson statistic of {dw:.3f} suggests positive autocorrelation "
            f"among residuals. Successive residuals tend to be similar."
        )
    elif dw <= 2.5:
        body = (
            f"A Durbin-Watson statistic of {dw:.3f} falls within the acceptable range "
            f"(1.5 - 2.5), suggesting no significant autocorrelation."
        )
    else:
        body = (
            f"A Durbin-Watson statistic of {dw:.3f} suggests negative autocorrelation "
            f"among residuals. Successive residuals tend to alternate in sign."
        )
    return Interpretation(title="Durbin-Watson", body=body)


# ─── Phase 3: Recommended Checks and Additional Validators ──────────────────

_RECOMMENDED_CHECKS: dict[str, list[str]] = {
    "t_test": ["normality", "equal_variance", "sample_size"],
    "paired_t_test": ["normality", "sample_size"],
    "anova": ["normality", "equal_variance", "sample_size"],
    "linear_regression": [
        "normality", "homoscedasticity", "independence",
        "multicollinearity", "sample_size",
    ],
    "correlation": ["normality", "sample_size", "outliers"],
    "chi_square": ["expected_frequencies", "sample_size"],
    "mann_whitney": ["sample_size"],
    "normality_test": ["sample_size"],
    "descriptive": ["missing_data", "outliers"],
}


def get_recommended_checks(analysis_type: str) -> list[str]:
    """Return the list of recommended validation checks for a given analysis type."""
    return _RECOMMENDED_CHECKS.get(analysis_type, [])


def run_recommended_checks(
    analysis_type: str,
    data: pd.DataFrame,
    params: dict | None = None,
) -> list[ValidationCheck]:
    """Run all recommended checks for *analysis_type* against *data*.

    *params* may supply:
        - column: target numeric column name
        - group_column: grouping column name
        - predictors: list of predictor column names
        - residuals: pre-computed residuals array
        - X: pre-computed design matrix
        - contingency: pre-computed contingency table (np.ndarray)
    """
    params = params or {}
    recommended = get_recommended_checks(analysis_type)
    results: list[ValidationCheck] = []

    # Determine a default numeric column
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    default_col = params.get("column") or (numeric_cols[0] if numeric_cols else None)

    # Map analysis_type -> test_type key used in _MIN_N
    _type_map = {
        "t_test": "t-test",
        "paired_t_test": "paired-t",
        "anova": "anova",
        "linear_regression": "regression",
        "correlation": "correlation",
        "chi_square": "chi-square",
        "mann_whitney": "t-test",
        "normality_test": "t-test",
        "descriptive": "t-test",
    }

    for check_name in recommended:
        try:
            if check_name == "normality":
                col = default_col
                if col is not None:
                    arr = pd.to_numeric(data[col], errors="coerce").dropna().values
                    results.append(check_normality(arr, label=col))

            elif check_name == "equal_variance":
                group_col = params.get("group_column")
                if group_col and default_col:
                    groups = [
                        grp[default_col].dropna().values
                        for _, grp in data.groupby(group_col)
                    ]
                    results.append(check_equal_variance(*groups))

            elif check_name == "sample_size":
                test_key = _type_map.get(analysis_type, "t-test")
                results.append(check_sample_size(len(data), test_key))

            elif check_name == "homoscedasticity":
                residuals = params.get("residuals")
                X = params.get("X")
                if residuals is not None and X is not None:
                    results.append(check_homoscedasticity(residuals, X))
                elif default_col and len(numeric_cols) >= 2:
                    # Attempt a quick OLS fit
                    try:
                        import statsmodels.api as sm
                        y = data[default_col].dropna()
                        preds = [c for c in numeric_cols if c != default_col]
                        X_df = data[preds].loc[y.index].dropna()
                        common = y.index.intersection(X_df.index)
                        y = y.loc[common].values
                        X_arr = sm.add_constant(X_df.loc[common].values)
                        model = sm.OLS(y, X_arr).fit()
                        results.append(check_homoscedasticity(model.resid, X_arr))
                    except Exception:
                        pass

            elif check_name == "independence":
                residuals = params.get("residuals")
                if residuals is not None:
                    results.append(check_independence(residuals))
                elif default_col and len(numeric_cols) >= 2:
                    try:
                        import statsmodels.api as sm
                        y = data[default_col].dropna()
                        preds = [c for c in numeric_cols if c != default_col]
                        X_df = data[preds].loc[y.index].dropna()
                        common = y.index.intersection(X_df.index)
                        y = y.loc[common].values
                        X_arr = sm.add_constant(X_df.loc[common].values)
                        model = sm.OLS(y, X_arr).fit()
                        results.append(check_independence(model.resid))
                    except Exception:
                        pass

            elif check_name == "multicollinearity":
                predictors = params.get("predictors")
                if predictors:
                    X_df = data[predictors].dropna()
                    results.append(check_multicollinearity(X_df, predictors))
                elif len(numeric_cols) >= 2:
                    cols = [c for c in numeric_cols if c != default_col]
                    if cols:
                        X_df = data[cols].dropna()
                        results.append(check_multicollinearity(X_df, cols))

            elif check_name == "outliers":
                col = default_col
                if col is not None:
                    arr = pd.to_numeric(data[col], errors="coerce").dropna().values
                    results.append(check_outlier_proportion(arr))

            elif check_name == "missing_data":
                results.append(check_missing_data(data))

            elif check_name == "expected_frequencies":
                contingency = params.get("contingency")
                if contingency is not None:
                    results.append(check_expected_frequencies(contingency))

        except Exception:
            # Skip checks that error out rather than crashing the whole run
            pass

    return results


def check_linearity(X: np.ndarray, y: np.ndarray) -> ValidationCheck:
    """Check whether residuals show a non-linear pattern vs. fitted values.

    Fits OLS (X vs y), then regresses residuals-squared on fitted values.
    A significant relationship suggests non-linearity.
    """
    import statsmodels.api as sm

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    mask = np.all(np.isfinite(np.column_stack([X_arr, y_arr.reshape(-1, 1)])), axis=1)
    X_arr, y_arr = X_arr[mask], y_arr[mask]

    if len(y_arr) < X_arr.shape[1] + 3:
        return ValidationCheck(
            name="Linearity",
            status="warn",
            detail="Not enough observations to assess linearity",
        )

    try:
        X_const = sm.add_constant(X_arr)
        model = sm.OLS(y_arr, X_const).fit()
        fitted = model.fittedvalues
        resid_sq = model.resid ** 2

        # Regress residuals² on fitted values
        diag_X = sm.add_constant(fitted)
        diag_model = sm.OLS(resid_sq, diag_X).fit()
        r2 = diag_model.rsquared
        p = diag_model.f_pvalue

        if p >= 0.05:
            return ValidationCheck(
                name="Linearity",
                status="pass",
                detail=f"No significant non-linear pattern detected (p = {p:.4f}, R² = {r2:.4f})",
            )
        if r2 < 0.1:
            return ValidationCheck(
                name="Linearity",
                status="warn",
                detail=f"Mild non-linear pattern (p = {p:.4f}, R² = {r2:.4f})",
                suggestion="Consider adding polynomial terms or a transformation",
            )
        return ValidationCheck(
            name="Linearity",
            status="fail",
            detail=f"Non-linear pattern detected (p = {p:.4f}, R² = {r2:.4f})",
            suggestion="A linear model may be inappropriate — consider polynomial or non-linear regression",
        )
    except Exception:
        return ValidationCheck(
            name="Linearity",
            status="warn",
            detail="Could not assess linearity",
        )


def check_group_balance(data: pd.DataFrame, group_col: str) -> ValidationCheck:
    """Check balance of group sizes.  Ratio > 3:1 warns, > 5:1 fails."""
    counts = data[group_col].value_counts()
    if len(counts) < 2:
        return ValidationCheck(
            name="Group Balance",
            status="warn",
            detail=f"Only {len(counts)} group(s) found in '{group_col}'",
        )
    largest = counts.iloc[0]
    smallest = counts.iloc[-1]
    ratio = largest / smallest if smallest > 0 else float("inf")
    detail = (
        f"Group sizes in '{group_col}': largest = {largest}, smallest = {smallest} "
        f"(ratio {ratio:.1f}:1)"
    )
    if ratio > 5:
        return ValidationCheck(
            name="Group Balance",
            status="fail",
            detail=detail,
            suggestion="Severe imbalance — consider resampling or weighted analysis",
        )
    if ratio > 3:
        return ValidationCheck(
            name="Group Balance",
            status="warn",
            detail=detail,
            suggestion="Moderate imbalance — results may be affected; consider balanced designs",
        )
    return ValidationCheck(name="Group Balance", status="pass", detail=detail)


# ─── Column Profiling ────────────────────────────────────────────────────────

def check_constant_column(series: pd.Series) -> ValidationCheck:
    """Flag a column with only 1 unique value."""
    n_unique = series.nunique(dropna=True)
    if n_unique <= 1:
        return ValidationCheck(
            name=f"Constant Column ({series.name})",
            status="fail",
            detail=f"Column '{series.name}' has {n_unique} unique value(s)",
            suggestion="Consider dropping this column — it provides no information",
        )
    return ValidationCheck(
        name=f"Constant Column ({series.name})",
        status="pass",
        detail=f"Column '{series.name}' has {n_unique} unique values",
    )


def check_high_cardinality(series: pd.Series, threshold: float = 0.95) -> ValidationCheck:
    """Flag columns where almost every value is unique (likely an ID column)."""
    n_valid = series.dropna().shape[0]
    if n_valid == 0:
        return ValidationCheck(
            name=f"High Cardinality ({series.name})",
            status="warn",
            detail="No non-null values to check",
        )
    ratio = series.nunique(dropna=True) / n_valid
    if ratio >= threshold:
        return ValidationCheck(
            name=f"High Cardinality ({series.name})",
            status="warn",
            detail=f"Column '{series.name}' has {ratio:.1%} unique values — likely an ID column",
            suggestion="Consider excluding from analysis",
        )
    return ValidationCheck(
        name=f"High Cardinality ({series.name})",
        status="pass",
        detail=f"Column '{series.name}' has {ratio:.1%} unique values",
    )


def profile_column(series: pd.Series) -> dict:
    """Return a profiling summary dict for a single column."""
    n_total = len(series)
    n_missing = int(series.isna().sum())
    n_valid = n_total - n_missing
    completeness = (n_valid / n_total * 100) if n_total > 0 else 0
    n_unique = int(series.nunique(dropna=True))

    profile = {
        "name": series.name,
        "dtype": str(series.dtype),
        "n_total": n_total,
        "n_missing": n_missing,
        "pct_missing": round(n_missing / n_total * 100, 1) if n_total > 0 else 0,
        "completeness": round(completeness, 1),
        "n_unique": n_unique,
    }

    if pd.api.types.is_numeric_dtype(series):
        valid = series.dropna()
        profile["distribution_type"] = "numeric"
        if len(valid) > 0:
            profile["min"] = float(valid.min())
            profile["max"] = float(valid.max())
            profile["mean"] = float(valid.mean())
            profile["std"] = float(valid.std())
            profile["median"] = float(valid.median())
            # Outlier count via IQR
            q1, q3 = float(np.percentile(valid, 25)), float(np.percentile(valid, 75))
            iqr = q3 - q1
            n_outliers = int(((valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)).sum())
            profile["n_outliers"] = n_outliers
    else:
        profile["distribution_type"] = "categorical"
        if n_valid > 0:
            top = series.value_counts().iloc[0]
            top_val = series.value_counts().index[0]
            profile["top_value"] = str(top_val)
            profile["top_count"] = int(top)

    return profile
