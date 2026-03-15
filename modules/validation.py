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
