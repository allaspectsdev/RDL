"""
Time Series Analysis Module - Decomposition, ACF/PACF, ARIMA, forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import statsmodels.api as sm
    from statsmodels.tsa.seasonal import seasonal_decompose, STL
    from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_SM = True
except ImportError:
    HAS_SM = False


def render_time_series(df: pd.DataFrame):
    """Render time series analysis interface."""
    if df is None or df.empty:
        st.warning("No data loaded.")
        return
    if not HAS_SM:
        st.error("statsmodels is required for time series analysis.")
        return

    tabs = st.tabs([
        "Exploration", "Decomposition", "Stationarity",
        "ACF/PACF", "Smoothing", "ARIMA/SARIMA", "Forecast Comparison",
    ])

    with tabs[0]:
        _render_exploration(df)
    with tabs[1]:
        _render_decomposition(df)
    with tabs[2]:
        _render_stationarity(df)
    with tabs[3]:
        _render_acf_pacf(df)
    with tabs[4]:
        _render_smoothing(df)
    with tabs[5]:
        _render_arima(df)
    with tabs[6]:
        _render_forecast_comparison(df)


def _get_ts_data(df, date_col, value_col):
    """Prepare time series data."""
    data = df[[date_col, value_col]].dropna().copy()
    if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
        try:
            data[date_col] = pd.to_datetime(data[date_col])
        except Exception:
            st.error(f"Cannot convert '{date_col}' to datetime.")
            return None
    data = data.sort_values(date_col).set_index(date_col)
    # Try to infer frequency
    try:
        data = data.asfreq(pd.infer_freq(data.index))
    except Exception:
        pass
    return data[value_col]


def _select_ts_columns(df, prefix):
    """Common column selection for time series."""
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    c1, c2 = st.columns(2)
    date_col = c1.selectbox("Date/Time column:", date_cols + all_cols, key=f"{prefix}_date")
    value_col = c2.selectbox("Value column:", num_cols, key=f"{prefix}_val")
    return date_col, value_col


def _render_exploration(df: pd.DataFrame):
    """Time series exploration with rolling stats."""
    date_col, value_col = _select_ts_columns(df, "exp")

    ts = _get_ts_data(df, date_col, value_col)
    if ts is None:
        return

    # Main time series plot with range slider
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode="lines",
                             name=value_col, line=dict(color="steelblue")))
    fig.update_layout(title=f"Time Series: {value_col}",
                      xaxis=dict(rangeslider=dict(visible=True)),
                      height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Rolling statistics
    st.markdown("#### Rolling Statistics")
    window = st.slider("Window size:", 2, min(100, len(ts) // 2), 12, key="exp_window")

    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts.index, y=ts.values, name="Original",
                             line=dict(color="steelblue", width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=ts.index, y=rolling_mean.values, name=f"Rolling Mean ({window})",
                             line=dict(color="red", width=2)))
    fig.add_trace(go.Scatter(x=ts.index, y=rolling_std.values, name=f"Rolling Std ({window})",
                             line=dict(color="green", width=2)))
    fig.update_layout(title="Rolling Statistics", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Lag plot
    with st.expander("Lag Plot"):
        lag = st.slider("Lag:", 1, 20, 1, key="exp_lag")
        lag_data = pd.DataFrame({"y(t)": ts.values[lag:], "y(t-{})".format(lag): ts.values[:-lag]})
        fig = px.scatter(lag_data, x=f"y(t-{lag})", y="y(t)", title=f"Lag Plot (lag={lag})")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    with st.expander("Summary"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{ts.mean():.4f}")
        c2.metric("Std Dev", f"{ts.std():.4f}")
        c3.metric("Min", f"{ts.min():.4f}")
        c4.metric("Max", f"{ts.max():.4f}")


def _render_decomposition(df: pd.DataFrame):
    """Time series decomposition."""
    date_col, value_col = _select_ts_columns(df, "dec")

    ts = _get_ts_data(df, date_col, value_col)
    if ts is None:
        return

    c1, c2, c3 = st.columns(3)
    model_type = c1.selectbox("Model:", ["additive", "multiplicative"], key="dec_model")
    method = c2.selectbox("Method:", ["Classical", "STL"], key="dec_method")
    period = c3.number_input("Period:", 2, 365, 12, key="dec_period")

    if st.button("Decompose", key="run_dec"):
        ts_clean = ts.dropna()
        if len(ts_clean) < 2 * period:
            st.error(f"Need at least {2 * period} data points for period={period}.")
            return

        try:
            if method == "Classical":
                result = seasonal_decompose(ts_clean, model=model_type, period=period)
            else:
                result = STL(ts_clean, period=period).fit()

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
                                vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=ts_clean.index, y=ts_clean.values, name="Observed",
                                     line=dict(color="steelblue")), row=1, col=1)
            fig.add_trace(go.Scatter(x=ts_clean.index, y=result.trend, name="Trend",
                                     line=dict(color="red")), row=2, col=1)
            fig.add_trace(go.Scatter(x=ts_clean.index, y=result.seasonal, name="Seasonal",
                                     line=dict(color="green")), row=3, col=1)
            fig.add_trace(go.Scatter(x=ts_clean.index, y=result.resid, name="Residual",
                                     line=dict(color="purple")), row=4, col=1)
            fig.update_layout(height=800, title=f"Decomposition ({model_type}, period={period})")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Decomposition error: {e}")


def _render_stationarity(df: pd.DataFrame):
    """Stationarity testing."""
    date_col, value_col = _select_ts_columns(df, "stat")

    ts = _get_ts_data(df, date_col, value_col)
    if ts is None:
        return

    if st.button("Run Stationarity Tests", key="run_stat"):
        ts_clean = ts.dropna()

        # ADF test
        adf_result = adfuller(ts_clean, autolag="AIC")
        st.markdown("#### Augmented Dickey-Fuller Test")
        st.write(f"**Test Statistic:** {adf_result[0]:.4f}")
        st.write(f"**p-value:** {adf_result[1]:.6f}")
        st.write(f"**Lags Used:** {adf_result[2]}")
        for key, val in adf_result[4].items():
            st.write(f"  Critical Value ({key}): {val:.4f}")
        if adf_result[1] < 0.05:
            st.success("Series is stationary (reject H₀ of unit root)")
        else:
            st.warning("Series is non-stationary (cannot reject H₀ of unit root)")

        # KPSS test
        st.markdown("#### KPSS Test")
        try:
            kpss_result = kpss(ts_clean, regression="c", nlags="auto")
            st.write(f"**Test Statistic:** {kpss_result[0]:.4f}")
            st.write(f"**p-value:** {kpss_result[1]:.6f}")
            for key, val in kpss_result[3].items():
                st.write(f"  Critical Value ({key}): {val:.4f}")
            if kpss_result[1] > 0.05:
                st.success("Series is stationary (cannot reject H₀ of stationarity)")
            else:
                st.warning("Series is non-stationary (reject H₀ of stationarity)")
        except Exception as e:
            st.warning(f"KPSS test error: {e}")

    # Differencing
    st.markdown("#### Differencing")
    diff_order = st.selectbox("Order:", [1, 2], key="stat_diff")
    if st.button("Apply Differencing", key="apply_diff"):
        ts_clean = ts.dropna()
        ts_diff = ts_clean.diff(diff_order).dropna()

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Original", f"Differenced (order={diff_order})"))
        fig.add_trace(go.Scatter(x=ts_clean.index, y=ts_clean.values, name="Original"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts_diff.index, y=ts_diff.values, name="Differenced"), row=2, col=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Test differenced series
        adf_diff = adfuller(ts_diff, autolag="AIC")
        st.write(f"**ADF on differenced series:** stat={adf_diff[0]:.4f}, p={adf_diff[1]:.6f}")
        if adf_diff[1] < 0.05:
            st.success("Differenced series is stationary.")


def _render_acf_pacf(df: pd.DataFrame):
    """ACF and PACF analysis."""
    date_col, value_col = _select_ts_columns(df, "acf")

    ts = _get_ts_data(df, date_col, value_col)
    if ts is None:
        return

    n_lags = st.slider("Number of lags:", 5, min(100, len(ts) // 2), 40, key="acf_lags")

    if st.button("Compute ACF/PACF", key="run_acf"):
        ts_clean = ts.dropna()

        acf_vals = acf(ts_clean, nlags=n_lags, fft=True)
        pacf_vals = pacf(ts_clean, nlags=min(n_lags, len(ts_clean) // 2 - 1))

        # Confidence interval (95%)
        ci = 1.96 / np.sqrt(len(ts_clean))

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"))

        # ACF
        for i, val in enumerate(acf_vals):
            fig.add_trace(go.Scatter(x=[i, i], y=[0, val], mode="lines",
                                     line=dict(color="steelblue", width=2), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(acf_vals))), y=acf_vals, mode="markers",
                                 marker=dict(color="steelblue", size=6), name="ACF"), row=1, col=1)
        fig.add_hline(y=ci, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-ci, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=0, line_color="black", row=1, col=1)

        # PACF
        for i, val in enumerate(pacf_vals):
            fig.add_trace(go.Scatter(x=[i, i], y=[0, val], mode="lines",
                                     line=dict(color="steelblue", width=2), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(pacf_vals))), y=pacf_vals, mode="markers",
                                 marker=dict(color="steelblue", size=6), name="PACF"), row=2, col=1)
        fig.add_hline(y=ci, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-ci, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0, line_color="black", row=2, col=1)

        fig.update_layout(height=600)
        fig.update_xaxes(title_text="Lag", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation guide
        with st.expander("Interpretation Guide"):
            st.markdown("""
            **ACF pattern → Model suggestion:**
            - Exponential decay → AR model
            - Cut off after lag q → MA(q) model
            - Slow decay → Need differencing

            **PACF pattern → Model suggestion:**
            - Cut off after lag p → AR(p) model
            - Exponential decay → MA model
            """)

        # Ljung-Box test
        st.markdown("#### Ljung-Box Test (White Noise)")
        lb_lags = min(20, n_lags)
        lb_result = acorr_ljungbox(ts_clean, lags=lb_lags, return_df=True)
        st.dataframe(lb_result.round(4), use_container_width=True)
        if lb_result["lb_pvalue"].iloc[-1] > 0.05:
            st.success("Series may be white noise (no significant autocorrelation).")
        else:
            st.info("Significant autocorrelation detected — suitable for time series modeling.")


def _render_smoothing(df: pd.DataFrame):
    """Exponential smoothing methods."""
    date_col, value_col = _select_ts_columns(df, "smooth")

    ts = _get_ts_data(df, date_col, value_col)
    if ts is None:
        return

    method = st.selectbox("Method:", [
        "Simple Moving Average", "Exponential Moving Average",
        "Holt (Linear Trend)", "Holt-Winters (Trend + Seasonal)",
    ], key="smooth_method")

    ts_clean = ts.dropna()

    if method == "Simple Moving Average":
        window = st.slider("Window:", 2, min(50, len(ts_clean) // 2), 7, key="sma_window")
        smoothed = ts_clean.rolling(window=window).mean()
        label = f"SMA({window})"

    elif method == "Exponential Moving Average":
        span = st.slider("Span:", 2, min(50, len(ts_clean) // 2), 12, key="ema_span")
        smoothed = ts_clean.ewm(span=span).mean()
        label = f"EMA(span={span})"

    elif method == "Holt (Linear Trend)":
        if st.button("Fit Holt's Model", key="fit_holt"):
            try:
                model = ExponentialSmoothing(ts_clean, trend="add", seasonal=None).fit(optimized=True)
                smoothed = model.fittedvalues
                label = "Holt's Linear"

                st.write(f"**Smoothing level (α):** {model.params['smoothing_level']:.4f}")
                st.write(f"**Smoothing trend (β):** {model.params['smoothing_trend']:.4f}")
                st.write(f"**AIC:** {model.aic:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
                return
        else:
            return

    elif method == "Holt-Winters (Trend + Seasonal)":
        c1, c2 = st.columns(2)
        seasonal_type = c1.selectbox("Seasonal type:", ["add", "mul"], key="hw_seasonal")
        period = c2.number_input("Season period:", 2, 365, 12, key="hw_period")

        if st.button("Fit Holt-Winters", key="fit_hw"):
            if len(ts_clean) < 2 * period:
                st.error(f"Need at least {2 * period} observations.")
                return
            try:
                model = ExponentialSmoothing(ts_clean, trend="add", seasonal=seasonal_type,
                                            seasonal_periods=period).fit(optimized=True)
                smoothed = model.fittedvalues
                label = f"Holt-Winters ({seasonal_type})"

                st.write(f"**α (level):** {model.params['smoothing_level']:.4f}")
                st.write(f"**β (trend):** {model.params['smoothing_trend']:.4f}")
                st.write(f"**γ (seasonal):** {model.params['smoothing_seasonal']:.4f}")
                st.write(f"**AIC:** {model.aic:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
                return
        else:
            return

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts_clean.index, y=ts_clean.values, name="Original",
                             line=dict(color="steelblue", width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=ts_clean.index, y=smoothed.values, name=label,
                             line=dict(color="red", width=2)))
    fig.update_layout(title=f"Smoothing: {label}", height=500)
    st.plotly_chart(fig, use_container_width=True)


def _render_arima(df: pd.DataFrame):
    """ARIMA/SARIMA modeling and forecasting."""
    date_col, value_col = _select_ts_columns(df, "arima")

    ts = _get_ts_data(df, date_col, value_col)
    if ts is None:
        return

    ts_clean = ts.dropna()

    st.markdown("#### ARIMA Order (p, d, q)")
    c1, c2, c3 = st.columns(3)
    p = c1.number_input("p (AR):", 0, 10, 1, key="arima_p")
    d = c2.number_input("d (diff):", 0, 3, 1, key="arima_d")
    q = c3.number_input("q (MA):", 0, 10, 1, key="arima_q")

    use_seasonal = st.checkbox("Seasonal (SARIMA)", value=False, key="arima_seasonal")
    if use_seasonal:
        st.markdown("#### Seasonal Order (P, D, Q, s)")
        c1, c2, c3, c4 = st.columns(4)
        P = c1.number_input("P:", 0, 5, 1, key="arima_P")
        D = c2.number_input("D:", 0, 2, 1, key="arima_D")
        Q = c3.number_input("Q:", 0, 5, 1, key="arima_Q")
        s = c4.number_input("s:", 2, 365, 12, key="arima_s")

    forecast_steps = st.number_input("Forecast steps:", 1, 365, 12, key="arima_steps")
    train_pct = st.slider("Training data %:", 50, 95, 80, key="arima_train_pct")

    if st.button("Fit Model", key="fit_arima"):
        n_train = int(len(ts_clean) * train_pct / 100)
        train = ts_clean[:n_train]
        test = ts_clean[n_train:]

        try:
            with st.spinner("Fitting ARIMA model..."):
                if use_seasonal:
                    model = SARIMAX(train, order=(p, d, q),
                                    seasonal_order=(P, D, Q, s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit(disp=0)
                else:
                    model = ARIMA(train, order=(p, d, q)).fit()

            # Model summary
            st.markdown("#### Model Summary")
            c1, c2, c3 = st.columns(3)
            c1.metric("AIC", f"{model.aic:.2f}")
            c2.metric("BIC", f"{model.bic:.2f}")
            c3.metric("Log Likelihood", f"{model.llf:.2f}")

            # Coefficients
            coef_df = pd.DataFrame({
                "Parameter": model.params.index,
                "Coefficient": model.params.values,
                "Std Error": model.bse.values,
                "p-value": model.pvalues.values,
            }).round(6)
            st.dataframe(coef_df, use_container_width=True, hide_index=True)

            # Diagnostics
            with st.expander("Diagnostic Plots"):
                resid = model.resid
                fig = make_subplots(rows=2, cols=2,
                                    subplot_titles=("Residuals", "Histogram", "QQ Plot", "ACF"))
                fig.add_trace(go.Scatter(y=resid, mode="lines", name="Residuals"), row=1, col=1)
                fig.add_trace(go.Histogram(x=resid, name="Histogram"), row=1, col=2)

                sorted_resid = np.sort(resid)
                n = len(sorted_resid)
                theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
                fig.add_trace(go.Scatter(x=theoretical, y=sorted_resid, mode="markers",
                                         marker=dict(size=3), name="QQ"), row=2, col=1)
                fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode="lines",
                                         line=dict(dash="dash", color="red")), row=2, col=1)

                acf_vals = acf(resid, nlags=20)
                for i, val in enumerate(acf_vals):
                    fig.add_trace(go.Scatter(x=[i, i], y=[0, val], mode="lines",
                                             line=dict(color="steelblue"), showlegend=False), row=2, col=2)

                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            # Forecast
            if len(test) > 0:
                forecast = model.forecast(len(test))
                # Metrics
                mae = np.mean(np.abs(test.values - forecast.values))
                rmse = np.sqrt(np.mean((test.values - forecast.values) ** 2))
                mape = np.mean(np.abs((test.values - forecast.values) / test.values)) * 100 if (test.values != 0).all() else np.nan

                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mae:.4f}")
                c2.metric("RMSE", f"{rmse:.4f}")
                c3.metric("MAPE", f"{mape:.2f}%")

            # Full forecast
            forecast_full = model.get_forecast(steps=forecast_steps + len(test))
            fc_mean = forecast_full.predicted_mean
            fc_ci = forecast_full.conf_int()

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Train",
                                     line=dict(color="steelblue")))
            if len(test) > 0:
                fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Test",
                                         line=dict(color="green")))
            fig.add_trace(go.Scatter(x=fc_mean.index, y=fc_mean.values, name="Forecast",
                                     line=dict(color="red", width=2)))
            fig.add_trace(go.Scatter(x=fc_ci.index.tolist() + fc_ci.index.tolist()[::-1],
                                     y=fc_ci.iloc[:, 1].tolist() + fc_ci.iloc[:, 0].tolist()[::-1],
                                     fill="toself", fillcolor="rgba(255,0,0,0.1)",
                                     line=dict(color="rgba(255,0,0,0)"), name="95% CI"))
            fig.update_layout(title="Forecast", height=500)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Model fitting error: {e}")

    # Auto-ARIMA (simple grid search)
    with st.expander("Auto-ARIMA (Grid Search)"):
        if st.button("Run Auto-ARIMA", key="run_auto_arima"):
            with st.spinner("Searching for best ARIMA order..."):
                best_aic = np.inf
                best_order = (0, 0, 0)
                results = []
                for pi in range(4):
                    for di in range(3):
                        for qi in range(4):
                            try:
                                model = ARIMA(ts_clean, order=(pi, di, qi)).fit()
                                results.append({"p": pi, "d": di, "q": qi,
                                                "AIC": model.aic, "BIC": model.bic})
                                if model.aic < best_aic:
                                    best_aic = model.aic
                                    best_order = (pi, di, qi)
                            except Exception:
                                pass

                st.success(f"**Best order: ARIMA{best_order}** (AIC={best_aic:.2f})")
                results_df = pd.DataFrame(results).sort_values("AIC")
                st.dataframe(results_df.head(10).round(2), use_container_width=True, hide_index=True)


def _render_forecast_comparison(df: pd.DataFrame):
    """Compare multiple forecasting models."""
    date_col, value_col = _select_ts_columns(df, "comp")

    ts = _get_ts_data(df, date_col, value_col)
    if ts is None:
        return

    train_pct = st.slider("Training %:", 50, 90, 80, key="comp_train")

    if st.button("Compare Models", key="run_comp"):
        ts_clean = ts.dropna()
        n_train = int(len(ts_clean) * train_pct / 100)
        train = ts_clean[:n_train]
        test = ts_clean[n_train:]

        if len(test) == 0:
            st.error("No test data available.")
            return

        results = []
        forecasts = {}

        # Naive (last value)
        naive_fc = pd.Series([train.iloc[-1]] * len(test), index=test.index)
        forecasts["Naive"] = naive_fc
        results.append(_eval_forecast("Naive", test, naive_fc))

        # SMA
        sma_fc = pd.Series([train.rolling(7).mean().iloc[-1]] * len(test), index=test.index)
        forecasts["SMA(7)"] = sma_fc
        results.append(_eval_forecast("SMA(7)", test, sma_fc))

        # EMA
        ema_val = train.ewm(span=12).mean().iloc[-1]
        ema_fc = pd.Series([ema_val] * len(test), index=test.index)
        forecasts["EMA(12)"] = ema_fc
        results.append(_eval_forecast("EMA(12)", test, ema_fc))

        # ARIMA(1,1,1)
        try:
            model = ARIMA(train, order=(1, 1, 1)).fit()
            arima_fc = model.forecast(len(test))
            forecasts["ARIMA(1,1,1)"] = arima_fc
            results.append(_eval_forecast("ARIMA(1,1,1)", test, arima_fc, model.aic))
        except Exception:
            pass

        # ARIMA(2,1,2)
        try:
            model = ARIMA(train, order=(2, 1, 2)).fit()
            arima_fc = model.forecast(len(test))
            forecasts["ARIMA(2,1,2)"] = arima_fc
            results.append(_eval_forecast("ARIMA(2,1,2)", test, arima_fc, model.aic))
        except Exception:
            pass

        # Holt
        try:
            model = ExponentialSmoothing(train, trend="add").fit(optimized=True)
            holt_fc = model.forecast(len(test))
            forecasts["Holt"] = holt_fc
            results.append(_eval_forecast("Holt", test, holt_fc, model.aic))
        except Exception:
            pass

        # Results table
        results_df = pd.DataFrame(results).sort_values("RMSE")
        st.markdown("#### Model Comparison")
        st.dataframe(results_df.round(4), use_container_width=True, hide_index=True)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train.index, y=train.values, name="Train",
                                 line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=test.index, y=test.values, name="Test",
                                 line=dict(color="black", width=2)))
        colors = px.colors.qualitative.Set1
        for i, (name, fc) in enumerate(forecasts.items()):
            fig.add_trace(go.Scatter(x=fc.index, y=fc.values, name=name,
                                     line=dict(color=colors[i % len(colors)], dash="dash")))
        fig.update_layout(title="Forecast Comparison", height=500)
        st.plotly_chart(fig, use_container_width=True)


def _eval_forecast(name, actual, forecast, aic=None):
    """Evaluate forecast accuracy."""
    actual_vals = actual.values
    forecast_vals = forecast.values[:len(actual_vals)]
    errors = actual_vals - forecast_vals
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors / actual_vals)) * 100 if (actual_vals != 0).all() else np.nan
    result = {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
    if aic is not None:
        result["AIC"] = aic
    return result
