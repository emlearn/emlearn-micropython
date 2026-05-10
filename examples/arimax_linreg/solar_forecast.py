#!/usr/bin/env python3
"""
ARIMAX Solar Forecasting Pipeline
==================================
Loads UniSolar solar generation and weather data, builds ARIMAX models
per site using sktime, and evaluates continuous 24-hour-ahead forecasts.

Usage:
    python solar_forecast.py --data-dir examples/datasets/unisolar/ [--seed 42] [--hop 24]
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------------------------------------------------------
# Metrics & Logging
# ---------------------------------------------------------------------------

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import structlog
from tqdm import tqdm
from sktime.performance_metrics.forecasting import (
    mean_absolute_error as _mae,
    mean_squared_error as _mse,
)

structlog.configure(
    cache_logger_on_first_use=True,
    wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WINDOW_HOURS = 336       # 14 days × 24h
HOP_HOURS = 12           # between window starts (overridable via --hop)
FORECAST_STEPS = 24      # predict next 24 hours ahead
MAX_GAP_HOURS = 4        # max forward/back fill for gaps
MODEL_ORDER = (2, 1, 0)  # non-seasonal ARIMA(p,d,q)
SEASONAL_ORDER = (1, 1, 0, 24)  # seasonal ARIMA(P,D,Q,s=24h diurnal cycle)


# ---------------------------------------------------------------------------
# 1. Data Loading & Preprocessing
# ---------------------------------------------------------------------------

def load_solar_generation(data_dir, site_id):
    """Load per-site solar_generation.npy as structured float32 array."""
    path = os.path.join(data_dir, "site_{:02d}".format(site_id), "solar_generation.npy")
    arr = np.load(path)
    timestamps = pd.to_datetime(arr["timestamp"], unit="s").tz_localize(None)
    generation = arr["generation"].astype(np.float64)
    return pd.DataFrame({"generation_kw": generation}, index=timestamps)


def load_weather_features(data_dir, campus_id):
    """Load per-campus weather_features.npy as structured float32 array."""
    path = os.path.join(data_dir, "campus_{:02d}".format(campus_id), "weather_features.npy")
    arr = np.load(path)
    timestamps = pd.to_datetime(arr["timestamp"], unit="s").tz_localize(None)
    cols = {
        "air_temp":       arr["air_temp"].astype(np.float64),
        "apparent_temp":  arr["apparent_temp"].astype(np.float64),
        "dew_point_temp": arr["dew_point_temp"].astype(np.float64),
        "rel_humidity":   arr["rel_humidity"].astype(np.float64),
        "wind_speed":     arr["wind_speed"].astype(np.float64),
        "wind_direction": arr["wind_direction"].astype(np.float64),
    }
    return pd.DataFrame(cols, index=timestamps)


def load_metadata(data_dir):
    """Load metadata.json → {site_id_int: campus_id_int}."""
    with open(os.path.join(data_dir, "metadata.json")) as f:
        meta = json.load(f)
    return {int(k): int(v) for k, v in meta["site_to_campus"].items()}


def resample_hourly(df):
    """Resample an irregular-frequency DataFrame to strict hourly."""
    df = df.sort_index()
    df_hr = df.resample("h").mean()
    return df_hr


def fill_gaps(solar_df, weather_df, max_gap_hours=MAX_GAP_HOURS):
    """
    Gap filling strategy:
      - Solar generation: nighttime hours (UTC <6am / >=7pm) → set to 0 kW.
        Remaining daytime gaps are filled via ffill/bfill up to max_gap_hours.
      - Weather features: ffill + bfill up to max_gap_hours.
    Returns filled DataFrames and a count of remaining NaN rows.
    """
    # --- Solar: fill nighttime → 0, then ffill/bfill daytime gaps ---
    hour = solar_df.index.hour
    night_mask = (hour < 6) | (hour >= 19)

    solar_filled = solar_df.copy()
    night_nan = solar_filled["generation_kw"].isna() & night_mask
    solar_filled.loc[night_nan, "generation_kw"] = 0.0

    # Remaining daytime gaps → ffill + bfill (up to max_gap_hours)
    solar_filled["generation_kw"] = (
        solar_filled["generation_kw"]
        .ffill(limit=max_gap_hours)
        .bfill(limit=max_gap_hours)
    )

    remaining_solar_rows = int(solar_filled["generation_kw"].isna().sum())

    # --- Weather: forward-fill then back-fill ---
    weather_filled = weather_df.copy()
    for col in weather_filled.columns:
        weather_filled[col] = (weather_filled[col]
                               .ffill(limit=max_gap_hours)
                               .bfill(limit=max_gap_hours))
    remaining_weather_rows = int(weather_filled.isna().any(axis=1).sum())

    return solar_filled, weather_filled, remaining_solar_rows, remaining_weather_rows


# ---------------------------------------------------------------------------
# 2. Window Construction
# ---------------------------------------------------------------------------

def build_windows_with_nan_check(series_df, col="generation_kw", hop=None):
    """
    Build overlapping windows and drop any containing NaN values.
    Returns a list of (start, end) slices and the count dropped.
    This is fast — just a vectorized NaN mask scan.
    """
    h = hop if hop is not None else HOP_HOURS
    total = len(series_df)
    values = series_df[col].values
    nan_mask = np.isnan(values)

    windows = []
    dropped = 0

    for start in range(0, total - WINDOW_HOURS + 1, h):
        end = start + WINDOW_HOURS
        if nan_mask[start:end].any():
            dropped += 1
            continue
        windows.append((start, end))

    return windows, dropped


# ---------------------------------------------------------------------------
# 3. Train/Validation/Test Split (Site-Level Holdout)
# ---------------------------------------------------------------------------

def split_sites(site_ids, train_ratio=0.70, val_ratio=0.15, seed=42):
    """Random site-level holdout split."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(site_ids))
    n_train = int(round(len(site_ids) * train_ratio))
    n_val = int(round(len(site_ids) * val_ratio))

    train = [site_ids[i] for i in idx[:n_train]]
    val   = [site_ids[i] for i in idx[n_train:n_train + n_val]]
    test  = [site_ids[i] for i in idx[n_train + n_val:]]
    return train, val, test




# ---------------------------------------------------------------------------
# 4–5. Continuous Forecast Loop with ARIMAX
# ---------------------------------------------------------------------------

def build_site_data(solar_df, weather_df, hop=HOP_HOURS):
    """
    Prepare a single site's data: build window list from the filled solar series.
    Returns (windows_list, dropped_count) or (None, 0) if no valid windows.
    Fast — just scans NaN mask.
    """
    solar = solar_df.copy()
    windows, total_dropped = build_windows_with_nan_check(solar, "generation_kw", hop=hop)
    return windows, total_dropped


def forecast_site_generator(site_id, group, windows, solar_df, weather_df, hop=HOP_HOURS):
    """
    Generator that yields one result dict per window.
    Each yield contains:
      site_id, group,
      predictions (list), actuals (list) — aligned with forecast_timestamps
      forecast_timestamps: list of pd.Timestamp for each pred/actual pair
      train_start, train_end: index positions into solar series for the training window
      train_timestamps: timestamps covering the training window
      fit_time_ms, pred_time_ms, error_msg or None

    Usage in main loop:
        for win_result in forecast_site_generator(...):
            total_elapsed = time.time() - overall_start
            if args.timeout and total_elapsed > args.timeout:
                break  # ← timeout abort handled by caller
            # accumulate results...
    """
    solar = solar_df.copy()
    solar_vals = solar["generation_kw"].values  # cached for speed

    fit_times = []
    pred_times = []
    prep_times = []
    exception_count = 0

    with tqdm(windows, desc=f"  site {site_id}", leave=False, ncols=100) as win_bar:
        for (w_start, w_end) in win_bar:
            # --- Data preparation ---
            t_prep = time.time()
            y_train = pd.Series(solar_vals[w_start:w_end], index=solar.index[w_start:w_end])
            if y_train.isna().any() or len(y_train) < 24:
                continue

            X_train = weather_df.loc[y_train.index, ["air_temp"]].dropna()
            common_idx = y_train.index.intersection(X_train.index)
            if len(common_idx) < 24:
                continue

            y_aligned_vals = y_train.loc[common_idx].values
            X_aligned_vals = X_train.loc[common_idx].values

            last_train_ts = solar.index[w_end - 1]
            future_idx = pd.date_range(
                start=last_train_ts + pd.Timedelta(hours=1),
                periods=FORECAST_STEPS, freq="h"
            )
            last_temp = float(X_aligned_vals[-1][0]) if len(X_aligned_vals) > 0 else 0.0
            prep_times.append(time.time() - t_prep)

            # --- Fit SARIMAX ---
            t_fit = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = SARIMAX(
                        y_aligned_vals,
                        exog=X_aligned_vals,
                        order=MODEL_ORDER,
                        seasonal_order=SEASONAL_ORDER,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = mod.fit(disp=False, maxiter=100)
                fit_times.append(time.time() - t_fit)
            except Exception as e:
                exception_count += 1
                if exception_count <= 3:
                    print(
                        f"  [ERROR] {site_id} [{w_start}:{w_end}] fit: {e}", flush=True,
                    )
                continue

            # --- Predict ---
            t_pred = time.time()
            try:
                y_pred_vals = res.forecast(
                    steps=FORECAST_STEPS,
                    exog=np.array([[last_temp]] * FORECAST_STEPS),
                )
                pred_times.append(time.time() - t_pred)
            except Exception as e:
                exception_count += 1
                if exception_count <= 3:
                    print(
                        f"  [ERROR] {site_id} [{w_start}:{w_end}] predict: {e}", flush=True,
                    )
                continue

            # Collect predictions vs actuals (finite only)
            preds = []
            actuals = []
            for i, ts in enumerate(future_idx):
                if i >= len(y_pred_vals) or ts not in solar.index:
                    continue
                pv = float(y_pred_vals[i])
                av = float(solar_vals[solar.index.get_loc(ts)])
                if np.isfinite(pv) and np.isfinite(av):
                    preds.append(pv)
                    actuals.append(av)

            yield {
                "site_id":           site_id,
                "group":             group,
                "predictions":       preds,
                "actuals":           actuals,
                "forecast_timestamps": [ts for ts in future_idx if ts in solar.index],
                "train_start":       w_start,
                "train_end":         w_end,
            }

    # Report timing summary once per site (after generator exhausted or aborted)
    n_fit = len(fit_times)
    t_fit_avg = np.mean(fit_times) if fit_times else 0
    t_pred_avg = np.mean(pred_times) if pred_times else 0
    t_prep_avg = np.mean(prep_times) if prep_times else 0
    print(
        f"     [timing] {site_id}: "
        f"fit={t_fit_avg*1000:.0f}ms pred={t_pred_avg*1000:.0f}ms "
        f"prep={t_prep_avg*1000:.0f}ms fits={n_fit} errs={exception_count}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# 6. Evaluation Metrics
# ---------------------------------------------------------------------------

def mean_absolute_error(y_true, y_pred):
    """MAE in original units (kW)."""
    return float(np.mean(np.abs(np.array(y_pred) - np.array(y_true))))


def root_mean_squared_error(y_true, y_pred):
    """RMSE in original units."""
    return float(np.sqrt(np.mean((np.array(y_pred) - np.array(y_true)) ** 2)))


def smape_score(y_true, y_pred):
    """Symmetric MAPE — scale-independent, handles near-zero well."""
    yt = np.array(y_true)
    yp = np.array(y_pred)
    denom = np.abs(yt) + np.abs(yp)
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(2.0 * np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_site_timeseries(site_id, group, solar_df, weather_df,
                         window_forecasts):
    """
    Build a plotly figure with three subplots (sharing X axis):

      Row 1 — Solar generation:
        • Full timeseries as a thin line (light grey)
        • Each prediction window drawn as its own dashed trace so that
          no lines bridge across discontinuous / unconnected windows.
        • Actuals at forecast times plotted as orange dots.

      Row 2 — Air temperature (°C) sharing X axis.

      Row 3 — Prediction error (actual − predicted) per window,
              sharing X axis.

    Returns a ``go.Figure``.
    """
    solar_idx = solar_df.index
    solar_vals = solar_df["generation_kw"].values
    temp_vals  = weather_df["air_temp"].values
    n = len(solar_idx)

    # ---- Collect all-indices markers (for legends / hover) ----
    has_actual = np.zeros(n, dtype=bool)
    actual_vals_arr = np.full(n, np.nan)

    traces = []  # list of go.Scatter objects

    for widx, wf in enumerate(window_forecasts):
        fts = wf["forecast_timestamps"]   # pd.Timestamp per forecast step
        preds  = wf["predictions"]
        actuals = wf["actuals"]
        if not fts or len(fts) == 0:
            continue

        x = list(fts)
        y_pred = list(preds)
        y_actual = [av for av in actuals]
        n_pts = len(x)

        # --- Plot prediction as its own trace (no bridging between windows) ---
        traces.append(go.Scatter(
            x=x, y=y_pred,
            name=f"Pred #{widx + 1}",
            mode="lines",
            line=dict(color="#1f77b4", width=2, dash="dash"),
            legendgroup=str(widx),          # group for click-to-toggle
            showlegend=(widx == 0),        # only first in legend column
        ))

        # --- Actuals at forecast times (markers) ---
        traces.append(go.Scatter(
            x=x, y=y_actual,
            name="Actual" if widx == 0 else None,
            mode="markers",
            marker=dict(color="#ff7f0e", size=4, opacity=0.6),
            legendgroup=str(widx),
            showlegend=(widx == 0),
        ))

        # --- Mark actuals for the full-timeseries overlay ---
        for ts, av in zip(x, y_actual):
            idx = solar_idx.get_loc(ts)
            has_actual[idx] = True
            if np.isfinite(av):
                actual_vals_arr[idx] = av

    # ---- Build per-window error traces (row 3) ----
    for widx, wf in enumerate(window_forecasts):
        fts = wf["forecast_timestamps"]
        preds  = wf["predictions"]
        actuals = wf["actuals"]
        if not fts or len(fts) == 0:
            continue
        x = list(fts)
        error = [av - pv for av, pv in zip(actuals, preds)]
        traces.append(go.Scatter(
            x=x, y=error,
            name=f"Error #{widx + 1}",
            mode="lines+markers",
            line=dict(color="#d62728", width=1.5),
            marker=dict(size=4),
            legendgroup=str(widx),
            showlegend=(widx == 0),
        ))

    # ---- Global solar generation trace ----
    trace_solar = go.Scatter(
        x=solar_idx, y=solar_vals,
        name="Solar generation",
        line=dict(color="#e0e0e0", width=1),
        hoverinfo="skip",
        showlegend=False,
    )

    # ---- Actuals overlay for row 1 ----
    trace_actuals = go.Scatter(
        x=solar_idx[has_actual], y=actual_vals_arr[has_actual],
        name="Actual (at forecast time)",
        mode="markers",
        marker=dict(color="#ff7f0e", size=4, opacity=0.5),
        showlegend=False,
    )

    # ---- Temperature trace ----
    trace_temp = go.Scatter(
        x=solar_idx, y=temp_vals,
        name="Air temperature (°C)",
        line=dict(color="#2ca02c", width=1.5),
    )

    # ---- Build figure ----
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.47, 0.26, 0.27],
        subplot_titles=("Solar Generation (kW)",
                        "Air Temperature (°C)",
                        "Prediction Error (kW)"),
    )

    fig.add_trace(trace_solar,          row=1, col=1)
    fig.add_trace(trace_actuals,        row=1, col=1)
    for tr in traces:
        if any("Error #" in name for name in [tr.name] if tr.name):
            fig.add_trace(tr, row=3, col=1)
        else:
            fig.add_trace(tr, row=1, col=1)

    # temperature is always row 2
    fig.add_trace(trace_temp, row=2, col=1)

    fig.update_layout(
        title=f"Site {site_id} ({group}) — Forecast Overlay",
        height=800,
        width=1200,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="right", x=1),
    )

    fig.update_yaxes(title_text="kW", row=1, col=1)
    fig.update_yaxes(title_text="°C", row=2, col=1)
    fig.update_yaxes(title_text="kW", row=3, col=1)
    return fig

def main():
    parser = argparse.ArgumentParser(description="ARIMAX Solar Forecasting Pipeline")
    parser.add_argument("--data-dir", required=True, help="Path to unisolar data directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sites", type=int, default=None,
                        help="Limit number of sites (for debugging)")
    parser.add_argument("--hop", type=int, default=HOP_HOURS,
                        help=f"Window hop in hours (default {HOP_HOURS})")
    parser.add_argument("--subsample", type=int, default=10,
                        help="Randomly subsample at most N windows per site "
                             "(default 10, use -1 for no limit)")
    parser.add_argument("--until",
                        help="Only use data before this datetime (YYYY-MM-DD or full ISO) "
                             "for both training and testing.")
    parser.add_argument("--max-time-pct", type=float, default=1.0,
                        help="Only use the first N% of available data time range "
                             "(e.g. 20 for first 20%%). 1.0 = full range.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Abort after N seconds; summarize results so far.")
    parser.add_argument("--plot", action="store_true",
                        help="Plot per-site timeseries + predictions (saves HTML in out/).")
    parser.add_argument("--plot-sites", type=str, default=None,
                        help="Comma-separated site IDs to plot (default: all processed sites).")
    args = parser.parse_args()

    data_dir = args.data_dir
    plot_dir = None
    if args.plot:
        plot_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "plot_output",
        )
    hop = args.hop
    overall_start = time.time()
    print(f"Data: {data_dir}  |  Hop: {hop}h  |  Seed: {args.seed}")

    # --- Load metadata & site list ---
    site_to_campus = load_metadata(data_dir)
    all_sites = sorted(site_to_campus.keys())
    if args.max_sites:
        all_sites = all_sites[:args.max_sites]
    n_total = len(all_sites)
    print(f"Sites to process: {n_total}")

    # --- Site split ---
    train_sites, val_sites, test_sites = split_sites(all_sites, seed=args.seed)
    site_groups = {}
    for s in train_sites:
        site_groups[s] = "train"
    for s in val_sites:
        site_groups[s] = "val"
    for s in test_sites:
        site_groups[s] = "test"

    print(f"Train: {len(train_sites)} | Val: {len(val_sites)} | Test: {len(test_sites)}\n")

    # --- Process each site (with generator + timeout support) ---
    results = []
    total_windows_available = 0
    total_windows_subsampled = 0

    timed_out = False

    with tqdm(all_sites, desc="Sites", ncols=100) as site_bar:
        for site_id in site_bar:
            campus_id = site_to_campus[site_id]
            group = site_groups[site_id]
            site_bar.set_description(f"Site {site_id:>2} ({campus_id:>2})")

            # --- Load & prep (fast) ---
            solar_df = load_solar_generation(data_dir, site_id)
            weather_df = load_weather_features(data_dir, campus_id)
            solar_hr = resample_hourly(solar_df)
            weather_hr = resample_hourly(weather_df)
            solar_filled, weather_filled, s_nan, w_nan = fill_gaps(solar_hr, weather_hr)

            # --- Time-range truncation ---
            max_dt = solar_filled.index.max()
            min_dt = solar_filled.index.min()
            data_range = (max_dt - min_dt).total_seconds()
            if args.until:
                cutoff = pd.to_datetime(args.until, utc=True).tz_localize(None)
                solar_filled = solar_filled.loc[solar_filled.index <= cutoff]
                weather_filled = weather_filled.loc[weather_filled.index <= cutoff]
            if args.max_time_pct < 1.0:
                target_end = min_dt + pd.Timedelta(seconds=data_range * args.max_time_pct / 100.0)
                solar_filled = solar_filled.loc[solar_filled.index <= target_end]
                weather_filled = weather_filled.loc[weather_filled.index <= target_end]

            site_bar.set_postfix({"hr": len(solar_filled), "s_nans": s_nan, "w_nans": w_nan})

            # --- Build windows (fast) ---
            all_windows, total_dropped_site = build_site_data(
                solar_filled, weather_filled, hop=hop
            )
            if all_windows is None or len(all_windows) == 0:
                site_bar.set_postfix_str("→ no valid windows")
                continue

            # --- Optional random subsample of windows ---
            if args.subsample > 0 and len(all_windows) > args.subsample:
                rng = np.random.RandomState(args.seed)
                subsample_idx = rng.choice(len(all_windows), size=args.subsample, replace=False)
                subsample_idx.sort()
                windows = [all_windows[i] for i in subsample_idx]
            else:
                windows = list(all_windows)
            total_windows_available += len(all_windows)
            total_windows_subsampled += len(windows)

            # --- Forecast loop with generator + timeout check ---
            site_preds = []
            site_actuals = []
            window_forecasts = []  # list of dicts for plotting: timestamps, preds, actuals

            for win_result in forecast_site_generator(
                site_id, group, windows, solar_filled, weather_filled, hop=hop
            ):
                if args.timeout is not None:
                    elapsed = time.time() - overall_start
                    if elapsed > args.timeout:
                        timed_out = True
                        break
                site_preds.extend(win_result["predictions"])
                site_actuals.extend(win_result["actuals"])
                window_forecasts.append(win_result)

            # --- Record results (unified: same path whether we continue or abort) ---
            if not site_preds:
                continue  # no forecasts produced, nothing to record
            results.append({
                "site_id":           site_id,
                "group":             group,
                "predictions":       site_preds,
                "actuals":           site_actuals,
                "windows_ok":        len(windows),
                "windows_available": total_windows_subsampled - len(windows) + len(all_windows),
                "window_forecasts":  window_forecasts,   # for plotting
                "solar_df":          solar_filled.copy(), # for plotting
                "weather_df":        weather_filled.copy(),# for plotting
            })

            # If timed out, print message and stop the outer site loop
            if timed_out:
                print(f"\n⚠️  Timeout after {time.time()-overall_start:.0f}s — stopping.", flush=True)
                break

    # --- Evaluation ---
    if not results:
        print("\nERROR: No valid results produced!")
        sys.exit(1)

    # ---- Per-site metrics ----
    rows = []
    for r in results:
        preds, actuals = r["predictions"], r["actuals"]
        if len(preds) < 2:
            continue
        rows.append({
            "site_id":      r["site_id"],
            "group":        r["group"],
            "n_forecasts":  len(preds),
            "windows_ok":   r["windows_ok"],
            "mae_kw":       mean_absolute_error(actuals, preds),
            "rmse_kw":      root_mean_squared_error(actuals, preds),
            "smape_pct":    smape_score(actuals, preds) * 100,
        })

    metrics_df = pd.DataFrame(rows)

    # ---- Print Summary Report ----
    print("\n" + "=" * 80)
    reason = "TIMED OUT" if timed_out else "COMPLETED"
    print(f"  ARIMAX SOLAR FORECASTING — {reason}")
    print("=" * 80)

    subsample_info = (
        f" and subsampled to max {args.subsample} per site"
        if args.subsample > 0 else ""
    )
    valid_windows = total_windows_subsampled
    print(f"\nSites processed:       {len(results)}")
    print(f"Window hop:            {hop}h")
    print(f"Total windows available:{total_windows_available}")
    print(f"Windows subsampled{subsample_info}: {total_windows_subsampled} "
          f"(from {total_windows_available} total)")

    # Per-site table
    print("\nPer-Site Metrics:")
    print("-" * 80)
    hdr = f"{'site':>5} {'group':>5} {'windows':>7} {'fcsts':>6} {'MAE(kW)':>10} " \
          f"{'RMSE(kW)':>10} {'sMAPE%':>8}"
    print(hdr)
    print("-" * 80)
    for _, row in metrics_df.iterrows():
        print(f"{int(row['site_id']):>5} {row['group']:>5} {int(row['windows_ok']):>7} "
              f"{int(row['n_forecasts']):>6} {row['mae_kw']:>10.4f} "
              f"{row['rmse_kw']:>10.4f} {row['smape_pct']:>8.2f}")

    # Aggregate by group
    print("\nAggregate Metrics by Group:")
    print("-" * 80)
    for grp in ["train", "val", "test"]:
        gdf = metrics_df[metrics_df["group"] == grp]
        if gdf.empty:
            continue
        print(f"  {grp.upper():>5}: n={len(gdf):2d}  "
              f"MAE:  mean={gdf['mae_kw'].mean():.4f} median={gdf['mae_kw'].median():.4f} kW  "
              f"RMSE: mean={gdf['rmse_kw'].mean():.4f} median={gdf['rmse_kw'].median():.4f} kW  "
              f"sMAPE={gdf['smape_pct'].mean():.2f}%")

    # Overall (pooled across all sites)
    print("\nOverall Aggregate:")
    print("-" * 80)
    all_p, all_a = [], []
    for r in results:
        all_p.extend(r["predictions"])
        all_a.extend(r["actuals"])

    print(f"  Total forecasts: {len(all_p)}")
    print(f"  MAE:  {mean_absolute_error(all_a, all_p):.4f} kW")
    print(f"  RMSE: {root_mean_squared_error(all_a, all_p):.4f} kW")
    print(f"  sMAPE: {smape_score(all_a, all_p) * 100:.2f}%")

    # ---- Plotting (optional) ----
    if args.plot and plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

        # Determine which sites to plot
        if args.plot_sites:
            plot_site_ids = {int(s.strip()) for s in args.plot_sites.split(",")}
            results_to_plot = [r for r in results if r["site_id"] in plot_site_ids]
        else:
            results_to_plot = results

        print(f"\nSaving {len(results_to_plot)} site plots to: {plot_dir}")
        for r in results_to_plot:
            sid = int(r["site_id"])
            solar_df_plot = r.get("solar_df")
            weather_df_plot = r.get("weather_df")
            if solar_df_plot is None or not r.get("window_forecasts"):
                print(f"  Site {sid}: skipped (no data / no forecasts for plotting)")
                continue

            fig = plot_site_timeseries(
                sid, r["group"],
                solar_df_plot, weather_df_plot,
                r["window_forecasts"],
            )
            out_path = os.path.join(plot_dir, f"site_{sid}.html")
            fig.write_html(out_path)  # self-contained HTML
            print(f"  Site {sid}: saved → {out_path}")

    # ---- Save detailed results ----
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forecast_results.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nPer-site results saved to: {out_csv}")


if __name__ == "__main__":
    main()
