"""Exploratory data analysis of NASA SMAP telemetry across the entire dataset.

Uses sliding windows (length 30, hop 10) to extract time-series features from
every channel, then compares anomalous vs non-anomalous windows using Cohen's d
(effect size) to rank feature discrimination power.

Outputs:
  - Console report with top discriminating features
  - Per-channel feature dataframe saved as parquet
  - Feature analysis summary (Cohen's d rankings) saved as parquet

Dataset: NASA SMAP MSL telemetry
Source: https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap
Labels: labeled_anomalies.csv (per-channel anomaly region indices)
"""

import ast
import os
import time

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# ── Configuration ──────────────────────────────────────────────────────────────

DATA_DIR = '/workspace/examples/datasets/smap_msl/data_raw/test'
LABELS_FILE = '/workspace/examples/datasets/smap_msl/labeled_anomalies.csv'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

WINDOW_LENGTH = 30       # window size in timesteps
HOP = 10                 # stride between windows


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_feature_names(n_cols):
    """Return list of feature names for all columns."""
    names = []
    for c in range(n_cols):
        label = 'col%d' % (c + 1)
        names.extend([
            f'{label}_mean', f'{label}_std', f'{label}_median',
            f'{label}_min', f'{label}_max', f'{label}_range',
            f'{label}_skew', f'{label}_kurt', f'{label}_Q1',
            f'{label}_Q3', f'{label}_trend', f'{label}_CV',
            f'{label}_max_diff', f'{label}_mean_abs_diff',
        ])
    return names


def extract_features(win):
    """Extract time-series window statistics for all columns.

    Parameters
    ----------
    win : ndarray of shape (WINDOW_LENGTH, n_cols)

    Returns
    -------
    feats : ndarray — flattened feature vector
    """
    feats = []
    for c in range(win.shape[1]):
        col_data = win[:, c]
        mean_v = np.mean(col_data)
        std_v = np.std(col_data)
        median_v = np.median(col_data)
        min_v, max_v = float(col_data.min()), float(col_data.max())
        range_v = max_v - min_v

        with np.errstate(invalid='ignore', divide='ignore'):
            skew_v = stats.skew(col_data) if std_v > 1e-12 else 0.0
            kurt_v = stats.kurtosis(col_data) if std_v > 1e-12 else 0.0

        q1, q3 = np.percentile(col_data, [25, 75])
        trend_v = col_data[-1] - col_data[0]
        cv_v = std_v / (abs(mean_v) + 1e-12) if mean_v != 0 else 0.0

        diffs = np.diff(col_data)
        max_diff_v = float(np.max(np.abs(diffs)))
        mean_abs_diff_v = float(np.mean(np.abs(diffs)))

        feats.extend([
            mean_v, std_v, median_v, min_v, max_v, range_v,
            skew_v, kurt_v, q1, q3, trend_v, cv_v,
            max_diff_v, mean_abs_diff_v,
        ])
    return np.array(feats)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    print('=== Exploratory Analysis: SMAP Time-Series Anomaly Detection ===')
    print('Window length = %d, Hop = %d' % (WINDOW_LENGTH, HOP))
    print()

    # Load anomaly labels
    df_labels = pd.read_csv(LABELS_FILE)
    n_channels = len(df_labels)
    print('Channels with anomaly labels: %d\n' % n_channels)

    # Per-channel summary accumulators
    channel_summaries = []
    total_timesteps = 0
    total_anom_steps = 0

    for _idx, (_, row) in enumerate(tqdm(df_labels.iterrows(), desc='Processing channels', total=n_channels)):
        chan = row['chan_id']
        anomaly_seq_str = row['anomaly_sequences']
        anom_regions = ast.literal_eval(anomaly_seq_str)

        path = os.path.join(DATA_DIR, f'{chan}.npy')
        if not os.path.exists(path):
            tqdm.write('  WARNING: file not found for %s — skipping' % chan)
            continue

        data = np.load(path)
        n_ts, n_cols = data.shape
        feat_names = make_feature_names(n_cols)
        n_feats = len(feat_names)

        # Build anomaly mask
        is_anom = np.zeros(n_ts, dtype=bool)
        for region in anom_regions:
            is_anom[region[0]:region[1] + 1] = True

        n_anom_steps = int(is_anom.sum())
        total_anom_steps += n_anom_steps
        total_timesteps += n_ts

        # Sliding window iteration with HOP stride
        feat_sums_a = np.zeros(n_feats)
        feat_sqsums_a = np.zeros(n_feats)
        counts_a = 0

        feat_sums_n = np.zeros(n_feats)
        feat_sqsums_n = np.zeros(n_feats)
        counts_n = 0

        w_start = WINDOW_LENGTH
        while w_start + HOP <= n_ts:
            win = data[w_start - WINDOW_LENGTH: w_start]
            feats_arr = extract_features(win)

            if is_anom[w_start]:
                feat_sums_a += feats_arr
                feat_sqsums_a += feats_arr ** 2
                counts_a += 1
            else:
                feat_sums_n += feats_arr
                feat_sqsums_n += feats_arr ** 2
                counts_n += 1

            w_start += HOP

        # Compute means per feature
        anom_means = feat_sums_a / max(counts_a, 1)
        norm_means = feat_sums_n / max(counts_n, 1)

        channel_summaries.append({
            'channel': chan,
            'n_timesteps': n_ts,
            'n_cols': n_cols,
            'n_anom_steps': n_anom_steps,
            'pct_anom_steps': n_anom_steps / max(n_ts, 1) * 100,
            'n_anom_windows': counts_a,
            'n_norm_windows': counts_n,
            'n_total_windows': counts_a + counts_n,
            'feat_names': feat_names,
            'anom_means': anom_means,
            'norm_means': norm_means,
            'feat_sqsums_a': feat_sqsums_a,
            'feat_sqsums_n': feat_sqsums_n,
        })

    total_windows = sum(cs['n_total_windows'] for cs in channel_summaries)
    elapsed = time.time() - t_start
    print('\nTotal windows analyzed: %s (%.1fs)\n' % (total_windows, elapsed))

    # ── Compute Cohen's d across entire dataset (pooled) ────────────────────────

    feat_types = [
        'mean', 'std', 'median', 'min', 'max', 'range',
        'skew', 'kurt', 'Q1', 'Q3', 'trend', 'CV',
        'max_diff', 'mean_abs_diff',
    ]

    feat_analysis = []

    for ftype in tqdm(feat_types, desc='Computing effect sizes'):
        anom_values = []
        norm_values = []
        wca_all = []
        wcn_all = []
        pooled_var_sum = 0.0
        total_df = 0

        for cs in channel_summaries:
            fnames = cs['feat_names']
            anom_m = cs['anom_means']
            norm_m = cs['norm_means']
            sqsum_a = cs['feat_sqsums_a']
            sqsum_n = cs['feat_sqsums_n']

            for j, fname in enumerate(fnames):
                if fname.endswith('_%s' % ftype):
                    na = cs['n_anom_windows']
                    nn = cs['n_norm_windows']

                    # Store per-channel values
                    anom_values.append(anom_m[j])
                    norm_values.append(norm_m[j])
                    wca_all.append(na)
                    wcn_all.append(nn)

                    # Compute per-channel variance from sum and sum-of-squares
                    var_a = 0.0
                    if na > 1:
                        var_a = sqsum_a[j] / na - anom_m[j] ** 2
                        var_a = max(var_a, 0.0)

                    var_n = 0.0
                    if nn > 1:
                        var_n = sqsum_n[j] / nn - norm_m[j] ** 2
                        var_n = max(var_n, 0.0)

                    # Accumulate pooled variance
                    pooled_var_sum += (na - 1) * var_a + (nn - 1) * var_n
                    total_df += (na - 1) + (nn - 1)

        if not anom_values:
            continue

        anom_values = np.array(anom_values)
        norm_values = np.array(norm_values)
        wca_arr = np.array(wca_all, dtype=float)
        wcn_arr = np.array(wcn_all, dtype=float)

        # Weighted mean difference
        total_w = wca_arr + wcn_arr
        weighted_diff = np.average(anom_values - norm_values, weights=total_w + 1e-12)

        # Pooled standard deviation
        if total_df > 0:
            pooled_std = np.sqrt(pooled_var_sum / total_df)
        else:
            pooled_std = 1e-12

        d = weighted_diff / pooled_std if pooled_std > 1e-12 else 0.0
        anom_mean_agg = np.average(anom_values, weights=wca_arr + 1e-12)
        norm_mean_agg = np.average(norm_values, weights=wcn_arr + 1e-12)

        feat_analysis.append({
            'feature_type': ftype,
            'cohens_d': abs(d),
            'd_signed': d,
            'weighted_diff': weighted_diff,
            'pooled_std': pooled_std,
            'anom_mean': anom_mean_agg,
            'norm_mean': norm_mean_agg,
        })

    # Sort by absolute Cohen's d descending
    feat_analysis.sort(key=lambda x: x['cohens_d'], reverse=True)

    # ── Report top 10 features ────────────────────────────────────────────────────

    print('=' * 80)
    print('  TOP FEATURES DISCRIMINATING ANOMALOUS vs NORMAL WINDOWS (Cohen\'s d)')
    print('=' * 80)
    print()
    print(f'{"Rank":<5} {"Feature Type":<16} {"|d|":>7}   '
          f'{"Anom Mean":>12} {"Norm Mean":>12} {"Diff":>12}')
    print('-' * 70)

    for rank, fa in enumerate(feat_analysis[:10], 1):
        sign = '+' if fa['d_signed'] >= 0 else '-'
        d_str = f'{sign}{abs(fa["weighted_diff"]):.4f}'
        print(f'{rank:<5} {fa["feature_type"]:<16} {fa["cohens_d"]:>7.3f}   '
              f'{fa["anom_mean"]:>12.4f} {fa["norm_mean"]:>12.4f} {d_str}')

    # ── Full ranking ──────────────────────────────────────────────────────────────

    print()
    print('=' * 80)
    print('  FULL RANKING -- all feature types')
    print('=' * 80)
    print()
    print(f'{"Rank":<5} {"Feature Type":<16} {"|d|":>7}   '
          f'{"Anom Mean":>12} {"Norm Mean":>12} {"Diff":>12} {"Direction"}')
    print('-' * 90)

    for rank, fa in enumerate(feat_analysis, 1):
        sign = '+' if fa['d_signed'] >= 0 else '-'
        d_str = f'{sign}{abs(fa["weighted_diff"]):.4f}'
        direction = 'UP   higher in anomaly' if fa['d_signed'] > 0 else 'DOWN lower in anomaly'
        print(f'{rank:<5} {fa["feature_type"]:<16} {fa["cohens_d"]:>7.3f}   '
              f'{fa["anom_mean"]:>12.4f} {fa["norm_mean"]:>12.4f} {d_str}  {direction}')

    # ── Per-channel summary table ────────────────────────────────────────────────

    print()
    print('=' * 80)
    print('  PER-CHANNEL SUMMARY')
    print('=' * 80)
    print()
    print(f'{"Channel":<8} {"Timesteps":>9} {"Cols":>4} '
          f'{"Anom Steps":>11} {"% Anom":>7} '
          f'{"Win (anom)":>11} {"Win (norm)":>11}')
    print('-' * 65)

    for cs in channel_summaries:
        pct = f"{cs['pct_anom_steps']:.1f}%"
        timesteps_str = '{:,}'.format(cs['n_timesteps'])
        anom_str = '{:,}'.format(cs['n_anom_steps'])
        wanom_str = '{:,}'.format(cs['n_anom_windows'])
        wnorm_str = '{:,}'.format(cs['n_norm_windows'])
        print('%-8s %9s %4d %11s %7s %11s %11s' % (
            cs['channel'], timesteps_str, cs['n_cols'],
            anom_str, pct,
            wanom_str, wnorm_str))

    # ── Save outputs to parquet ──────────────────────────────────────────────────

    out_path_features = os.path.join(OUTPUT_DIR, 'window_features.parquet')
    out_path_analysis = os.path.join(OUTPUT_DIR, 'feature_analysis.parquet')

    print('\nSaving per-channel stats to: %s' % out_path_features)

    # Build a flat table for parquet: one row per channel with aggregated stats
    rows = []
    for cs in channel_summaries:
        r = {
            'channel': cs['channel'],
            'n_timesteps': cs['n_timesteps'],
            'n_cols': cs['n_cols'],
            'n_anom_steps': cs['n_anom_steps'],
            'pct_anom_steps': cs['pct_anom_steps'],
            'n_anom_windows': cs['n_anom_windows'],
            'n_norm_windows': cs['n_norm_windows'],
        }

        feat_names_list = cs['feat_names']
        for j, fn in enumerate(feat_names_list):
            r['anom_%s' % fn] = cs['anom_means'][j]
            r['norm_%s' % fn] = cs['norm_means'][j]

            # Per-channel Cohen's d for each feature
            na = cs['n_anom_windows']
            nn = cs['n_norm_windows']
            mean_a = cs['anom_means'][j]
            mean_n = cs['norm_means'][j]

            var_a = 0.0
            if na > 1:
                var_a = cs['feat_sqsums_a'][j] / na - mean_a ** 2
                var_a = max(var_a, 0.0)
            var_n = 0.0
            if nn > 1:
                var_n = cs['feat_sqsums_n'][j] / nn - mean_n ** 2
                var_n = max(var_n, 0.0)

            pooled_var = ((na - 1) * var_a + (nn - 1) * var_n) / (na + nn - 2) if (na + nn - 2) > 0 else 0.0
            d_val = (mean_a - mean_n) / np.sqrt(pooled_var) if pooled_var > 0 else 0.0

            r['d_%s' % fn] = d_val

        rows.append(r)

    df_channel_stats = pd.DataFrame(rows)
    df_channel_stats.to_parquet(out_path_features, engine='pyarrow', index=False)

    # Feature analysis summary (Cohen's d rankings across all channels)
    df_analysis = pd.DataFrame(feat_analysis)
    df_analysis['rank'] = range(1, len(df_analysis) + 1)
    cols_out = ['rank', 'feature_type', 'cohens_d', 'd_signed',
                'weighted_diff', 'pooled_std', 'anom_mean', 'norm_mean']
    df_analysis = df_analysis[cols_out]
    df_analysis.to_parquet(out_path_analysis, engine='pyarrow', index=False)

    print(f'Saved: {out_path_features}')
    print(f'Saved: {out_path_analysis}\n')

    # ── Summary statistics ────────────────────────────────────────────────────────

    print('=' * 80)
    print('  DATASET OVERVIEW')
    print('=' * 80)
    print(f'Total channels processed : {len(channel_summaries)}')
    print(f'Total timesteps          : {total_timesteps:,}')
    pct_total = total_anom_steps / max(total_timesteps, 1) * 100
    print(f'Total anomalous steps    : {total_anom_steps:,} ({pct_total:.2f}%)')
    print(f'Total windows (HOP={HOP})   : {total_windows:,}')
    print(f'Processing time          : {elapsed:.1f}s\n')


if __name__ == '__main__':
    main()
