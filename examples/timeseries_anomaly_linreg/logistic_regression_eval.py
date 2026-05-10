"""Systematic logistic regression evaluation for anomaly detection on SMAP telemetry.

Fast, vectorised window feature extraction using numpy stride tricks + scipy batch ops,
with multiprocessing across channels. Then systematically tests every combination of 1,
2, 3 features from the top-10 candidates identified by explore_enhanced.py.

Top-10 candidate feature types (combined Borda ranking):
    rank  feature        description
    1     max_diff       Max absolute consecutive difference
    2     Q3             75th percentile
    3     std            Standard deviation
    4     skew           Skewness
    5     trend          Linear slope
    6     range          Max - Min
    7     median         Median
    8     mean           Mean
    9     min            Minimum
    10   kurt           Kurtosis

Usage:
    python3 examples/timeseries_anomaly_linreg/logistic_regression_eval.py

Outputs:
    - Console report with per-feature-set ROC-AUC, F1, precision, recall, PR-AUC
    - parquet file of all results
"""

import ast
import multiprocessing as mp
import os
import time
from functools import partial
from itertools import combinations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as spstats
warnings.filterwarnings('ignore', 'Precision loss')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_LENGTH = 30
HOP = 10
N_COLS = 25          # use only first 25 columns (standard across most channels)
DATA_DIR = '/workspace/examples/datasets/smap_msl/data_raw/test/'
LABELS_FILE = '/workspace/examples/datasets/smap_msl/labeled_anomalies.csv'
OUTPUT_DIR = '/workspace/examples/timeseries_anomaly_linreg/'
N_WORKERS = min(mp.cpu_count() or 4, 8)

# Top-10 candidate features from combined Borda ranking
TOP10_FEATURE_TYPES = [
    'max_diff',     # rank 1
    'Q3',           # rank 2
    'std',          # rank 3
    'skew',         # rank 4
    'trend',        # rank 5
    'range',        # rank 6
    'median',       # rank 7
    'mean',         # rank 8
    'min',          # rank 9
    'kurt',         # rank 10
]

# Full 14-feature order — mirrors explore_enhanced.py (index → name mapping)
FEATURE_TYPE_NAMES = [
    'mean', 'std', 'median', 'min', 'max', 'range',
    'skew', 'kurt', 'Q1', 'Q3', 'trend', 'CV',
    'max_diff', 'mean_abs_diff',
]

N_FOLDS = 5
N_REPETITIONS = 1  # single pass for systematic comparison (use repeat later for final tuning)

# ---------------------------------------------------------------------------
# Fast vectorised feature extraction using stride tricks
# ---------------------------------------------------------------------------


def _extract_all_14_batch(windows):
    """Extract all 14 features for ALL windows in one call.

    Parameters
    ----------
    windows : ndarray of shape (n_windows, WINDOW_LENGTH, n_cols)

    Returns
    -------
    feats : ndarray of shape (n_windows, n_cols, 14) — same order as FEATURE_TYPE_NAMES
    """
    # Basic stats: all (n_windows, n_cols)
    means = np.mean(windows, axis=1)
    stds = np.std(windows, axis=1, ddof=0)
    medians = np.median(windows, axis=1)
    minis = np.min(windows, axis=1)
    maxis = np.max(windows, axis=1)
    ranges = maxis - minis

    # Shape stats via scipy (vectorised over all windows × cols)
    with np.errstate(invalid='ignore'):
        skews = spstats.skew(windows, axis=1)
        kurts = spstats.kurtosis(windows, axis=1)

    Q1 = np.percentile(windows, 25, axis=1)
    Q3 = np.percentile(windows, 75, axis=1)

    # Trend (slope of linear fit per window×col) — vectorised normal equations
    x = np.arange(WINDOW_LENGTH, dtype=np.float64)
    sx = x.sum(); sxx = (x ** 2).sum(); n = WINDOW_LENGTH
    denom_val = max(n * sxx - sx ** 2, 1e-12)
    slope_num = ((x[None, :, None] * windows).sum(axis=1)) - sx * means
    trends = slope_num / denom_val

    # CV (coefficient of variation)
    cvs = stds / (np.abs(means) + 1e-12)

    # Diff stats
    diffs = np.diff(windows, axis=1)                # (n_windows, W-1, n_cols)
    max_diffs = np.max(np.abs(diffs), axis=1)
    mean_abs_diffs = np.mean(np.abs(diffs), axis=1)

    # Stack into (n_windows, n_cols, 14) matching FEATURE_TYPE_NAMES order
    feats = np.stack([means, stds, medians, minis, maxis, ranges,
                      skews, kurts, Q1, Q3, trends, cvs,
                      max_diffs, mean_abs_diffs], axis=2)

    return feats


def _build_channel_windows(row):
    """Build windows for a single channel using stride tricks + batch feature extraction.

    Returns (X, y) where X shape is (n_hop_windows, N_COLS*14), y shape is (n_hop_windows,).
    If the channel has fewer than N_COLS columns, remaining columns are zero-padded.
    """
    chan = row['chan_id']
    anom_regions = ast.literal_eval(row['anomaly_sequences'])

    path = os.path.join(DATA_DIR, f'{chan}.npy')
    if not os.path.exists(path):
        return None, None

    data = np.load(path)
    n_ts, n_cols_actual = data.shape[:2]
    n_cols_use = min(n_cols_actual, N_COLS)

    # Build anomaly mask (label at window-end timestep w_start)
    is_anom = np.zeros(n_ts, dtype=bool)
    for region in anom_regions:
        is_anom[region[0]:region[1] + 1] = True

    win_n = n_ts - WINDOW_LENGTH + 1
    if win_n <= 0:
        return None, None

    # Create view of all windows using stride tricks — zero-copy!
    windows = np.lib.stride_tricks.as_strided(
        data[:win_n + WINDOW_LENGTH - 1],
        shape=(win_n, WINDOW_LENGTH, n_cols_use),
        strides=(data.strides[0], data.strides[0], data.strides[1]),
    )

    # Pad to N_COLS if needed (zero-fill). Zero-filled channels produce constant-zero columns
    # which cause NaN in skew/kurt/cv. We'll handle this by replacing zero-column features later.
    if n_cols_use < N_COLS:
        pad = np.zeros((win_n, WINDOW_LENGTH, N_COLS - n_cols_use), dtype=data.dtype)
        windows = np.concatenate([windows, pad], axis=2)

    # Batch extract all features — one scipy call for all windows at once!
    all_feats = _extract_all_14_batch(windows)  # (win_n, N_COLS, 14)

    # Stride index i → window data[i:i+WINDOW_LENGTH]. Original uses w_start where
    # window = data[w_start-WINDOW_LENGTH:w_start], label = is_anom[w_start].
    # So stride_idx = w_start - WINDOW_LENGTH. w_start runs: W, W+H, W+2H, ... while w_start+H <= n_ts.
    hop_indices = np.arange(0, n_ts - WINDOW_LENGTH, HOP)  # stride indices matching original logic
    if len(hop_indices) == 0:
        return None, None

    hop_feats = all_feats[hop_indices]            # (n_hop, N_COLS, 14)

    # Replace NaN/Inf from zero-padded channels and division-by-zero in CV
    hop_feats = np.where(np.isfinite(hop_feats), hop_feats, 0.0)

    hop_windows_end = WINDOW_LENGTH + hop_indices  # timestep positions for anomaly labels
    y = is_anom[hop_windows_end].astype(int)

    # Reshape: (n_hop_windows, N_COLS * 14) for ML
    X = hop_feats.reshape(hop_feats.shape[0], -1)
    return X, y


# ---------------------------------------------------------------------------
# Feature index helpers
# ---------------------------------------------------------------------------


def get_feature_indices(feature_types):
    """Map feature type names to their indices in the 14-feature vector."""
    return [FEATURE_TYPE_NAMES.index(ft) for ft in feature_types]


def build_feature_indices(feature_types):
    """Return the list of X column indices used by the selected feature types.

    For N_COLS columns and a feature type at position p in FEATURE_TYPE_NAMES:
      X index for channel c = c * 14 + p
    """
    target_indices = get_feature_indices(feature_types)
    all_indices = []
    for col in range(N_COLS):
        for p in target_indices:
            all_indices.append(col * 14 + p)
    return all_indices


# ---------------------------------------------------------------------------
# Evaluate a feature subset with stratified k-fold CV
# ---------------------------------------------------------------------------

def evaluate_feature_subset(X, y, feat_indices, feat_name):
    """Train logistic regression with repeated stratified k-fold CV."""
    X_sub = X[:, feat_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    all_aucs, all_f1s, all_precs, all_recalls, all_pr_aucs = [], [], [], [], []

    for _rep in range(N_REPETITIONS):
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, y)):
            clf = LogisticRegression(
                C=1.0, max_iter=2000, class_weight='balanced',
                solver='lbfgs', random_state=42 + _rep * 100 + fold_idx,
            )
            clf.fit(X_scaled[train_idx], y[train_idx])

            y_prob = clf.predict_proba(X_scaled[test_idx])[:, 1]
            y_pred = clf.predict(X_scaled[test_idx])

            try:
                auc = roc_auc_score(y[test_idx], y_prob)
            except ValueError:
                auc = np.nan

            all_aucs.append(auc)
            all_f1s.append(f1_score(y[test_idx], y_pred, zero_division=0))
            all_precs.append(precision_score(y[test_idx], y_pred, zero_division=0))
            all_recalls.append(recall_score(y[test_idx], y_pred, zero_division=0))
            all_pr_aucs.append(average_precision_score(y[test_idx], y_prob))

    return {
        'feature_set': feat_name,
        'n_features': len(feat_indices),
        'auc_mean': float(np.nanmean(all_aucs)),
        'auc_std': float(np.nanstd(all_aucs)),
        'f1_mean': float(np.nanmean(all_f1s)),
        'f1_std': float(np.nanstd(all_f1s)),
        'precision_mean': float(np.nanmean(all_precs)),
        'precision_std': float(np.nanstd(all_precs)),
        'recall_mean': float(np.nanmean(all_recalls)),
        'recall_std': float(np.nanstd(all_recalls)),
        'pr_auc_mean': float(np.nanmean(all_pr_aucs)),
        'pr_auc_std': float(np.nanstd(all_pr_aucs)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print('=' * 80)
    print('  LOGISTIC REGRESSION EVALUATION — SMAP Telemetry Anomaly Detection')
    print('=' * 80)
    print(f'Window length: {WINDOW_LENGTH}, Hop: {HOP}')
    print(f'Columns used : {N_COLS} (padded/truncated to fixed size)')
    print(f'Multiprocess workers: {N_WORKERS}')
    print(f'Stratified K-fold  : {N_FOLDS}, Repeats: {N_REPETITIONS}')
    print(f'Model              : LogisticRegression (balanced, C=1.0, lbfgs)')
    print()

    # Step 1: Build dataset using multiprocessing + stride tricks
    print('Step 1/3: Building per-window dataset (parallel, vectorised)...')
    df_labels = pd.read_csv(LABELS_FILE)

    all_X, all_y = [], []
    meta_list = []

    # Convert to list of dicts for picklable rows
    labels_list = df_labels.to_dict('records')

    with mp.Pool(N_WORKERS) as pool:
        results = list(tqdm(
            pool.imap(_build_channel_windows, labels_list),
            total=len(labels_list),
            desc='Channels processed'
        ))

    for X_ch, y_ch in results:
        if X_ch is None:
            continue
        all_X.append(X_ch)
        all_y.append(y_ch)
        n_anom = int(y_ch.sum())
        meta_list.append({'channel': df_labels.iloc[len(all_y)-1]['chan_id'],
                          'n_windows': len(y_ch), 'n_anom': n_anom})

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    elapsed = time.time() - t_start
    print(f'  Total windows : {len(y):,}')
    print(f'  Anomalous     : {int(y.sum()):,} ({y.mean()*100:.2f}%)')
    print(f'  Feature dims  : {X.shape[1]} (14 × {N_COLS})')
    print(f'  Channels      : {len(meta_list)}')
    print(f'  Elapsed       : {elapsed:.1f}s\n')

    # Step 2: Build index maps for each feature type
    feat_type_indices = {}
    for ft in TOP10_FEATURE_TYPES:
        feat_type_indices[ft] = build_feature_indices([ft])

    # Step 3a: Single features (C(10,1) = 10)
    print('=' * 80)
    print('  PART A — SINGLE FEATURE EVALUATION')
    print(f'         {N_COLS} channels × {N_COLS*14} raw dims → {N_COLS} feature dims each')
    print('=' * 80)

    results = []
    for ft in TOP10_FEATURE_TYPES:
        feat_idx = feat_type_indices[ft]
        r = evaluate_feature_subset(X, y, feat_idx, ft)
        results.append(r)
        elapsed = time.time() - t_start
        print(f'  [{elapsed:.1f}s] {ft:<14s} '
              f'AUC={r["auc_mean"]:.4f} (±{r["auc_std"]:.4f})  '
              f'F1={r["f1_mean"]:.4f}  P={r["precision_mean"]:.4f}  R={r["recall_mean"]:.4f}  '
              f'PR-AUC={r["pr_auc_mean"]:.4f}')

    # Step 3b: Pairwise (C(10,2) = 45)
    print()
    print('=' * 80)
    print('  PART B — PAIRWISE FEATURE COMBINATIONS')
    print(f'         {N_COLS*2} feature dims each')
    print('=' * 80)

    for combo in combinations(TOP10_FEATURE_TYPES, 2):
        feat_name = f'{combo[0]}+{combo[1]}'
        feat_indices = sorted(set(
            feat_type_indices[combo[0]] + feat_type_indices[combo[1]]
        ))
        r = evaluate_feature_subset(X, y, feat_indices, feat_name)
        results.append(r)
        elapsed = time.time() - t_start
        print(f'  [{elapsed:.1f}s] {feat_name:<26s} '
              f'AUC={r["auc_mean"]:.4f} (±{r["auc_std"]:.4f})  '
              f'F1={r["f1_mean"]:.4f}')

    # Step 3c: Triples — top-6 by single AUC (C(6,3) = 20)
    print()
    print('=' * 80)
    print('  PART C — TRIPLE FEATURE COMBINATIONS')
    print(f'         {N_COLS*3} feature dims each')
    print('=' * 80)

    top6 = sorted(results[:len(TOP10_FEATURE_TYPES)],
                  key=lambda x: x['auc_mean'], reverse=True)[:6]
    top6_names = [r['feature_set'] for r in top6]
    print(f'  Using top-6 singles by AUC:')
    for rank, name in enumerate(top6_names, 1):
        r_single = results[[i for i, r in enumerate(results) if r['feature_set'] == name][0]]
        print(f'    {rank}. {name}  AUC={r_single["auc_mean"]:.4f}')

    for combo in combinations(top6_names, 3):
        feat_name = '+'.join(combo)
        feat_indices = sorted(set(
            feat_type_indices[combo[0]] + feat_type_indices[combo[1]]
            + feat_type_indices[combo[2]]
        ))
        r = evaluate_feature_subset(X, y, feat_indices, feat_name)
        results.append(r)
        elapsed = time.time() - t_start
        print(f'  [{elapsed:.1f}s] {feat_name:<40s} '
              f'AUC={r["auc_mean"]:.4f} (±{r["auc_std"]:.4f})  '
              f'F1={r["f1_mean"]:.4f}')

    elapsed = time.time() - t_start
    print(f'\nTotal feature sets evaluated: {len(results)} in {elapsed:.1f}s')

    # =========================================================================
    # REPORT — ranked by AUC
    # =========================================================================

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('auc_mean', ascending=False).reset_index(drop=True)
    df_results['final_rank'] = range(1, len(df_results) + 1)

    print()
    print('=' * 80)
    print('  SUMMARY — ALL FEATURE SETS RANKED BY ROC-AUC')
    print('=' * 80)

    # Separate into single/pair/triple groups for clean display
    for group_label, filter_fn in [
        ('SINGLE FEATURES',       lambda r: len(r['feature_set']) <= 6),
        ('PAIRWISE COMBINATIONS', lambda r: '+' in r['feature_set'] and r['feature_set'].count('+') == 1),
        ('TRIPLE COMBINATIONS',   lambda r: r['feature_set'].count('+') >= 2),
    ]:
        mask = df_results.apply(filter_fn, axis=1)
        subset = df_results[mask].sort_values('auc_mean', ascending=False)
        if len(subset) == 0:
            continue
        print(f'\n  --- {group_label} (N={len(subset)}) ---')
        hdr = f'{"Rank":<5} {"Feature Set":<40s} {"AUC":>7} {"±Std":>6} {"F1":>6} {"P":>6} {"R":>6} {"PR-AUC":>7}'
        print(f'  {hdr}')
        print('  ' + '-' * (len(hdr) - 2))
        for i, (_, row) in enumerate(subset.iterrows(), 1):
            print(f'  {row["final_rank"]:<5} {row["feature_set"]:<40s} '
                  f'{row["auc_mean"]:>6.4f} ±{row["auc_std"]:>5.4f}  '
                  f'{row["f1_mean"]:>6.4f}  '
                  f'{row["precision_mean"]:>6.4f}  '
                  f'{row["recall_mean"]:>6.4f}  '
                  f'{row["pr_auc_mean"]:>6.4f}')

    # Best overall
    best = df_results.iloc[0]
    print()
    print('=' * 80)
    print('  🏆 OVERALL BEST')
    print('=' * 80)
    print(f'  Feature set : {best["feature_set"]}')
    print(f'  N features  : {int(best["n_features"])} ({N_COLS} cols × {int(best["n_features"]/N_COLS)} feat types)')
    print(f'  ROC-AUC     : {best["auc_mean"]:.4f} ± {best["auc_std"]:.4f}')
    print(f'  F1          : {best["f1_mean"]:.4f} ± {best["f1_std"]:.4f}')
    print(f'  Precision   : {best["precision_mean"]:.4f} ± {best["precision_std"]:.4f}')
    print(f'  Recall      : {best["recall_mean"]:.4f} ± {best["recall_std"]:.4f}')
    print(f'  PR-AUC      : {best["pr_auc_mean"]:.4f} ± {best["pr_auc_std"]:.4f}')
    print()

    # =========================================================================
    # Save results
    # =========================================================================

    out_path = os.path.join(OUTPUT_DIR, 'logistic_regression_results.parquet')
    df_results.to_parquet(out_path, engine='pyarrow', index=False)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
