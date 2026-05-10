"""Enhanced exploratory data analysis of NASA SMAP telemetry.

Computes five metrics comparing anomalous vs normal sliding windows across
all 82 channels, each capturing a different aspect of distributional
difference:

  Cohen's d        — linear location shift
  KS statistic     — maximum CDF gap (any shape difference)
  KL divergence    — information-theoretic shape difference
  Energy ratio     — variance / signal-power ratio
  MI               — mutual information (captures ANY dependency, including
                     non-linear, multimodal, deterministic)

All per-type metric computations are parallelised with joblib, driven by a
tqdm progress hook so you always see percentage + time remaining.

Outputs:
  - Console report with all five metric rankings plus a combined Borda rank
  - Per-channel parquet (window_features.parquet)
  - Combined metrics parquet (combined_metrics.parquet)
"""

import ast
import os
import time

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WINDOW_LENGTH = 30
HOP = 10
DATA_DIR = '/workspace/examples/datasets/smap_msl/data_raw/test/'
LABELS_FILE = '/workspace/examples/datasets/smap_msl/labeled_anomalies.csv'
OUTPUT_DIR = '/workspace/examples/timeseries_anomaly_linreg/'
N_JOBS = -1  # use all available CPU cores

# ---------------------------------------------------------------------------
# Feature extraction — 14 features per 1-D window
# ---------------------------------------------------------------------------


def make_feature_names(n_cols):
    """Return 14 * n_cols feature names, e.g. ['0_mean', '0_std', ..., 24_max']."""
    names = []
    for col_idx in range(n_cols):
        names.extend([
            f'{col_idx}_mean',     f'{col_idx}_std',
            f'{col_idx}_median',   f'{col_idx}_min',
            f'{col_idx}_max',      f'{col_idx}_range',
            f'{col_idx}_skew',     f'{col_idx}_kurt',
            f'{col_idx}_Q1',       f'{col_idx}_Q3',
            f'{col_idx}_trend',    f'{col_idx}_CV',
            f'{col_idx}_max_diff', f'{col_idx}_mean_abs_diff',
        ])
    return names


def _extract_col_features(col_data):
    """Extract 14 features from a single 1-D column (shape: WINDOW_LENGTH)."""
    col = np.asarray(col_data, dtype=np.float64)
    n = len(col)

    mean_v = float(np.mean(col))
    std_v = float(np.std(col))
    median_v = float(np.median(col))
    min_v = float(np.min(col))
    max_v = float(np.max(col))
    range_v = max_v - min_v

    with np.errstate(invalid='ignore', divide='ignore'):
        if std_v > 1e-12:
            skew_v = float(stats.skew(col))
            kurt_v = float(stats.kurtosis(col))
        else:
            skew_v = 0.0
            kurt_v = 0.0

    q1 = float(np.percentile(col, 25))
    q3 = float(np.percentile(col, 75))

    # Linear trend via least-squares slope
    x = np.arange(n, dtype=np.float64)
    sx = x.sum()
    sxx = (x ** 2).sum()
    sxy = (x * col).sum()
    denom = n * sxx - sx ** 2
    trend_v = float(sxy * n - sx * (col.sum())) / max(denom, 1e-12) if abs(denom) > 1e-12 else 0.0

    cv_v = std_v / abs(mean_v) if abs(mean_v) > 1e-12 else 0.0

    diffs = np.diff(col)
    max_diff_v = float(np.max(np.abs(diffs)))
    mean_abs_diff_v = float(np.mean(np.abs(diffs)))

    return np.array([
        mean_v, std_v, median_v, min_v, max_v, range_v,
        skew_v, kurt_v, q1, q3, trend_v, cv_v,
        max_diff_v, mean_abs_diff_v,
    ])


def extract_features(win):
    """Extract time-series window statistics for all columns.

    Parameters
    ----------
    win : ndarray of shape (WINDOW_LENGTH, n_cols)

    Returns
    -------
    feats : ndarray — flattened feature vector (14 * n_cols,)
    """
    feats = []
    n_cols = win.shape[1]
    for c in range(n_cols):
        feats.extend(_extract_col_features(win[:, c]))
    return np.array(feats)


# ---------------------------------------------------------------------------
# Per-channel processing (embarrassingly parallel)
# ---------------------------------------------------------------------------

def process_single_channel(row):
    """Load one channel's data, build anomaly mask, slide windows, and
    accumulate sums for anomalous / normal groups.  Returns a dict that
    is later joined into ``channel_summaries``."""
    chan = row['chan_id']
    anomaly_seq_str = row['anomaly_sequences']
    anom_regions = ast.literal_eval(anomaly_seq_str)

    path = os.path.join(DATA_DIR, f'{chan}.npy')
    if not os.path.exists(path):
        return None

    data = np.load(path)
    n_ts, n_cols = data.shape
    feat_names = make_feature_names(n_cols)
    n_feats = len(feat_names)

    # Build anomaly mask
    is_anom = np.zeros(n_ts, dtype=bool)
    for region in anom_regions:
        is_anom[region[0]:region[1] + 1] = True

    n_anom_steps = int(is_anom.sum())

    feat_sums_a = np.zeros(n_feats)
    feat_sqsums_a = np.zeros(n_feats)
    counts_a = 0

    feat_sums_n = np.zeros(n_feats)
    feat_sqsums_n = np.zeros(n_feats)
    counts_n = 0

    w_start = WINDOW_LENGTH
    while w_start + HOP <= n_ts:
        feats_arr = extract_features(data[w_start - WINDOW_LENGTH:w_start])

        if is_anom[w_start]:
            feat_sums_a += feats_arr
            feat_sqsums_a += feats_arr ** 2
            counts_a += 1
        else:
            feat_sums_n += feats_arr
            feat_sqsums_n += feats_arr ** 2
            counts_n += 1

        w_start += HOP

    anom_means = feat_sums_a / max(counts_a, 1)
    norm_means = feat_sums_n / max(counts_n, 1)

    return {
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
    }


# ---------------------------------------------------------------------------
# Distributional metrics (non-linear / shape-aware)
# ---------------------------------------------------------------------------

def compute_kl_divergence(anom_vals, norm_vals, n_bins=40):
    """Histogram-based KL(P_anom || P_norm) with Laplace smoothing."""
    all_vals = np.concatenate([anom_vals, norm_vals])
    vmin, vmax = all_vals.min(), all_vals.max()
    if vmax - vmin < 1e-12:
        return 0.0

    counts_a, _ = np.histogram(anom_vals, bins=n_bins, range=(vmin, vmax))
    counts_n, _ = np.histogram(norm_vals, bins=n_bins, range=(vmin, vmax))

    p_a = (counts_a + 1.0) / (counts_a.sum() + n_bins)
    p_n = (counts_n + 1.0) / (counts_n.sum() + n_bins)
    return float(np.sum(p_a * np.log(p_a / p_n)))


def compute_ks_single(a_arr, n_arr):
    """Two-sample KS statistic."""
    result = stats.ks_2samp(a_arr, n_arr)
    return float(result.statistic)


def compute_energy_ratio_per_channel(anom_sqsum, anom_count, norm_sqsum, norm_count):
    """Log-ratio of per-window mean-squared values (symmetric magnitude)."""
    mean_sq_a = anom_sqsum / max(anom_count, 1)
    mean_sq_n = norm_sqsum / max(norm_count, 1)
    if mean_sq_n < 1e-12:
        return 0.0
    return abs(np.log(mean_sq_a / mean_sq_n))


# ---------------------------------------------------------------------------
# Mutual Information (MI) — captures ANY dependency via nearest-neighbours
# ---------------------------------------------------------------------------

def estimate_mi_kraskov_1d(anom_vals, norm_vals, k=5):
    """Estimate MI between group label (binary) and feature value using the
    Kraskov-Stogbauer-Grassberger (KSG) nearest-neighbour estimator.

    MI = H(X) + H(Y) - H(X,Y),  where X is group membership, Y is the
    continuous feature.  This captures any dependency — linear, non-linear,
    multimodal, deterministic — not just mean shifts.

    Uses the Kraskov 1st estimator:
      MI ≈ (1/n) Σ_i [ ψ(n_in_ball_i + 1) - ψ(k + 1)
                        + ψ(n_same_label_i + 1) - ψ(n_diff_label_i + 1) ]
           + log(n_n) - log(n_a)
    """
    combined_arr = np.concatenate([anom_vals, norm_vals])
    n_a = len(anom_vals)
    n_n = len(norm_vals)
    n_total = n_a + n_n

    if n_total < k + 2:
        return 0.0

    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    nn_model.fit(combined_arr.reshape(-1, 1))

    dists, _ = nn_model.kneighbors(combined_arr.reshape(-1, 1))
    knn_dists = dists[:, -1] + 1e-12

    # Binary labels: 0 for anomalous, 1 for normal
    labels = np.concatenate([np.zeros(n_a), np.ones(n_n)])
    indices = np.arange(n_total)

    mi_sum = 0.0
    for i in range(n_total):
        r_i = knn_dists[i]
        # Count total points in ball (excluding self)
        n_in_ball = int(np.sum(
            (np.abs(combined_arr - combined_arr[i]) <= r_i) & (indices != i)
        ))
        # Count how many have the SAME label as point i
        same_label_count = int(np.sum(
            (labels == labels[i]) & (indices != i)
            & (np.abs(combined_arr - combined_arr[i]) <= r_i)
        ))
        # Diff label count
        diff_label_count = n_in_ball - same_label_count

        # Digamma approx: ψ(x+1) ≈ log(x + 0.5) for large x; exact for small
        psi_same = _digamma(same_label_count + 1)
        psi_diff = _digamma(max(diff_label_count, 1)) if diff_label_count > 0 else 0.0

        mi_sum += _digamma(n_in_ball + 1) - _digamma(k + 1)
        mi_sum += psi_same - psi_diff

    # Correct for binary variable marginal
    p_a = n_a / n_total
    p_n = n_n / n_total
    h_group = -(p_a * np.log(p_a + 1e-12) + p_n * np.log(p_n + 1e-12))

    avg_mi = mi_sum / n_total + h_group
    return float(max(avg_mi, 0.0))


def _digamma(x):
    """Accurate digamma ψ(x) for x > 0."""
    if x <= 0:
        return 0.0
    # Use recurrence to bring into range [1, 2], then polynomial approximation
    result = -0.5772156649015329  # Euler-Mascheroni constant, ψ(1)
    while x < 7:
        if x > 1:
            result += 1 / (x - 1)
        x += 1
    # Asymptotic expansion for x >= 7
    inv = 1.0 / x
    return np.log(x - 0.5) + inv * (1.0 / 12.0 - inv * inv * (1.0 / 360.0))


# ---------------------------------------------------------------------------
# Parallel metric computation per feature type
# ---------------------------------------------------------------------------

def _compute_metrics_for_type(ftype, channel_summaries):
    """One task: accumulate per-channel arrays for feature type FTYPE, then
    compute all five metrics.  This is the unit that gets parallelised."""
    anom_vals = []
    norm_vals = []
    pvar_sum = 0.0
    df_total = 0

    for cs in channel_summaries:
        feat_names = cs['feat_names']
        anom_m = cs['anom_means']
        norm_m = cs['norm_means']
        sq_a = cs['feat_sqsums_a']
        sq_n = cs['feat_sqsums_n']
        na = cs['n_anom_windows']
        nn = cs['n_norm_windows']

        for j, fname in enumerate(feat_names):
            if fname.endswith('_%s' % ftype):
                anom_vals.append(anom_m[j])
                norm_vals.append(norm_m[j])

                var_a = 0.0
                if na > 1:
                    var_a = max(sq_a[j] / na - anom_m[j] ** 2, 0.0)
                var_n = 0.0
                if nn > 1:
                    var_n = max(sq_n[j] / nn - norm_m[j] ** 2, 0.0)

                pvar_sum += (na - 1) * var_a + (nn - 1) * var_n
                df_total += (na - 1) + (nn - 1)

    if not anom_vals:
        return None

    a_arr = np.array(anom_vals, dtype=np.float64)
    n_arr = np.array(norm_vals, dtype=np.float64)

    # Cohen's d
    anom_agg = float(np.mean(a_arr))
    norm_agg = float(np.mean(n_arr))
    pooled_std = np.sqrt(pvar_sum / max(df_total, 1))
    d = (anom_agg - norm_agg) / pooled_std if pooled_std > 1e-12 else 0.0

    # KS statistic
    ks_stat = compute_ks_single(a_arr, n_arr)

    # KL divergence
    kl_div = compute_kl_divergence(a_arr, n_arr, n_bins=40)

    # Energy ratio (log-ratio of per-window squared means)
    a_sq = a_arr ** 2
    n_sq = n_arr ** 2
    valid_idx = np.where((n_sq > 1e-12) & (a_sq > 1e-12))[0]
    energy_score = abs(float(np.mean(np.log(a_sq[valid_idx] / n_sq[valid_idx])))) \
        if len(valid_idx) > 0 else 0.0

    # Mutual information (KSG nearest-neighbour estimator)
    mi_val = estimate_mi_kraskov_1d(a_arr, n_arr, k=5)

    return {
        'feature_type': ftype,
        'cohens_d': abs(d),
        'd_signed': d,
        'ks_stat': ks_stat,
        'kl_divergence': kl_div,
        'energy_ratio_score': energy_score,
        'mi_value': mi_val,
        'anom_mean': anom_agg,
        'norm_mean': norm_agg,
        'weighted_diff': anom_agg - norm_agg,
    }


def _wrapper(ftype_cs):
    """Module-level wrapper for multiprocessing.Pool.imap (metrics) — picklable."""
    ftype, cs_data = ftype_cs
    return _compute_metrics_for_type(ftype, cs_data)


def _channel_wrapper(row):
    """Module-level wrapper for multiprocessing.Pool.imap (channels) — picklable."""
    return process_single_channel(row)


def compute_all_parallel(feat_types, channel_summaries):
    """Single-level multiprocessing.Pool.imap across feature types.

    Each task independently loops over all channels for one feature type,
    accumulating arrays then computing all 5 metrics.  Results stream back
    through Pool.imap so tqdm shows live percentage + ETA.
    """
    import multiprocessing as mp

    print('Computing metrics (all 5 per feature type, parallel)...')

    n_workers = max(1, os.cpu_count() or 4)

    with mp.Pool(n_workers) as pool:
        results = []
        task_pairs = [(ft, channel_summaries) for ft in feat_types]
        for r in tqdm(pool.imap(_wrapper, task_pairs), total=len(feat_types), desc='All metrics'):
            if r is not None:
                results.append(r)

    return results


# ---------------------------------------------------------------------------
# Combined ranking (Borda score across all 5 metrics)
# ---------------------------------------------------------------------------

def build_combined_ranking(metric_results):
    """Build a combined Borda-style ranking from the per-metric ranked lists."""
    # Build rank dicts for each metric (lower = better → score closer to 0)
    ft_set = {r['feature_type'] for r in metric_results}

    def sort_key(results, metric_name):
        return sorted(results, key=lambda x: x[metric_name], reverse=True)

    ranked_d = sort_key(metric_results, 'cohens_d')
    ranked_ks = sort_key(metric_results, 'ks_stat')
    ranked_kl = sort_key(metric_results, 'kl_divergence')
    ranked_e = sort_key(metric_results, 'energy_ratio_score')
    ranked_mi = sort_key(metric_results, 'mi_value')

    rank_d = {r['feature_type']: float(i) for i, r in enumerate(ranked_d)}
    rank_ks = {r['feature_type']: float(i) for i, r in enumerate(ranked_ks)}
    rank_kl = {r['feature_type']: float(i) for i, r in enumerate(ranked_kl)}
    rank_e = {r['feature_type']: float(i) for i, r in enumerate(ranked_e)}
    rank_mi = {r['feature_type']: float(i) for i, r in enumerate(ranked_mi)}

    combined = []
    for ft in sorted(ft_set):
        norm_rank = (
            rank_d.get(ft, 1) + rank_ks.get(ft, 1) + rank_kl.get(ft, 1) +
            rank_e.get(ft, 1) + rank_mi.get(ft, 1)
        ) / 5.0

        combined.append({
            'feature_type': ft,
            'cohens_d_rank': rank_d.get(ft, '-'),
            'cohens_d_value': next((r['cohens_d'] for r in metric_results if r['feature_type'] == ft), 0),
            'ks_stat_rank': rank_ks.get(ft, '-'),
            'ks_stat_value': next((r['ks_stat'] for r in metric_results if r['feature_type'] == ft), 0),
            'kl_div_rank': rank_kl.get(ft, '-'),
            'kl_div_value': next((r['kl_divergence'] for r in metric_results if r['feature_type'] == ft), 0),
            'energy_rank': rank_e.get(ft, '-'),
            'energy_value': next((r['energy_ratio_score'] for r in metric_results if r['feature_type'] == ft), 0),
            'mi_rank': rank_mi.get(ft, '-'),
            'mi_value': next((r['mi_value'] for r in metric_results if r['feature_type'] == ft), 0),
            'combined_norm_rank': norm_rank,
        })

    combined.sort(key=lambda x: x['combined_norm_rank'])
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()

    print('=== Enhanced Exploratory Analysis: SMAP Telemetry ===')
    print(f'Window length = {WINDOW_LENGTH}, Hop = {HOP}')
    cpu_count = os.cpu_count() or 4
    print(f'Parallel workers (multiprocessing): {cpu_count} (all cores)\n')

    # Load anomaly labels
    df_labels = pd.read_csv(LABELS_FILE)
    n_channels = len(df_labels)
    print(f'Channels with anomaly labels: {n_channels}\n')

    feat_types = [
        'mean', 'std', 'median', 'min', 'max', 'range',
        'skew', 'kurt', 'Q1', 'Q3', 'trend', 'CV',
        'max_diff', 'mean_abs_diff',
    ]

    # Step 1 — process channels in parallel (embarrassingly independent).
    # Use multiprocessing.Pool.imap for streaming results into tqdm progress bar.
    import multiprocessing as mp

    print('Step 1/{}: processing channels (parallel)...'.format(len(feat_types) + 2))

    channel_count = len(df_labels)
    rows_list = [row for _, row in df_labels.iterrows()]
    n_workers = max(1, os.cpu_count() or 4)

    with mp.Pool(n_workers) as pool:
        channel_summaries = [r for r in tqdm(pool.imap(_channel_wrapper, rows_list),
                                             total=channel_count, desc='Channels loaded') if r is not None]

    total_channels = len(channel_summaries)
    print('Channels loaded: {}\n'.format(total_channels))

    # Step 2 — aggregate per-channel stats for reporting
    total_timesteps = sum(cs['n_timesteps'] for cs in channel_summaries)
    total_anom_steps = sum(cs['n_anom_steps'] for cs in channel_summaries)
    total_windows = sum(cs['n_total_windows'] for cs in channel_summaries)
    elapsed = time.time() - t_start

    print(f'Total windows analyzed: {total_windows:,} ({elapsed:.1f}s)\n')

    # Step 3 — compute all 5 metrics in parallel across feature types.
    # Use multiprocessing.Pool.imap for streaming progress into tqdm.
    print('Step {}/{}: computing metrics (parallel)...'.format(3, len(feat_types) + 2))
    metric_results = compute_all_parallel(feat_types, channel_summaries)

    elapsed = time.time() - t_start

    # Sort each metric for reporting
    metric_results.sort(key=lambda x: x['cohens_d'], reverse=True)

    # Step 4 — combined ranking
    combined = build_combined_ranking(metric_results)

    # =========================================================================
    # REPORT
    # =========================================================================

    # --- Cohen's d ---
    d_rows = sorted(metric_results, key=lambda x: x['cohens_d'], reverse=True)
    print('=' * 80)
    print('  1. COHEN\'S D — Linear location shift (baseline)')
    print('=' * 80)
    print(f'{"Rank":<5} {"Feature":<16} {"|d|":>7}   {"Anom":>12} {"Norm":>12} {"Diff":>12}')
    print('-' * 70)
    for _i, r in enumerate(d_rows[:10], 1):
        sign = '+' if r['d_signed'] >= 0 else '-'
        d_str = f'{sign}{abs(r["weighted_diff"]):.4f}'
        print(f'{_i:<5} {r["feature_type"]:<16} {r["cohens_d"]:>7.3f}   '
              f'{r["anom_mean"]:>12.4f} {r["norm_mean"]:>12.4f} {d_str}')
    print()

    # --- KS Statistic ---
    ks_rows = sorted(metric_results, key=lambda x: x['ks_stat'], reverse=True)
    print('=' * 80)
    print('  2. KS STATISTIC — Maximum CDF gap (any distributional difference)')
    print('=' * 80)
    print(f'{"Rank":<5} {"Feature":<16} {"KS":>7}   {"Anom":>12} {"Norm":>12} {"Diff":>12}')
    print('-' * 70)
    for _i, r in enumerate(ks_rows[:10], 1):
        sign = '+' if r['weighted_diff'] >= 0 else '-'
        d_str = f'{sign}{abs(r["weighted_diff"]):.4f}'
        print(f'{_i:<5} {r["feature_type"]:<16} {r["ks_stat"]:>7.4f}   '
              f'{r["anom_mean"]:>12.4f} {r["norm_mean"]:>12.4f} {d_str}')
    print()

    # --- KL Divergence ---
    kl_rows = sorted(metric_results, key=lambda x: x['kl_divergence'], reverse=True)
    print('=' * 80)
    print('  3. KL DIVERGENCE — Distribution shape (non-linear, information-theoretic)')
    print('=' * 80)
    print(f'{"Rank":<5} {"Feature":<16} {"KL":>12}   {"Anom":>12} {"Norm":>12} {"Diff":>12}')
    print('-' * 70)
    for _i, r in enumerate(kl_rows[:10], 1):
        sign = '+' if r['weighted_diff'] >= 0 else '-'
        d_str = f'{sign}{abs(r["weighted_diff"]):.4f}'
        print(f'{_i:<5} {r["feature_type"]:<16} {r["kl_divergence"]:>12.6f}   '
              f'{r["anom_mean"]:>12.4f} {r["norm_mean"]:>12.4f} {d_str}')
    print()

    # --- Energy Ratio ---
    e_rows = sorted(metric_results, key=lambda x: x['energy_ratio_score'], reverse=True)
    print('=' * 80)
    print('  4. ENERGY RATIO — Variance / signal-power difference')
    print('=' * 80)
    print(f'{"Rank":<5} {"Feature":<16} {"|log(r)|":>10}')
    print('-' * 40)
    for _i, r in enumerate(e_rows[:10], 1):
        print(f'{_i:<5} {r["feature_type"]:<16} {r["energy_ratio_score"]:>10.4f}')
    print()

    # --- MI — Mutual Information ---
    mi_rows = sorted(metric_results, key=lambda x: x['mi_value'], reverse=True)
    print('=' * 80)
    print('  5. MUTUAL INFORMATION (KSG estimator) — Any non-linear dependency')
    print('=' * 80)
    print(f'{"Rank":<5} {"Feature":<16} {"MI":>12}   {"Anom":>12} {"Norm":>12}')
    print('-' * 60)
    for _i, r in enumerate(mi_rows[:10], 1):
        print(f'{_i:<5} {r["feature_type"]:<16} {r["mi_value"]:>12.6f}   '
              f'{r["anom_mean"]:>12.4f} {r["norm_mean"]:>12.4f}')
    print()

    # --- Combined Borda Ranking ---
    print('=' * 80)
    print('  6. COMBINED RANKING — Normalised rank-aggregate (all 5 metrics)')
    print('     Lower combined score = consistently high across ALL methods')
    print('=' * 80)
    hdr = f'{"Rank":<5} {"Feature":<16} {"d-rk":>4} {"|d|":>7}   '
    hdr += f'{"ks-rk":>4} {"KS":>7}   '
    hdr += f'{"kl-rk":>4} {"KL":>10}   '
    hdr += f'{"e-rk":>4} {"Energy":>9}   '
    hdr += f'{"mi-rk":>4} {"MI":>10}   '
    hdr += f'{"Combined":>9}'
    print(hdr)
    print('-' * len(hdr))
    for i, r in enumerate(combined[:15], 1):
        def fmt_r(v):
            return int(round(v)) if isinstance(v, (int, float)) else v

        d_r = f"{fmt_r(r['cohens_d_rank'])}"
        ks_r = f"{fmt_r(r['ks_stat_rank'])}"
        kl_r = f"{fmt_r(r['kl_div_rank'])}"
        e_r = f"{fmt_r(r['energy_rank'])}"
        mi_r = f"{fmt_r(r['mi_rank'])}"
        print(f'{i:<5} {r["feature_type"]:<16} '
              f'{d_r:>5} {r["cohens_d_value"]:>7.3f}   '
              f'{ks_r:>5} {r["ks_stat_value"]:>7.4f}   '
              f'{kl_r:>5} {r["kl_div_value"]:>10.6f}   '
              f'{e_r:>5} {r["energy_value"]:>9.4f}   '
              f'{mi_r:>5} {r["mi_value"]:>10.6f}   '
              f'{r["combined_norm_rank"]:>9.4f}')
    print()

    # =========================================================================
    # PARQUET OUTPUT
    # =========================================================================

    out_path_features = os.path.join(OUTPUT_DIR, 'window_features.parquet')
    out_path_analysis = os.path.join(OUTPUT_DIR, 'feature_analysis.parquet')
    out_path_combined = os.path.join(OUTPUT_DIR, 'combined_metrics.parquet')

    df_analysis = pd.DataFrame(metric_results)
    df_analysis['rank'] = range(1, len(df_analysis) + 1)
    df_analysis.to_parquet(out_path_analysis, engine='pyarrow', index=False)

    combined_df = pd.DataFrame(combined)
    combined_df['final_rank'] = range(1, len(combined_df) + 1)
    combined_df.to_parquet(out_path_combined, engine='pyarrow', index=False)

    # Per-channel feature parquet (same as original explore.py)
    all_rows = []
    for cs in channel_summaries:
        row = {
            'channel': cs['channel'],
            'n_timesteps': cs['n_timesteps'],
            'n_cols': cs['n_cols'],
            'n_anom_steps': cs['n_anom_steps'],
            'pct_anom_steps': cs['pct_anom_steps'],
            'n_anom_windows': cs['n_anom_windows'],
            'n_norm_windows': cs['n_norm_windows'],
        }
        for j, fname in enumerate(cs['feat_names']):
            row[fname] = float(cs['anom_means'][j])
        all_rows.append(row)

    pd.DataFrame(all_rows).to_parquet(out_path_features, engine='pyarrow', index=False)

    print(f'Saved: {out_path_features}')
    print(f'Saved: {out_path_analysis}')
    print(f'Saved: {out_path_combined}\n')

    # =========================================================================
    # SUMMARY STATISTICS
    # =========================================================================

    pct_total = total_anom_steps / max(total_timesteps, 1) * 100
    elapsed = time.time() - t_start

    print('=' * 80)
    print('  DATASET OVERVIEW')
    print('=' * 80)
    print(f'Total channels processed : {total_channels}')
    print(f'Total timesteps          : {total_timesteps:,}')
    print(f'Total anomalous steps    : {total_anom_steps:,} ({pct_total:.2f}%)')
    print(f'Total windows (HOP={HOP})  : {total_windows:,}')
    print(f'Processing time          : {elapsed:.1f}s\n')


if __name__ == '__main__':
    main()
