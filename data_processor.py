"""
Synthetic data generator v2 - improved for all 4 scoring criteria:
  1. Structure     - exact dtype/range/category preservation
  2. Privacy       - no row duplicates; calibrated noise per column type
  3. Discrimination - Gaussian copula preserves multivariate correlations
  4. Distribution  - empirical marginals for categoricals; copula for numerics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, LabelEncoder

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ─── column classification ─────────────────────────────────────────────────

def is_likely_numeric_str(series):
    """True if object col is mostly numeric values (possibly with 'Unknown')."""
    sample = series.dropna().head(300)
    converted = pd.to_numeric(sample.replace('Unknown', np.nan), errors='coerce')
    return converted.notna().mean() > 0.6

def classify_columns(df):
    date_cols     = [c for c in df.columns if '(YYYY-MM-DD)' in c]
    time_cols     = [c for c in df.columns if '(HH:mm)' in c]
    obj_cols      = list(df.select_dtypes(include=['object', 'str']).columns)
    mixed_num_cols = [c for c in obj_cols
                      if c not in date_cols and c not in time_cols
                      and is_likely_numeric_str(df[c])]
    true_cat_cols  = [c for c in obj_cols
                      if c not in date_cols and c not in time_cols
                      and c not in mixed_num_cols]
    raw_num_cols   = list(df.select_dtypes(include=['int64', 'float64']).columns)
    return date_cols, time_cols, mixed_num_cols, true_cat_cols, raw_num_cols

# ─── missingness analysis ──────────────────────────────────────────────────

def compute_miss_rates(df, raw_num_cols, mixed_num_cols, date_cols, time_cols, true_cat_cols):
    """Return dict: col -> (nan_rate, unknown_rate) for mixed; col -> nan_rate otherwise."""
    rates = {}
    for c in raw_num_cols + date_cols + time_cols:
        rates[c] = {'nan': df[c].isna().mean(), 'unknown': 0.0}
    for c in mixed_num_cols:
        rates[c] = {
            'nan':     df[c].isna().mean(),
            'unknown': (df[c] == 'Unknown').mean(),
        }
    for c in true_cat_cols:
        blank_rate = df[c].isna().mean() + (df[c] == '<BLANK>').mean()
        rates[c] = {'nan': blank_rate, 'unknown': 0.0}
    return rates

# ─── encode numeric columns to float matrix ────────────────────────────────

def encode_numerics(df, date_cols, time_cols, mixed_num_cols, raw_num_cols):
    """Return float DataFrame of all numeric-ish columns, with NaN filled by median.
    Very sparse columns (< 15% non-null) are flagged separately for empirical sampling."""
    work = df.copy()
    ref_date = pd.Timestamp("1970-01-01")

    for c in date_cols:
        work[c] = (pd.to_datetime(work[c], errors='coerce') - ref_date) / pd.Timedelta(days=1)

    for c in time_cols:
        t = pd.to_datetime(work[c], format='%H:%M', errors='coerce')
        work[c] = t.dt.hour * 60 + t.dt.minute

    for c in mixed_num_cols:
        work[c] = pd.to_numeric(work[c].replace('Unknown', np.nan), errors='coerce')

    all_num = raw_num_cols + mixed_num_cols + date_cols + time_cols
    num_df = work[all_num].copy().astype(float)

    # Identify sparse cols to exclude from copula (use empirical sampling instead)
    sparse_cols = [c for c in all_num if num_df[c].notna().mean() < 0.15]
    dense_cols  = [c for c in all_num if c not in sparse_cols]

    # For copula: fill NaN with column median
    dense_df = num_df[dense_cols].copy()
    for c in dense_cols:
        med = dense_df[c].median()
        if pd.isna(med): med = 0.0
        dense_df[c] = dense_df[c].fillna(med)

    return dense_df, dense_cols, sparse_cols, num_df

# ─── Gaussian copula ───────────────────────────────────────────────────────

def fit_and_sample_copula(num_df: pd.DataFrame, n_samples: int) -> pd.DataFrame:
    """Fit Gaussian copula and return n_samples synthetic rows."""
    matrix = num_df.values.astype(float)
    n, p = matrix.shape

    # Per-column QuantileTransformer → N(0,1) marginals
    qt = QuantileTransformer(
        output_distribution='normal',
        n_quantiles=min(1000, n),
        random_state=RANDOM_SEED
    )
    G = qt.fit_transform(matrix)  # (n, p)

    mu  = G.mean(axis=0)
    cov = np.cov(G.T) + np.eye(p) * 1e-4   # regularise

    # Robust sampling via eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-8)
    L = eigvecs @ np.diag(np.sqrt(eigvals))

    z = rng.standard_normal((n_samples, p))
    G_synth = z @ L.T + mu

    # CRITICAL: clip to prevent extreme values from collapsing discrete cols
    G_synth = np.clip(G_synth, -4.5, 4.5)

    X_synth = qt.inverse_transform(G_synth)
    return pd.DataFrame(X_synth, columns=num_df.columns)

# ─── categorical synthesis: frequency-based ────────────────────────────────

def synthesize_categoricals(df, true_cat_cols, miss_rates, n_samples):
    """
    Sample each categorical column from its empirical marginal distribution.
    Preserves exact proportions including NaN.
    """
    cat_synth = {}
    for c in true_cat_cols:
        # Get value counts including NaN
        vc = df[c].value_counts(dropna=True, normalize=True)
        nan_rate = miss_rates[c]['nan']

        values = list(vc.index)
        probs  = list(vc.values)

        # Allocate NaN slots
        non_nan_rate = 1.0 - nan_rate
        probs_adj = [p * non_nan_rate for p in probs]

        # Sample
        chosen_idx = rng.choice(len(values), size=n_samples, p=probs_adj / np.array(probs_adj).sum())
        col_vals = np.array(values)[chosen_idx].astype(object)

        # Inject NaN
        if nan_rate > 0:
            nan_mask = rng.random(n_samples) < nan_rate
            col_vals[nan_mask] = None

        cat_synth[c] = col_vals

    return pd.DataFrame(cat_synth)

# ─── decode numerics back to original formats ──────────────────────────────

def synthesize_sparse_numerics(df, sparse_cols, miss_rates, n_samples):
    """Sample sparse numeric cols from their empirical distribution."""
    result = {}
    for c in sparse_cols:
        # Handle mixed cols that contain 'Unknown'
        raw = df[c].replace('Unknown', np.nan)
        non_null = pd.to_numeric(raw, errors='coerce').dropna()
        nan_rate = miss_rates.get(c, {}).get('nan', 0.0)
        if len(non_null) == 0:
            result[c] = np.full(n_samples, np.nan)
            continue
        sampled = rng.choice(non_null.values, size=n_samples, replace=True).astype(float)
        mask = rng.random(n_samples) < nan_rate
        sampled[mask] = np.nan
        result[c] = sampled
    return pd.DataFrame(result)


def decode_numerics(synth_num_df, df_orig, date_cols, time_cols,
                    mixed_num_cols, raw_num_cols, miss_rates):
    out = synth_num_df.copy()

    # Restore missingness for numeric/date/time cols
    for c in raw_num_cols + date_cols + time_cols:
        nan_rate = miss_rates.get(c, {}).get('nan', 0.0)
        if nan_rate > 0:
            mask = rng.random(len(out)) < nan_rate
            out.loc[mask, c] = np.nan

    # Raw numeric: enforce original dtype and range
    orig_int_cols = set(df_orig.select_dtypes(include='int64').columns)
    for c in raw_num_cols:
        orig_min = df_orig[c].min()
        orig_max = df_orig[c].max()
        out[c] = out[c].clip(orig_min, orig_max)
        if c in orig_int_cols:
            out[c] = out[c].round(0)
        else:
            # Preserve decimal precision of original
            sample_str = df_orig[c].dropna().astype(str).head(50)
            decimals = sample_str.str.extract(r'\.(\d+)')[0].str.len().median()
            decimals = 2 if pd.isna(decimals) else int(decimals)
            out[c] = out[c].round(decimals)

    # Mixed numeric/string cols
    for c in mixed_num_cols:
        orig_num = pd.to_numeric(df_orig[c].replace('Unknown', np.nan), errors='coerce')
        orig_min, orig_max = orig_num.min(), orig_num.max()
        out[c] = out[c].clip(orig_min, orig_max)

        # Determine decimal precision
        sample_str = orig_num.dropna().astype(str).head(50)
        decimals = sample_str.str.extract(r'\.(\d+)')[0].str.len().median()
        decimals = 0 if pd.isna(decimals) else int(decimals)
        out[c] = out[c].round(decimals)

        # Convert to object dtype for string output
        col_obj = out[c].astype(object)

        # Inject 'Unknown' at correct rate
        unknown_rate = miss_rates[c]['unknown']
        nan_rate     = miss_rates[c]['nan']
        unknown_mask = rng.random(len(out)) < unknown_rate
        nan_mask     = (~unknown_mask) & (rng.random(len(out)) < nan_rate)
        col_obj[unknown_mask] = 'Unknown'
        col_obj[nan_mask] = np.nan

        # Format remaining numerics as strings matching original precision
        num_mask = ~unknown_mask & ~nan_mask
        if decimals == 0:
            col_obj[num_mask] = col_obj[num_mask].apply(lambda x: str(int(x)) if pd.notna(x) else x)
        else:
            fmt = f'{{:.{decimals}f}}'
            col_obj[num_mask] = col_obj[num_mask].apply(lambda x: fmt.format(float(x)) if pd.notna(x) else x)

        out[c] = col_obj

    # Date columns
    for c in date_cols:
        valid = out[c].notna()
        out[c] = out[c].astype(object)
        days = out.loc[valid, c].round().astype(int)
        # Clip to plausible date range from original
        orig_days = (pd.to_datetime(df_orig[c], errors='coerce') - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
        lo, hi = int(orig_days.min() - 365), int(orig_days.max() + 365)
        days = days.clip(lo, hi)
        dates = pd.to_datetime(days, unit='D', origin='1970-01-01', errors='coerce').dt.strftime('%Y-%m-%d')
        out.loc[valid, c] = dates

    # Time columns
    for c in time_cols:
        valid = out[c].notna()
        out[c] = out[c].astype(object)
        mins = out.loc[valid, c].round().astype(int).clip(0, 1439)
        h = (mins // 60).astype(str).str.zfill(2)
        m = (mins % 60).astype(str).str.zfill(2)
        out.loc[valid, c] = h + ':' + m

    return out

# ─── privacy: deduplicate against original ─────────────────────────────────

def ensure_privacy(synth_num: np.ndarray, orig_num: np.ndarray, col_stds: np.ndarray):
    """Perturb synthetic rows that are too close to any original row."""
    safe_std = np.where(col_stds == 0, 1.0, col_stds)
    s_norm = synth_num / safe_std
    o_norm = orig_num  / safe_std

    # Use a random sample of originals for speed
    idx = rng.choice(len(o_norm), size=min(1000, len(o_norm)), replace=False)
    o_sample = o_norm[idx]

    for i in range(len(s_norm)):
        dists = np.abs(o_sample - s_norm[i]).mean(axis=1)
        if dists.min() < 0.02:
            noise = rng.normal(0, 0.03, size=synth_num.shape[1]) * safe_std
            synth_num[i] += noise

    return synth_num

# ─── main pipeline ─────────────────────────────────────────────────────────

def synthesize(filepath, n_samples=None):
    print("Loading data...")
    df = pd.read_csv(filepath, low_memory=False)
    if n_samples is None:
        n_samples = len(df)
    print(f"Original shape: {df.shape}")

    date_cols, time_cols, mixed_num_cols, true_cat_cols, raw_num_cols = classify_columns(df)
    print(f"  raw_num={len(raw_num_cols)}, mixed={len(mixed_num_cols)}, "
          f"categorical={len(true_cat_cols)}, date={len(date_cols)}, time={len(time_cols)}")

    miss_rates = compute_miss_rates(df, raw_num_cols, mixed_num_cols,
                                     date_cols, time_cols, true_cat_cols)

    # ── 1. Numeric synthesis via Gaussian copula ──
    print("Encoding numeric columns...")
    dense_df, dense_cols, sparse_cols, full_num_df = encode_numerics(
        df, date_cols, time_cols, mixed_num_cols, raw_num_cols
    )
    print(f"  Dense (copula): {len(dense_cols)}, Sparse (empirical): {len(sparse_cols)}")

    print(f"Fitting + sampling Gaussian copula ({dense_df.shape[1]} numeric cols)...")
    synth_dense_df = fit_and_sample_copula(dense_df, n_samples)

    print("Applying privacy noise...")
    orig_matrix = dense_df.values
    col_stds = orig_matrix.std(axis=0)
    synth_matrix = ensure_privacy(synth_dense_df.values.copy(), orig_matrix, col_stds)
    synth_dense_df = pd.DataFrame(synth_matrix, columns=dense_df.columns)

    # Synthesize sparse columns empirically
    print("Synthesizing sparse numeric columns...")
    synth_sparse_df = synthesize_sparse_numerics(df, sparse_cols, miss_rates, n_samples)

    # Merge dense + sparse into one numeric dataframe
    all_num_synth = pd.concat([synth_dense_df, synth_sparse_df], axis=1)

    print("Decoding numeric columns...")
    synth_num_decoded = decode_numerics(
        all_num_synth, df, date_cols, time_cols, mixed_num_cols, raw_num_cols, miss_rates
    )

    # ── 2. Categorical synthesis via frequency sampling ──
    print("Synthesizing categorical columns...")
    synth_cat_df = synthesize_categoricals(df, true_cat_cols, miss_rates, n_samples)

    # ── 3. Assemble in original column order ──
    print("Assembling output...")
    result = pd.DataFrame(index=range(n_samples))
    for c in df.columns:
        if c in synth_num_decoded.columns:
            result[c] = synth_num_decoded[c].values
        elif c in synth_cat_df.columns:
            result[c] = synth_cat_df[c].values
        else:
            result[c] = np.nan

    print(f"Done. Output shape: {result.shape}")
    return result


# ─── quality check ─────────────────────────────────────────────────────────

def quality_report(orig_path, synth):
    orig = pd.read_csv(orig_path, low_memory=False)
    print("\n========== QUALITY REPORT ==========")
    print(f"Columns match:  {list(orig.columns) == list(synth.columns)}")
    print(f"Row count:      {len(synth)} (orig: {len(orig)})")

    # Missingness
    print("\n-- Missingness (cols with >1% orig missing) --")
    for c in orig.columns:
        o_m = orig[c].isna().mean()
        s_m = synth[c].isna().mean()
        if o_m > 0.01 or abs(o_m - s_m) > 0.05:
            flag = ' ⚠' if abs(o_m - s_m) > 0.05 else ''
            print(f"  {c[:50]:50s}  orig={o_m:.3f}  synth={s_m:.3f}{flag}")

    # Categorical distributions
    print("\n-- Categorical distributions (first 8 cols) --")
    _, _, mixed, cats, _ = classify_columns(orig)
    for c in (cats + mixed)[:8]:
        if c not in synth.columns: continue
        print(f"  {c[:50]}:")
        o_vc = orig[c].value_counts(normalize=True).head(4).round(3).to_dict()
        s_vc = synth[c].value_counts(normalize=True).head(4).round(3).to_dict()
        print(f"    orig:  {o_vc}")
        print(f"    synth: {s_vc}")

    # Numeric distribution
    print("\n-- Numeric distributions (sample) --")
    num_cols = orig.select_dtypes(include=['int64', 'float64']).columns[:6]
    for c in num_cols:
        if c not in synth.columns: continue
        s_col = pd.to_numeric(synth[c], errors='coerce')
        print(f"  {c[:45]:45s}  orig μ={orig[c].mean():.2f} σ={orig[c].std():.2f}"
              f"  synth μ={s_col.mean():.2f} σ={s_col.std():.2f}")

    # Privacy: exact row matches
    print("\n-- Privacy --")
    num_only_orig  = orig.select_dtypes(include=['int64','float64']).fillna(-9999)
    num_only_synth = synth[num_only_orig.columns].apply(pd.to_numeric, errors='coerce').fillna(-9999)
    orig_hashes  = set(pd.util.hash_pandas_object(num_only_orig,  index=False))
    synth_hashes =     pd.util.hash_pandas_object(num_only_synth, index=False)
    n_exact = synth_hashes.isin(orig_hashes).sum()
    print(f"  Exact numeric row duplicates: {n_exact} / {len(synth)}")
    print("====================================")


if __name__ == '__main__':
    ORIG_PATH = '/mnt/user-data/uploads/data.csv'
    OUT_PATH  = '/mnt/user-data/outputs/data.csv'

    synth = synthesize(ORIG_PATH)
    synth.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")

    quality_report(ORIG_PATH, synth)
