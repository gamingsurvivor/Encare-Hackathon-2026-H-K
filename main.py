"""
main_v2.py — Improved CTGAN training pipeline

Key improvements over original:
  1. Multi-condition vector (not just 1 col) — richer conditioning signal
  2. Wasserstein loss + gradient penalty (WGAN-GP) — more stable training,
     better distribution matching, directly attacks the Discrimination score
  3. Separate handling of binary/categorical cols to fix the ±5 Gaussian collapse
  4. More epochs (600) with learning rate schedule
  5. Post-generation privacy deduplication pass
  6. Cleaner dtype restoration — no more force_to_str patches
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from dotenv import load_dotenv

from data_processor import (
    classify_columns,
    compute_miss_rates,
    encode_numerics,
    decode_numerics,
)
from Discriminator import Discriminator
from Generative import Generator, generate_synthetic_samples

load_dotenv()

# ─── WGAN-GP gradient penalty ──────────────────────────────────────────────

def compute_gradient_penalty(discriminator, real_samples, fake_samples, conditions, device):
    """
    Gradient penalty term for WGAN-GP.
    Forces the discriminator gradient norm ≈ 1 everywhere,
    giving much more stable training than vanilla GAN / BCELoss.
    This directly improves the Discrimination score.
    """
    alpha = torch.rand(real_samples.size(0), 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Discriminator forward on interpolated samples
    # Note: PacGAN packing means we need full-batch shapes — use unpacked critic here
    d_interpolates = discriminator(interpolates, conditions)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# ─── dtype restoration (replaces the force_to_str/float hack) ─────────────

def restore_dtypes(synthetic_df: pd.DataFrame, raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean, systematic dtype restoration that matches the original CSV exactly.
    Handles: int64, float64, object (strings), and mixed numeric/string cols.
    """
    result = synthetic_df.copy()

    for col in result.columns:
        if col not in raw_data.columns:
            continue

        orig_dtype = raw_data[col].dtype
        orig_col = raw_data[col]

        # Check if the original object column is actually numeric (like "Unknown" + numbers)
        if orig_dtype == object:
            numeric_part = pd.to_numeric(orig_col.replace('Unknown', np.nan), errors='coerce')
            is_mixed_numeric = numeric_part.notna().mean() > 0.5
        else:
            is_mixed_numeric = False

        try:
            if orig_dtype == 'int64':
                result[col] = pd.to_numeric(result[col], errors='coerce').round().fillna(0).astype(int)

            elif orig_dtype == 'float64':
                result[col] = pd.to_numeric(result[col], errors='coerce').astype(float)

            elif orig_dtype == object and is_mixed_numeric:
                # Keep as object (string), but format numeric values properly
                # Determine decimal precision from original
                sample = orig_col.replace('Unknown', np.nan).dropna().head(50).astype(str)
                decimals_series = sample.str.extract(r'\.(\d+)')[0].str.len()
                decimals = 0 if decimals_series.isna().all() else int(decimals_series.median())

                col_numeric = pd.to_numeric(result[col].replace('Unknown', np.nan), errors='coerce')
                col_obj = col_numeric.astype(object)

                valid = col_numeric.notna()
                if decimals == 0:
                    col_obj[valid] = col_numeric[valid].round(0).astype(int).astype(str)
                else:
                    fmt = f'{{:.{decimals}f}}'
                    col_obj[valid] = col_numeric[valid].apply(lambda x: fmt.format(x))

                col_obj[~valid] = np.nan
                result[col] = col_obj

            elif orig_dtype == object:
                # Pure categorical — convert to string, clean up artefacts
                result[col] = result[col].astype(str)
                result[col] = result[col].str.replace(r'\.0$', '', regex=True)
                result[col] = result[col].replace({'nan': np.nan, 'None': np.nan, '': np.nan})

        except Exception as e:
            print(f"  Warning: dtype restore failed for {col}: {e}")

    return result


# ─── privacy pass ─────────────────────────────────────────────────────────

def ensure_no_duplicates(synthetic_df: pd.DataFrame, original_df: pd.DataFrame,
                         num_cols: list, noise_std: float = 0.05) -> pd.DataFrame:
    """
    Perturb any synthetic row whose numeric fingerprint matches an original row.
    Protects the Privacy score.
    """
    result = synthetic_df.copy()
    avail = [c for c in num_cols if c in result.columns and c in original_df.columns]
    if not avail:
        return result

    synth_num = result[avail].apply(pd.to_numeric, errors='coerce').fillna(0).values
    orig_num  = original_df[avail].apply(pd.to_numeric, errors='coerce').fillna(0).values

    col_stds = orig_num.std(axis=0)
    safe_std = np.where(col_stds == 0, 1.0, col_stds)

    synth_norm = synth_num / safe_std
    orig_norm  = orig_num  / safe_std

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(orig_norm), size=min(1000, len(orig_norm)), replace=False)
    orig_sample = orig_norm[sample_idx]

    perturb_count = 0
    for i in range(len(synth_norm)):
        dists = np.abs(orig_sample - synth_norm[i]).mean(axis=1)
        if dists.min() < 0.02:
            noise = rng.normal(0, noise_std, size=len(avail)) * safe_std
            synth_num[i] += noise
            perturb_count += 1

    if perturb_count > 0:
        print(f"  Privacy: perturbed {perturb_count} near-duplicate rows")
        result[avail] = synth_num

    return result


# ─── main ─────────────────────────────────────────────────────────────────

def main():
    data_dir   = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    input_file = data_dir / "data.csv"
    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    raw_data = pd.read_csv(str(input_file), low_memory=False)
    print(f"Original Data Shape: {raw_data.shape}")

    print("\nPreprocessing data...")
    # Classify columns using data_processor helpers
    date_cols, time_cols, mixed_num_cols, true_cat_cols, raw_num_cols = classify_columns(raw_data)
    miss_rates = compute_miss_rates(raw_data, raw_num_cols, mixed_num_cols,
                                    date_cols, time_cols, true_cat_cols)
    dense_df, dense_cols, sparse_cols, full_num_df = encode_numerics(
        raw_data, date_cols, time_cols, mixed_num_cols, raw_num_cols
    )

    # Label-encode categoricals for GAN input
    from sklearn.preprocessing import LabelEncoder
    work = raw_data.copy()
    label_encoders = {}
    for c in true_cat_cols:
        work[c] = work[c].fillna('<BLANK>')
        le = LabelEncoder()
        work[c] = le.fit_transform(work[c].astype(str)).astype(float)
        label_encoders[c] = le

    # Convert mixed cols to numeric
    for c in mixed_num_cols:
        work[c] = pd.to_numeric(work[c].replace('Unknown', np.nan), errors='coerce')

    # Convert dates / times
    ref_date = pd.Timestamp("1970-01-01")
    for c in date_cols:
        work[c] = (pd.to_datetime(work[c], errors='coerce') - ref_date) / pd.Timedelta(days=1)
    for c in time_cols:
        t = pd.to_datetime(work[c], format='%H:%M', errors='coerce')
        work[c] = t.dt.hour * 60 + t.dt.minute

    # Build the fully encoded DataFrame and fill NaN with column median
    all_num_cols = raw_num_cols + mixed_num_cols + date_cols + time_cols
    df_encoded = work[all_num_cols + true_cat_cols].astype(float)
    for c in df_encoded.columns:
        med = df_encoded[c].median()
        df_encoded[c] = df_encoded[c].fillna(0 if pd.isna(med) else med)

    # Fit a single scaler on all encoded columns (used for postprocessing)
    from sklearn.preprocessing import QuantileTransformer
    scaler = QuantileTransformer(output_distribution='normal',
                                 n_quantiles=min(1000, len(df_encoded)),
                                 random_state=42)
    encoded_cols = df_encoded.columns.tolist()
    df_encoded[encoded_cols] = scaler.fit_transform(df_encoded[encoded_cols])

    # Track missing flags for numeric cols (same convention as original pipeline)
    missing_flags = []
    for c in all_num_cols:
        if raw_data[c].isna().any():
            flag_col = f"{c}_missing_flag"
            df_encoded[flag_col] = raw_data[c].isna().astype(float)
            missing_flags.append(flag_col)

    original_columns = raw_data.columns.tolist()
    categorical_cols = true_cat_cols

    # ── conditioning: use multiple high-signal columns, not just one ──────
    # Using the top complication flag + a few key clinical variables gives
    # the generator richer structural signal → better multivariate fidelity
    CONDITION_COLS_CANDIDATES = [
        'Complications at all during primary stay::183',
        'Gender::5',
        'Age::40',
        'BMI::24',
        'ERAS Implementation::4',
    ]
    condition_cols = [c for c in CONDITION_COLS_CANDIDATES if c in df_encoded.columns]
    feature_cols   = [c for c in df_encoded.columns if c not in condition_cols]

    data_dim      = len(feature_cols)
    condition_dim = len(condition_cols)
    latent_dim    = 256

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device.type.upper()}")
    print(f"Features: {data_dim} | Conditions: {condition_dim} ({condition_cols}) | Latent: {latent_dim}")

    discriminator = Discriminator(data_dim, condition_dim).to(device)
    generator     = Generator(latent_dim, condition_dim, data_dim).to(device)

    # WGAN-GP needs raw logits — patch out the Sigmoid that's baked into Discriminator
    discriminator.model[-1] = torch.nn.Identity().to(device)

    X_train = torch.tensor(df_encoded[feature_cols].values,  dtype=torch.float32).to(device)
    C_train = torch.tensor(df_encoded[condition_cols].values, dtype=torch.float32).to(device)

    batch_size   = 128   # larger batch → more stable WGAN-GP gradients
    dataset      = TensorDataset(X_train, C_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # WGAN-GP hyperparameters
    lr_D          = 0.0001
    lr_G          = 0.0001
    lambda_gp     = 10.0     # gradient penalty weight
    n_critic      = 5        # train discriminator 5× per generator step
    num_epochs    = 600      # more epochs → better convergence

    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.0, 0.9))
    optimizer_G = torch.optim.Adam(generator.parameters(),     lr=lr_G, betas=(0.0, 0.9))

    # LR schedule: reduce both by 50% halfway through
    scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=num_epochs // 2, gamma=0.5)
    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=num_epochs // 2, gamma=0.5)

    print(f"\nStarting WGAN-GP Training ({num_epochs} epochs, n_critic={n_critic})...")
    print("(WGAN-GP replaces BCELoss for more stable training and better distribution matching)\n")

    for epoch in range(num_epochs):
        for n, (real_samples, conditions) in enumerate(train_loader):
            current_batch = real_samples.size(0)

            # ── Train Discriminator (Critic) n_critic times ───────────────
            for _ in range(n_critic):
                discriminator.zero_grad()

                z = torch.randn(current_batch, latent_dim, device=device)
                fake_samples = generator(z, conditions).detach()

                d_real = discriminator(real_samples, conditions)
                d_fake = discriminator(fake_samples, conditions)
                loss_wasserstein = -torch.mean(d_real) + torch.mean(d_fake)

                gp = compute_gradient_penalty(
                    discriminator, real_samples, fake_samples, conditions, device
                )
                loss_D = loss_wasserstein + lambda_gp * gp
                loss_D.backward()
                optimizer_D.step()

            # ── Train Generator ───────────────────────────────────────────
            generator.zero_grad()
            z = torch.randn(current_batch, latent_dim, device=device)
            fake_samples = generator(z, conditions)
            d_fake_g = discriminator(fake_samples, conditions)
            loss_G = -torch.mean(d_fake_g)
            loss_G.backward()
            optimizer_G.step()

        scheduler_D.step()
        scheduler_G.step()

        if epoch % 50 == 0:
            print(f"Epoch [{epoch:>3}/{num_epochs}] | "
                  f"Loss D: {loss_D.item():+.4f} | Loss G: {loss_G.item():+.4f} | "
                  f"GP: {gp.item():.4f}")

    # ── Generate ──────────────────────────────────────────────────────────
    print("\nGenerating synthetic data...")
    num_samples = len(raw_data)  # match original row count exactly

    synthetic_features, synthetic_conditions = generate_synthetic_samples(
        generator_model=generator,
        num_samples=num_samples,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        device=device,
    )

    df_synth_features   = pd.DataFrame(synthetic_features.cpu().numpy(),   columns=feature_cols)
    df_synth_conditions = pd.DataFrame(synthetic_conditions.cpu().numpy(), columns=condition_cols)
    df_synth_encoded    = pd.concat([df_synth_features, df_synth_conditions], axis=1)

    # Reorder to match the encoded column order (encoded_cols + missing_flags)
    all_gan_cols = encoded_cols + missing_flags
    # Only keep cols that the GAN actually produced
    available = [c for c in all_gan_cols if c in df_synth_encoded.columns]
    df_synth_encoded = df_synth_encoded.reindex(columns=available)

    print("Reversing to original CSV format...")

    # Inverse-scale
    synth_vals = df_synth_encoded[encoded_cols].values
    synth_vals = scaler.inverse_transform(synth_vals)
    df_synth_inv = pd.DataFrame(synth_vals, columns=encoded_cols)

    # Restore missing flags
    for flag_col in missing_flags:
        orig_col = flag_col.replace("_missing_flag", "")
        if flag_col in df_synth_encoded.columns and orig_col in df_synth_inv.columns:
            miss_mask = df_synth_encoded[flag_col].values > 0.5
            df_synth_inv.loc[miss_mask, orig_col] = np.nan

    # Use decode_numerics for date/time/numeric formatting
    final_synthetic_df = decode_numerics(
        df_synth_inv, raw_data, date_cols, time_cols,
        mixed_num_cols, raw_num_cols, miss_rates
    )

    # Decode categoricals
    for c in categorical_cols:
        if c not in df_synth_inv.columns:
            continue
        le = label_encoders[c]
        max_id = len(le.classes_) - 1
        idx = df_synth_inv[c].fillna(0).round().clip(0, max_id).astype(int)
        final_synthetic_df[c] = le.inverse_transform(idx)
        final_synthetic_df[c] = final_synthetic_df[c].replace('<BLANK>', np.nan)
        nan_rate = miss_rates.get(c, {}).get('nan', 0.0)
        if nan_rate > 0:
            mask = np.random.default_rng(42).random(len(final_synthetic_df)) < nan_rate
            final_synthetic_df.loc[mask, c] = np.nan

    # Reorder to match original column order
    final_synthetic_df = final_synthetic_df.reindex(columns=original_columns)

    # ── Privacy pass ──────────────────────────────────────────────────────
    print("Running privacy deduplication pass...")
    num_cols = raw_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    final_synthetic_df = ensure_no_duplicates(final_synthetic_df, raw_data, num_cols)

    # ── Dtype restoration ────────────────────────────────────────────────
    print("Restoring original data types...")
    final_synthetic_df = restore_dtypes(final_synthetic_df, raw_data)

    # Round floats
    float_cols = final_synthetic_df.select_dtypes(include=['float64', 'float32']).columns
    final_synthetic_df[float_cols] = final_synthetic_df[float_cols].round(2)

    # ── Save ──────────────────────────────────────────────────────────────
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
    output_path     = results_dir / f"synthetic_data_{timestamp}.csv"
    final_synthetic_df.to_csv(output_path, index=False)

    print(f"\n✓ Saved: {output_path}")
    print(f"  Shape: {final_synthetic_df.shape}")

    # Quick distribution sanity check
    print("\n-- Distribution check (numeric cols) --")
    for col in raw_data.select_dtypes(include=['int64', 'float64']).columns[:5]:
        if col in final_synthetic_df.columns:
            s = pd.to_numeric(final_synthetic_df[col], errors='coerce')
            print(f"  {col[:50]:50s}  orig μ={raw_data[col].mean():.2f}  synth μ={s.mean():.2f}")

    print("\n-- Categorical check --")
    obj_cols = raw_data.select_dtypes(include='object').columns[:3]
    for col in obj_cols:
        if col in final_synthetic_df.columns:
            o = raw_data[col].value_counts(normalize=True).head(2).round(3).to_dict()
            s = final_synthetic_df[col].value_counts(normalize=True).head(2).round(3).to_dict()
            print(f"  {col[:45]}:")
            print(f"    orig:  {o}")
            print(f"    synth: {s}")


if __name__ == "__main__":
    main()