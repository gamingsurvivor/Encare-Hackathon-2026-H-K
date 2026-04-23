import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

# Import your custom modules
from data_processor import (
    load_data, 
    preprocess_for_synthesis, 
    postprocess_synthetic_data
)
from Discriminator import Discriminator
from Generative import Generator, generate_synthetic_samples

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    """
    Standard WGAN-GP Gradient Penalty to enforce Lipschitz constraint.
    """
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size()).to(device)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def main():
    # --- Phase 1: Setup & Data Loading ---
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    input_file = data_dir / "data.csv"
    if not input_file.exists():
        print("Error: data/data.csv not found.")
        return

    raw_data = load_data(str(input_file))
    original_columns = raw_data.columns.tolist()

    # --- Phase 2: Preprocessing ---
    print("\nPreprocessing data...")
    # df_encoded contains original columns + missing flags
    df_encoded, scaler, date_cols, time_cols, label_encoders, categorical_cols, missing_flags, unknown_flags = preprocess_for_synthesis(raw_data)    
    all_feature_cols = df_encoded.columns.tolist()

    # Feature Pruning (Removes columns with 0 variance)
    variances = df_encoded[all_feature_cols].var()
    feature_cols = variances[variances > 0.0].index.tolist()
    pruned_cols = [c for c in all_feature_cols if c not in feature_cols]
    
    # Store defaults for pruned columns to re-insert later
    pruned_defaults = {c: df_encoded[c].mode()[0] for c in pruned_cols}
    print(f"Training on {len(feature_cols)} features. Pruned {len(pruned_cols)} constant columns.")

    # --- Phase 3: GAN Configuration ---
    data_dim = len(feature_cols)
    latent_dim = 256
    batch_size = 256
    num_epochs = 1000
    n_critic = 5
    lambda_gp = 5
    lr = 0.00005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device.type.upper()}")

    generator = Generator(latent_dim, data_dim).to(device)
    discriminator = Discriminator(data_dim).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))

    X_train = torch.tensor(df_encoded[feature_cols].values, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # --- Phase 4: Training Loop (WGAN-GP) ---
    print("\nStarting WGAN-GP Training...")
    for epoch in range(num_epochs):
        for i, (real_samples,) in enumerate(train_loader):
            
            # 1. Update Discriminator (Critic)
            optimizer_D.zero_grad()
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = generator(noise)

            real_validity = discriminator(real_samples)
            fake_validity = discriminator(fake_samples.detach())
            gp = compute_gradient_penalty(discriminator, real_samples, fake_samples.detach(), device)
            
            # WGAN-GP Loss: Wasserstein Distance + Gradient Penalty
            loss_D = torch.mean(fake_validity) - torch.mean(real_validity) + lambda_gp * gp
            loss_D.backward()
            optimizer_D.step()

            # 2. Update Generator
            if i % n_critic == 0:
                optimizer_G.zero_grad()
                gen_samples = generator(torch.randn(batch_size, latent_dim).to(device))
                loss_G = -torch.mean(discriminator(gen_samples))
                loss_G.backward()
                optimizer_G.step()

        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{num_epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

    # --- Phase 5: Generation & Shape Reconstruction ---
    print("\nGenerating 8000 samples...")
    num_final_samples = 8000
    massive_features = generate_synthetic_samples(generator, num_final_samples, latent_dim, device)
    
    # Create DataFrame from learned features (307 columns)
    df_synth_learned = pd.DataFrame(massive_features.cpu().numpy(), columns=feature_cols)

    # RE-INSERT PRUNED COLUMNS: This is the fix for the Shape ValueError
    print("Re-inserting pruned constant columns...")
    for col, default_val in pruned_defaults.items():
        df_synth_learned[col] = default_val

    # Ensure all original columns + missing flags are present (341 total)
    full_col_list = original_columns + missing_flags + unknown_flags
    df_full_synth = pd.DataFrame(index=range(num_final_samples), columns=full_col_list)

    for col in df_synth_learned.columns:
        if col in df_full_synth.columns:
            df_full_synth[col] = df_synth_learned[col]
            
    # Fill any remaining gaps (e.g., specific missing flags) with 0
    df_full_synth = df_full_synth.fillna(0)

    # --- Phase 6: Post-processing ---
    print("Applying medical logic and datatype alignment...")
    # Pass it to post-processing
    final_synthetic_df = postprocess_synthetic_data(
        synthetic_tensor=df_full_synth.values,
        original_columns=original_columns,
        scaler_bundle=scaler, 
        date_cols=date_cols, 
        time_cols=time_cols,
        label_encoders=label_encoders, 
        categorical_cols=categorical_cols, 
        missing_flags=missing_flags,
        unknown_flags=unknown_flags,     # <--- ADD THIS HERE
        raw_df=raw_data 
    )

    # --- Phase 7: Save Results ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = results_dir / f"synthetic_data_{timestamp}.csv"
    final_synthetic_df.to_csv(output_path, index=False)
    
    # Save a copy as 'raw_gan_output' for the fix_validator.py script
    final_synthetic_df.to_csv("results/raw_gan_output.csv", index=False)

    print(f"\nExecution successful. Saved to: {output_path}")
    print("IMPORTANT: Now run 'python fix_validator.py' to finalize the file!")

if __name__ == "__main__":
    main()