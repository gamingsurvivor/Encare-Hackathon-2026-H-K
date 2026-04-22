import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from dotenv import load_dotenv

from CSVTester import SyntheticDataDiscriminator
from data_processor import load_data, preprocess_for_synthesis, postprocess_synthetic_data
from Discriminator import Discriminator
from Generative import Generator, generate_synthetic_samples

load_dotenv()

def compute_gradient_penalty(D, real_samples, fake_samples, conditions, device):
    alpha = torch.rand(real_samples.size(0), 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, conditions)
    fake = torch.ones(d_interpolates.size()).to(device)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_correlation_matrix(x):
    x_centered = x - torch.mean(x, dim=0)
    std = torch.std(x, dim=0) + 1e-8
    x_standardized = x_centered / std
    corr_matrix = torch.matmul(x_standardized.T, x_standardized) / (x.size(0) - 1)
    return corr_matrix

def main():
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    input_file = data_dir / "data.csv"
    if not input_file.exists():
        print("Error: CSV not found.")
        return

    raw_data = load_data(str(input_file))
    print(f"Original Data Shape: {raw_data.shape}")

    print("\nPreprocessing data...")
    df_encoded, scaler, date_cols, time_cols, label_encoders, categorical_cols, missing_flags = preprocess_for_synthesis(raw_data)
    original_columns = raw_data.columns.tolist()
    
    TARGET_CONDITION = 'Complications at all during primary stay::183'
    condition_cols = [TARGET_CONDITION]
    all_feature_cols = [col for col in df_encoded.columns.tolist() if col != TARGET_CONDITION]

    # --- Phase 1.5: FEATURE PRUNING ---
    print("\nPruning low-variance columns...")
    variances = df_encoded[all_feature_cols].var()
    feature_cols = variances[variances > 0.0].index.tolist()
    pruned_cols = [c for c in all_feature_cols if c not in feature_cols]
    pruned_defaults = {c: df_encoded[c].mode()[0] for c in pruned_cols}
    print(f"Kept {len(feature_cols)} features. Pruned {len(pruned_cols)} useless features.")

    # --- Phase 2: GAN Setup & Class Balancing ---
    data_dim = len(feature_cols)
    condition_dim = len(condition_cols)
    latent_dim = 512 
    pack_size = 8

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nInitializing WGAN-GP Models on: {device.type.upper()}...")

    discriminator = Discriminator(data_dim, condition_dim, pack_size=pack_size).to(device)
    generator = Generator(latent_dim, condition_dim, data_dim).to(device)

    X_train = torch.tensor(df_encoded[feature_cols].values, dtype=torch.float32).to(device)
    C_train = torch.tensor(df_encoded[condition_cols].values, dtype=torch.float32).to(device)

    # --- AMPLIFIER: Make rare complications "Louder" ---
    raw_target = raw_data[TARGET_CONDITION].astype(str).fillna('Unknown')
    class_counts = raw_target.value_counts()
    sample_weights = raw_target.map(lambda x: 1.0 / class_counts[x]).values.copy()
    
    complication_cols = [c for c in feature_cols if df_encoded[c].nunique() == 2]
    rare_row_mask = (df_encoded[complication_cols] > 0.5).any(axis=1).values
    sample_weights[rare_row_mask] *= 5.0
    
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    batch_size = 256
    dataset = TensorDataset(X_train, C_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True) 

    lr = 0.0001
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))

    # --- Phase 3: WGAN-GP Training Loop ---
    num_epochs = 300
    n_critic = 5 
    lambda_gp = 10 
    
    print("\nStarting WGAN-GP Training...")
    
    for epoch in range(num_epochs):
        for n, (real_samples, conditions) in enumerate(train_loader):
            current_batch_size = real_samples.size(0)
            
            # --- 1. Train Discriminator ---
            discriminator.zero_grad()
            latent_space_samples = torch.randn(current_batch_size, latent_dim).to(device)
            generated_samples = generator(latent_space_samples, conditions)
            
            real_validity = discriminator(real_samples, conditions)
            fake_validity = discriminator(generated_samples.detach(), conditions)
            gradient_penalty = compute_gradient_penalty(discriminator, real_samples.data, generated_samples.data, conditions.data, device)
            
            loss_discriminator = torch.mean(fake_validity) - (torch.mean(real_validity) * 0.9) + lambda_gp * gradient_penalty
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # --- 2. Train Generator ---
            if n % n_critic == 0:
                generator.zero_grad()
                latent_space_samples = torch.randn(current_batch_size, latent_dim).to(device)
                generated_samples = generator(latent_space_samples, conditions)
                
                # 1. Base WGAN Loss
                fake_validity = discriminator(generated_samples, conditions)
                loss_generator_base = -torch.mean(fake_validity)
                
                # 2. Correlation Overdrive (Preserves 2D Relationships)
                real_corr = compute_correlation_matrix(real_samples)
                fake_corr = compute_correlation_matrix(generated_samples)
                corr_loss = torch.nn.functional.mse_loss(fake_corr, real_corr)
                
                # 3. Statistical Feature Matching (Prevents Mode Collapse & Fixes 1D Shapes)
                # This mathematically forces the GAN to match the exact center and spread of the real data
                mean_loss = torch.nn.functional.mse_loss(generated_samples.mean(dim=0), real_samples.mean(dim=0))
                std_loss = torch.nn.functional.mse_loss(generated_samples.std(dim=0) + 1e-8, real_samples.std(dim=0) + 1e-8)
                
                # Balance the constraints
                lambda_corr = 5.0 
                lambda_fm = 10.0 
                
                loss_generator = loss_generator_base + (lambda_corr * corr_loss) + (lambda_fm * (mean_loss + std_loss))
                
                loss_generator.backward()
                optimizer_generator.step()

        if epoch % 10 == 0:
            print(f"Epoch: [{epoch}/{num_epochs}] | Loss D: {loss_discriminator.item():.4f} | Loss G: {loss_generator.item():.4f}")

    print("\n[UPGRADE] Over-Generating and Cherry-Picking the best PacGAN packs...")
    OVERGENERATE_MULTIPLIER = 10
    num_final_samples = 8000
    total_to_generate = num_final_samples * OVERGENERATE_MULTIPLIER
    
    massive_features, massive_conditions = generate_synthetic_samples(
        generator_model=generator, num_samples=total_to_generate, 
        latent_dim=latent_dim, condition_dim=condition_dim, device=device
    )

    discriminator.eval()
    with torch.no_grad():
        critic_scores = discriminator(massive_features, massive_conditions).squeeze()

    num_packs_to_keep = num_final_samples // pack_size
    _, top_pack_indices = torch.topk(critic_scores, num_packs_to_keep)

    features_packed = massive_features.view(-1, pack_size, massive_features.size(-1))
    conditions_packed = massive_conditions.view(-1, pack_size, massive_conditions.size(-1))

    elite_features_packed = features_packed[top_pack_indices]
    elite_conditions_packed = conditions_packed[top_pack_indices]

    elite_features = elite_features_packed.view(num_final_samples, -1)
    elite_conditions = elite_conditions_packed.view(num_final_samples, -1)

    print(f"Successfully cherry-picked the top {num_final_samples} patients based on Critic scores!")

    df_synth_features = pd.DataFrame(elite_features.cpu().numpy(), columns=feature_cols)
    df_synth_conditions = pd.DataFrame(elite_conditions.cpu().numpy(), columns=condition_cols)
    
    for col, default_val in pruned_defaults.items():
        df_synth_features[col] = default_val

    df_synth_encoded = pd.concat([df_synth_features, df_synth_conditions], axis=1)
    all_encoded_cols = original_columns + missing_flags
    df_synth_encoded = df_synth_encoded[all_encoded_cols]

    print("Reversing data to original CSV format...")
    final_synthetic_df = postprocess_synthetic_data(
        synthetic_tensor=df_synth_encoded.values,
        original_columns=original_columns,
        scaler_bundle=scaler, date_cols=date_cols, time_cols=time_cols,
        label_encoders=label_encoders, categorical_cols=categorical_cols, missing_flags=missing_flags,
        raw_df=raw_data 
    )

    output_filename = f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    output_path = results_dir / output_filename
    final_synthetic_df.to_csv(output_path, index=False)
    
    print(f"\nExecution successful. Raw file saved to: {output_path}")
    print("REMINDER: Run 'python fix_validator.py' to format the columns before uploading!")

if __name__ == "__main__":
    main()