import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from dotenv import load_dotenv

# Make sure your imports match the files we just created
from data_processor import load_data, preprocess_for_synthesis, postprocess_synthetic_data
from validator import run_evaluation_report
from approaches.example import run_random_sample

from Discriminator import Discriminator
from Generative import Generator, generate_synthetic_samples

load_dotenv()

def main():
    # Define directory paths
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True) # Ensure results directory exists

    input_file = data_dir / "data.csv"
    
    # ==========================================
    # Phase 1: Data Loading and Preprocessing
    # ==========================================
    if not input_file.exists():
        print(f"Error: {input_file} not found. Please place the CSV in the 'data' folder.")
        return

    raw_data = load_data(str(input_file))
    print(f"Original Data Shape: {raw_data.shape}")

    # Run the robust preprocessing pipeline
    print("\nPreprocessing data...")
    # [!] FIX: Unpack cols_to_scale directly from the function
    df_encoded, scaler, date_cols, time_cols, cols_to_scale = preprocess_for_synthesis(raw_data)
    
    # We need to capture these for the post-processor
    encoded_cols = df_encoded.columns.tolist()
    categorical_cols = raw_data.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_scale = [col for col in encoded_cols if df_encoded[col].max() > 1 or df_encoded[col].min() < 0]

    # ==========================================
    # Phase 2: GAN Setup & PyTorch DataLoaders
    # ==========================================
    # Choose what we want to condition the generation on
    # Because of one-hot encoding, the prefix will grab all related columns (e.g., _Yes, _No, _Unknown)
    TARGET_CONDITION_PREFIX = 'Complications at all during primary stay::183_'
    condition_cols = [col for col in encoded_cols if col.startswith(TARGET_CONDITION_PREFIX)]
    feature_cols = [col for col in encoded_cols if col not in condition_cols]

    if not condition_cols:
        print(f"Warning: Could not find condition columns starting with '{TARGET_CONDITION_PREFIX}'. Check your headers.")
        return

    data_dim = len(feature_cols)
    condition_dim = len(condition_cols)
    latent_dim = 100

    print(f"\nInitializing Models...")
    print(f"Features Dim: {data_dim} | Condition Dim: {condition_dim} | Latent Dim: {latent_dim}")

    discriminator = Discriminator(data_dim, condition_dim)
    generator = Generator(latent_dim, condition_dim, data_dim)

    X_train = torch.tensor(df_encoded[feature_cols].values, dtype=torch.float32)
    C_train = torch.tensor(df_encoded[condition_cols].values, dtype=torch.float32)

    if torch.isnan(X_train).any() or torch.isnan(C_train).any():
        print("CRITICAL ERROR: NaNs detected in the final tensor! Preprocessing failed.")
        return

    # Create DataLoader
    batch_size = 64
    dataset = TensorDataset(X_train, C_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizers and Loss
    lr = 0.0002
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()

    # ==========================================
    # Phase 3: Training Loop
    # ==========================================
    num_epochs = 300
    print("\nStarting GAN Training...")
    
    for epoch in range(num_epochs):
        for n, (real_samples, conditions) in enumerate(train_loader):
            current_batch_size = real_samples.size(0)
            
            # --- 1. Train Discriminator ---
            discriminator.zero_grad()
            
            # Generate fake data
            latent_space_samples = torch.randn(current_batch_size, latent_dim)
            generated_samples = generator(latent_space_samples, conditions)
            
            # Labels for calculating loss
            real_labels = torch.ones(current_batch_size, 1)
            fake_labels = torch.zeros(current_batch_size, 1)
            
            # Discriminator predictions
            output_real = discriminator(real_samples, conditions)
            output_fake = discriminator(generated_samples.detach(), conditions)
            
            # Discriminator loss
            loss_real = loss_function(output_real, real_labels)
            loss_fake = loss_function(output_fake, fake_labels)
            loss_discriminator = (loss_real + loss_fake) / 2
            
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # --- 2. Train Generator ---
            generator.zero_grad()
            
            # Ask discriminator to evaluate the fake data again (with updated weights)
            output_discriminator_generated = discriminator(generated_samples, conditions)
            
            # Generator loss (it wants the discriminator to output '1' for its fake data)
            loss_generator = loss_function(output_discriminator_generated, real_labels)
            
            loss_generator.backward()
            optimizer_generator.step()

        # Print progress every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: [{epoch}/{num_epochs}] | Loss D: {loss_discriminator.item():.4f} | Loss G: {loss_generator.item():.4f}")

    # ==========================================
    # Phase 4: Generate Synthetic Data
    # ==========================================
    print("\nGenerating synthetic data...")
    num_samples_to_generate = 8000
    
    # Use the helper function from Generative.py
    synthetic_features, synthetic_conditions = generate_synthetic_samples(
        generator_model=generator, 
        num_samples=num_samples_to_generate, 
        latent_dim=latent_dim, 
        condition_dim=condition_dim
    )

    # Combine features and conditions back into a single dataframe
    # We must ensure the columns are in the exact same order as df_encoded for the post-processor
    df_synth_features = pd.DataFrame(synthetic_features.cpu().numpy(), columns=feature_cols)
    df_synth_conditions = pd.DataFrame(synthetic_conditions.cpu().numpy(), columns=condition_cols)
    
    df_synth_encoded = pd.concat([df_synth_features, df_synth_conditions], axis=1)
    df_synth_encoded = df_synth_encoded[encoded_cols] # Reorder to match original encoding

    # ==========================================
    # Phase 5: Post-Processing & Export
    # ==========================================
    print("Reversing data to original CSV format...")
    final_synthetic_df = postprocess_synthetic_data(
        synthetic_tensor=df_synth_encoded.values,
        original_df=raw_data,
        scaler=scaler,
        cols_to_scale=cols_to_scale,
        encoded_cols=encoded_cols,
        categorical_cols=categorical_cols,
        date_cols=date_cols,
        time_cols=time_cols
    )

    # Optional: Run Evaluation Report (if your validator script handles dataframes directly)
    # run_evaluation_report(raw_data, final_synthetic_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"synthetic_data_{timestamp}.csv"
    output_path = results_dir / output_filename

    final_synthetic_df.to_csv(output_path, index=False)
    
    print(f"\nExecution successful.")
    print(f"Synthetic file saved to: {output_path}")

if __name__ == "__main__":
    main()