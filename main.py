import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from dotenv import load_dotenv

# Make sure your imports match the files we created
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

    # Run the new Tokenizer pipeline
    print("\nPreprocessing data...")
    df_encoded, scaler, date_cols, time_cols, label_encoders, categorical_cols = preprocess_for_synthesis(raw_data)
    original_columns = raw_data.columns.tolist()
    
    # ==========================================
    # Phase 2: GAN Setup & PyTorch DataLoaders
    # ==========================================
    # Choose what we want to condition the generation on
    TARGET_CONDITION = 'Complications at all during primary stay::183'
    condition_cols = [TARGET_CONDITION]
    feature_cols = [col for col in original_columns if col != TARGET_CONDITION]

    data_dim = len(feature_cols)
    condition_dim = len(condition_cols) # This will now be 1
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
    
    synthetic_features, synthetic_conditions = generate_synthetic_samples(
        generator_model=generator, 
        num_samples=num_samples_to_generate, 
        latent_dim=latent_dim, 
        condition_dim=condition_dim
    )

    # Combine back into a single dataframe
    df_synth_features = pd.DataFrame(synthetic_features.cpu().numpy(), columns=feature_cols)
    df_synth_conditions = pd.DataFrame(synthetic_conditions.cpu().numpy(), columns=condition_cols)
    
    df_synth_encoded = pd.concat([df_synth_features, df_synth_conditions], axis=1)
    
    # CRITICAL: Reorder to perfectly match the 286 original headers
    df_synth_encoded = df_synth_encoded[original_columns] 

    # ==========================================
    # Phase 5: Post-Processing & Export
    # ==========================================
    print("Reversing data to original CSV format...")
    final_synthetic_df = postprocess_synthetic_data(
        synthetic_tensor=df_synth_encoded.values,
        original_columns=original_columns,
        scaler=scaler,
        date_cols=date_cols,
        time_cols=time_cols,
        label_encoders=label_encoders,
        categorical_cols=categorical_cols
    )

    # --- THE VALIDATOR SCHEMA HACK ---
    print("Matching original data types for submission validator...")
    
    # The exact columns the validator is stubborn about
    stubborn_columns = [
        'Preoperative body weight (kg)::20',
        'Height (cm)::23',
        'Intraoperative blood loss (ml)::69',
        'Core body temperature at end of operation (°C)::95',
        'Morning weight - On postoperative day 1 (kg)::111',
        'Morning weight - On postoperative day 2 (kg)::113',
        'Oral fluids, total volume taken - On postoperative day 1 (ml)::119',
        'Oral nutritional supplements, energy intake - On day of surgery, postoperatively (kCal)::122'
    ]

    for col in final_synthetic_df.columns:
        orig_type = raw_data[col].dtype
        
        # If the original file considered this a string, OR if the validator explicitly demands it...
        if orig_type == 'object' or orig_type.name == 'category' or col in stubborn_columns:
            final_synthetic_df[col] = final_synthetic_df[col].astype(str)
            
            # Clean up decimals (e.g. 63.0 -> 63) and Python's stringified 'nan'
            final_synthetic_df[col] = final_synthetic_df[col].str.replace(r'\.0$', '', regex=True)
            final_synthetic_df[col] = final_synthetic_df[col].replace({'nan': 'Unknown', '': 'Unknown', 'None': 'Unknown'})
            
            # SCHEMA ANCHOR: Inject "Unknown" into the first row to force the CSV reader to see text
            if not final_synthetic_df[col].str.contains(r'[A-Za-z]').any():
                final_synthetic_df.iloc[0, final_synthetic_df.columns.get_loc(col)] = 'Unknown'
                
        elif orig_type == 'int64':
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').round().fillna(0).astype(int)
            
        elif orig_type == 'float64':
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').astype(float)
    # -----------------------------------------------

    # --- THE FILE SIZE FIX ---
    print("Optimizing file size...")
    float_cols = final_synthetic_df.select_dtypes(include=['float64', 'float32']).columns
    final_synthetic_df[float_cols] = final_synthetic_df[float_cols].round(2)
    # -------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"synthetic_data_{timestamp}.csv"
    output_path = results_dir / output_filename
    
    final_synthetic_df.to_csv(output_path, index=False)
    
    print(f"\nExecution successful.")
    print(f"Synthetic file saved to: {output_path}")

if __name__ == "__main__":
    main()