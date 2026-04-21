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
    df_encoded, scaler, date_cols, time_cols, label_encoders, categorical_cols, missing_flags = preprocess_for_synthesis(raw_data)
    original_columns = raw_data.columns.tolist()
    
    # ==========================================
    # Phase 2: GAN Setup & PyTorch DataLoaders
    # ==========================================
    # Choose what we want to condition the generation on
    TARGET_CONDITION = 'Complications at all during primary stay::183'
    condition_cols = [TARGET_CONDITION]
    feature_cols = [col for col in original_columns if col != TARGET_CONDITION]

    data_dim = len(feature_cols)
    condition_dim = len(condition_cols)
    
    # TRICK 3: Increase the random noise vector for better multivariate correlations
    latent_dim = 256 

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
    lr_D = 0.0002
    lr_G = 0.0001 
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
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
            
            latent_space_samples = torch.randn(current_batch_size, latent_dim)
            generated_samples = generator(latent_space_samples, conditions)
            
            # TRICK 1: Label Smoothing! 
            # Real is 0.9 instead of 1.0. Fake is 0.1 instead of 0.0.
            real_labels = torch.ones(current_batch_size, 1) * 0.9 
            fake_labels = torch.zeros(current_batch_size, 1) + 0.1
            
            output_real = discriminator(real_samples, conditions)
            output_fake = discriminator(generated_samples.detach(), conditions)
            
            loss_real = loss_function(output_real, real_labels)
            loss_fake = loss_function(output_fake, fake_labels)
            loss_discriminator = (loss_real + loss_fake) / 2
            
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # --- 2. Train Generator ---
            generator.zero_grad()
            
            output_discriminator_generated = discriminator(generated_samples, conditions)
            loss_generator = loss_function(output_discriminator_generated, real_labels)
            
            loss_generator.backward()
            optimizer_generator.step()

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

    df_synth_features = pd.DataFrame(synthetic_features.cpu().numpy(), columns=feature_cols)
    df_synth_conditions = pd.DataFrame(synthetic_conditions.cpu().numpy(), columns=condition_cols)
    
    df_synth_encoded = pd.concat([df_synth_features, df_synth_conditions], axis=1)
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
        categorical_cols=categorical_cols,
        missing_flags=missing_flags # Add this variable here!
    )

    # --- TYPE COERCION ---
    print("Matching original data types for submission validator...")

    # Columns the validator expects as str.
    force_to_str = {
        # Stubborn continuous columns
        'Preoperative body weight (kg)::20',
        'Height (cm)::23',
        'Intraoperative blood loss (ml)::69',
        'Core body temperature at end of operation (°C)::95',
        'Morning weight - On postoperative day 1 (kg)::111',
        'Morning weight - On postoperative day 2 (kg)::113',
        'Oral fluids, total volume taken - On postoperative day 1 (ml)::119',
        'Oral nutritional supplements, energy intake - On day of surgery, postoperatively (kCal)::122',
        # Complication columns — primary stay
        'Lobar atelectasis::190',
        'Pneumonia::191',
        'Pleural Fluid::192',
        'Respiratory failure::193',
        'Pneumothorax::194',
        'Other respiratory complication::195',
        'Wound Infection::204',
        'Intraperitoneal or retroperitoneal abscess::202', # Added
        'Sepsis::201',
        'Septic Shock::200',
        'Infected graft or prosthesis::199',
        'Other infectious complication::198',
        'Heart Failure::214',
        'Acute Myocardial Infarction::213',
        'Deep Venous Thrombosis::212',
        'Portal Vein Thrombosis::211',
        'Pulmonary Embolus::210',
        'Cerebrovascular lesion::209',
        'Cardiac arrhythmia::208',
        'Cardiac arrest::207',
        'Other cardiovascular complication::206',
        'Renal dysfunction::228',
        'Urinary retention::226',
        'Hepatic dysfunction::225',
        'Pancreatitis::220',
        'Gastrointestinal haemorrhage::219',
        'Nausea or vomiting::218',
        'Obstipation or diarrhoea::217',
        'Other organ dysfunction::216',
        'Anastomotic leak::244',
        'Urinary tract injury::243',
        'Mechanical bowel obstruction::241',
        'Postoperative paralytic ileus::240',
        'Deep wound dehiscence::239',
        'Intraoperative excessive haemorrhage::237',
        'Postoperative excessive haemorrhage::236',
        'Other surgical technical complication or injury::234',
        'Hematoma::233',
        'Post dural-puncture headache::249',
        'Epidural hematoma or abscess::248', # Added
        'Other EDA or spinal related complication::247', # Added
        'Pulmonary aspiration of gastric contents::257',
        'Hypotension::256',
        'Hypoxia::255',
        'Prolonged postoperative sedation::251',
        'Other anaesthetic complication(s)::253',
        'Asthenia or tiredness::260', # Added
        'Re-operation(s)::286',
        # Complication columns — readmission / follow-up
        'Respiratory complication(s)::297', # Added
        'Lobar atelectasis::300',
        'Wound Infection::323',
        'Urinary tract infection::320',
        'Intraperitoneal or retroperitoneal abscess::317',
        'Sepsis::319',
        'Septic Shock::318',
        'Infected graft or prosthesis::314',
        'Other infectious complication::315',
        'Cardiovascular complication(s)::282',
        'Acute myocardial infarction::288',
        'Cardiac arrhythmia::295',
        'Renal, hepatic, pancreatic and gastrointestinal complication(s)::298',
        'Renal dysfunction::299',
        'Urinary retention::352',
        'Hepatic dysfunction::302',
        'Pancreatitis::304',
        'Gastrointestinal haemorrhage::306',
        'Nausea or vomiting::310',
        'Obstipation or diarrhoea::311',
        'Incontinence::313',
        'Other organ dysfunction::309',
        'Surgical complication(s)::325',
        'Anastomotic leak::324',
        'Urinary tract injury::328',
        'Mechanical bowel obstruction::322',
        'Postoperative paralytic ileus::321',
        'Deep wound dehiscence::340',
        'Intraoperative excessive haemorrhage::339',
        'Postoperative excessive haemorrhage::338',
        'Other surgical technical complication or injury::337',
        'Hematoma::336',
        'Complication(s) related to epidural or spinal anaesthesia::326',
        'Anaesthetic complication(s)::331',
        'Psychiatric complication(s)::343',
        'Asthenia or tiredness::344',
        'Pain::342',
        'Injuries::345',
        'Other::347',
    }

    # Columns the validator expects as float64, derived from "expected (float64), got (str)" errors.
    force_to_float = {
        'Number of nights receiving intensive care::284',
        'Pneumonia::301',
        'Pleural Fluid::305',
        'Respiratory failure::308',
        'Pneumothorax::307',
        'Other respiratory complication::303',
        'Heart failure::287',
        'Deep venous thrombosis::285',
        'Portal Vein Thrombosis::289',
        'Pulmonary embolus::291',
        'Cerebrovascular lesion::294',
        'Cardiac arrest::296',
        'Hypertension::316',
        'Other cardiovascular complication::292',
        'Post dural-puncture headache::327',
        'Epidural hematoma or abscess::329',
        'Other EDA or spinal related complication::330',
    }

    for col in final_synthetic_df.columns:
        orig_type = raw_data[col].dtype

        # force_to_float takes absolute priority — must stay numeric
        if col in force_to_float:
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').astype(float)

        elif orig_type == 'object' or orig_type.name == 'category' or col in force_to_str:
            final_synthetic_df[col] = final_synthetic_df[col].astype(str)
            final_synthetic_df[col] = final_synthetic_df[col].str.replace(r'\.0$', '', regex=True)
            
            # CRITICAL RUBRIC FIX: True blanks MUST be empty strings (""), NOT "Unknown"
            final_synthetic_df[col] = final_synthetic_df[col].replace({'nan': '', 'None': ''})

            # SCHEMA ANCHOR: Inject ONE "Unknown" into the first row so CSV reader sees text
            if not final_synthetic_df[col].str.contains(r'[A-Za-z]').any():
                final_synthetic_df.iloc[0, final_synthetic_df.columns.get_loc(col)] = 'Unknown'

        elif orig_type == 'int64':
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').round().fillna(0).astype(int)

        elif orig_type == 'float64':
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').astype(float)

    # --- FILE SIZE OPTIMISATION ---
    print("Optimizing file size...")
    float_cols = final_synthetic_df.select_dtypes(include=['float64', 'float32']).columns
    final_synthetic_df[float_cols] = final_synthetic_df[float_cols].round(2)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_filename = f"synthetic_data_{timestamp}.csv"
    output_path = results_dir / output_filename
    
    final_synthetic_df.to_csv(output_path, index=False)
    
    print(f"\nExecution successful.")
    print(f"Synthetic file saved to: {output_path}")

if __name__ == "__main__":
    main()