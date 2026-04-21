import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from dotenv import load_dotenv

from data_processor import load_data, preprocess_for_synthesis, postprocess_synthetic_data
from Discriminator import Discriminator
from Generative import Generator, generate_synthetic_samples

load_dotenv()

def main():
    data_dir = Path("data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    input_file = data_dir / "data.csv"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found. Please place the CSV in the 'data' folder.")
        return

    raw_data = load_data(str(input_file))
    print(f"Original Data Shape: {raw_data.shape}")

    print("\nPreprocessing data...")
    df_encoded, scaler, date_cols, time_cols, label_encoders, categorical_cols, missing_flags = preprocess_for_synthesis(raw_data)
    original_columns = raw_data.columns.tolist()
    
    # ==========================================
    # Phase 2: GAN Setup & PyTorch DataLoaders
    # ==========================================
    TARGET_CONDITION = 'Complications at all during primary stay::183'
    condition_cols = [TARGET_CONDITION]
    feature_cols = [col for col in df_encoded.columns.tolist() if col != TARGET_CONDITION]
    
    data_dim = len(feature_cols)
    condition_dim = len(condition_cols)
    latent_dim = 256 
    pack_size = 8

    # --- THE GPU UPGRADE ---
    # Detect Nvidia (cuda), Apple Silicon (mps), or fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nInitializing CTGAN-Lite Models on: {device.type.upper()}...")
    print(f"Features Dim: {data_dim} | Condition Dim: {condition_dim} | Latent Dim: {latent_dim}")

    # Initialize and immediately push models to the GPU
    discriminator = Discriminator(data_dim, condition_dim, pack_size=pack_size).to(device)
    generator = Generator(latent_dim, condition_dim, data_dim).to(device)

    # Push the entire training dataset into GPU memory (tabular data is small enough to fit)
    X_train = torch.tensor(df_encoded[feature_cols].values, dtype=torch.float32).to(device)
    C_train = torch.tensor(df_encoded[condition_cols].values, dtype=torch.float32).to(device)
    # -----------------------

    batch_size = 64
    dataset = TensorDataset(X_train, C_train)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) 

    lr_D = 0.0002
    lr_G = 0.0001
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    loss_function = nn.BCELoss()
    num_epochs = 300
    print("\nStarting CTGAN-Lite Training...")
    
    for epoch in range(num_epochs):
        for n, (real_samples, conditions) in enumerate(train_loader):
            current_batch_size = real_samples.size(0)
            packed_batch_size = current_batch_size // pack_size
            
           # --- Train Discriminator ---
            discriminator.zero_grad()
            
            # --- GPU UPGRADE: Send noise and labels to device ---
            latent_space_samples = torch.randn(current_batch_size, latent_dim).to(device)
            generated_samples = generator(latent_space_samples, conditions)
            
            real_labels = (torch.ones(packed_batch_size, 1) * 0.9).to(device)
            fake_labels = (torch.zeros(packed_batch_size, 1) + 0.1).to(device)
            # ----------------------------------------------------
            
            output_real = discriminator(real_samples, conditions)
            output_fake = discriminator(generated_samples.detach(), conditions)
            
            loss_real = loss_function(output_real, real_labels)
            loss_fake = loss_function(output_fake, fake_labels)
            loss_discriminator = (loss_real + loss_fake) / 2
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # --- Train Generator ---
            generator.zero_grad()
            output_discriminator_generated = discriminator(generated_samples, conditions)
            loss_generator = loss_function(output_discriminator_generated, real_labels)
            loss_generator.backward()
            optimizer_generator.step()

        if epoch % 10 == 0:
            print(f"Epoch: [{epoch}/{num_epochs}] | Loss D: {loss_discriminator.item():.4f} | Loss G: {loss_generator.item():.4f}")

    print("\nGenerating synthetic data...")
    num_samples_to_generate = 8000
    
    synthetic_features, synthetic_conditions = generate_synthetic_samples(
        generator_model=generator, 
        num_samples=num_samples_to_generate, 
        latent_dim=latent_dim, 
        condition_dim=condition_dim,
        device=device
    )

    df_synth_features = pd.DataFrame(synthetic_features.cpu().numpy(), columns=feature_cols)
    df_synth_conditions = pd.DataFrame(synthetic_conditions.cpu().numpy(), columns=condition_cols)
    
    df_synth_encoded = pd.concat([df_synth_features, df_synth_conditions], axis=1)
    all_encoded_cols = original_columns + missing_flags
    df_synth_encoded = df_synth_encoded[all_encoded_cols]

    print("Reversing data to original CSV format...")
    final_synthetic_df = postprocess_synthetic_data(
        synthetic_tensor=df_synth_encoded.values,
        original_columns=original_columns,
        scaler=scaler,
        date_cols=date_cols,
        time_cols=time_cols,
        label_encoders=label_encoders,
        categorical_cols=categorical_cols,
        missing_flags=missing_flags
    )

    print("Matching original data types for submission validator...")

    force_to_str = {
        # Stubborn continuous & clinical columns
        'Preoperative body weight (kg)::20', 'Height (cm)::23', 'Intraoperative blood loss (ml)::69',
        'Core body temperature at end of operation (°C)::95', 'Morning weight - On postoperative day 1 (kg)::111',
        'Morning weight - On postoperative day 2 (kg)::113', 'Oral fluids, total volume taken - On postoperative day 1 (ml)::119',
        'Oral nutritional supplements, energy intake - On day of surgery, postoperatively (kCal)::122',
        'Was iron replacement treatment given?::38',
        # ---> THE NEW NUTRITION & VAS HOLDOUTS <---
        'Oral nutritional supplements, energy intake - On postoperative day 1 (kCal)::123',
        'Patient-reported maximum pain (VAS) - On day of surgery (cm)::158',
        'Patient-reported maximum nausea (VAS) - On day of surgery (cm)::162',
        'Patient-reported maximum pain (VAS) - On postoperative day 1 (cm)::159',
        'Termination of smoking (no. of weeks before surgery)::25', 'Standard units per week::41',
        'Termination of alcohol (no of weeks before surgery)::26', 'Last HbA1c value (Unknown)::28',
        'Distance from anal verge::1840', 'Level of insertion::89', 
        'IV volume of crystalloids intraoperatively (ml)::97', 'IV volume of colloids intraoperatively (ml)::99',
        'IV volume of blood products intra-operatively (ml)::100', 
        'Intravenous fluids, volume infused - On day of surgery, postoperatively (ml)::106',
        'Mobilisation - On postoperative day 1::137', 'Strong opioids given within 48 hrs postoperatively::152',
        'Successful block?::150', 'Use of peripheral opioid receptor antagonist::153',
        'Patient-reported maximum nausea (VAS) - On postoperative day 1 (cm)::163',
        'Grading of most severe complication::186', 'Urinary tract infection::203',
        'Grading of most severe complication::290', 'Date of last chemotherapy treatment (YYYY-MM-DD)::29',
        'Termination of epidural analgesia (YYYY-MM-DD)::147', 'Nasogastric tube reinserted date (YYYY-MM-DD)::461',
        
        # SURGICAL COLUMNS
        'Type of anastomosis::67', 'Anastomotic technique::68', 'Length of incision (cm)::64',

        # Complication columns — primary stay
        'Lobar atelectasis::190', 'Pneumonia::191', 'Pleural Fluid::192', 'Respiratory failure::193',
        'Pneumothorax::194', 'Other respiratory complication::195', 'Wound Infection::204',
        'Intraperitoneal or retroperitoneal abscess::202', 'Sepsis::201', 'Septic Shock::200',
        'Infected graft or prosthesis::199', 'Other infectious complication::198', 'Heart Failure::214',
        'Acute Myocardial Infarction::213', 'Deep Venous Thrombosis::212', 'Portal Vein Thrombosis::211',
        'Pulmonary Embolus::210', 'Cerebrovascular lesion::209', 'Cardiac arrhythmia::208',
        'Cardiac arrest::207', 'Other cardiovascular complication::206', 'Renal dysfunction::228',
        'Urinary retention::226', 'Hepatic dysfunction::225', 'Pancreatitis::220',
        'Gastrointestinal haemorrhage::219', 'Nausea or vomiting::218', 'Obstipation or diarrhoea::217',
        'Other organ dysfunction::216', 'Anastomotic leak::244', 'Urinary tract injury::243',
        'Mechanical bowel obstruction::241', 'Postoperative paralytic ileus::240', 'Deep wound dehiscence::239',
        'Intraoperative excessive haemorrhage::237', 'Postoperative excessive haemorrhage::236',
        'Other surgical technical complication or injury::234', 'Hematoma::233', 'Post dural-puncture headache::249',
        'Epidural hematoma or abscess::248', 'Other EDA or spinal related complication::247',
        'Pulmonary aspiration of gastric contents::257', 'Hypotension::256', 'Hypoxia::255',
        'Prolonged postoperative sedation::251', 'Other anaesthetic complication(s)::253', 'Asthenia or tiredness::260',
        'Re-operation(s)::286', 
        
        # Complication columns — readmission / follow-up
        'Infectious complication(s)::312',
        'Respiratory complication(s)::297', 'Lobar atelectasis::300', 'Wound Infection::323',
        'Urinary tract infection::320', 'Intraperitoneal or retroperitoneal abscess::317', 'Sepsis::319',
        'Septic Shock::318', 'Infected graft or prosthesis::314', 'Other infectious complication::315',
        'Cardiovascular complication(s)::282', 'Acute myocardial infarction::288', 'Cardiac arrhythmia::295',
        'Renal, hepatic, pancreatic and gastrointestinal complication(s)::298', 'Renal dysfunction::299',
        'Urinary retention::352', 'Hepatic dysfunction::302', 'Pancreatitis::304', 'Gastrointestinal haemorrhage::306',
        'Nausea or vomiting::310', 'Obstipation or diarrhoea::311', 'Incontinence::313', 'Other organ dysfunction::309',
        'Surgical complication(s)::325', 'Anastomotic leak::324', 'Urinary tract injury::328',
        'Mechanical bowel obstruction::322', 'Postoperative paralytic ileus::321', 'Deep wound dehiscence::340',
        'Intraoperative excessive haemorrhage::339', 'Postoperative excessive haemorrhage::338',
        'Other surgical technical complication or injury::337', 'Hematoma::336',
        'Complication(s) related to epidural or spinal anaesthesia::326', 'Anaesthetic complication(s)::331',
        'Psychiatric complication(s)::343', 'Asthenia or tiredness::344', 'Pain::342', 'Injuries::345', 'Other::347',
    }

    force_to_float = {
        'Number of nights receiving intensive care::284', 'Pneumonia::301', 'Pleural Fluid::305',
        'Respiratory failure::308', 'Pneumothorax::307', 'Other respiratory complication::303',
        'Heart failure::287', 'Deep venous thrombosis::285', 'Portal Vein Thrombosis::289',
        'Pulmonary embolus::291', 'Cerebrovascular lesion::294', 'Cardiac arrest::296',
        'Hypertension::316', 'Other cardiovascular complication::292', 'Post dural-puncture headache::327',
        'Epidural hematoma or abscess::329', 'Other EDA or spinal related complication::330',
    }

    for col in final_synthetic_df.columns:
        orig_type = raw_data[col].dtype

        if col in force_to_float:
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').astype(float)

        elif orig_type == 'object' or orig_type.name == 'category' or col in force_to_str:
            final_synthetic_df[col] = final_synthetic_df[col].astype(str)
            final_synthetic_df[col] = final_synthetic_df[col].str.replace(r'\.0$', '', regex=True)
            final_synthetic_df[col] = final_synthetic_df[col].replace({'nan': '', 'None': ''})

            if not final_synthetic_df[col].str.contains(r'[A-Za-z]').any():
                final_synthetic_df.iloc[0, final_synthetic_df.columns.get_loc(col)] = 'Unknown'

        elif orig_type == 'int64':
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').round().fillna(0).astype(int)

        elif orig_type == 'float64':
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').astype(float)

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