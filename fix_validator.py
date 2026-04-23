import pandas as pd
import glob
import os
from data_processor import align_datatypes_strictly

import re

import pandas as pd
import numpy as np

def format_synthetic_csv_for_portal(input_csv, output_csv):
    print("Formatting CSV for the final portal submission...")
    
    # Read the synthetic data as text first to avoid Pandas converting things to float
    df = pd.read_csv(input_csv, dtype=str)
    
    # 1. The Integer Columns (Must have NO decimals)
    # Year of Birth, Age, etc.
    integer_cols = ['Year of Birth::18', 'Age::40', 'Termination of smoking (no. of weeks before surgery)::25',
                    'Standard units per week::41', 'Termination of alcohol (no of weeks before surgery)::26',
                    'Length of incision (cm)::64', 'Time to passage of flatus (nights)::129',
                    'Time to passage of stool (nights)::131', 'Time to tolerating solid food (nights)::133',
                    'Time to termination of urinary drainage (nights)::141', 'Time to recovery of ADL ability (nights)::143',
                    'Time to termination of epidural analgesia (nights)::149', 'Time to pain control with oral analgesics (nights)::156',
                    'Length of stay (nights in hospital after primary operation)::179', 'Number of nights receiving intensive care::184',
                    'Time between operation and follow-up (nights)::235', 'Length of stay for readmissions::354',
                    'Total length of stay (nights)::353']

    for col in integer_cols:
        if col in df.columns:
            # Safely convert to float, then round, then convert to string integer (no .0)
            df[col] = pd.to_numeric(df[col], errors='coerce').round().astype('Int64').astype(str)
            df[col] = df[col].replace('<NA>', np.nan)

    # 2. The 1-Decimal Columns
    # Height, Core Body Temp
    one_decimal_cols = ['Height (cm)::23', 'Core body temperature at end of operation (°C)::95']
    for col in one_decimal_cols:
        if col in df.columns:
            # If it's a number, format it to exactly 1 decimal place (e.g. 162.1)
            numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
            df.loc[numeric_mask, col] = pd.to_numeric(df.loc[numeric_mask, col]).map('{:.1f}'.format)

    # 3. The 2-Decimal Columns
    # Weight, BMI, Weight Change
    two_decimal_cols = ['Weight 6 months prior to admission (kg)::21', 'Preoperative body weight (kg)::20', 'BMI::24',
                        'Morning weight - On postoperative day 1 (kg)::111', 'Weight change day 1 (kg)::112',
                        'Morning weight - On postoperative day 2 (kg)::113', 'Weight change day 2 (kg)::116',
                        'Morning weight - On postoperative day 3 (kg)::114', 'Weight change day 3 (kg)::117']
    for col in two_decimal_cols:
        if col in df.columns:
            numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
            df.loc[numeric_mask, col] = pd.to_numeric(df.loc[numeric_mask, col]).map('{:.2f}'.format)

    # 4. Clean up any 'nan' strings back to empty CSV cells
    df = df.replace('nan', '')

    # Save without the pandas index
    df.to_csv(output_csv, index=False)
    print(f"Successfully formatted and saved to {output_csv}")

# --- Run the Formatter ---
if __name__ == "__main__":
    input_file = "results/official_submission_ready.csv" # Your GAN output
    output_file = "results/FINAL_PORTAL_SUBMISSION.csv"  # The file you upload
    format_synthetic_csv_for_portal(input_file, output_file)
    

def main():
    print("Final Blueprint Alignment...")
    raw_data = pd.read_csv("data/data.csv", low_memory=False)
    
    list_of_files = glob.glob('results/synthetic_data_*.csv')
    if not list_of_files:
        latest_file = 'results/raw_gan_output.csv'
    else:
        latest_file = max(list_of_files, key=os.path.getctime)
        
    print(f"Processing: {latest_file}")
    df_synth = pd.read_csv(latest_file, low_memory=False)

    if all(c in df_synth.columns for c in ['BMI::24', 'Preoperative body weight (kg)::20', 'Height (cm)::23']):
        w = pd.to_numeric(df_synth['Preoperative body weight (kg)::20'], errors='coerce')
        h = pd.to_numeric(df_synth['Height (cm)::23'], errors='coerce') / 100
        df_synth['BMI::24'] = (w / (h**2)).round(2)

    final_output = align_datatypes_strictly(df_synth, raw_data)

    output_path = "results/official_submission_ready.csv"
    
    # The crucial part: save without pandas generating "nan" strings
    final_output.to_csv(output_path, index=False, na_rep='')
    
    print(f"\n[SUCCESS] File perfectly aligned! Mismatches eliminated.")

if __name__ == "__main__":
    main()
    # Change these paths to point to your actual files
    input_file = "results/official_submission_ready.csv" # Your GAN output
    output_file = "results/FINAL_PORTAL_SUBMISSION.csv"  # The file you upload
    format_synthetic_csv_for_portal(input_file, output_file)