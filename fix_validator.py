import pandas as pd
import glob
import os
from data_processor import align_datatypes_strictly

import re

def scrub_zero_decimals(input_path, output_path):
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # THE MAGIC REGEX:
    # \b(\d+)   : Captures the whole number part (e.g., "1938" or "81")
    # \.0+      : Matches exactly ".0" (or ".00")
    # (?=[,\n]) : Ensures the number is immediately followed by a comma or a new line
    cleaned_text = re.sub(r'\b(\d+)\.0+(?=[,\n\r])', r'\1', text)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)
        
    print(f"Scrub complete! Saved strictly formatted data to {output_path}")

    

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
    input_csv = "results/official_submission_ready.csv"
    output_csv = "results/final_submission_scrubbed.csv"
    
    scrub_zero_decimals(input_csv, output_csv)