import pandas as pd
import glob
import os
from data_processor import align_datatypes_strictly

import re
import numpy as np

def format_synthetic_csv_for_portal(input_csv, output_csv):
    print("Forging CSV formatting for final portal submission (Preserving Text)...")
    
    # Read as pure strings so Pandas doesn't auto-guess the types yet
    df = pd.read_csv(input_csv, dtype=str)
    
    def safe_format(val, fmt_type):
        """Safely formats numbers while PROTECTING text like 'Unknown'"""
        if pd.isna(val) or str(val).strip().lower() in ['nan', 'none', '<na>', '']:
            return ''
        
        val_str = str(val).strip()
        
        # BODYGUARD: If it's text (like "Unknown", "Not applicable"), DO NOT touch it!
        if any(c.isalpha() for c in val_str):
            return val_str
            
        try:
            num = float(val)
            if fmt_type == 'dot_zero':
                return f"{num:.1f}" # e.g. 1951.0
            elif fmt_type == 'integer':
                return f"{int(round(num))}" # e.g. 4
            elif fmt_type == 'one_decimal':
                return f"{num:.1f}" # e.g. 2.3
            elif fmt_type == 'two_decimal':
                return f"{num:.2f}" # e.g. 72.69
            else:
                return val_str
        except ValueError:
            return val_str

    # 1. The ".0" Columns (Integers that the real dataset formatted with .0)
    dot_zero_cols = [
        'Year of Birth::18', 'Age::40', 'Time to passage of flatus (nights)::129', 
        'Time to passage of stool (nights)::131', 'Time to tolerating solid food (nights)::133', 
        'Time to termination of urinary drainage (nights)::141', 'Time to recovery of ADL ability (nights)::143', 
        'Time to termination of epidural analgesia (nights)::149', 'Time to pain control with oral analgesics (nights)::156', 
        'Length of stay (nights in hospital after primary operation)::179', 'Number of nights receiving intensive care::184', 
        'Time between operation and follow-up (nights)::235', 'Length of stay for readmissions::354', 
        'Total length of stay (nights)::353'
    ]
    
    # 2. Strict Integer Columns (No decimals allowed!)
    integer_cols = [
        'Termination of smoking (no. of weeks before surgery)::25',
        'Termination of alcohol (no of weeks before surgery)::26',
        'Length of incision (cm)::64'
    ]

    # 3. One Decimal Columns
    one_decimal_cols = [
        'Height (cm)::23', 'Core body temperature at end of operation (°C)::95',
        'Standard units per week::41'
    ]

    # 4. Two Decimal Columns
    two_decimal_cols = [
        'Weight 6 months prior to admission (kg)::21', 'Preoperative body weight (kg)::20', 'BMI::24',
        'Morning weight - On postoperative day 1 (kg)::111', 'Weight change day 1 (kg)::112',
        'Morning weight - On postoperative day 2 (kg)::113', 'Weight change day 2 (kg)::116',
        'Morning weight - On postoperative day 3 (kg)::114', 'Weight change day 3 (kg)::117',
        'Intraoperative blood loss (ml)::69', 'IV volume of crystalloids intraoperatively (ml)::97',
        'IV volume of colloids intraoperatively (ml)::99', 'IV volume of blood products intra-operatively (ml)::100',
        'Total IV volume of fluids intra-operatively (ml)::101'
    ]

    # Apply the safe formatter to each column group
    for col in dot_zero_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_format(x, 'dot_zero'))
            
    for col in integer_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_format(x, 'integer'))

    for col in one_decimal_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_format(x, 'one_decimal'))

    for col in two_decimal_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_format(x, 'two_decimal'))

    # Clean up any lingering 'nan' strings back to empty CSV cells
    df = df.replace(['nan', '<NA>', 'None'], '')

    # Save to final output
    df.to_csv(output_csv, index=False)
    print(f"Successfully forged formatting. Ready for submission: {output_csv}")
    

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
    format_synthetic_csv_for_portal("results/official_submission_ready.csv", "results/FINAL_PORTAL_SUBMISSION.csv")
