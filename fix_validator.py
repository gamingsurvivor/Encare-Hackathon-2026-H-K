import pandas as pd
import numpy as np
import glob
import os

def main():
    print("Loading datasets for Fully Autonomous Alignment...")
    # The blueprint
    raw_data = pd.read_csv("data/data.csv", low_memory=False)
    
    # Locate the most recent file in the results folder automatically
    list_of_files = glob.glob('results/synthetic_data_*.csv')
    if not list_of_files:
        print("Could not find a generated CSV in results/ folder!")
        return
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Aligning the newest file: {latest_file}")
    
    final_synthetic_df = pd.read_csv(latest_file, low_memory=False)

    print("Mirroring original schema automatically...")
    
    for col in final_synthetic_df.columns:
        if col not in raw_data.columns:
            continue
            
        orig_type = raw_data[col].dtype

        # CASE 1: Original was Text/Categorical
        if orig_type == 'object' or orig_type.name == 'category':
            final_synthetic_df[col] = final_synthetic_df[col].astype(str)
            final_synthetic_df[col] = final_synthetic_df[col].str.replace(r'\.0$', '', regex=True)
            final_synthetic_df[col] = final_synthetic_df[col].replace({'nan': '', 'None': '', '<BLANK>': '', 'NaN': ''})
            
            # THE CSV SURVIVAL TRICK: Auto-inject 'Unknown' if the text column is just numbers
            if not final_synthetic_df[col].str.contains(r'[A-Za-z]').any():
                final_synthetic_df.iloc[0, final_synthetic_df.columns.get_loc(col)] = 'Unknown'
                
        # CASE 2: Original was Whole Numbers
        elif pd.api.types.is_integer_dtype(orig_type):
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').round().fillna(0).astype(int)

        # CASE 3: Original was Decimals
        elif pd.api.types.is_float_dtype(orig_type):
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').astype(float)

    output_path = "results/official_submission_ready.csv"
    final_synthetic_df.to_csv(output_path, index=False)
    print(f"\nDone! Clean file generated: {output_path}")
    print(">>> UPLOAD THIS FILE TO THE PORTAL: official_submission_ready.csv")

if __name__ == "__main__":
    main()