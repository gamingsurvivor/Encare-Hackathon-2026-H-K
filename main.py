import datetime
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from data_processor import load_data, preprocess_for_synthesis
from validator import run_evaluation_report
from approaches.example import run_random_sample


load_dotenv()

def main():
    # Define directory paths using pathlib for cross-platform compatibility
    data_dir = Path("data")
    results_dir = Path("results")

    # File paths
    input_file = data_dir / "synthetic-data-hackaton-sample.csv"
    
    # Phase 1: Data loading and cleaning
    if not input_file.exists():
        print(f"Error: {input_file} not found. Please place the CSV in the 'data' folder.")
        return

    raw_data = load_data(str(input_file))
    df_clean = preprocess_for_synthesis(raw_data, max_columns=20)

    # Phase 2: Synthetic Data Generation
    # Gaussian Copula is used as a baseline approach
    print("\nGenerate synthetic data: ")
    synthetic_df = run_random_sample(df_clean, num_samples=200)

    # Phase 3: Quality Assurance
    # Compares the cleaned source data against the generated output
    run_evaluation_report(raw_data, synthetic_df)

    # Phase 4: Export
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Create the unique filename
    output_filename = f"synthetic_records_{timestamp}.csv"
    output_path = results_dir / output_filename

    # Save to the specific folder
    synthetic_df.to_csv(output_path, index=False)

    print(f"\nExecution successful.")
    print(f"Synthetic file saved to: {output_path}")

if __name__ == "__main__":
    main()