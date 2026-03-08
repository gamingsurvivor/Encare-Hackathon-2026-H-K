import pandas as pd
import numpy as np

def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw dataset."""
    print(f"Reading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    return df

def preprocess_for_synthesis(df: pd.DataFrame, max_columns: int = 50) -> pd.DataFrame:
    
    "Placeholder for your data processing logic. This is where you can do feature selection, cleaning, and imputation before passing the data to the synthesis model."

    # --- 1. Feature selection, choose specific columns for the model to target ---
    key_columns = ['Age::40', 'Gender::5', 'BMI::24']

    available_cols = [c for c in key_columns if c in df.columns]
    
    # Create the new feature dataset
    df_subset = df[available_cols].copy()

    # --- 2. Data processing and cleaning (Replace with your own logic) ---
    
    # Example: Clip outliers so the the model doesn't learn "impossible" ages
    if 'Age::40' in df_subset.columns:
        df_subset['Age::40'] = df_subset['Age::40'].clip(lower=18, upper=100)

    # Example: Normalize categories so 'male' and 'Male' are the same
    if 'Gender::5' in df_subset.columns:
        df_subset['Gender::5'] = df_subset['Gender::5'].astype(str).str.upper().str.strip()

    # --- 3. Handling null-values ---
    
    # Another idea could be to handle null-values by filling them with a placeholder value or using imputation techniques.
    
    return df_subset