import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch


def load_data(filepath: str) -> pd.DataFrame:
    """Load the ERAS dataset and strip the trailing ghost column."""
    # Load the file
    df = pd.read_csv(filepath, sep=',', low_memory=False)

    return df

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_for_synthesis(df):
    print("Initial shape:", df.shape)
    df_clean = df.copy()

    # 1. Handle "Unknown" strings
    df_clean.replace('Unknown', np.nan, inplace=True)

    # 2. Convert Dates to Integers (Days since 1970-01-01)
    date_cols = [col for col in df_clean.columns if '(YYYY-MM-DD)' in col]
    reference_date = pd.Timestamp("1970-01-01")

    for col in date_cols:
        # Convert string to datetime, invalid parsing becomes NaN
        temp_dates = pd.to_datetime(df_clean[col], errors='coerce')
        # Calculate days since 1970. (Creates a float to allow NaNs)
        df_clean[col] = (temp_dates - reference_date) / pd.Timedelta(days=1)

    # 3. Convert Times to Integers (Minutes since midnight)
    time_cols = [col for col in df_clean.columns if '(HH:mm)' in col]
    
    for col in time_cols:
        temp_times = pd.to_datetime(df_clean[col], format='%H:%M', errors='coerce')
        # Convert to total minutes (e.g., 14:30 becomes 870)
        df_clean[col] = temp_times.dt.hour * 60 + temp_times.dt.minute

    # 4. Separate Columns by Type
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns

    for col in numerical_cols:
        median_val = df_clean[col].median()
        # If the column was 100% missing, the median is NaN. Fallback to 0.
        if pd.isna(median_val):
            df_clean[col] = df_clean[col].fillna(0)
        else:
            df_clean[col] = df_clean[col].fillna(median_val)

    # Fill categorical NaNs with a new category 'Missing'
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Missing')

    # --- NEW: PREVENT DIMENSIONALITY EXPLOSION ---
    # Drop categorical columns with more than 50 unique values (e.g., IDs, notes)
    cols_to_drop = []
    for col in categorical_cols:
        if df_clean[col].nunique() > 50:
            print(f"Dropping high-cardinality column to save memory: {col} ({df_clean[col].nunique()} unique values)")
            cols_to_drop.append(col)
            
    df_clean.drop(columns=cols_to_drop, inplace=True)
    # Update categorical_cols list so get_dummies doesn't look for dropped columns
    categorical_cols = [c for c in categorical_cols if c not in cols_to_drop]

    # 5. Encode Categorical Variables (One-Hot Encoding)
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=False)

    # 6. Encode Categorical Variables
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=False)
    
    # Convert booleans to 1s and 0s
    bool_cols = df_encoded.select_dtypes(include='bool').columns
    df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

    # 7. Scale Numerical Data to [0, 1]
    # THIS IS MANDATORY: Dates turned into integers will be ~20,000. 
    # PyTorch will crash if these aren't scaled down to between 0 and 1.
    scaler = MinMaxScaler()
    cols_to_scale = [col for col in df_encoded.columns if df_encoded[col].max() > 1 or df_encoded[col].min() < 0]
    
    # SAFETY CHECK: Only transform if there are actually columns to scale
    if len(cols_to_scale) > 0:
        df_encoded[cols_to_scale] = scaler.fit_transform(df_encoded[cols_to_scale])

    print("Final shape after encoding:", df_encoded.shape)
    
    # [!] FIX: Return cols_to_scale at the end
    return df_encoded, scaler, date_cols, time_cols, cols_to_scale


def postprocess_synthetic_data(synthetic_tensor, original_df, scaler, cols_to_scale, encoded_cols, categorical_cols, date_cols, time_cols):
    """
    Reverses the preprocessing steps to return a dataframe identical in structure to the original CSV.
    """
    # 1. Convert PyTorch tensor to Numpy, then to Pandas DataFrame
    # (Assuming synthetic_tensor is on CPU and detached)
    if isinstance(synthetic_tensor, torch.Tensor):
        synthetic_array = synthetic_tensor.cpu().detach().numpy()
    else:
        synthetic_array = synthetic_tensor
        
    df_synth = pd.DataFrame(synthetic_array, columns=encoded_cols)

    # 2. Reverse the MinMaxScaler for continuous columns
    if len(cols_to_scale) > 0:
        df_synth[cols_to_scale] = scaler.inverse_transform(df_synth[cols_to_scale])

    # 3. Reverse One-Hot Encoding for Categorical Variables
    for col in categorical_cols:
        prefix = f"{col}_"
        # Find all generated dummy columns associated with this original category
        dummy_cols = [c for c in encoded_cols if c.startswith(prefix)]
        
        if dummy_cols:
            # The GAN outputs probabilities (0 to 1). We pick the category with the highest probability.
            max_col_names = df_synth[dummy_cols].idxmax(axis=1)
            
            # Remove the prefix to get the actual category name back (e.g., 'Gender::5_Male' -> 'Male')
            df_synth[col] = max_col_names.str.replace(prefix, "", regex=False)
            
            # If the model chose the 'Missing' category we created, convert it back to NaN or 'Unknown'
            df_synth[col] = df_synth[col].replace('Missing', 'Unknown')
            
            # Drop the one-hot columns as we no longer need them
            df_synth = df_synth.drop(columns=dummy_cols)

    # 4. Reverse Dates (Days since 1970 back to YYYY-MM-DD)
    for col in date_cols:
        df_synth[col] = df_synth[col].round().astype(int)
        df_synth[col] = pd.to_datetime(df_synth[col], unit='D', origin='1970-01-01', errors='coerce').dt.strftime('%Y-%m-%d')
        # Handle potential NaNs generated by out-of-bounds math
        df_synth[col] = df_synth[col].fillna('Unknown')

    # 5. Reverse Times (Minutes since midnight back to HH:mm)
    for col in time_cols:
        df_synth[col] = df_synth[col].round().astype(int)
        df_synth[col] = df_synth[col].clip(0, 1439) # Force into a 24-hour clock
        
        hours = df_synth[col] // 60
        minutes = df_synth[col] % 60
        df_synth[col] = hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(2)

    # 6. Reorder columns to exactly match the original CSV headers
    # Drop any leftover encoded columns that shouldn't be in the final output
    final_columns = original_df.columns.tolist()
    df_final = df_synth[final_columns]
    
    # 7. Final cleanup: Ensure original numerical columns don't have bizarre decimal places (e.g., Age 63.45 -> 63)
    # We round float columns that were originally integers if needed, but keeping them as floats is generally safer for CSV export
    
    return df_final

