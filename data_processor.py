import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_for_synthesis(df):
    print("Initial shape:", df.shape)
    df_clean = df.copy()

    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    date_cols = [col for col in df_clean.columns if '(YYYY-MM-DD)' in col]
    reference_date = pd.Timestamp("1970-01-01")
    for col in date_cols:
        temp_dates = pd.to_datetime(df_clean[col], errors='coerce')
        df_clean[col] = (temp_dates - reference_date) / pd.Timedelta(days=1)
        if col in categorical_cols: categorical_cols.remove(col)
        if col not in numerical_cols: numerical_cols.append(col)

    time_cols = [col for col in df_clean.columns if '(HH:mm)' in col]
    for col in time_cols:
        temp_times = pd.to_datetime(df_clean[col], format='%H:%M', errors='coerce')
        df_clean[col] = temp_times.dt.hour * 60 + temp_times.dt.minute
        if col in categorical_cols: categorical_cols.remove(col)
        if col not in numerical_cols: numerical_cols.append(col)

    # --- THE MASKING FIX FOR NUMERICAL DATA ---
    missing_flags = []
    for col in numerical_cols:
        if df_clean[col].isna().any():
            # 1. Create a binary mask (1 = hidden/missing, 0 = has value)
            flag_col = f"{col}_missing_flag"
            df_clean[flag_col] = df_clean[col].isna().astype(float)
            missing_flags.append(flag_col)
            
            # 2. Temporarily fill with median to fix the scaling distortion
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(0 if pd.isna(median_val) else median_val)

    # Categorical Missingness (Leave as explicit token)
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('<BLANK>')

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # Scale EVERYTHING (Including our new flags)
    scaler = MinMaxScaler()
    all_cols = df_clean.columns.tolist()
    df_clean[all_cols] = scaler.fit_transform(df_clean[all_cols])

    print("Final shape after Masking & Scaling:", df_clean.shape)

    return df_clean, scaler, date_cols, time_cols, label_encoders, categorical_cols, missing_flags


def postprocess_synthetic_data(synthetic_tensor, original_columns, scaler, date_cols, time_cols, label_encoders, categorical_cols, missing_flags):
    # Rebuild dataframe with the extra flag columns
    all_cols = original_columns + missing_flags
    df_synth = pd.DataFrame(synthetic_tensor, columns=all_cols)

    # 1. Reverse the 0-1 Scaling
    df_synth[all_cols] = scaler.inverse_transform(df_synth[all_cols])

    # 2. ENFORCE THE RUBRIC: Restore structural missingness perfectly
    for flag_col in missing_flags:
        orig_col = flag_col.replace("_missing_flag", "")
        
        # If the GAN learned the flag > 0.5, the logic dictates this cell should be blank
        df_synth.loc[df_synth[flag_col] > 0.5, orig_col] = np.nan
        
    # Drop the temporary flags so they don't export to the CSV
    df_synth = df_synth[original_columns]

    # Clean up numerical formatting
    numerical_cols = [col for col in original_columns if col not in categorical_cols and col not in date_cols and col not in time_cols]
    for col in numerical_cols:
        df_synth[col] = df_synth[col].round(2)

    # 3. Reverse Tokenization
    for col in categorical_cols:
        if col in label_encoders and col in df_synth.columns:
            le = label_encoders[col]
            max_class_id = len(le.classes_) - 1
            df_synth[col] = df_synth[col].round().clip(0, max_class_id).astype(int)
            
            df_synth[col] = le.inverse_transform(df_synth[col])
            # Restore true blanks
            df_synth[col] = df_synth[col].replace('<BLANK>', np.nan)

    # 4. Reverse Dates
    for col in date_cols:
        if col in df_synth.columns:
            valid_mask = df_synth[col].notna()
            df_synth[col] = df_synth[col].astype(object)
            dates = pd.to_datetime(df_synth.loc[valid_mask, col].round().astype(int), unit='D', origin='1970-01-01', errors='coerce').dt.strftime('%Y-%m-%d')
            df_synth.loc[valid_mask, col] = dates

    # 5. Reverse Times
    for col in time_cols:
        if col in df_synth.columns:
            valid_mask = df_synth[col].notna()
            df_synth[col] = df_synth[col].astype(object)
            valid_times = df_synth.loc[valid_mask, col].round().astype(int).clip(0, 1439)
            hours = valid_times // 60
            minutes = valid_times % 60
            df_synth.loc[valid_mask, col] = hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(2)

    return df_synth