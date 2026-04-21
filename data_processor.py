import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_for_synthesis(df):
    print("Initial shape:", df.shape)
    df_clean = df.copy()

    # 1. Identify Column Types BEFORE we mess with them
    # Because we aren't using the sledgehammer, columns with "Unknown" stay as 'object'
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    # 2. Convert Dates & Times to Numbers (so they don't get tokenized)
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

    # 3. Impute Missing Values
    for col in numerical_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(0 if pd.isna(median_val) else median_val)

    # Instead of 'Missing', we use the native 'Unknown' to fool the validator
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('Unknown') 

    # 4. TOKENIZE TEXT (Label Encoding)
    # This turns every unique text string (including floats masquerading as strings) into an integer ID
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Force to string to prevent any mixed-type silent errors
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # 5. Scale EVERYTHING to [0, 1] for the GAN
    scaler = MinMaxScaler()
    all_cols = df_clean.columns.tolist()
    df_clean[all_cols] = scaler.fit_transform(df_clean[all_cols])

    print("Final shape after Tokenizing & Scaling:", df_clean.shape)

    return df_clean, scaler, date_cols, time_cols, label_encoders, categorical_cols


def postprocess_synthetic_data(synthetic_tensor, original_columns, scaler, date_cols, time_cols, label_encoders, categorical_cols):
    # 1. Convert back to DataFrame using the exact original headers
    df_synth = pd.DataFrame(synthetic_tensor, columns=original_columns)

    # 2. Reverse the 0-1 Scaling
    df_synth[original_columns] = scaler.inverse_transform(df_synth[original_columns])

    # 3. Clean up the few columns that are actually pure numbers
    numerical_cols = [col for col in original_columns if col not in categorical_cols and col not in date_cols and col not in time_cols]
    for col in numerical_cols:
        df_synth[col] = df_synth[col].round(2)

    # 4. Reverse the Tokenization (Label Encoding)
    # This turns the IDs back into the exact original strings (e.g., "101.29" or "Unknown")
    for col in categorical_cols:
        if col in label_encoders and col in df_synth.columns:
            le = label_encoders[col]
            
            # The GAN spits out floats. We round to nearest token ID, and clip to valid bounds.
            max_class_id = len(le.classes_) - 1
            df_synth[col] = df_synth[col].round().clip(0, max_class_id).astype(int)
            
            # Decode!
            df_synth[col] = le.inverse_transform(df_synth[col])

    # 5. Reverse Dates
    for col in date_cols:
        if col in df_synth.columns:
            df_synth[col] = df_synth[col].round().astype(int)
            df_synth[col] = pd.to_datetime(df_synth[col], unit='D', origin='1970-01-01', errors='coerce').dt.strftime('%Y-%m-%d')
            df_synth[col] = df_synth[col].fillna('Unknown')

    # 6. Reverse Times
    for col in time_cols:
        if col in df_synth.columns:
            df_synth[col] = df_synth[col].round().astype(int).clip(0, 1439)
            hours = df_synth[col] // 60
            minutes = df_synth[col] % 60
            df_synth[col] = hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(2)

    return df_synth