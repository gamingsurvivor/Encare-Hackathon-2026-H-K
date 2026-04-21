import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_for_synthesis(df):
    print("Initial shape:", df.shape)
    df_clean = df.copy()

    # 1. Identify Column Types
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

    # 3. EXPLICIT MISSINGNESS (NO IMPUTATION)
    # The rubric explicitly states 'Unknown' is a valid answer. We leave it completely alone!
    
    # For numerical data, we flag NaNs with an impossible number (-999). 
    # The GAN will learn to generate -999 when the logic dictates the cell should be empty.
    for col in numerical_cols:
        df_clean[col] = df_clean[col].fillna(-999)

    # For text data, we flag NaNs with a specific token string.
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('<BLANK>') 

    # 4. TOKENIZE TEXT (Label Encoding)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # 5. Scale EVERYTHING to [0, 1] for the GAN
    scaler = MinMaxScaler()
    all_cols = df_clean.columns.tolist()
    df_clean[all_cols] = scaler.fit_transform(df_clean[all_cols])

    print("Final shape after Tokenizing & Scaling:", df_clean.shape)

    return df_clean, scaler, date_cols, time_cols, label_encoders, categorical_cols


def postprocess_synthetic_data(synthetic_tensor, original_columns, scaler, date_cols, time_cols, label_encoders, categorical_cols):
    df_synth = pd.DataFrame(synthetic_tensor, columns=original_columns)

    # 1. Reverse the 0-1 Scaling
    df_synth[original_columns] = scaler.inverse_transform(df_synth[original_columns])

    # 2. Reverse Pure Numerical Columns & Restore Missing Values
    numerical_cols = [col for col in original_columns if col not in categorical_cols and col not in date_cols and col not in time_cols]
    for col in numerical_cols:
        # If the GAN generated a number anywhere near our -999 flag, it means "This should be empty"
        df_synth.loc[df_synth[col] < -500, col] = np.nan
        df_synth[col] = df_synth[col].round(2)

    # 3. Reverse Tokenization & Restore Missing Values
    for col in categorical_cols:
        if col in label_encoders and col in df_synth.columns:
            le = label_encoders[col]
            
            max_class_id = len(le.classes_) - 1
            df_synth[col] = df_synth[col].round().clip(0, max_class_id).astype(int)
            
            # Decode back to exact text
            df_synth[col] = le.inverse_transform(df_synth[col])
            
            # Turn our explicit structural flag back into a true empty cell
            df_synth[col] = df_synth[col].replace('<BLANK>', np.nan)

    # 4. Reverse Dates
    for col in date_cols:
        if col in df_synth.columns:
            # Reverse the flag for dates
            df_synth.loc[df_synth[col] < -500, col] = np.nan
            
            valid_mask = df_synth[col].notna()
            
            # THE FIX: Cast the column to object so Pandas allows us to insert strings
            df_synth[col] = df_synth[col].astype(object)
            
            # We only format the valid dates, leaving NaNs alone
            dates = pd.to_datetime(df_synth.loc[valid_mask, col].round().astype(int), unit='D', origin='1970-01-01', errors='coerce').dt.strftime('%Y-%m-%d')
            df_synth.loc[valid_mask, col] = dates

    # 5. Reverse Times
    for col in time_cols:
        if col in df_synth.columns:
            df_synth.loc[df_synth[col] < -500, col] = np.nan
            
            valid_mask = df_synth[col].notna()
            
            # THE FIX: Cast the column to object so Pandas allows us to insert strings
            df_synth[col] = df_synth[col].astype(object)
            
            valid_times = df_synth.loc[valid_mask, col].round().astype(int).clip(0, 1439)
            hours = valid_times // 60
            minutes = valid_times % 60
            
            df_synth.loc[valid_mask, col] = hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(2)

    return df_synth