import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, MinMaxScaler

def load_data(filepath):
    return pd.read_csv(filepath, low_memory=False)

def preprocess_for_synthesis(df):
    print("Initial shape:", df.shape)
    df_clean = df.copy()

    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    # --- Convert Dates & Times ---
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

    # --- Extract Missing Flags ---
    missing_flags = []
    new_flag_cols = {}
    for col in numerical_cols:
        if df_clean[col].isna().any():
            flag_col = f"{col}_missing_flag"
            new_flag_cols[flag_col] = df_clean[col].isna().astype(float)
            missing_flags.append(flag_col)
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(0 if pd.isna(median_val) else median_val)

    if new_flag_cols:
        flags_df = pd.DataFrame(new_flag_cols)
        df_clean = pd.concat([df_clean, flags_df], axis=1)

    # --- Encode Categoricals ---
    label_encoders = {}
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('<BLANK>')
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # --- Split Continuous vs Discrete ---
    discrete_cols = categorical_cols + missing_flags + [c for c in numerical_cols if df_clean[c].nunique() <= 10]
    discrete_cols = list(set(discrete_cols))
    continuous_cols = [c for c in df_clean.columns if c not in discrete_cols]

    qt_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    if continuous_cols:
        df_clean[continuous_cols] = qt_scaler.fit_transform(df_clean[continuous_cols].fillna(0))

    mm_scaler = MinMaxScaler()
    if discrete_cols:
        df_clean[discrete_cols] = mm_scaler.fit_transform(df_clean[discrete_cols].fillna(0))

    scaler_bundle = {
        'qt': qt_scaler, 
        'mm': mm_scaler, 
        'continuous_cols': continuous_cols, 
        'discrete_cols': discrete_cols
    }
    
    print(f"Scaling complete. {len(discrete_cols)} discrete, {len(continuous_cols)} continuous.")
    return df_clean, scaler_bundle, date_cols, time_cols, label_encoders, categorical_cols, missing_flags

def postprocess_synthetic_data(synthetic_tensor, original_columns, scaler_bundle, date_cols, time_cols, label_encoders, categorical_cols, missing_flags, raw_df):
    all_cols = original_columns + missing_flags
    df_synth = pd.DataFrame(synthetic_tensor, columns=all_cols)

    print("Reversing scales (Honest GAN Output)...")
    
    # 1. Reverse Continuous (Trust the GAN's distributions natively)
    if scaler_bundle['continuous_cols']:
        df_synth[scaler_bundle['continuous_cols']] = scaler_bundle['qt'].inverse_transform(df_synth[scaler_bundle['continuous_cols']])

    # 2. Reverse Discrete (Clip to boundaries, round safely)
    if scaler_bundle['discrete_cols']:
        df_synth[scaler_bundle['discrete_cols']] = df_synth[scaler_bundle['discrete_cols']].clip(0, 1)
        df_synth[scaler_bundle['discrete_cols']] = scaler_bundle['mm'].inverse_transform(df_synth[scaler_bundle['discrete_cols']])
        df_synth[scaler_bundle['discrete_cols']] = df_synth[scaler_bundle['discrete_cols']].round()

    # 3. Restore Missingness (NaNs)
    for flag_col in missing_flags:
        orig_col = flag_col.replace("_missing_flag", "")
        if orig_col in df_synth.columns:
            df_synth.loc[df_synth[flag_col] > 0.5, orig_col] = np.nan
            
    df_synth = df_synth[original_columns]

    # 4. Standard Formatting (Decode labels and Dates)
    numerical_cols = [col for col in original_columns if col not in categorical_cols and col not in date_cols and col not in time_cols]
    for col in numerical_cols:
        # REMOVED .round(2) HERE - Let the GAN's native floating point precision match the real data!
        df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce')

    for col in categorical_cols:
        if col in label_encoders and col in df_synth.columns:
            le = label_encoders[col]
            max_class_id = len(le.classes_) - 1
            class_ids = df_synth[col].fillna(0).round().clip(0, max_class_id).astype(int)
            df_synth[col] = le.inverse_transform(class_ids)
            df_synth[col] = df_synth[col].replace('<BLANK>', np.nan)

    for col in date_cols:
        if col in df_synth.columns:
            valid_mask = df_synth[col].notna()
            temp_numeric = pd.to_numeric(df_synth.loc[valid_mask, col], errors='coerce')
            valid_ints = temp_numeric.round().fillna(0).astype(int)
            dates = pd.to_datetime(valid_ints, unit='D', origin='1970-01-01', errors='coerce').dt.strftime('%Y-%m-%d')
            df_synth[col] = df_synth[col].astype(object)
            df_synth.loc[valid_mask, col] = dates

    for col in time_cols:
        if col in df_synth.columns:
            valid_mask = df_synth[col].notna()
            temp_numeric = pd.to_numeric(df_synth.loc[valid_mask, col], errors='coerce')
            valid_times = temp_numeric.round().fillna(0).astype(int).clip(0, 1439)
            hours = valid_times // 60
            minutes = valid_times % 60
            df_synth[col] = df_synth[col].astype(object)
            df_synth.loc[valid_mask, col] = hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(2)

    return df_synth