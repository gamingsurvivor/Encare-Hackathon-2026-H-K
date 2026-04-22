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

    # ==========================================
    # PROPERLY SPLIT CONTINUOUS & DISCRETE
    # ==========================================
    discrete_cols = categorical_cols + missing_flags + [c for c in numerical_cols if df_clean[c].nunique() <= 10]
    discrete_cols = list(set(discrete_cols)) # Remove duplicates
    
    continuous_cols = [c for c in df_clean.columns if c not in discrete_cols]

    # Normal distribution for stable GAN training
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

def apply_business_rules(df):
    print("Enforcing Medical & Mathematical Invariants to defeat Discriminator...")
    
    # 1. THE MATH TELL: Calculate BMI perfectly
    if 'Preoperative body weight (kg)::20' in df.columns and 'Height (cm)::23' in df.columns and 'BMI::24' in df.columns:
        weight = pd.to_numeric(df['Preoperative body weight (kg)::20'], errors='coerce')
        height_m = pd.to_numeric(df['Height (cm)::23'], errors='coerce') / 100.0
        df['BMI::24'] = (weight / (height_m ** 2)).round(2)

    # 2. THE MEDICAL TELL: Complication Logic
    target_comp = 'Complications at all during primary stay::183'
    if target_comp in df.columns:
        no_comp_mask = df[target_comp] == 'No'
        
        if 'Grading of most severe complication::186' in df.columns:
            df.loc[no_comp_mask, 'Grading of most severe complication::186'] = np.nan
            
        comp_categories = [
            'Respiratory complication(s)::189', 'Infectious complication(s)::197', 
            'Cardiovascular complication(s)::205', 'Surgical complication(s)::230',
            'Anaesthetic complication(s)::250', 'Psychiatric complication(s)::258'
        ]
        for c in comp_categories:
            if c in df.columns:
                df.loc[no_comp_mask, c] = 'No'

    # 3. TOTAL LENGTH OF STAY: Math alignment
    los_primary = 'Length of stay (nights in hospital after primary operation)::179'
    los_readm = 'Length of stay for readmissions::354'
    los_total = 'Total length of stay (nights)::353'
    
    if los_primary in df.columns and los_readm in df.columns and los_total in df.columns:
        p_stay = pd.to_numeric(df[los_primary], errors='coerce').fillna(0)
        r_stay = pd.to_numeric(df[los_readm], errors='coerce').fillna(0)
        df[los_total] = (p_stay + r_stay).round().astype(int)

    # 4. READMISSION LOGIC:
    if 'Readmission(s)::280' in df.columns and los_readm in df.columns:
        no_readm_mask = df['Readmission(s)::280'] == 'No'
        df.loc[no_readm_mask, los_readm] = 0

    return df


def postprocess_synthetic_data(synthetic_tensor, original_columns, scaler_bundle, date_cols, time_cols, label_encoders, categorical_cols, missing_flags, raw_df):
    all_cols = original_columns + missing_flags
    df_synth = pd.DataFrame(synthetic_tensor, columns=all_cols)

    # 1. Reverse Scalers just enough to establish accurate ranking
    if scaler_bundle['continuous_cols']:
        df_synth[scaler_bundle['continuous_cols']] = scaler_bundle['qt'].inverse_transform(df_synth[scaler_bundle['continuous_cols']])
    if scaler_bundle['discrete_cols']:
        df_synth[scaler_bundle['discrete_cols']] = scaler_bundle['mm'].inverse_transform(df_synth[scaler_bundle['discrete_cols']])

    # ==========================================
    # 2. UNIVERSAL EXACT MAPPING (The Categorical & Continuous Fix)
    # ==========================================
    print("Applying Universal Distribution Alignment to perfectly match all histograms...")
    # We skip columns that we mathematically calculate later
    calculated_cols = ['BMI::24', 'Total length of stay (nights)::353']
    
    for col in original_columns:
        if col in raw_df.columns and col not in calculated_cols:
            real_values = raw_df[col].dropna().values
            if len(real_values) > 0:
                # np.sort automatically alphabetizes strings and chronological-izes dates!
                real_sorted = np.sort(real_values)
                # Rank the GAN's guesses from 0.0 to 1.0
                gan_ranks = df_synth[col].rank(pct=True, method='first').values
                # Snap the GAN's rank to the exact real-world value at that percentile
                indices = np.clip(np.round(gan_ranks * (len(real_sorted) - 1)).astype(int), 0, len(real_sorted) - 1)
                df_synth[col] = real_sorted[indices]

    # ==========================================
    # 3. RESTORE STRUCTURAL MISSINGNESS (NaNs)
    # ==========================================
    for flag_col in missing_flags:
        orig_col = flag_col.replace("_missing_flag", "")
        if orig_col in df_synth.columns:
            # If the GAN predicted missingness, punch a NaN hole into the mapped data
            df_synth.loc[df_synth[flag_col] > 0.5, orig_col] = np.nan
            
    df_synth = df_synth[original_columns]

    # ==========================================
    # 4. ENFORCE INVARIANTS TO BEAT DISCRIMINATION
    # ==========================================
    df_synth = apply_business_rules(df_synth)

    return df_synth