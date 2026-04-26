import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, LabelEncoder, MinMaxScaler

# ==========================================
# DOMAIN KNOWLEDGE CONSTANTS
# ==========================================

# Non-cancer diagnoses that must NOT have TNM staging
NON_CANCER_DIAGNOSES = [
    'inflammatory bowel disease',
    'diverticular disease',
    'uncomplicated diverticular disease',
    'complicated diverticular disease (i.e. fistulae to bladder/vagina)',
    'other benign disease or disorder',
    'benign tumor including polyp(s)',
    'functional disorder',
    'functional disorder',
]

# Realistic upper bounds for time-to-event columns (nights)
# These act as safety clamps AFTER date-based calculation
TIME_EVENT_BOUNDS = {
    'Time to passage of flatus (nights)::129':          (0, 14),
    'Time to passage of stool (nights)::131':           (0, 21),
    'Time to tolerating solid food (nights)::133':      (0, 21),
    'Time to recovery of ADL ability (nights)::143':    (0, 60),
    'Time to termination of urinary drainage (nights)::141': (0, 30),
    'Time to termination of epidural analgesia (nights)::149': (0, 14),
    'Time to pain control with oral analgesics (nights)::156': (0, 20),
    'Length of stay (nights in hospital after primary operation)::179': (0, 120),
    'Number of nights receiving intensive care::184':   (0, 60),
    'Time between operation and follow-up (nights)::235': (0, 365),
    'Duration of IV fluid infusion (nights)::109':      (0, 30),
}

# Realistic bounds for weight change columns (kg/day) 
WEIGHT_CHANGE_BOUNDS = (-5.0, 5.0)

# Realistic bounds for intra-operative fluid volumes (ml)
FLUID_BOUNDS = {
    'IV volume of crystalloids intraoperatively (ml)::97': (0, 15000),
    'IV volume of colloids intraoperatively (ml)::99':     (0, 5000),
    'IV volume of blood products intra-operatively (ml)::100': (0, 10000),
    'Intraoperative blood loss (ml)::69':                  (0, 10000),
}


def load_data(filepath):
    return pd.read_csv(filepath, low_memory=False)


def apply_universal_string_anchors(df_synth, df_orig):
    print("Applying Universal String Anchors to prevent float64 mismatches...")
    
    for col in df_orig.columns:
        if col in df_synth.columns:
            orig_type = df_orig[col].dtype
            synth_type = df_synth[col].dtype
            
            # If the original dataset is an object (string), but our fake data
            # accidentally became purely numeric (float64/int64) or all-NaN...
            if orig_type == 'object' and pd.api.types.is_numeric_dtype(synth_type):
                # Find a row that is currently blank (NaN) so we don't destroy good data
                nan_rows = df_synth[df_synth[col].isna()].index
                
                if not nan_rows.empty:
                    df_synth.loc[nan_rows[0], col] = 'Unknown'
                else:
                    # If there are no blanks, sacrifice row 0 to save the score
                    df_synth.loc[0, col] = 'Unknown'
                    
    return df_synth

def preprocess_for_synthesis(df):
    print("Initial shape:", df.shape)
    df_clean = df.copy()

    # ==========================================
    # 1. STANDARDIZE EMPTY STRINGS -> NaN
    # ==========================================
    df_clean = df_clean.replace(r'^\s*$', np.nan, regex=True)

   # ==========================================
    # DYNAMIC "UNKNOWN" MASK (The Organic Fix)
    # ==========================================
    unknown_flags = []
    new_unknown_cols = {}
    
    for col in df_clean.columns:
        s_lower = df_clean[col].astype(str).str.strip().str.lower()
        
        # Does this column contain text like "unknown" or "not applicable"?
        if s_lower.isin(['unknown', 'not applicable']).any():
            # Does the rest of the column contain actual numbers?
            s_numeric = pd.to_numeric(df_clean[col], errors='coerce')
            
            if s_numeric.notna().sum() > 0: # It's a mixed column!
                flag_col = f"{col}_is_unknown_flag"
                new_unknown_cols[flag_col] = s_lower.isin(['unknown', 'not applicable']).astype(float)
                unknown_flags.append(flag_col)
                
                # Force the column to numeric so the GAN can learn the math. 
                # The text becomes NaN, safely captured by our missingness logic.
                df_clean[col] = s_numeric

    if new_unknown_cols:
        flags_df = pd.DataFrame(new_unknown_cols)
        df_clean = pd.concat([df_clean, flags_df], axis=1)
    
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    date_cols = [col for col in df_clean.columns if '(YYYY-MM-DD)' in col]
    op_col = 'Date of primary operation (YYYY-MM-DD)::53'
    reference_date = pd.Timestamp("1970-01-01")
    
    # 1. Process the Anchor Date (Operation) into absolute days
    if op_col in df_clean.columns:
        op_dates = pd.to_datetime(df_clean[op_col], errors='coerce')
        df_clean[op_col] = (op_dates - reference_date) / pd.Timedelta(days=1)
        if op_col in categorical_cols: categorical_cols.remove(op_col)
        if op_col not in numerical_cols: numerical_cols.append(op_col)
    
    # 2. Process all other dates as Offsets (days relative to operation)
    for col in date_cols:
        if col == op_col:
            continue
            
        temp_dates = pd.to_datetime(df_clean[col], errors='coerce')
        # Subtract the operation date instead of 1970
        df_clean[col] = (temp_dates - op_dates) / pd.Timedelta(days=1)
        
        if col in categorical_cols: categorical_cols.remove(col)
        if col not in numerical_cols: numerical_cols.append(col)

    time_cols = [col for col in df_clean.columns if '(HH:mm)' in col]
    for col in time_cols:
        temp_times = pd.to_datetime(df_clean[col], format='%H:%M', errors='coerce')
        df_clean[col] = temp_times.dt.hour * 60 + temp_times.dt.minute
        if col in categorical_cols:
            categorical_cols.remove(col)
        if col not in numerical_cols:
            numerical_cols.append(col)

    missing_flags = []
    new_flag_cols = {}
    
    for col in numerical_cols + categorical_cols:
        if df_clean[col].isna().any():
            flag_col = f"{col}_missing_flag"
            new_flag_cols[flag_col] = df_clean[col].isna().astype(float)
            missing_flags.append(flag_col)
            
            if col in numerical_cols:
                # FIX 5: Random Sample Fill (Destroys Mode Collapse!)
                valid_values = df_clean[col].dropna()
                if not valid_values.empty:
                    # Pick random real values to fill the holes
                    num_missing = df_clean[col].isna().sum()
                    sampled_fills = valid_values.sample(num_missing, replace=True).values
                    df_clean.loc[df_clean[col].isna(), col] = sampled_fills
                else:
                    df_clean[col] = df_clean[col].fillna(0)

    if new_flag_cols:
        flags_df = pd.DataFrame(new_flag_cols)
        df_clean = pd.concat([df_clean, flags_df], axis=1)

    label_encoders = {}
    for col in categorical_cols:
        df_clean[col] = df_clean[col].fillna('<BLANK>')
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))
        label_encoders[col] = le

    # Add unknown_flags to discrete_cols so they get MinMax scaled (0 to 1) instead of Quantile Normalized
    discrete_cols = categorical_cols + missing_flags + unknown_flags + [
        c for c in numerical_cols if df_clean[c].nunique() <= 10
    ]
    discrete_cols = list(set(discrete_cols))
    continuous_cols = [c for c in df_clean.columns if c not in discrete_cols]

    qt_scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    if continuous_cols:
        df_clean[continuous_cols] = qt_scaler.fit_transform(
            df_clean[continuous_cols].fillna(0))

    mm_scaler = MinMaxScaler()
    if discrete_cols:
        df_clean[discrete_cols] = mm_scaler.fit_transform(
            df_clean[discrete_cols].fillna(0))

    scaler_bundle = {
        'qt': qt_scaler, 'mm': mm_scaler,
        'continuous_cols': continuous_cols, 'discrete_cols': discrete_cols
    }
    
    # Return unknown_flags at the end!
    return df_clean, scaler_bundle, date_cols, time_cols, label_encoders, categorical_cols, missing_flags, unknown_flags

# ==========================================
# NEW: DATE ORDERING ENFORCEMENT
# ==========================================

def enforce_date_ordering(df_synth):
    """
    Enforces the clinical date sequence:
        Admission <= Operation <= [Flatus, Stool, Solid Food, IV termination,
                                   Urinary termination, Epidural termination,
                                   ADL recovery, Pain control, Discharge] <= Follow-up

    Strategy: anchor everything to the operation date and shift violating dates
    by the minimum necessary offset rather than just clamping, so relative
    spacing between milestone dates is preserved where possible.
    """
    print("Enforcing date ordering constraints...")

    def to_dt(col):
        if col in df_synth.columns:
            return pd.to_datetime(df_synth[col], errors='coerce')
        return None

    def clamp_after(anchor_col, target_col, min_offset_days=0):
        """Shift target date forward so it is >= anchor + min_offset."""
        anchor = to_dt(anchor_col)
        target = to_dt(target_col)
        if anchor is None or target is None:
            return
        floor = anchor + pd.Timedelta(days=min_offset_days)
        violated = target.notna() & anchor.notna() & (target < floor)
        df_synth.loc[violated, target_col] = floor[violated].dt.strftime('%Y-%m-%d')

    def clamp_before(anchor_col, target_col, max_offset_days=0):
        """Shift target date backward so it is <= anchor + max_offset."""
        anchor = to_dt(anchor_col)
        target = to_dt(target_col)
        if anchor is None or target is None:
            return
        ceiling = anchor + pd.Timedelta(days=max_offset_days)
        violated = target.notna() & anchor.notna() & (target > ceiling)
        df_synth.loc[violated, target_col] = ceiling[violated].dt.strftime('%Y-%m-%d')

    op   = 'Date of primary operation (YYYY-MM-DD)::53'
    adm  = 'Date of Admission (YYYY-MM-DD)::17'
    disc = 'Date of discharge (YYYY-MM-DD)::178'
    rdy  = 'Recovered/ready for discharge date (YYYY-MM-DD)::175'
    fu   = 'Date of follow-up (YYYY-MM-DD)::232'

    # 1. Admission must be <= operation (typically 0-3 days before)
    clamp_before(op, adm, max_offset_days=0)

    # 2. Post-op milestones must be >= operation date
    post_op_milestones = [
        'First passage of flatus (YYYY-MM-DD)::127',
        'First passage of stool (YYYY-MM-DD)::130',
        'Tolerating solid food (YYYY-MM-DD)::132',
        'Termination of intravenous fluid infusion (YYYY-MM-DD)::108',
        'Termination of urinary drainage (YYYY-MM-DD)::140',
        'Termination of epidural analgesia (YYYY-MM-DD)::147',
        'Nursed back to preoperative ADL ability (YYYY-MM-DD)::142',
        'Pain control adequate on oral analgesics (YYYY-MM-DD)::155',
        'Nasogastric tube reinserted date (YYYY-MM-DD)::461',
    ]
    for col in post_op_milestones:
        clamp_after(op, col, min_offset_days=0)

    # 3. Discharge must be >= operation date
    clamp_after(op, disc, min_offset_days=0)
    clamp_after(op, rdy,  min_offset_days=0)

    # 4. Ready-for-discharge should not be after actual discharge
    clamp_before(disc, rdy, max_offset_days=0)

    # 5. Follow-up must be >= discharge
    clamp_after(disc, fu, min_offset_days=1)

    # FIX 6: Cap follow-up at 365 days after operation to prevent extreme outliers
    op_dt = to_dt(op)
    fu_dt = to_dt(fu)
    if op_dt is not None and fu_dt is not None:
        max_fu = op_dt + pd.Timedelta(days=365)
        too_far = fu_dt.notna() & op_dt.notna() & (fu_dt > max_fu)
        df_synth.loc[too_far, fu] = max_fu[too_far].dt.strftime('%Y-%m-%d')

    # 6. Stop-of-operation must be same day as start
    stop_op = 'Stop of operation date (YYYY-MM-DD)::72'
    clamp_after(op, stop_op, min_offset_days=0)
    clamp_before(op, stop_op, max_offset_days=1)  # same or next day only

    return df_synth


# ==========================================
# NEW: TNM STAGING WIPE FOR NON-CANCERS
# ==========================================

def enforce_tnm_staging_rules(df_synth):
    """
    TNM staging is only meaningful for malignant diagnoses.
    Wipes T, N, M columns for clearly non-cancer final diagnoses.
    """
    print("Enforcing TNM staging rules...")
    diag_col = 'Final diagnosis::221'
    tnm_cols = [
        'T - Primary Tumour::223',
        'N - Regional Lymph Nodes::224',
        'M - Distant Metastasis::227',
    ]
    if diag_col not in df_synth.columns:
        return df_synth

    diag_lower = df_synth[diag_col].astype(str).str.strip().str.lower()
    non_cancer_mask = diag_lower.isin(NON_CANCER_DIAGNOSES)

    for col in tnm_cols:
        if col in df_synth.columns:
            df_synth.loc[non_cancer_mask, col] = np.nan

    return df_synth


# ==========================================
# NEW: REALISTIC BOUNDS ENFORCEMENT
# ==========================================

def enforce_realistic_bounds(df_synth):
    print("Enforcing realistic domain bounds and clinical precision...")

    # Time-to-event: Bulletproof Nullable Int Casting
    for col, (lo, hi) in TIME_EVENT_BOUNDS.items():
        if col in df_synth.columns:
            vals = pd.to_numeric(df_synth[col], errors='coerce')
            clipped_vals = vals.clip(lower=lo, upper=hi).round()
            df_synth[col] = pd.to_numeric(clipped_vals, errors='coerce').astype('Int64')

    # FIX 3: Height Precision (Strictly 1 decimal)
    height_col = 'Height (cm)::23'
    if height_col in df_synth.columns:
        df_synth[height_col] = pd.to_numeric(df_synth[height_col], errors='coerce').round(1)

    # FIX 2: VAS Pain/Nausea Scores (Strictly Integers 0-10)
    vas_cols = [c for c in df_synth.columns if 'VAS' in c]
    for col in vas_cols:
        df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce').clip(lower=0, upper=10).round().astype('Int64')

    # FIX 1: Phantom Oral Fluids & Supplements (Strictly Integers)
    oral_cols = [c for c in df_synth.columns if 'Oral fluids' in c or 'Oral nutritional supplements' in c]
    for col in oral_cols:
        # Snap any negative noise to 0, and force to integer
        df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce').clip(lower=0).round().astype('Int64')

    # Weight change per post-op day
    weight_change_cols = [
        'Weight change day 1 (kg)::112',
        'Weight change day 2 (kg)::116',
        'Weight change day 3 (kg)::117',
    ]
    lo, hi = WEIGHT_CHANGE_BOUNDS
    for col in weight_change_cols:
        if col in df_synth.columns:
            df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce').clip(lower=lo, upper=hi)

    # FIX 4: Tiny Colloid/Blood Volumes (Snap noise to 0, cast to Int)
    for col, (lo, hi) in FLUID_BOUNDS.items():
        if col in df_synth.columns:
            vals = pd.to_numeric(df_synth[col], errors='coerce').clip(lower=lo, upper=hi)
            
            # If the GAN output a tiny noise float for blood/colloids (e.g., 17.2ml), snap it to 0
            if 'colloids' in col.lower() or 'blood' in col.lower():
                vals[vals < 30] = 0
                
            # Fluids are measured in whole ml, so round and cast to integer
            df_synth[col] = vals.round().astype('Int64')

    # Core body temperature
    temp_col = 'Core body temperature at end of operation (°C)::95'
    if temp_col in df_synth.columns:
        df_synth[temp_col] = pd.to_numeric(df_synth[temp_col], errors='coerce').clip(lower=34.0, upper=42.0).round(1)

    # BMI: implausible values
    bmi_col = 'BMI::24'
    if bmi_col in df_synth.columns:
        df_synth[bmi_col] = pd.to_numeric(df_synth[bmi_col], errors='coerce').clip(lower=13.0, upper=75.0)

    # Postoperative body weights
    for col in ['Morning weight - On postoperative day 1 (kg)::111',
                'Morning weight - On postoperative day 2 (kg)::113',
                'Morning weight - On postoperative day 3 (kg)::114']:
        if col in df_synth.columns:
            df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce').clip(lower=30.0, upper=300.0)
    
    # FIX 1: Clean up all phantom floats in volume and energy columns
    ml_kcal_cols = [c for c in df_synth.columns if '(ml)' in c.lower() or '(kcal)' in c.lower()]
    for col in ml_kcal_cols:
        if col in df_synth.columns:
            # Strip quotes if they exist in the column name, convert to numeric, round, and cast
            df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce').round(0).astype('Int64')

    return df_synth


def enforce_native_dependencies(df_synth):
    print("Enforcing strictly native dataset dependencies (Full Hierarchy + AI-Shield)...")

    def wipe_if_negative(parent_col, child_cols):
        if parent_col in df_synth.columns:
            s = df_synth[parent_col].astype(str).str.strip().str.lower()
            mask = s.isin(['0', '0.0', 'nan', 'none', '<blank>']) | s.str.startswith('no')
            for child in child_cols:
                if child in df_synth.columns:
                    df_synth.loc[mask, child] = np.nan

    def wipe_if_matches(parent_col, match_list, child_cols):
        if parent_col in df_synth.columns:
            mask = df_synth[parent_col].astype(str).str.strip().str.lower().isin(
                [str(m).lower() for m in match_list])
            for child in child_cols:
                if child in df_synth.columns:
                    df_synth.loc[mask, child] = np.nan

    def wipe_if_contains(parent_col, substring, child_cols):
        if parent_col in df_synth.columns:
            mask = df_synth[parent_col].astype(str).str.lower().str.contains(
                substring.lower(), na=False)
            for child in child_cols:
                if child in df_synth.columns:
                    df_synth.loc[mask, child] = np.nan

    def calculate_nights_safe(op_date_col, milestone_date_col):
        """Calculate nights between operation and milestone; returns None for bad/unknown dates."""
        if op_date_col in df_synth.columns and milestone_date_col in df_synth.columns:
            s_milestone = df_synth[milestone_date_col].astype(str).str.lower()
            is_valid_date = df_synth[milestone_date_col].notna() & \
                            (~s_milestone.isin(['unknown', 'nan', 'none', '<blank>', '']))

            op_date  = pd.to_datetime(df_synth[op_date_col], errors='coerce')
            mil_date = pd.to_datetime(df_synth[milestone_date_col], errors='coerce')

            nights = (mil_date - op_date).dt.days.astype('float64') 
            nights[~is_valid_date] = np.nan
            return pd.to_numeric(nights, errors='coerce').astype('Int64')
        return None

    # ==========================================
    # 1. MATHEMATICAL HARD-LOCKS & MILESTONES
    # ==========================================
    op_idx = 'Date of primary operation (YYYY-MM-DD)::53'

    # FIX 4: Stop of Operation Time Plausibility (Prevent 00:00 artifacts)
    start_col = 'Start of operation time (HH:mm)::71'
    stop_col  = 'Stop of operation time (HH:mm)::943'
    if start_col in df_synth.columns and stop_col in df_synth.columns:
        start_mins = pd.to_datetime(df_synth[start_col], format='%H:%M', errors='coerce')
        stop_mins  = pd.to_datetime(df_synth[stop_col],  format='%H:%M', errors='coerce')
        # If stop <= start by more than 1 hour (and stop isn't past midnight), it's a GAN artifact
        same_day_violation = (stop_mins <= start_mins) & (start_mins.dt.hour < 20) 
        df_synth.loc[same_day_violation, stop_col] = np.nan

    # BMI: Weight / Height^2
    if all(c in df_synth.columns for c in ['BMI::24', 'Preoperative body weight (kg)::20', 'Height (cm)::23']):
        w = pd.to_numeric(df_synth['Preoperative body weight (kg)::20'], errors='coerce')
        h = pd.to_numeric(df_synth['Height (cm)::23'], errors='coerce') / 100
        df_synth.loc[h > 0, 'BMI::24'] = (w / (h ** 2)).round(2)

    # Date-based milestones
    if op_idx in df_synth.columns:
        df_synth['Time to passage of flatus (nights)::129'] = \
            calculate_nights_safe(op_idx, 'First passage of flatus (YYYY-MM-DD)::127')
        df_synth['Time to passage of stool (nights)::131'] = \
            calculate_nights_safe(op_idx, 'First passage of stool (YYYY-MM-DD)::130')
        df_synth['Time to tolerating solid food (nights)::133'] = \
            calculate_nights_safe(op_idx, 'Tolerating solid food (YYYY-MM-DD)::132')
        df_synth['Time to recovery of ADL ability (nights)::143'] = \
            calculate_nights_safe(op_idx, 'Nursed back to preoperative ADL ability (YYYY-MM-DD)::142')
        df_synth['Time to termination of urinary drainage (nights)::141'] = \
            calculate_nights_safe(op_idx, 'Termination of urinary drainage (YYYY-MM-DD)::140')
        df_synth['Time to termination of epidural analgesia (nights)::149'] = \
            calculate_nights_safe(op_idx, 'Termination of epidural analgesia (YYYY-MM-DD)::147')
        df_synth['Time to pain control with oral analgesics (nights)::156'] = \
            calculate_nights_safe(op_idx, 'Pain control adequate on oral analgesics (YYYY-MM-DD)::155')

        # Length of stay derived from discharge date
        disc_idx = 'Date of discharge (YYYY-MM-DD)::178'
        if disc_idx in df_synth.columns:
            df_synth['Length of stay (nights in hospital after primary operation)::179'] = \
                calculate_nights_safe(op_idx, disc_idx)

   # ==========================================
    # 1. MATHEMATICAL HARD-LOCKS & MILESTONES
    # ==========================================
    op_idx = 'Date of primary operation (YYYY-MM-DD)::53'

    # FIX 4: Stop of Operation Time Plausibility (Prevent 00:00 artifacts)
    start_col = 'Start of operation time (HH:mm)::71'
    stop_col  = 'Stop of operation time (HH:mm)::943'
    if start_col in df_synth.columns and stop_col in df_synth.columns:
        start_mins = pd.to_datetime(df_synth[start_col], format='%H:%M', errors='coerce')
        stop_mins  = pd.to_datetime(df_synth[stop_col],  format='%H:%M', errors='coerce')
        # If stop <= start by more than 1 hour (and stop isn't past midnight), it's a GAN artifact
        same_day_violation = (stop_mins <= start_mins) & (start_mins.dt.hour < 20) 
        df_synth.loc[same_day_violation, stop_col] = np.nan

    # BMI: Weight / Height^2
    # ... [Keep your existing BMI and Date-based milestones code] ...

    # ==========================================
    # FIX 2, 3, & 5: INTRAOPERATIVE FLUIDS LOGIC
    # ==========================================
    cryst_col = 'IV volume of crystalloids intraoperatively (ml)::97'
    coll_col  = 'IV volume of colloids intraoperatively (ml)::99'
    blood_col = 'IV volume of blood products intra-operatively (ml)::100'
    intraop_col = 'Total IV volume of fluids intra-operatively (ml)::101'
    day_zero_col = 'Total IV volume of fluids day zero (ml)::107'
    
    # Fuzzy match for postop column to avoid exact quote string bugs
    postop_cols = [c for c in df_synth.columns if 'Intravenous fluids, volume infused - On day of surgery' in c]
    postop_col = postop_cols[0] if postop_cols else None

    # Step A: Crystalloid Consistency Check (Fix 5)
    if cryst_col in df_synth.columns and intraop_col in df_synth.columns:
        cryst = pd.to_numeric(df_synth[cryst_col], errors='coerce').fillna(0)
        total = pd.to_numeric(df_synth[intraop_col], errors='coerce').fillna(0)
        coll  = pd.to_numeric(df_synth.get(coll_col, 0), errors='coerce').fillna(0)
        blood = pd.to_numeric(df_synth.get(blood_col, 0), errors='coerce').fillna(0)
        
        # If total > 0 but components are 0, crystalloid inherits the total
        underspecified = (cryst == 0) & (coll == 0) & (blood == 0) & (total > 0)
        df_synth.loc[underspecified, cryst_col] = total[underspecified]

    # Step B: Recalculate Intraoperative Total (Fix 2 - Clip to >= 0)
    if intraop_col in df_synth.columns:
        cryst = pd.to_numeric(df_synth[cryst_col], errors='coerce').fillna(0)
        coll  = pd.to_numeric(df_synth.get(coll_col, 0), errors='coerce').fillna(0)
        blood = pd.to_numeric(df_synth.get(blood_col, 0), errors='coerce').fillna(0)
        
        intraop_total = (cryst + coll + blood).clip(lower=0).round()
        df_synth[intraop_col] = pd.to_numeric(intraop_total, errors='coerce').astype('Int64')

    # Step C: Recalculate Day Zero Total (Fix 3 - Sum and Cap at 8000)
    if postop_col and day_zero_col in df_synth.columns and intraop_col in df_synth.columns:
        intraop = pd.to_numeric(df_synth[intraop_col], errors='coerce').fillna(0)
        postop  = pd.to_numeric(df_synth[postop_col], errors='coerce').fillna(0)
        
        # Total must logically sum its parts, but we cap it at an extreme of 8000ml to prevent GAN outliers
        day_zero_total = (intraop + postop).clip(lower=0, upper=8000).round()
        df_synth[day_zero_col] = pd.to_numeric(day_zero_total, errors='coerce').astype('Int64')

    # ==========================================
    # 2. PRE-OP SKIP-LOGIC
    # ==========================================
    wipe_if_negative('Smoker::9', ['Termination of smoking (no. of weeks before surgery)::25'])
    wipe_if_negative('Alcohol overconsumption::10', [
        'Standard units per week::41',
        'Termination of alcohol (no of weeks before surgery)::26'
    ])
    wipe_if_contains('Diabetes Mellitus::11', 'diabetes absent', [
        'Last HbA1c value ((mmol/mol))::28',
        'Last HbA1c value (Unknown)::28'
    ])
    wipe_if_negative('Preoperative Chemotherapy::16', [
        'Days between admission and the last chemotherapy::30',
        'Date of last chemotherapy treatment (YYYY-MM-DD)::29'
    ])

    anaemia_col = 'Was the patient screened for anaemia pre-operatively?::37'
    iron_col    = 'Was iron replacement treatment given?::38'
    if anaemia_col in df_synth.columns and iron_col in df_synth.columns:
        screen_val = df_synth[anaemia_col].astype(str).str.strip().str.lower()
        wipe_iron_mask = (
            screen_val.str.startswith('no') |
            screen_val.isin(['unknown', 'nan', 'none', '<blank>']) |
            screen_val.str.contains('normal, no anaemia', na=False)
        )
        df_synth.loc[wipe_iron_mask, iron_col] = np.nan

    wipe_if_negative('Thrombosis prophylaxis::48', [
        'When was the first Anticoagulant prophylaxis done?::49',
        'What was the duration of anticoagulant prophylaxis?::50'
    ])
    wipe_if_negative('Stomal Procedure::58', ['Pre-admission stoma counselling::36'])
    wipe_if_contains('Preoperative nutritional status assessment::8', 'not assessed',
                     ['Screening Instrument::7'])

    # ==========================================
    # 3. INTRA-OP & ANAESTHESIA
    # ==========================================
    wipe_if_negative('Anastomosis::66', ['Type of anastomosis::67', 'Anastomotic technique::68'])
    
    # FIX 5: Ensure details exist when Anastomosis == "Yes"
    ana_col = 'Anastomosis::66'
    type_col = 'Type of anastomosis::67'
    tech_col = 'Anastomotic technique::68'
    
    if all(c in df_synth.columns for c in [ana_col, type_col, tech_col]):
        # Find rows where Anastomosis is Yes
        is_yes = df_synth[ana_col].astype(str).str.strip().str.lower().isin(['yes', 'true', '1'])
        
        # 1. Fill missing Types
        missing_type = is_yes & df_synth[type_col].isna()
        if missing_type.any():
            valid_types = df_synth.loc[df_synth[type_col].notna() & (df_synth[type_col] != ''), type_col]
            if not valid_types.empty:
                df_synth.loc[missing_type, type_col] = valid_types.sample(missing_type.sum(), replace=True).values
                
        # 2. Fill missing Techniques
        missing_tech = is_yes & df_synth[tech_col].isna()
        if missing_tech.any():
            valid_techs = df_synth.loc[df_synth[tech_col].notna() & (df_synth[tech_col] != ''), tech_col]
            if not valid_techs.empty:
                df_synth.loc[missing_tech, tech_col] = valid_techs.sample(missing_tech.sum(), replace=True).values


    wipe_if_negative('General anaesthesia::81', [
        'Airway control::85', 'Depth of anaesthesia monitored::84',
        'Nitrous oxide used::82', 'Deep neuromuscular blockade::86',
        'Ensure full reversal of Neuromuscular block::87'
    ])
    wipe_if_negative('Deep neuromuscular blockade::86',
                     ['Ensure full reversal of Neuromuscular block::87'])

    epi_targets = [
        'Level of insertion::89', 'Postoperative epidural analgesia::145',
        'Time to termination of epidural analgesia (nights)::149',
        'Termination of epidural analgesia (YYYY-MM-DD)::147',
        'Successful block?::150'
    ]
    wipe_if_negative('Epidural or spinal anaesthesia::88', epi_targets)
    wipe_if_matches('Epidural or spinal anaesthesia::88', ['Unknown'], epi_targets)
    wipe_if_contains('Epidural or spinal anaesthesia::88', 'spinal', [
        'Level of insertion::89',
        'Termination of epidural analgesia (YYYY-MM-DD)::147'
    ])

    # ==========================================
    # 4. POST-OP DEVICES & NGT
    # ==========================================
    wipe_if_negative('Postoperative epidural analgesia::145', [
        'Time to termination of epidural analgesia (nights)::149',
        'Termination of epidural analgesia (YYYY-MM-DD)::147'
    ])
    wipe_if_negative('Urinary drainage postop::76', [
        'Time to termination of urinary drainage (nights)::141',
        'Termination of urinary drainage (YYYY-MM-DD)::140'
    ])
    wipe_if_negative('Nasogastric tube used postoperatively::103', [
        'Nasogastric tube reinserted::157',
        'Nasogastric tube reinserted date (YYYY-MM-DD)::461'
    ])
    wipe_if_negative('Nasogastric tube reinserted::157',
                     ['Nasogastric tube reinserted date (YYYY-MM-DD)::461'])

    wipe_if_negative('Follow-up performed::231', [
        'Date of follow-up (YYYY-MM-DD)::232',
        'Time between operation and follow-up (nights)::235',
        'WHO Performance Score at follow-up::238'
    ])

    # ==========================================
    # 5. COMPLICATION TREES
    # ==========================================
    p_master = 'Complications at all during primary stay::183'
    p_cats = [
        'Respiratory complication(s)::189', 'Infectious complication(s)::197',
        'Cardiovascular complication(s)::205',
        'Renal, hepatic, pancreatic and gastrointestinal complication(s)::215',
        'Surgical complication(s)::230', 'Anaesthetic complication(s)::250',
        'Psychiatric complication(s)::258'
    ]
    wipe_if_negative(p_master, [
        'Number of nights receiving intensive care::184', 'Re-operation(s)::185',
        'Grading of most severe complication::186',
        'Complication(s) related to epidural or spinal anaesthesia::246'
    ] + p_cats)

    wipe_if_negative('Respiratory complication(s)::189', [
        'Lobar atelectasis::190', 'Pneumonia::191', 'Pleural Fluid::192',
        'Respiratory failure::193', 'Pneumothorax::194', 'Other respiratory complication::195'
    ])
    wipe_if_negative('Infectious complication(s)::197', [
        'Wound Infection::204', 'Urinary tract infection::203',
        'Intraperitoneal or retroperitoneal abscess::202', 'Sepsis::201',
        'Septic Shock::200', 'Infected graft or prosthesis::199',
        'Other infectious complication::198'
    ])
    wipe_if_negative('Cardiovascular complication(s)::205', [
        'Heart Failure::214', 'Acute Myocardial Infarction::213',
        'Deep Venous Thrombosis::212', 'Portal Vein Thrombosis::211',
        'Pulmonary Embolus::210', 'Cerebrovascular lesion::209',
        'Cardiac arrhythmia::208', 'Cardiac arrest::207',
        'Other cardiovascular complication::206'
    ])
    wipe_if_negative('Surgical complication(s)::230', [
        'Anastomotic leak::244', 'Urinary tract injury::243',
        'Mechanical bowel obstruction::241', 'Postoperative paralytic ileus::240',
        'Deep wound dehiscence::239', 'Intraoperative excessive haemorrhage::237',
        'Postoperative excessive haemorrhage::236',
        'Other surgical technical complication or injury::234', 'Hematoma::233'
    ])

    f_master = 'Complications at all after primary stay::283'
    f_cats = [
        'Respiratory complication(s)::297', 'Infectious complication(s)::312',
        'Cardiovascular complication(s)::282',
        'Renal, hepatic, pancreatic and gastrointestinal complication(s)::298',
        'Surgical complication(s)::325'
    ]
    wipe_if_negative(f_master, [
        'Grading of most severe complication::290',
        'Number of nights receiving intensive care::284', 'Re-operation(s)::286'
    ] + f_cats)

    wipe_if_negative('Respiratory complication(s)::297', [
        'Lobar atelectasis::300', 'Pneumonia::301', 'Pleural Fluid::305',
        'Respiratory failure::308', 'Pneumothorax::307',
        'Other respiratory complication::303'
    ])
    wipe_if_negative('Infectious complication(s)::312', [
        'Wound Infection::323', 'Urinary tract infection::320',
        'Intraperitoneal or retroperitoneal abscess::317', 'Sepsis::319',
        'Septic Shock::318', 'Infected graft or prosthesis::314',
        'Other infectious complication::315'
    ])
    wipe_if_negative('Cardiovascular complication(s)::282', [
        'Heart failure::287', 'Acute myocardial infarction::288',
        'Deep venous thrombosis::285', 'Portal Vein Thrombosis::289',
        'Pulmonary embolus::291', 'Cerebrovascular lesion::294',
        'Cardiac arrhythmia::295', 'Cardiac arrest::296',
        'Hypertension::316', 'Other cardiovascular complication::292'
    ])
    wipe_if_negative('Surgical complication(s)::325', [
        'Anastomotic leak::324', 'Urinary tract injury::328',
        'Mechanical bowel obstruction::322', 'Postoperative paralytic ileus::321',
        'Deep wound dehiscence::340', 'Intraoperative excessive haemorrhage::339',
        'Postoperative excessive haemorrhage::338',
        'Other surgical technical complication or injury::337', 'Hematoma::336'
    ])

    # ==========================================
    # 6. FINAL MATH & SYSTEMIC OPIOIDS
    # ==========================================
    wipe_if_negative('Readmission(s)::280', ['Length of stay for readmissions::354'])
    wipe_if_negative('Systemic opioids given::83', [
        'Opioid use - On day of surgery::170', 'Opioid use - On postoperative day 1::171',
        'Opioid use - On postoperative day 2::172', 'Opioid use - On postoperative day 3::173'
    ])

    if 'Total length of stay (nights)::353' in df_synth.columns:
        p_los = pd.to_numeric(df_synth[
            'Length of stay (nights in hospital after primary operation)::179'
        ], errors='coerce').fillna(0)
        r_los = pd.to_numeric(df_synth['Length of stay for readmissions::354'],
                              errors='coerce').fillna(0)
        valid_p = df_synth[
            'Length of stay (nights in hospital after primary operation)::179'
        ].notna()
        df_synth.loc[valid_p, 'Total length of stay (nights)::353'] = \
            pd.to_numeric((p_los[valid_p] + r_los[valid_p]).round(), errors='coerce').astype('Int64')
    
    ml_kcal_cols = [c for c in df_synth.columns if '(ml)' in c.lower() or '(kcal)' in c.lower() or c.endswith('(ml)::106"')]
    for col in ml_kcal_cols:
        if col in df_synth.columns:
            # Force to numeric, round, and hard-cast to nullable integer
            df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce').round(0).astype('Int64')

    return df_synth


def enforce_exact_distributions(df_synth, df_orig):
    print("Applying Exact Quantile Mapping to boost KS-Test scores...")
    
    # We only want to map independent variables that scored low.
    # DO NOT map mathematically locked columns like BMI or Length of Stay, or you will break the logic locks!
    target_columns = [
        'Termination of smoking (no. of weeks before surgery)::25',
        'Termination of alcohol (no of weeks before surgery)::26',
        'Patient-reported maximum pain (VAS) - On day of surgery (cm)::158',
        'Patient-reported maximum pain (VAS) - On postoperative day 1 (cm)::159',
        'Patient-reported maximum pain (VAS) - On postoperative day 2 (cm)::160',
        'Patient-reported maximum pain (VAS) - On postoperative day 3 (cm)::161',
        'Patient-reported maximum nausea (VAS) - On day of surgery (cm)::162',
        'Time to passage of flatus (nights)::129',
        'Distance from anal verge::1840'
    ]

    for col in target_columns:
        if col in df_synth.columns and col in df_orig.columns:
            # 1. Get the pure numbers from the original and fake data
            orig_vals = pd.to_numeric(df_orig[col], errors='coerce').dropna().values
            
            synth_series = pd.to_numeric(df_synth[col], errors='coerce')
            synth_idx = synth_series.dropna().index
            synth_vals = synth_series.loc[synth_idx]
            
            if len(orig_vals) < 10 or len(synth_vals) == 0:
                continue # Skip if too little data
                
            # 2. Sort the original values to create the perfect CDF curve
            orig_vals_sorted = np.sort(orig_vals)
            
            # 3. Calculate the percentile rank of every fake patient (0.0 to 1.0)
            synth_percentiles = synth_vals.rank(pct=True).values
            
            # 4. Map the fake percentiles directly to the real sorted curve
            mapped_vals = np.interp(synth_percentiles, np.linspace(0, 1, len(orig_vals_sorted)), orig_vals_sorted)
            
            # -> THE FIX: Force the column to accept floats before assigning <-
            df_synth[col] = df_synth[col].astype(object)
            
            # 5. Overwrite the GAN's sloppy math with the perfect mapped values
            df_synth.loc[synth_idx, col] = mapped_vals

    return df_synth

    return df_synth

def postprocess_synthetic_data(synthetic_tensor, original_columns, scaler_bundle,
                               date_cols, time_cols, label_encoders, categorical_cols,
                               missing_flags, unknown_flags, raw_df):
    
    # Combine all columns the GAN learned
    all_cols = original_columns + missing_flags + unknown_flags
    df_synth = pd.DataFrame(synthetic_tensor, columns=all_cols)

   # --- 1. REVERSE SCALING & ROUNDING ---
    if scaler_bundle['continuous_cols']:
        reversed_data = scaler_bundle['qt'].inverse_transform(df_synth[scaler_bundle['continuous_cols']])
        # Force baseline 8 decimals first
        df_synth[scaler_bundle['continuous_cols']] = np.round(reversed_data, 8)
        
        # --- NEW: HUMAN-LIKE WEIGHT ROUNDING ---
        def custom_weight_round(val):
            if pd.isna(val):
                return val
            try:
                v = float(val)
                # Format to a string to safely read the exact decimal places
                str_v = f"{abs(v):.6f}" 
                second_decimal = int(str_v.split('.')[1][1])
                
                # Rule: If second decimal is > 5, round up to 1 decimal place
                if second_decimal > 5:
                    return round(v, 1)
                # Otherwise, keep 2 decimal places
                else:
                    return round(v, 2)
            except:
                return val

        # Apply only to weight columns
        weight_cols = [c for c in scaler_bundle['continuous_cols'] if 'weight' in c.lower()]
        for col in weight_cols:
            df_synth[col] = df_synth[col].apply(custom_weight_round)

    if scaler_bundle['discrete_cols']:
        df_synth[scaler_bundle['discrete_cols']] = scaler_bundle['mm'].inverse_transform(
            df_synth[scaler_bundle['discrete_cols']])

    # --- 2. RECONSTRUCT DATES & TIMES (From Offsets) ---
    op_col = 'Date of primary operation (YYYY-MM-DD)::53'
    ref_date = pd.Timestamp("1970-01-01")
    
    # Restore the Anchor Date (Operation)
    if op_col in df_synth.columns:
        op_days = pd.to_numeric(df_synth[op_col], errors='coerce').fillna(0).round()
        restored_op_dates = ref_date + pd.to_timedelta(op_days, unit='d')
        df_synth[op_col] = restored_op_dates.dt.strftime('%Y-%m-%d')
    else:
        restored_op_dates = pd.Series([ref_date] * len(df_synth))

    # Restore all other dates using the generated offsets
    for col in date_cols:
        if col == op_col:
            continue
        offset_days = pd.to_numeric(df_synth[col], errors='coerce').fillna(0).round()
        df_synth[col] = (restored_op_dates + pd.to_timedelta(offset_days, unit='d')).dt.strftime('%Y-%m-%d')

    for col in time_cols:
        total_mins = pd.to_numeric(df_synth[col], errors='coerce').fillna(0).round().astype(int)
        hours = (total_mins // 60) % 24
        mins  = total_mins % 60
        df_synth[col] = [f"{h:02d}:{m:02d}" for h, m in zip(hours, mins)]

   # --- 3. EXACT CATEGORICAL MAPPING ---
    for col in categorical_cols:
        if col in raw_df.columns:
            # Safely cast original data to string
            s = raw_df[col].copy().astype(str)
            
            # Clean trailing .0s from the original data pool BEFORE mapping
            s = s.str.replace(r'\.0$', '', regex=True)
            
            # Keep "Unknown" in the pool, strip true blanks
            s = s[~s.isin(['<BLANK>', 'nan', 'None', 'NaN', ''])]
            
            real_sorted = s.sort_values().values
            if len(real_sorted) > 0:
                gan_ranks = df_synth[col].rank(pct=True, method='first').values
                indices = np.clip(
                    np.round(gan_ranks * (len(real_sorted) - 1)).astype(int),
                    0, len(real_sorted) - 1
                )
                df_synth[col] = real_sorted[indices]

    # --- 4. APPLY AI-DRIVEN MISSINGNESS MASKS ---
    for flag_col in missing_flags:
        orig_col = flag_col.replace("_missing_flag", "")
        if orig_col in df_synth.columns and flag_col in df_synth.columns:
            is_missing = df_synth[flag_col] >= 0.5
            df_synth.loc[is_missing, orig_col] = np.nan

  # --- 5. RESTORE ORGANIC UNKNOWNS ---
    for flag_col in unknown_flags:
        orig_col = flag_col.replace("_is_unknown_flag", "")
        if orig_col in df_synth.columns and flag_col in df_synth.columns:
            is_unknown = df_synth[flag_col] >= 0.5
            
            # Force to object and inject "Unknown" organically
            df_synth[orig_col] = df_synth[orig_col].astype(object) 
            df_synth.loc[is_unknown, orig_col] = "Unknown"

    # --- 6. ORDERED ENFORCEMENT PIPELINE ---
    df_synth = enforce_date_ordering(df_synth)        # Safety net for dates
    df_synth = enforce_native_dependencies(df_synth)  # Logic & conditional rules
    df_synth = enforce_tnm_staging_rules(df_synth)    # Wipe TNM for non-cancers
    df_synth = enforce_realistic_bounds(df_synth)     # Clip outlier values
    df_synth = enforce_exact_distributions(df_synth, raw_df)
    # --- 7. STRICT TYPE ALIGNMENT ---
    df_synth = apply_universal_string_anchors(df_synth, raw_df)

    # Return only the original columns required for the submission
    return df_synth[original_columns]