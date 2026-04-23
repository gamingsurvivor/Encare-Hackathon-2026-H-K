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


def align_datatypes_strictly(df_synth, df_orig):
    print("Applying Strict Datatype Alignment (The Anti-Inference CSV Lock)...")
    df_synth = df_synth[df_orig.columns].copy()

    for col in df_orig.columns:
        if col not in df_synth.columns:
            continue

        orig_dtype = df_orig[col].dtype

        try:
            # 1. Handle Numerics (Floats and Ints)
            if pd.api.types.is_numeric_dtype(orig_dtype):
                df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce')
                
                # --- DISGUISED INTEGER CHECK ---
                # If the original column had NaNs, Pandas loaded it as float64. 
                # We check if all non-NaN values are actually whole numbers.
                orig_clean = pd.to_numeric(df_orig[col], errors='coerce').dropna()
                is_disguised_int = not orig_clean.empty and (orig_clean % 1 == 0).all()

                if pd.api.types.is_integer_dtype(orig_dtype) or is_disguised_int:
                    # Round and cast to Pandas Nullable Integer
                    df_synth[col] = df_synth[col].round().astype('Int64')
                else:
                    df_synth[col] = df_synth[col].astype(orig_dtype)

            # 2. Handle Categoricals / Objects / Strings
            else:
                df_synth[col] = df_synth[col].astype(str)
                df_synth[col] = df_synth[col].replace(['nan', 'None', '<NA>', 'nan.0', '0.0', ''], np.nan)
                df_synth[col] = df_synth[col].str.replace(r'\.0$', '', regex=True)
                df_synth[col] = df_synth[col].astype(object)
                
                # --- THE CSV INFERENCE LOCK ---
                valid_synth = df_synth[col].dropna()
                
                if valid_synth.empty or pd.to_numeric(valid_synth, errors='coerce').notna().all():
                    orig_valid = df_orig[col].dropna().astype(str)
                    text_only = orig_valid[~pd.to_numeric(orig_valid, errors='coerce').notna()]
                    
                    if not text_only.empty:
                        safe_string = text_only.mode().iloc[0]
                        df_synth.loc[0, col] = safe_string

        except Exception as e:
            print(f"Warning: Fallback triggered on {col} - {e}")
            df_synth[col] = df_synth[col].astype(object)

    return df_synth

def preprocess_for_synthesis(df):
    print("Initial shape:", df.shape)
    df_clean = df.copy()

    # ==========================================
    # 1. STANDARDIZE EMPTY STRINGS -> NaN
    # ==========================================
    df_clean = df_clean.replace(r'^\s*$', np.nan, regex=True)

    # ==========================================
    # 2. THE "UNKNOWN" MASK
    # ==========================================
    unknown_flags = []
    new_unknown_cols = {}
    
    for col in df_clean.columns:
        col_lower = col.lower()
        # If this is supposed to be a continuous numerical column...
        if 'weight' in col_lower or '(ml)' in col_lower or '(cm)' in col_lower or '(kcal)' in col_lower or 'bmi' in col_lower or 'nights' in col_lower:
            
            # Check if the word "Unknown" was explicitly typed in this column
            if df_clean[col].astype(str).str.strip().str.lower().eq('unknown').any():
                flag_col = f"{col}_is_unknown_flag"
                # Create a binary flag: 1.0 if "Unknown", 0.0 otherwise
                new_unknown_cols[flag_col] = df_clean[col].astype(str).str.strip().str.lower().eq('unknown').astype(float)
                unknown_flags.append(flag_col)
            
            # Now it's safe to force to numeric. "Unknown" and empty strings become true NaN.
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Append our new mask columns to the dataframe
    if new_unknown_cols:
        flags_df = pd.DataFrame(new_unknown_cols)
        df_clean = pd.concat([df_clean, flags_df], axis=1)

    # ==========================================
    
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
    
    # FIX: Check ALL columns for missingness, not just numerical ones!
    for col in numerical_cols + categorical_cols:
        if df_clean[col].isna().any():
            flag_col = f"{col}_missing_flag"
            new_flag_cols[flag_col] = df_clean[col].isna().astype(float)
            missing_flags.append(flag_col)
            
            # If it's numerical, apply the Mean-fill to prevent zero-spikes
            if col in numerical_cols:
                mean_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(0 if pd.isna(mean_val) else mean_val)
            # (Categoricals are safely handled by the '<BLANK>' fill below this loop)

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
    print("Enforcing realistic domain bounds...")

    # Time-to-event: Bulletproof Nullable Int Casting
    for col, (lo, hi) in TIME_EVENT_BOUNDS.items():
        if col in df_synth.columns:
            vals = pd.to_numeric(df_synth[col], errors='coerce')
            clipped_vals = vals.clip(lower=lo, upper=hi).round()
            df_synth[col] = pd.to_numeric(clipped_vals, errors='coerce').astype('Int64')

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

    # Fluid volumes
    for col, (lo, hi) in FLUID_BOUNDS.items():
        if col in df_synth.columns:
            df_synth[col] = pd.to_numeric(df_synth[col], errors='coerce').clip(lower=lo, upper=hi)

    # Core body temperature
    temp_col = 'Core body temperature at end of operation (°C)::95'
    if temp_col in df_synth.columns:
        df_synth[temp_col] = pd.to_numeric(df_synth[temp_col], errors='coerce').clip(lower=34.0, upper=42.0)

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

    # Intraoperative Fluids Sum - BULLETPROOF CASTING
    if 'Total IV volume of fluids intra-operatively (ml)::101' in df_synth.columns:
        cryst = pd.to_numeric(df_synth['IV volume of crystalloids intraoperatively (ml)::97'],  errors='coerce').fillna(0)
        coll  = pd.to_numeric(df_synth['IV volume of colloids intraoperatively (ml)::99'],      errors='coerce').fillna(0)
        blood = pd.to_numeric(df_synth['IV volume of blood products intra-operatively (ml)::100'], errors='coerce').fillna(0)
        total_fluids = cryst + coll + blood
        df_synth['Total IV volume of fluids intra-operatively (ml)::101'] = pd.to_numeric(
            total_fluids.round(), errors='coerce'
        ).astype('Int64')

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

   # --- 5. EXACT UNKNOWNS (Distribution Matched) ---
    for flag_col in unknown_flags:
        orig_col = flag_col.replace("_is_unknown_flag", "")
        if orig_col in df_synth.columns and orig_col in raw_df.columns:
            # Count exactly how many 'Unknown's were in the original column
            num_unknowns = raw_df[orig_col].astype(str).str.strip().str.lower().eq('unknown').sum()
            
            if num_unknowns > 0:
                # Rank the GAN's flag outputs to find the most likely rows
                ranks = df_synth[flag_col].rank(method='first', ascending=False)
                
                # Force column to object type and inject the exact number of Unknowns
                df_synth[orig_col] = df_synth[orig_col].astype(object) 
                df_synth.loc[ranks <= num_unknowns, orig_col] = "Unknown"

    # --- 6. ORDERED ENFORCEMENT PIPELINE ---
    df_synth = enforce_date_ordering(df_synth)        # Safety net for dates
    df_synth = enforce_native_dependencies(df_synth)  # Logic & conditional rules
    df_synth = enforce_tnm_staging_rules(df_synth)    # Wipe TNM for non-cancers
    df_synth = enforce_realistic_bounds(df_synth)     # Clip outlier values

    # --- 7. STRICT TYPE ALIGNMENT ---
    df_synth = align_datatypes_strictly(df_synth, raw_df)

    # Return only the original columns required for the submission
    return df_synth[original_columns]