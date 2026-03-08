import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict

def run_evaluation_report(original_df: pd.DataFrame, synthetic_df: pd.DataFrame):
    """
    Executes a comprehensive comparison between the source synthetic dataset and the generated one.
    """
    print("\n" + "="*60)
    print("EVALUATION REPORT: SYNTHETIC DATA FIDELITY")
    print("="*60)

    # Distribution Similarity: Kolmogorov-Smirnov Test
    #Compares the CDF:s of two samples to evaluate how closely the synthetic data matches the original distribution for each numeric column.
    similarities = _compare_distributions(original_df, synthetic_df)
    print("\nStatistical Similarity Scores (1.0 = Identical):")
    for col, score in list(similarities.items())[:8]:
        print(f"  {col[:30]:30s}: {score:.3f}")

    # Clinical Logic: Range and Relationship Checks
    # Ensures the AI generated biologically plausible 'patients'
    validations = _validate_clinical_logic(synthetic_df)
    print("\nClinical Sanity Check Results:")
    for check, passed in validations.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {check:30s}: {status}")

def _compare_distributions(orig: pd.DataFrame, synth: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates 1 minus the KS-statistic for each numeric column.
    A score of 1.0 indicates identical cumulative distributions.
    """
    results = {}
    common_cols = list(set(orig.columns) & set(synth.columns))
    
    for col in common_cols:
        if orig[col].dtype in [np.float64, np.int64]:
            # Perform two-sample Kolmogorov-Smirnov test
            stat, _ = stats.ks_2samp(orig[col].dropna(), synth[col].dropna())
            results[col] = round(1 - stat, 3)
    return results

def _validate_clinical_logic(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Verifies that generated values obey known clinical relationships and bounds.
    """
    results = {}
    
    # Verify age boundaries
    if 'Age::40' in df.columns:
        ages = pd.to_numeric(df['Age::40'], errors='coerce')
        results['Age Within Range (18-100)'] = ((ages >= 18) & (ages <= 100)).all()

    # Verify mathematical relationship: BMI = kg / m^2
    # This confirms if the model learned correlations between separate columns
    weight_col = 'Preoperative body weight (kg)::20'
    height_col = 'Height (cm)::23'
    bmi_col = 'BMI::24'
    
    if all(c in df.columns for c in [weight_col, height_col, bmi_col]):
        w = pd.to_numeric(df[weight_col])
        h = pd.to_numeric(df[height_col]) / 100 # Convert cm to meters
        bmi = pd.to_numeric(df[bmi_col])
        
        expected_bmi = w / (h**2)
        # Check if 90% of rows are within 1.5 BMI units of the expected calculation
        is_consistent = (abs(bmi - expected_bmi).dropna() < 1.5).mean() > 0.90
        results['BMI Feature Correlation'] = is_consistent

    return results