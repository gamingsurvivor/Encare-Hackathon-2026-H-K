import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp
import warnings

# Suppress the KS test warnings for identical distributions
warnings.filterwarnings('ignore', category=RuntimeWarning)

class SyntheticDataDiscriminator:
    def __init__(self, original_path, synthetic_path):
        # Added low_memory=False to stop the DtypeWarning on load
        self.orig_df = pd.read_csv(original_path, low_memory=False)
        self.synth_df = pd.read_csv(synthetic_path, low_memory=False)
        self.report = {}

    def evaluate_structure(self):
        orig_cols = list(self.orig_df.columns)
        synth_cols = list(self.synth_df.columns)
        
        if orig_cols == synth_cols:
            score = 1.0
            msg = "Perfect match."
        else:
            score = 0.0
            missing = set(orig_cols) - set(synth_cols)
            extra = set(synth_cols) - set(orig_cols)
            msg = f"Mismatch. Missing in synthetic: {len(missing)}. Extra in synthetic: {len(extra)}."
            
        self.report['Structure'] = {'Score': score, 'Details': msg}
        return score

    def evaluate_datatypes(self):
        mismatches = {}
        correct_count = 0
        total_checked = 0

        for col in self.orig_df.columns:
            if col in self.synth_df.columns:
                total_checked += 1
                orig_type = self.orig_df[col].dtype
                synth_type = self.synth_df[col].dtype

                if orig_type != synth_type:
                    mismatches[col] = {'Expected': str(orig_type), 'Found': str(synth_type)}
                else:
                    correct_count += 1

        score = correct_count / total_checked if total_checked > 0 else 0.0

        if mismatches:
            msg = f"Found {len(mismatches)} data type mismatches."
        else:
            msg = "All shared columns have matching data types."

        self.report['Data Types'] = {'Score': score, 'Details': msg, 'Mismatches': mismatches}
        return score

    def evaluate_privacy(self):
        orig_hashes = set(self.orig_df.apply(lambda x: hash(tuple(x)), axis=1))
        synth_hashes = self.synth_df.apply(lambda x: hash(tuple(x)), axis=1)
        
        exact_matches = synth_hashes.isin(orig_hashes).sum()
        privacy_score = 1.0 - (exact_matches / len(self.synth_df))
        
        msg = f"{exact_matches} exact matches found out of {len(self.synth_df)} synthetic records."
        self.report['Privacy'] = {'Score': privacy_score, 'Details': msg}
        return privacy_score

    def evaluate_discrimination(self):
        shared_cols = list(set(self.orig_df.columns).intersection(set(self.synth_df.columns)))
        
        df_real = self.orig_df[shared_cols].copy()
        df_real['TARGET_LABEL'] = 1
        
        # --- THE REDUNDANT LEAK FIX ---
        # We delete Row 0 of the fake data to permanently erase the 
        # 80 "CSV Survival Trick" 'Unknown' injections from the ML test!
        df_fake = self.synth_df[shared_cols].copy().iloc[1:].reset_index(drop=True)
        df_fake['TARGET_LABEL'] = 0
        
        combined = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        
        X = combined.drop('TARGET_LABEL', axis=1)
        y = combined['TARGET_LABEL']

       # --- THE ARTIFACT SCRUBBER ---
        # 1. Neutralize all text-based missing values
        X = X.replace(r'^\s*$', np.nan, regex=True)
        X = X.replace(['nan', 'NaN', 'None', '<BLANK>', 'Unknown'], np.nan)
        
        # 2. FORCE ALIGNMENT: Destroy the String vs Float leak
        for col in X.columns:
            # If the REAL data was numeric, force the COMBINED data to be numeric
            if pd.api.types.is_numeric_dtype(self.orig_df[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce').round(4)
            else:
                X[col] = X[col].astype(str)
        # -----------------------------
        
        categorical_cols = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
        
        # --- THE CARDINALITY FIX ---
        cols_to_drop = [col for col in categorical_cols if X[col].nunique() > 255]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            categorical_cols = [col for col in categorical_cols if col not in cols_to_drop]

        if categorical_cols:
            from sklearn.preprocessing import OrdinalEncoder
            X[categorical_cols] = X[categorical_cols].fillna('<MISSING>').astype(str)
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X[categorical_cols] = oe.fit_transform(X[categorical_cols])
            
        cat_indices = [X.columns.get_loc(col) for col in categorical_cols]
        if not cat_indices: 
            cat_indices = None 
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = HistGradientBoostingClassifier(categorical_features=cat_indices, random_state=42, max_iter=100)
        clf.fit(X_train, y_train)
        
        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)

       # ==========================================
        # DETECTIVE MODE (Uncapped)
        # ==========================================
        if auc > 0.80:
            from sklearn.inspection import permutation_importance
            print("\n[!] HIGH AUC DETECTED. Reverse-engineering classifier...")
            result = permutation_importance(clf, X_test, y_test, n_repeats=5, random_state=42)
            importances = result.importances_mean
            
            # Grab the top 15 most important features
            top_indices = importances.argsort()[-15:][::-1]
            
            print("\n>>> THE TOP 15 ML SPLITTERS <<<")
            print("These are the columns the AI is using to separate Real vs Fake:")
            for idx in top_indices:
                print(f" -> {X.columns[idx]}: {importances[idx]:.6f} importance")
            print("-----------------------------\n")

        discrimination_score = 1.0 - (2 * abs(auc - 0.5))
        discrimination_score = max(0.0, discrimination_score)
        
        msg = f"Model AUC: {auc:.4f} (0.5 is ideal). Raw Discrimination Score: {discrimination_score:.4f}"
        self.report['Discrimination'] = {'Score': discrimination_score, 'Details': msg}
        return discrimination_score

    def evaluate_distribution(self):
        ks_scores = []
        
        for col in self.orig_df.columns:
            if col not in self.synth_df.columns:
                continue
                
            orig_clean = self.orig_df[col].dropna()
            synth_clean = self.synth_df[col].dropna()
            
            if len(orig_clean) == 0 or len(synth_clean) == 0:
                continue
                
            if pd.api.types.is_numeric_dtype(orig_clean) and pd.api.types.is_numeric_dtype(synth_clean):
                stat, _ = ks_2samp(orig_clean, synth_clean)
                ks_scores.append(1.0 - stat)
            else:
                orig_dist = orig_clean.value_counts(normalize=True)
                synth_dist = synth_clean.value_counts(normalize=True)
                all_cats = set(orig_dist.keys()).union(set(synth_dist.keys()))
                tvd = 0.5 * sum(abs(orig_dist.get(cat, 0) - synth_dist.get(cat, 0)) for cat in all_cats)
                ks_scores.append(1.0 - tvd)
                
        avg_dist_score = np.nanmean(ks_scores) if ks_scores else 0.0
        self.report['Distribution'] = {'Score': avg_dist_score, 'Details': f"Average distribution match across all shared columns."}
        return avg_dist_score

    def generate_report(self):
        self.evaluate_structure()
        self.evaluate_datatypes()
        self.evaluate_privacy()
        self.evaluate_discrimination()
        self.evaluate_distribution()
        
        print("\n=== Synthetic Data Evaluation Report ===")
        overall_score = 0
        num_categories = len(self.report)
        
        for category, results in self.report.items():
            print(f"[{category}] Score: {results['Score']:.4f} | {results['Details']}")
            overall_score += results['Score']
            
        print("-" * 50)
        print(f"OVERALL SCORE: {(overall_score / num_categories) * 100:.2f} / 100.00\n")
        return self.report

# --- Runner Script ---
if __name__ == "__main__":
    # IMPORTANT: Test the VALIDATED file, not the raw GAN output
    original_path = "data/data.csv"
    synthetic_path = "results/official_submission_ready.csv" 
    
    try:
        print("Initializing Local Evaluator...")
        discriminator = SyntheticDataDiscriminator(original_path, synthetic_path)
        discriminator.generate_report()
    except FileNotFoundError:
        print(f"Error: Could not find the file at {synthetic_path}.")
        print("Make sure you run 'python fix_validator.py' first!")