import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import ks_2samp

class SyntheticDataDiscriminator:
    def __init__(self, original_path, synthetic_path):
        self.orig_df = pd.read_csv(original_path)
        self.synth_df = pd.read_csv(synthetic_path)
        self.report = {}

    def evaluate_structure(self):
        """1a. Data Structure: Checks if columns, headers, and order match exactly."""
        orig_cols = list(self.orig_df.columns)
        synth_cols = list(self.synth_df.columns)
        
        if orig_cols == synth_cols:
            score = 1.0
            msg = "Perfect match."
        else:
            score = 0.0
            missing = set(orig_cols) - set(synth_cols)
            extra = set(synth_cols) - set(orig_cols)
            msg = f"Mismatch. Missing in synthetic: {missing}. Extra in synthetic: {extra}."
            
        self.report['Structure'] = {'Score': score, 'Details': msg}
        return score

    def evaluate_datatypes(self):
        """1b. Data Types: Checks if the data types in the synthetic data match the original data."""
        mismatches = {}
        correct_count = 0
        total_checked = 0

        for col in self.orig_df.columns:
            if col in self.synth_df.columns:
                total_checked += 1
                orig_type = self.orig_df[col].dtype
                synth_type = self.synth_df[col].dtype

                if orig_type != synth_type:
                    mismatches[col] = {
                        'Expected': str(orig_type),
                        'Found': str(synth_type)
                    }
                else:
                    correct_count += 1

        score = correct_count / total_checked if total_checked > 0 else 0.0

        if mismatches:
            msg = f"Found {len(mismatches)} data type mismatches. See details below."
        else:
            msg = "All shared columns have matching data types."

        self.report['Data Types'] = {
            'Score': score,
            'Details': msg,
            'Mismatches': mismatches
        }
        return score

    def _preprocess_data(self):
        """Prepares data by aligning columns and converting types for the ML model."""
        # We run this AFTER evaluating raw datatypes so we don't mask errors
        for col in self.orig_df.columns:
            if col not in self.synth_df.columns:
                continue
                
            # THE FIX: Check if either dataset contains text/objects
            if self.orig_df[col].dtype == 'object' or self.synth_df[col].dtype == 'object':
                if 'YYYY-MM-DD' in col or 'HH:mm' in col:
                    self.orig_df[col] = pd.to_datetime(self.orig_df[col], errors='coerce').astype('int64') // 10**9
                    self.synth_df[col] = pd.to_datetime(self.synth_df[col], errors='coerce').astype('int64') // 10**9
                else:
                    # Force to string first to neutralize floats, then convert to category
                    self.orig_df[col] = self.orig_df[col].astype(str).astype('category')
                    self.synth_df[col] = self.synth_df[col].astype(str).astype('category')

    def evaluate_privacy(self):
        """2. Privacy: Checks for exact row matches (memorization) between synthetic and original."""
        orig_hashes = set(self.orig_df.apply(lambda x: hash(tuple(x)), axis=1))
        synth_hashes = self.synth_df.apply(lambda x: hash(tuple(x)), axis=1)
        
        exact_matches = synth_hashes.isin(orig_hashes).sum()
        privacy_score = 1.0 - (exact_matches / len(self.synth_df))
        
        msg = f"{exact_matches} exact matches found out of {len(self.synth_df)} synthetic records."
        self.report['Privacy'] = {'Score': privacy_score, 'Details': msg}
        return privacy_score

    def evaluate_discrimination(self):
        """3. Discrimination: Trains a classifier to distinguish real vs. synthetic."""
        self._preprocess_data()
        
        # Label data: 1 for Original, 0 for Synthetic
        df_real = self.orig_df.copy()
        df_real['TARGET_LABEL'] = 1
        
        df_fake = self.synth_df.copy()
        df_fake['TARGET_LABEL'] = 0
        
        # Only keep columns that exist in both to avoid breaking the model
        shared_cols = list(set(self.orig_df.columns).intersection(set(self.synth_df.columns)))
        df_real = df_real[shared_cols + ['TARGET_LABEL']]
        df_fake = df_fake[shared_cols + ['TARGET_LABEL']]
        
        combined = pd.concat([df_real, df_fake], axis=0, ignore_index=True)
        
        X = combined.drop('TARGET_LABEL', axis=1)
        y = combined['TARGET_LABEL']
        
        # --- THE NUCLEAR FIX ---
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].astype(str).astype('category')
        
        # --- THE CARDINALITY FIX ---
        # Scikit-Learn's HistGradientBoosting crashes if a category > 255 unique values.
        # Drop these highly unique text columns (like exact dates/IDs) from the ML test.
        cols_to_drop = [col for col in X.columns if X[col].dtype.name == 'category' and X[col].nunique() > 255]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
        # ---------------------------
        
        categorical_features = [i for i, col in enumerate(X.columns) if X[col].dtype.name == 'category']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        clf = HistGradientBoostingClassifier(categorical_features=categorical_features, random_state=42)
        clf.fit(X_train, y_train)
        
        preds = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        
        discrimination_score = 1.0 - (2 * abs(auc - 0.5))
        
        msg = f"Model AUC: {auc:.4f} (0.5 is ideal). Discrimination capability: {1-discrimination_score:.4f}"
        self.report['Discrimination'] = {'Score': max(0, discrimination_score), 'Details': msg}
        return discrimination_score

    def evaluate_distribution(self):
        """4. Distribution: Uses Kolmogorov-Smirnov test for numerical and variation distance for categoricals."""
        ks_scores = []
        
        for col in self.orig_df.columns:
            if col not in self.synth_df.columns:
                continue
                
            orig_clean = self.orig_df[col].dropna()
            synth_clean = self.synth_df[col].dropna()
            
            # THE FIX: Skip columns that are completely empty so ks_2samp doesn't crash
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
                
        # THE FIX: Use np.nanmean to safely ignore any hidden NaNs
        avg_dist_score = np.nanmean(ks_scores) if ks_scores else 0.0
        self.report['Distribution'] = {'Score': avg_dist_score, 'Details': f"Average distribution match across all shared columns."}
        return avg_dist_score

    def generate_report(self):
        """Runs all checks and returns a formatted dictionary."""
        # Run raw structural checks first before any preprocessing changes the datatypes
        self.evaluate_structure()
        self.evaluate_datatypes()
        
        # Run the rest
        self.evaluate_privacy()
        self.evaluate_discrimination()
        self.evaluate_distribution()
        
        print("=== Synthetic Data Evaluation Report ===")
        overall_score = 0
        num_categories = len(self.report)
        
        for category, results in self.report.items():
            print(f"[{category}] Score: {results['Score']:.4f} | {results['Details']}")
            overall_score += results['Score']
            
            # Print specific datatype mismatches if they exist
            if category == 'Data Types' and results.get('Mismatches'):
                print("    -> Type Mismatches Found:")
                for col, types in results['Mismatches'].items():
                    print(f"       Column '{col}': Expected {types['Expected']}, but found {types['Found']}")
            
        print("-" * 50)
        print(f"OVERALL SCORE: {(overall_score / num_categories) * 100:.2f} / 100.00")
        return self.report

# --- How to run it ---
# if __name__ == "__main__":
#     discriminator = SyntheticDataDiscriminator("original_eras.csv", "synthetic_eras.csv")
#     discriminator.generate_report()