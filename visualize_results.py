import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import warnings

# Silence warnings for clean output
warnings.filterwarnings('ignore')

def visualize_distributions():
    print("Loading datasets...")
    # Update these paths to match your local setup
    orig_path = 'data/data.csv'
    synth_path = 'results/final_submission_scrubbed.csv'
    
    try:
        df_orig = pd.read_csv(orig_path, low_memory=False)
        df_synth = pd.read_csv(synth_path, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Could not find {orig_path} or {synth_path}")
        return

    output_pdf = "results/feature_distributions.pdf"
    
    with PdfPages(output_pdf) as pdf:
        features = df_orig.columns
        cols_per_page = 6
        rows_per_page = 3
        features_per_page = cols_per_page * rows_per_page
        
        print(f"Generating distributions for {len(features)} features...")
        
        for i in range(0, len(features), features_per_page):
            fig, axes = plt.subplots(rows_per_page, cols_per_page, figsize=(20, 12))
            axes = axes.flatten()
            
            page_features = features[i:i + features_per_page]
            
            for j, col in enumerate(page_features):
                ax = axes[j]
                
                # Clean data for plotting
                orig_data = pd.to_numeric(df_orig[col].astype(str).replace('Unknown', np.nan), errors='coerce').dropna()
                synth_data = pd.to_numeric(df_synth[col].astype(str).replace('Unknown', np.nan), errors='coerce').dropna()
                
                # HEURISTIC: Plot as numeric if it has many unique values, otherwise categorical
                if len(orig_data) > 0 and df_orig[col].nunique() > 10:
                    sns.kdeplot(orig_data, ax=ax, label='Real', fill=True, color='blue', alpha=0.3)
                    sns.kdeplot(synth_data, ax=ax, label='Synth', fill=True, color='orange', alpha=0.3)
                else:
                    # Categorical / Low-cardinality Bar Chart
                    top_cats = df_orig[col].value_counts().nlargest(5).index
                    orig_counts = df_orig[col].value_counts(normalize=True).reindex(top_cats).fillna(0)
                    synth_counts = df_synth[col].value_counts(normalize=True).reindex(top_cats).fillna(0)
                    
                    x = np.arange(len(top_cats))
                    width = 0.35
                    ax.bar(x - width/2, orig_counts, width, label='Real', color='blue', alpha=0.5)
                    ax.bar(x + width/2, synth_counts, width, label='Synth', color='orange', alpha=0.5)
                    ax.set_xticks(x)
                    ax.set_xticklabels([str(c)[:8] for c in top_cats], rotation=45, fontsize=8)

                # Clean up title (remove the numeric ID at the end)
                title = col.split('::')[0]
                ax.set_title(title[:25] + '...', fontsize=10)
                ax.legend(fontsize=8)
                ax.set_xlabel('')
                ax.set_ylabel('')

            # Hide unused subplots on the last page
            for k in range(len(page_features), features_per_page):
                axes[k].axis('off')
                
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"  Processed {min(i + features_per_page, len(features))}/{len(features)}...")

    print(f"\nSuccess! Open your report here: {output_pdf}")

if __name__ == "__main__":
    visualize_distributions()