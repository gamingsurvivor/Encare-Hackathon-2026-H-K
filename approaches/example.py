import pandas as pd
import numpy as np

def run_random_sample(df: pd.DataFrame, num_samples: int = 100) -> pd.DataFrame:
    """
    An example approach as placeholder. 
    It creates data by picking random values between the min and max 
    of the original columns. It completely ignores clinical logic.
    """
    synthetic_data = {}

    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            # Generate random numbers between the observed min and max
            low = df[col].min()
            high = df[col].max()
            synthetic_data[col] = np.random.uniform(low, high, num_samples)
        else:
            # For text columns, just pick existing words at random (shuffling)
            choices = df[col].dropna().unique()
            synthetic_data[col] = np.random.choice(choices, num_samples)

    return pd.DataFrame(synthetic_data)