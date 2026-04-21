import warnings
# Silence the Pandas DtypeWarnings so your console is clean!
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=FutureWarning)

# Import your tester class
from CSVTester import SyntheticDataDiscriminator

def main():
    print("Initializing Local Evaluator...")
    
    # 1. Point to your original data
    original_data_path = "data/data.csv"
    
    # 2. Point to the perfectly formatted synthetic data
    synthetic_data_path = "results/fixed_for_validator.csv" 
    
    print(f"Comparing:")
    print(f"  Real Data: {original_data_path}")
    print(f"  Fake Data: {synthetic_data_path}\n")
    
    try:
        # Load the files into your tester
        evaluator = SyntheticDataDiscriminator(original_data_path, synthetic_data_path)
        
        # RUN THE MASTER EVALUATION COMMAND
        evaluator.generate_report()
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please make sure you have run 'fix_validator.py' first so the file exists!")

if __name__ == "__main__":
    main()