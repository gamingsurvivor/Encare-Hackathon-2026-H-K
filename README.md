# Encare Synthetic Data Hackathon

This goal of this hackathon is to create new synthetic data of medical records

## Project Structure
- `/data`: Place your raw `synthetic-data-hackaton-sample.csv` here.
- `/results`: Synthetic outputs will be saved here with timestamps.
- `/examples`: Baseline generators (e.g., Random Sampler).
- `data_processor.py`: Cleaning and imputation logic.
- `validator.py`: Statistical (KS-test) and clinical validation.

## Setup Instructions

### 1. Create a Virtual Environment
Open your terminal in the project folder and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

Ensure your csv file is in the /data folder and run main.py

The repo also contains some validation for sense checks. The generated data passing these tests shall not be seen as an indicator for a high score.