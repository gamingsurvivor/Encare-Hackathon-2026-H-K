import pandas as pd

from CSVTester import SyntheticDataDiscriminator

def main():
    print("Loading datasets...")
    # 1. Load the original data just to read its inferred datatypes
    raw_data = pd.read_csv("data/data.csv")
    
    # 2. Load your MOST RECENT synthetic file that failed the validator
    # (Change this filename to whatever your last run generated)
    synthetic_file_path = "results/synthetic_data_20260421_2223.csv" 
    final_synthetic_df = pd.read_csv(synthetic_file_path)

    # 3. Your master override lists
    force_to_str = {
        'Preoperative body weight (kg)::20', 'Height (cm)::23', 'Intraoperative blood loss (ml)::69',
        'Core body temperature at end of operation (°C)::95', 'Morning weight - On postoperative day 1 (kg)::111',
        'Morning weight - On postoperative day 2 (kg)::113', 
        'Oral fluids, total volume taken - On day of surgery, postoperatively (ml)::118',
        'Oral fluids, total volume taken - On postoperative day 1 (ml)::119',
        'Oral nutritional supplements, energy intake - On day of surgery, postoperatively (kCal)::122',
        'Oral nutritional supplements, energy intake - On postoperative day 1 (kCal)::123',
        'Patient-reported maximum pain (VAS) - On day of surgery (cm)::158',
        'Patient-reported maximum pain (VAS) - On postoperative day 1 (cm)::159',
        'Patient-reported maximum nausea (VAS) - On day of surgery (cm)::162',
        'Patient-reported maximum nausea (VAS) - On postoperative day 1 (cm)::163',
        'Patient-reported maximum nausea (VAS) - On postoperative day 2 (cm)::164',
        'Patient-reported maximum nausea (VAS) - On postoperative day 3 (cm)::165',
        'Was iron replacement treatment given?::38',
        'Termination of smoking (no. of weeks before surgery)::25', 'Standard units per week::41',
        'Termination of alcohol (no of weeks before surgery)::26', 'Last HbA1c value (Unknown)::28',
        'Distance from anal verge::1840', 'Level of insertion::89', 
        'IV volume of crystalloids intraoperatively (ml)::97', 'IV volume of colloids intraoperatively (ml)::99',
        'IV volume of blood products intra-operatively (ml)::100', 
        'Intravenous fluids, volume infused - On day of surgery, postoperatively (ml)::106',
        'Mobilisation - On postoperative day 1::137', 'Strong opioids given within 48 hrs postoperatively::152',
        'Successful block?::150', 'Use of peripheral opioid receptor antagonist::153',
        'Grading of most severe complication::186', 'Urinary tract infection::203',
        'Grading of most severe complication::290', 'Date of last chemotherapy treatment (YYYY-MM-DD)::29',
        'Termination of epidural analgesia (YYYY-MM-DD)::147', 'Nasogastric tube reinserted date (YYYY-MM-DD)::461',
        'Type of anastomosis::67', 'Anastomotic technique::68', 'Length of incision (cm)::64',
        'Re-operation(s)::185', 'Lobar atelectasis::190', 'Pneumonia::191', 'Pleural Fluid::192', 'Respiratory failure::193',
        'Pneumothorax::194', 'Other respiratory complication::195', 'Wound Infection::204',
        'Intraperitoneal or retroperitoneal abscess::202', 'Sepsis::201', 'Septic Shock::200',
        'Infected graft or prosthesis::199', 'Other infectious complication::198', 'Heart Failure::214',
        'Acute Myocardial Infarction::213', 'Deep Venous Thrombosis::212', 'Portal Vein Thrombosis::211',
        'Pulmonary Embolus::210', 'Cerebrovascular lesion::209', 'Cardiac arrhythmia::208',
        'Cardiac arrest::207', 'Other cardiovascular complication::206', 'Renal dysfunction::228',
        'Urinary retention::226', 'Hepatic dysfunction::225', 'Pancreatitis::220',
        'Gastrointestinal haemorrhage::219', 'Nausea or vomiting::218', 'Obstipation or diarrhoea::217',
        'Other organ dysfunction::216', 'Anastomotic leak::244', 'Urinary tract injury::243',
        'Mechanical bowel obstruction::241', 'Postoperative paralytic ileus::240', 'Deep wound dehiscence::239',
        'Intraoperative excessive haemorrhage::237', 'Postoperative excessive haemorrhage::236',
        'Other surgical technical complication or injury::234', 'Hematoma::233', 'Post dural-puncture headache::249',
        'Epidural hematoma or abscess::248', 'Other EDA or spinal related complication::247',
        'Pulmonary aspiration of gastric contents::257', 'Hypotension::256', 'Hypoxia::255',
        'Prolonged postoperative sedation::251', 'Other anaesthetic complication(s)::253', 'Asthenia or tiredness::260',
        'Re-operation(s)::286', 'Infectious complication(s)::312',
        'Respiratory complication(s)::297', 'Lobar atelectasis::300', 'Wound Infection::323',
        'Urinary tract infection::320', 'Intraperitoneal or retroperitoneal abscess::317', 'Sepsis::319',
        'Septic Shock::318', 'Infected graft or prosthesis::314', 'Other infectious complication::315',
        'Cardiovascular complication(s)::282', 'Acute myocardial infarction::288', 'Cardiac arrhythmia::295',
        'Renal, hepatic, pancreatic and gastrointestinal complication(s)::298', 'Renal dysfunction::299',
        'Urinary retention::352', 'Hepatic dysfunction::302', 'Pancreatitis::304', 'Gastrointestinal haemorrhage::306',
        'Nausea or vomiting::310', 'Obstipation or diarrhoea::311', 'Incontinence::313', 'Other organ dysfunction::309',
        'Surgical complication(s)::325', 'Anastomotic leak::324', 'Urinary tract injury::328',
        'Mechanical bowel obstruction::322', 'Postoperative paralytic ileus::321', 'Deep wound dehiscence::340',
        'Intraoperative excessive haemorrhage::339', 'Postoperative excessive haemorrhage::338',
        'Other surgical technical complication or injury::337', 'Hematoma::336',
        'Complication(s) related to epidural or spinal anaesthesia::326', 'Anaesthetic complication(s)::331',
        'Psychiatric complication(s)::343', 'Asthenia or tiredness::344', 'Pain::342', 'Injuries::345', 'Other::347',
        'Preoperative body weight (kg)::20', 
        'Height (cm)::23', 
        'Termination of alcohol (no of weeks before surgery)::26', 
        'IV volume of crystalloids intraoperatively (ml)::97', 
        'IV volume of colloids intraoperatively (ml)::99', 
        'IV volume of blood products intra-operatively (ml)::100', 
        'Oral fluids, total volume taken - On postoperative day 1 (ml)::119', 
        'Oral nutritional supplements, energy intake - On day of surgery, postoperatively (kCal)::122', 
        'Oral nutritional supplements, energy intake - On postoperative day 1 (kCal)::123', 
        'Mobilisation - On postoperative day 1::137', 
        'Patient-reported maximum nausea (VAS) - On postoperative day 1 (cm)::163',
    }

    force_to_float = {
        'Number of nights receiving intensive care::284', 'Pneumonia::301', 'Pleural Fluid::305',
        'Respiratory failure::308', 'Pneumothorax::307', 'Other respiratory complication::303',
        'Heart failure::287', 'Deep venous thrombosis::285', 'Portal Vein Thrombosis::289',
        'Pulmonary embolus::291', 'Cerebrovascular lesion::294', 'Cardiac arrest::296',
        'Hypertension::316', 'Other cardiovascular complication::292', 'Post dural-puncture headache::327',
        'Epidural hematoma or abscess::329', 'Other EDA or spinal related complication::330',
    }

    print("Applying strict schema typing...")
    for col in final_synthetic_df.columns:
        orig_type = raw_data[col].dtype

        if col in force_to_float:
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').astype(float)

        elif orig_type == 'object' or orig_type.name == 'category' or col in force_to_str:
            final_synthetic_df[col] = final_synthetic_df[col].astype(str)
            final_synthetic_df[col] = final_synthetic_df[col].str.replace(r'\.0$', '', regex=True)
            final_synthetic_df[col] = final_synthetic_df[col].replace({'nan': '', 'None': ''})

            if not final_synthetic_df[col].str.contains(r'[A-Za-z]').any():
                final_synthetic_df.iloc[0, final_synthetic_df.columns.get_loc(col)] = 'Unknown'

        elif orig_type == 'int64':
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').round().fillna(0).astype(int)

        elif orig_type == 'float64':
            final_synthetic_df[col] = pd.to_numeric(final_synthetic_df[col], errors='coerce').astype(float)

    float_cols = final_synthetic_df.select_dtypes(include=['float64', 'float32']).columns
    final_synthetic_df[float_cols] = final_synthetic_df[float_cols].round(2)

    output_path = "results/fixed_for_validator.csv"
    final_synthetic_df.to_csv(output_path, index=False)
    
    print(f"Done! Upload this file: {output_path}")

if __name__ == "__main__":
    main()
    original_data_path = "data/data.csv"
    
    # We want to test the file that has been perfectly formatted
    synthetic_data_path = "results/fixed_for_validator.csv" 
    
    discriminator = SyntheticDataDiscriminator(original_data_path, synthetic_data_path)