from src.config import ECG_QA_TRAIN, PTBXL_METADATA
from src.ecgqa_loader import load_ecgqa_json, inspect_ecgqa_sample
from src.ptbxl_loader import load_ptbxl_metadata, describe_ecg_row, load_ecg_signal

print("ECG-QA:", ECG_QA_TRAIN)
print("PTB-XL:", PTBXL_METADATA)

#data = load_ecgqa_json(ECG_QA_TRAIN)
#print("Number of samples:", len(data))
#inspect_ecgqa_sample(data[0])

metadata = load_ptbxl_metadata()

describe_ecg_row(12, metadata)

#needs to prefer "lr" because "hr" is missing for this sample
signal = load_ecg_signal(12, metadata=metadata, prefer="lr")
print("Signal shape:", signal.shape)