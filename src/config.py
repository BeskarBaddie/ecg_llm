from pathlib import Path

# -----------------------------
# BASE PATHS
# -----------------------------

# Project root (this file lives in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory (you said data is in Documents)
DATA_ROOT = Path.home() / "Diss" / "Data"
# -----------------------------
# DATASET PATHS
# -----------------------------

# ECG-QA
ECG_QA_DIR = DATA_ROOT / "ecg-qa"
ECG_QA_TRAIN = ECG_QA_DIR / "ecgqa" / "ptbxl" / "template" / "train"
ECG_QA_VAL = ECG_QA_DIR / "ecgqa" / "ptbxl" / "template" / "val"
ECG_QA_TEST = ECG_QA_DIR / "ecgqa" / "ptbxl" / "template" / "test"

# PTB-XL
PTBXL_DIR = DATA_ROOT / "physionet.org" / "files" / "ptb-xl" / "1.0.3"
PTBXL_METADATA = PTBXL_DIR / "ptbxl_database.csv"

# -----------------------------
# OUTPUT PATHS
# -----------------------------

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories if they don’t exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# EXPERIMENT SETTINGS
# -----------------------------

# Number of samples for testing
NUM_SAMPLES = 100

# Device (used later for CSFM)
DEVICE = "cpu"  # change to "mps" or "cuda" later