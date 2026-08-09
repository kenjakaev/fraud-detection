from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = RAW_DATA_DIR / "train_identity.csv"

TRAIN_DF_PATH = PROCESSED_DATA_DIR / "train_merged.parquet"

RANDOM_SEED = 42
VAL_SIZE = 0.2