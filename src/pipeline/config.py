from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

TRAIN_TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.csv"
TRAIN_IDENTITY_PATH = RAW_DATA_DIR / "train_identity.csv"

TRAIN_MERGED_PATH = PROCESSED_DATA_DIR / "train_merged.parquet"
TRAIN_DF_PATH = PROCESSED_DATA_DIR / "final_train_df.parquet"

TEST_TRANSACTION_PATH = RAW_DATA_DIR / "test_transaction.csv"
TEST_IDENTITY_PATH = RAW_DATA_DIR / "test_identity.csv"

TEST_MERGED_PATH = PROCESSED_DATA_DIR / "test_merged.parquet"

CATBOOSTV1_PATH = MODELS_DIR / "catboost_fraud_model_v1.cbm"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_SEED = 42
VAL_SIZE = 0.2
