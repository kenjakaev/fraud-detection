import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from fastapi import FastAPI, HTTPException

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.pipeline.config import CATBOOSTV1_PATH, METADATA_PATH, TEST_MERGED_PATH
from src.pipeline.logger import logger

ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model and configurations...")
    try:
        model = CatBoostClassifier()
        model.load_model(CATBOOSTV1_PATH)
        test_df = pd.read_parquet(TEST_MERGED_PATH, engine="pyarrow")

        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            selected_features = model.feature_names_
            cat_features = metadata.get("cat_features", [])
            if not selected_features:
                logger.warning(
                    "The 'features' list in the metadata is empty or missing!"
                )

        ml_artifacts["test_df"] = test_df
        ml_artifacts["model"] = model
        ml_artifacts["features"] = selected_features
        ml_artifacts["cat_features"] = cat_features
        logger.info("The model has been successfully loaded into memory!")
    except Exception as e:
        logger.error(f"Critical error loading artifacts: {e}")
        raise RuntimeError("Failed to initialize ML model artifacts") from e

    yield

    logger.info("Resource cleanup on shutdown...")
    ml_artifacts.clear()


app = FastAPI(title="Anti-Fraud API", lifespan=lifespan)


@app.get("/health")
def health_check():
    is_ready = "model" in ml_artifacts
    return {"status": "ok" if is_ready else "error", "service": "anti-fraud-api"}


@app.post("/predict")
def predict_random_transaction():
    if "model" not in ml_artifacts or "test_df" not in ml_artifacts:
        raise HTTPException(status_code=500, detail="Model or data not loaded")

    model = ml_artifacts["model"]
    features = ml_artifacts["features"]
    cat_features = ml_artifacts["cat_features"]
    test_df = ml_artifacts["test_df"]

    logger.info("Trying to predict a random transaction")

    try:
        random_sample = test_df.sample(n=1)

        X = random_sample.reindex(columns=features, fill_value=0)

        for col in cat_features:
            if col in X.columns:
                X[col] = X[col].fillna(-1).astype(str)

        eval_pool = Pool(data=X, cat_features=cat_features)

        probabilities = model.predict_proba(eval_pool)
        fraud_prob = float(probabilities[0][1])

        tx_id = None
        if "TransactionID" in random_sample.columns:
            tx_id = int(random_sample["TransactionID"].values[0])

        threshold = 0.5
        is_fraud = fraud_prob >= threshold

        return {
            "transaction_id": tx_id,
            "fraud_probability": round(fraud_prob, 4),
            "is_fraud": is_fraud,
            "action": "DECLINE" if is_fraud else "APPROVE",
        }

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error calculating the prediction")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
