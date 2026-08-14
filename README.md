# 🛡️ IEEE-CIS Anti-Fraud ML Service

A production-ready microservice built with **FastAPI** and **CatBoost** designed for real-time fraud detection on financial transactions.

---

## 📌 Architecture & Key Features

- **ML Core**: Gradient Boosting model (**CatBoostClassifier**) trained on the IEEE-CIS Fraud Detection dataset to identify anomalous transactions.
- **FastAPI Framework**: Asynchronous REST API utilizing the modern `lifespan` context manager for heavy artifact initialization (model & data) at startup.
- **Robust Inference**: Dynamic categorical feature pre-processing and accurate column ordering enforced directly via `model.feature_names_`.
- **I/O Optimization**: High-speed Parquet data processing leveraging the **PyArrow** engine to minimize memory footprint and loading time.
- **Containerized**: Fully Dockerized application built on a lightweight `python:3.11-slim` base image with optimized layer caching and strict Prod/Dev dependency separation.

---

## 🛠️ Project Structure

```text
.
├── data/
│   ├── processed/           # Processed Parquet datasets
│   │   ├── final_train_df.parquet
│   │   ├── test_merged.parquet
│   │   └── train_merged.parquet
│   └── raw/                 # Original IEEE-CIS CSV raw files
│       ├── test_identity.csv
│       ├── test_transaction.csv
│       ├── train_identity.csv
│       └── train_transaction.csv
├── models/                  # Model artifacts and metadata
│   ├── catboost_fraud_model_v1.cbm
│   └── model_metadata.json
├── notebooks/               # EDA & Model Development Pipeline
│   ├── 01_eda_and_baseline.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_testing.ipynb
├── src/                     # Source Code
│   ├── api/
│   │   └── main.py          # FastAPI application & endpoints
│   └── pipeline/
│       ├── config.py        # Environment & path configurations
│       └── logger.py        # Centralized logging setup
├── .dockerignore            # Excluded build contexts
├── .gitignore               # Excluded cache
├── Dockerfile               # Docker container recipe
├── LICENSE                  # MIT License
├── pyproject.toml           # Project metadata & tool settings
├── README.md                # Project documentation
├── requirements.txt         # Production dependencies
└── requirements-dev.txt     # Development & testing tools
```

---

## 🚀 Quick Start with Docker

### 1. Build the Docker Image
Execute the following command in the root directory:

```bash
docker build -t anti-fraud-service:v1 .
```

### 2. Run the Container
Launch the container with port forwarding:

```bash
docker run -d -p 8000:8000 --name anti-fraud-app anti-fraud-service:v1
```

Once running, interactive OpenAPI/Swagger documentation is available at:  
👉 **`http://localhost:8000/docs`**

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | **Readiness Probe**: Verifies that artifacts are properly loaded into memory. |
| `POST` | `/predict` | **Inference**: Evaluates a transaction and returns fraud probability with action tags. |

### Sample Response (`POST /predict`)

```json
{
  "transaction_id": 3701968,
  "fraud_probability": 0.0001,
  "is_fraud": false,
  "action": "APPROVE"
}
```

---

## 💡 Future Improvements (High-Load Optimization)

- **Vectorized Preprocessing**: Replace Pandas DataFrame manipulations with **NumPy** or **Polars** arrays to reduce end-to-end inference latency.
- **Request Validation**: Implement strict Pydantic schemas (`BaseModel`) for payload parsing.
- **Production Scaling**: Export the CatBoost model to C++ Native API or deploy via **Triton Inference Server** for ultra-low latency (< 1ms) high-RPS production pipelines.

---

## 📜 License
Distributed under the MIT License.