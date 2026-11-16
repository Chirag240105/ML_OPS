# RiskStream MLOps Pipeline — Complete Project Specification & AWS Deployment Blueprint

> **Purpose of this document:**  
> This file contains the complete, ground-up architectural, technical, mathematical, data, model, API, and deployment specification for the **RiskStream Fraud Detection Pipeline**. Any AI model, LLM, Cloud Architect, or DevOps engineer reading this document will have 100% of the context required to assist with AWS infrastructure setup, CI/CD deployment, containerization, experiment tracking, and production monitoring without ambiguity.

---

## 1. Executive Summary & Project Identity

- **Project Name:** RiskStream
- **Sub-Project / Directory:** `ml-firstPipeline/`
- **Domain:** Financial Technology (FinTech) / Real-time Fraud Detection
- **Core Problem:** Detect fraudulent financial transactions (e.g., account drains, fraudulent transfers) in real time before authorization while maintaining ultra-low false-positive rates for routine transactions.
- **System Type:** Hybrid Inference Engine combining a **Deterministic Expert Rule Layer** with a **Machine Learning Scorer (XGBoost)** exposed via a **FastAPI REST API** packaged in a **Docker Container**.
- **Current Status:** 
  - Model trained, calibrated, and serialized (`riskstream_v1.joblib`).
  - Automated tests passing (9/9 unit and integration tests).
  - Docker container built and verified locally (`riskstream-api:latest`).
  - GitHub Actions CI configured for automated testing and image building.
  - Ready for cloud deployment on **AWS**.

---

## 2. AWS Environment & Account Context

When designing or executing AWS provisioning scripts (Terraform, CloudFormation, CDK, or AWS CLI), adhere to these parameters:

| Parameter | Configuration Value |
|---|---|
| **AWS Account ID** | `112997106414` |
| **IAM User Identity** | `arn:aws:iam::112997106414:user/admin-chirag` |
| **Default AWS Region** | `ap-south-1` (Asia Pacific - Mumbai) |
| **AWS CLI Version** | AWS CLI v2 (`2.37.12`) installed locally |
| **Target S3 Bucket** | `riskstream-mlops-artifacts-112997106414` (Region: `ap-south-1`) |
| **Target ECR Repository** | `112997106414.dkr.ecr.ap-south-1.amazonaws.com/riskstream-api` |
| **Inference Port** | `8000` |
| **Health Check Path** | `/health` (HTTP 200) |
| **Predict Path** | `/predict` (HTTP 200, POST JSON) |

---

## 3. Directory Structure & File Manifest

All project components reside under `ml-firstPipeline/`:

```text
ml-firstPipeline/
├── .github/
│   └── workflows/
│       └── ci.yaml              # GitHub Actions CI (pytest + docker build)
├── models/
│   ├── riskstream_v1.joblib     # Production model artifact (Pipeline + Rules + XGBoost)
│   └── xgb_booster.json         # Raw XGBoost booster exported from training
├── src/
│   └── riskstream_model.py      # Custom RiskStreamModel class definition & feature engineering
├── tests/
│   └── test_api.py              # Pytest suite (Unit tests with stub + Integration tests with joblib)
├── .dockerignore                # Excludes venv, pycache, tests, git from image
├── .gitignore                   # Excludes venv, pycache, db, logs from Git
├── Dockerfile                   # Multi-stage production container image (Python 3.13-slim)
├── Machine.MD                   # Project notes & exploratory data findings
├── main.py                      # FastAPI application with Pydantic validation & inference routes
├── rebuild_model.py             # Utility to reconstruct riskstream_v1.joblib from booster JSON
├── register_riskstream.py       # MLflow Model Registry registration script (pyfunc wrapper)
├── requirements.txt             # Production and testing Python dependencies
└── train.py                     # Legacy / baseline training script with MLflow logging
```

---

## 4. Dataset & Data Engineering Specification

### 4.1 Underlying Dataset: PaySim Synthetic Financial Dataset
- **Total Records:** 6,362,620 transactions
- **Total Features:** 11 raw columns
- **Class Imbalance:** Extreme fraud scarcity
  - Non-Fraud (`isFraud = 0`): 6,354,407 (~99.871%)
  - Fraud (`isFraud = 1`): 8,213 (~0.129%)
  - Imbalance Ratio: ~773:1 (Training set ratio: ~1,051:1)

### 4.2 Raw Features & Pre-Transaction Filtering (Data Leakage Prevention)

To prevent **Data Leakage**, the production model only accepts data available **at prediction time (before authorization)**:

| Feature Name | Type | Production Decision | Justification |
|---|---|---|---|
| `step` | Integer | Kept in offline analysis | Simulation hour (1 to 743). Excluded from API inference payload to decouple from synthetic simulation clocks. |
| `type` | String (Categorical) | **KEPT (Core Feature)** | Transaction type (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`). Fraud in PaySim occurs almost exclusively in `TRANSFER` and `CASH_OUT`. |
| `amount` | Float | **KEPT (Core Feature)** | Transaction amount in local currency. |
| `oldbalanceOrg` | Float | **KEPT (Core Feature)** | Sender's account balance before transaction. |
| `oldbalanceDest` | Float | **KEPT (Core Feature)** | Recipient's account balance before transaction. |
| `newbalanceOrig` | Float | **EXCLUDED (Leakage)** | Sender's balance *after* transaction. Not known before transaction authorization. |
| `newbalanceDest` | Float | **EXCLUDED (Leakage)** | Recipient's balance *after* transaction. Not known before transaction authorization. |
| `nameOrig` | String | **EXCLUDED** | High-cardinality unique customer account identifier. |
| `nameDest` | String | **EXCLUDED** | High-cardinality unique merchant/customer identifier. |
| `isFlaggedFraud` | Integer | **EXCLUDED** | Output of simple legacy rule engine (`amount > 200,000` on transfers). |
| `isFraud` | Integer (0/1) | **TARGET** | Ground truth label. |

---

## 5. Model Architecture & Pipeline Specification

The model implementation in `src/riskstream_model.py` follows a **Defense-in-Depth / Hybrid Pattern**:

```
                       Incoming Transaction JSON
                                  │
                                  ▼
                    Feature Engineering Pipeline
             (amount_to_balance, exact_drain, balance_gap, etc.)
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
       Deterministic Rule Layer             ML Scorer Pipeline
      - rule_exact_drain                   - ColumnTransformer (OHE + Passthrough)
      - rule_cap_10m                       - XGBClassifier (hist, scale_pos_weight=32)
      - rule_zero_cashout                          │
                 │                                 ▼
                 │                          fraud_score (0.0 to 1.0)
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                         Decision Aggregator
                     is_fraud = Rules OR (Score >= Threshold)
```

### 5.1 Engineered Features (`build_features`)
Computed on the raw columns (`type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`):
1. `amount_to_balance = amount / (oldbalanceOrg + 1.0)`: Measures the proportion of balance being moved.
2. `orig_zero = (oldbalanceOrg == 0)`: Binary flag identifying transactions originating from zero-balance accounts.
3. `exact_drain = (|amount - oldbalanceOrg| < 0.01) & (oldbalanceOrg > 0)`: Flags complete liquidation of sender funds (a signature PaySim fraud pattern).
4. `balance_gap = oldbalanceOrg - amount`: Post-transaction balance deficit or surplus.

### 5.2 Deterministic Heuristic Rules (`rule_flags`)
These catch hard-coded fraud patterns with 100% precision regardless of the ML score:
1. **`rule_exact_drain`**: Fired when `type` is in `['CASH_OUT', 'TRANSFER']` and `exact_drain == 1`.
2. **`rule_cap_10m`**: Fired when `type == 'TRANSFER'`, `amount == 10,000,000.0`, and `oldbalanceOrg > 10,000,000.0` (PaySim caps individual fraudulent transfers at 10 million).
3. **`rule_zero_cashout`**: Fired when `type == 'CASH_OUT'` and `amount == 0.0`.

### 5.3 Machine Learning Scorer (`Pipeline`)
- **Preprocessor:** `ColumnTransformer`
  - Numeric features (`amount`, `oldbalanceOrg`, `amount_to_balance`, `exact_drain`, `orig_zero`, `balance_gap`): `"passthrough"`
  - Categorical feature (`type`): `OneHotEncoder(categories=[["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]], handle_unknown="ignore")`
- **Estimator:** `xgboost.XGBClassifier`
  - `n_estimators`: 300
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `scale_pos_weight`: 32.0 (Balances precision vs. recall on positive fraud examples)
  - `tree_method`: `"hist"`
  - `random_state`: 42
  - Output: Continuous fraud probability score $\in [0.0, 1.0]$.

---

## 6. REST API Specification (`main.py`)

### 6.1 Framework & Server
- **Framework:** FastAPI (`^0.163.0`)
- **Server:** Uvicorn (`^0.54.0`)
- **Validation:** Pydantic v2
- **Port:** `8000` (configurable via host binding)
- **Lifespan Loader:** The model `riskstream_v1.joblib` is loaded once into memory (`app.state.model`) at application startup inside the FastAPI `lifespan` context.

### 6.2 Data Schemas

#### Input: `TransactionInput`
```json
{
  "type": "CASH_OUT",
  "amount": 146908.38,
  "oldbalanceOrg": 146908.38,
  "oldbalanceDest": 0.0
}
```
*Validation constraints:*
- `type`: Must be one of `["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]` (returns HTTP 422 on invalid).
- `amount`: `float >= 0.0` (returns HTTP 422 if negative).
- `oldbalanceOrg`: `float >= 0.0`.
- `oldbalanceDest`: `float >= 0.0`.

#### Output: `PredictionResponse`
```json
{
  "is_fraud": true,
  "fraud_score": 0.9999969005584717,
  "rules_fired": [
    "rule_exact_drain"
  ],
  "model_version": "riskstream_v1"
}
```

### 6.3 API Endpoints

| Method | Route | Description | Expected Status | Response Body |
|---|---|---|---|---|
| `GET` | `/` | Root readiness probe | `200 OK` | `{"status": "Ready to serve"}` |
| `GET` | `/health` | Cloud ALB / ECS health check | `200 OK` | `{"status": "healthy", "model_name": "riskstream", "model_version": "riskstream_v1"}` |
| `POST` | `/predict` | Single transaction fraud evaluation | `200 OK` | `PredictionResponse` object |
| `POST` | `/predict/batch` | Batch transaction fraud evaluation (up to 1,000 items) | `200 OK` | Array of `PredictionResponse` objects |

---

## 7. Containerization Specification (`Dockerfile`)

```dockerfile
# Base image matches Python 3.13 runtime
FROM python:3.13-slim

WORKDIR /app

# Cached dependency installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code & artifacts
COPY main.py .
COPY models/ models/
COPY src/ src/

# Crucial for unpickling custom riskstream_model.RiskStreamModel
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Key Container Characteristics:
- Image Size: ~428 MB (slim base image, no cache).
- No root GPU requirements (CPU-optimized `hist` tree method).
- Startup latency: < 2 seconds.
- Memory consumption: ~250MB RAM at runtime.

---

## 8. Automated Testing & CI/CD Pipeline

### 8.1 Local & Automated Test Suite (`tests/test_api.py`)
Tested via `pytest` (9 test cases):
1. `test_health_endpoint`: Asserts `/health` returns status healthy and model version.
2. `test_valid_prediction_returns_expected_keys`: Validates JSON schema response.
3. `test_invalid_type_returns_422`: Rejects unregistered transaction types.
4. `test_negative_amount_returns_422`: Rejects negative balances/amounts.
5. `test_missing_field_returns_422`: Rejects incomplete payloads.
6. `test_zero_amount_is_accepted`: Allows zero-dollar transactions.
7. `test_integration_exact_drain_flagged`: Real model flags account drains with `rule_exact_drain`.
8. `test_integration_cap_10m_flagged`: Real model flags 10M transfers with `rule_cap_10m`.
9. `test_integration_payment_not_flagged`: Routine payments are marked legitimate with 0 rules fired.

### 8.2 GitHub Actions Workflow (`.github/workflows/ci.yaml`)
- **Trigger:** On push or pull request to branch `main`.
- **Runner:** `ubuntu-latest`.
- **Steps:**
  1. Check out repository (`actions/checkout@v4`).
  2. Install Python 3.13 (`actions/setup-python@v5`).
  3. Install dependencies from `requirements.txt`.
  4. Run automated tests with `PYTHONPATH=src python -m pytest -v`.
  5. Build Docker image with `docker build -t riskstream-api:${{ github.sha }} .`.

---

## 9. AWS Deployment Architectures & Implementation Options

Here are the four standard AWS architectures for deploying this service, ordered from simplest/recommended to enterprise-grade.

---

### Option A: AWS App Runner (Recommended — Fast, Fully Managed, Low Ops)

**Why App Runner?** It provides direct container execution, automatic SSL/TLS, auto-scaling (down to minimum instances), built-in load balancing, and zero VPC configuration required.

```
Developer / Git ──> GitHub Actions CI/CD ──> Amazon ECR ──> AWS App Runner ──> Public HTTPS Endpoint
```

#### Step-by-Step CLI Setup for App Runner:
1. **Create ECR Repository:**
   ```bash
   aws ecr create-repository \
     --repository-name riskstream-api \
     --region ap-south-1 \
     --image-scanning-configuration scanOnPush=true
   ```

2. **Authenticate Docker and Push Image:**
   ```bash
   aws ecr get-login-password --region ap-south-1 | docker login --username AWS --password-stdin 112997106414.dkr.ecr.ap-south-1.amazonaws.com
   docker tag riskstream-api:latest 112997106414.dkr.ecr.ap-south-1.amazonaws.com/riskstream-api:v1
   docker push 112997106414.dkr.ecr.ap-south-1.amazonaws.com/riskstream-api:v1
   ```

3. **Create App Runner IAM Access Role:**
   Create trust policy file `ecr-trust.json`:
   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": { "Service": "build.apprunner.amazonaws.com" },
         "Action": "sts:AssumeRole"
       }
     ]
   }
   ```
   ```bash
   aws iam create-role --role-name AppRunnerECRAccessRole --assume-role-policy-document file://ecr-trust.json
   aws iam attach-role-policy --role-name AppRunnerECRAccessRole --policy-arn arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess
   ```

4. **Deploy App Runner Service:**
   Create `apprunner.json`:
   ```json
   {
     "ServiceName": "riskstream-fraud-api",
     "SourceConfiguration": {
       "AuthenticationConfiguration": {
         "AccessRoleArn": "arn:aws:iam::112997106414:role/AppRunnerECRAccessRole"
       },
       "AutoDeploymentsEnabled": true,
       "ImageRepository": {
         "ImageIdentifier": "112997106414.dkr.ecr.ap-south-1.amazonaws.com/riskstream-api:v1",
         "ImageConfiguration": {
           "Port": "8000",
           "RuntimeEnvironmentVariables": {
             "MODEL_VERSION": "riskstream_v1"
           }
         },
         "ImageRepositoryType": "ECR"
       }
     },
     "InstanceConfiguration": {
       "Cpu": "1 vCPU",
       "Memory": "2 GB"
     },
     "HealthCheckConfiguration": {
       "Protocol": "HTTP",
       "Path": "/health",
       "Interval": 10,
       "Timeout": 5,
       "HealthyThreshold": 1,
       "UnhealthyThreshold": 3
     }
   }
   ```
   ```bash
   aws apprunner create-service --cli-input-json file://apprunner.json --region ap-south-1
   ```

---

### Option B: Amazon ECS (Fargate) + Application Load Balancer (ALB)

**Best for:** Production VPC architecture, custom domain routing, private subnets, and WAF integration.

#### Components:
- **Networking:** 1 VPC, 2 Public Subnets (for ALB), 2 Private Subnets (for Fargate tasks), 1 NAT Gateway.
- **ALB:** Listens on port 80/443; forwards to Target Group (port 8000, Health check: `/health`).
- **ECS Cluster:** Fargate launch type (`FARGATE`).
- **Task Definition:** 0.5 vCPU (`512`), 1024 MB Memory (`1024`).
- **Service:** Desired count 2 (High Availability across AZs).

#### Task Definition JSON (`task-def.json`):
```json
{
  "family": "riskstream-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::112997106414:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "riskstream-api",
      "image": "112997106414.dkr.ecr.ap-south-1.amazonaws.com/riskstream-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        { "name": "MODEL_VERSION", "value": "riskstream_v1" }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/riskstream-api",
          "awslogs-region": "ap-south-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

---

### Option C: Amazon EC2 Single Host (Cost-Effective Learning / Development)

**Best for:** Quick prototyping with fixed cost (~$10/month on `t4g.small` or `t3.small`).

#### Setup Script (User Data):
```bash
#!/bin/bash
sudo apt-get update -y
sudo apt-get install -y docker.io awscli
sudo systemctl enable docker
sudo systemctl start docker

# Authenticate with ECR
aws ecr get-login-password --region ap-south-1 | sudo docker login --username AWS --password-stdin 112997106414.dkr.ecr.ap-south-1.amazonaws.com

# Pull and run container
sudo docker pull 112997106414.dkr.ecr.ap-south-1.amazonaws.com/riskstream-api:latest
sudo docker run -d --name riskstream-api --restart always -p 80:8000 112997106414.dkr.ecr.ap-south-1.amazonaws.com/riskstream-api:latest
```

---

## 10. MLflow Experiment Tracking & Remote S3 Storage Setup

To migrate the local SQLite/local artifacts MLflow instance into a centralized cloud setup:

### 10.1 Create S3 Artifact Bucket:
```bash
aws s3api create-bucket \
  --bucket riskstream-mlops-artifacts-112997106414 \
  --region ap-south-1 \
  --create-bucket-configuration LocationConstraint=ap-south-1
```

### 10.2 Block Public Access & Enable Encryption:
```bash
aws s3api put-public-access-block \
  --bucket riskstream-mlops-artifacts-112997106414 \
  --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

aws s3api put-bucket-encryption \
  --bucket riskstream-mlops-artifacts-112997106414 \
  --server-side-encryption-configuration '{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}'
```

### 10.3 Run MLflow with Remote S3 Backing:
```powershell
pip install boto3
$env:AWS_DEFAULT_REGION = "ap-south-1"

mlflow server `
  --backend-store-uri sqlite:///mlflow.db `
  --default-artifact-root s3://riskstream-mlops-artifacts-112997106414/mlflow/ `
  --host 0.0.0.0 `
  --port 7001
```

---

## 11. Required AWS IAM Permissions Summary

If provisioning AWS resources using the `admin-chirag` user or an automated CI/CD pipeline, ensure the following AWS Managed Policies or equivalent IAM statements are assigned:

1. **AmazonEC2ContainerRegistryPowerUser** (`arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryPowerUser`):
   - To create repositories, get auth tokens, and push/pull Docker images.
2. **AWSAppRunnerFullAccess** (`arn:aws:iam::aws:policy/AWSAppRunnerFullAccess`):
   - If deploying via Option A (App Runner).
3. **AmazonECS_FullAccess** (`arn:aws:iam::aws:policy/AmazonECS_FullAccess`):
   - If deploying via Option B (ECS Fargate).
4. **AmazonS3FullAccess** (scoped to `arn:aws:s3:::riskstream-mlops-artifacts-112997106414/*`):
   - To store and retrieve MLflow experiment artifacts and dataset snapshots.

---

## 12. Quick Reference: End-to-End Test Payloads

### Test Case 1: Known Fraud Pattern (Account Liquidation / Exact Drain)
- **Request (`POST /predict`):**
  ```json
  {
    "type": "CASH_OUT",
    "amount": 146908.38,
    "oldbalanceOrg": 146908.38,
    "oldbalanceDest": 0.0
  }
  ```
- **Expected Result:**
  ```json
  {
    "is_fraud": true,
    "fraud_score": 0.9999969,
    "rules_fired": ["rule_exact_drain"],
    "model_version": "riskstream_v1"
  }
  ```

### Test Case 2: Routine Legitimate Payment
- **Request (`POST /predict`):**
  ```json
  {
    "type": "PAYMENT",
    "amount": 9839.64,
    "oldbalanceOrg": 170136.0,
    "oldbalanceDest": 0.0
  }
  ```
- **Expected Result:**
  ```json
  {
    "is_fraud": false,
    "fraud_score": 0.000000026,
    "rules_fired": [],
    "model_version": "riskstream_v1"
  }
  ```
