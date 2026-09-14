# E-Commerce Customer Retention MLOps System

An end-to-end, batch-oriented MLOps platform that predicts non-contractual customer churn and prioritizes retention efforts based on Revenue at Risk. Built around the Brazilian Olist E-Commerce dataset, the system orchestrates feature engineering, model training, inference, and statistical monitoring.

The architecture emphasizes out-of-core data processing, strict point-in-time correctness, idempotent execution, and automated, ROI-gated model deployments.

## System Architecture

![E-Commerce Customer Retention MLOps System Architecture](docs/images/system-architecture.png)


## Core Pipelines

The system is decoupled into four independent pipelines, executed via GitHub Actions and mediated by Amazon S3 state pointers.

### 1. Data Pipeline (Feature Engineering)

Transforms raw, Hive-partitioned S3 data into an Analytical Base Table (ABT).

* **Out-of-Core Processing:** Uses DuckDB with the `httpfs` extension to read, execute joins, and write Parquet files directly from S3 to S3, bypassing local container memory limits.
* **Point-in-Time Correctness:** Enforces temporal boundaries via SQL. Future timestamps and status updates relative to the pipeline's `end_date` are explicitly masked to mathematically guarantee zero target leakage.
* **Data Contracts:** Performs structural schema and null checks against S3 partitions before executing feature engineering. Publishes deterministic telemetry (row counts, schema hashes) upon completion.

### 2. Training Pipeline (Optimization & Deployment)

Executes hyperparameter tuning, model calibration, and conditional deployment.

* **Model Pipeline:** Wraps an XGBoost estimator in a `CalibratedClassifierCV` (Isotonic regression) to output true probabilities necessary for expected ROI calculations. Uses a custom `CategoricalSchemaEnforcer` to guarantee categorical consistency between training and serving.
* **Hyperparameter Optimization:** Utilizes Optuna (`TPESampler`) for Bayesian search with early stopping.
* **Champion-Challenger Evaluation:** Evaluates the challenger model against the active champion using an Expected ROI (EROI) threshold. The challenger is deployed only if it exceeds the `eroi_hysteresis_margin`.
* **Atomic Deployment:** Writes serialized WORM artifacts (`model.pkl`, SHAP baselines) to S3, followed by a zero-downtime atomic overwrite of a global `model_state.json` routing pointer.

### 3. Inference Pipeline (Batch Scoring)

Executes daily batch scoring on the active customer base.

* **Logic Parity:** Uses the exact same DuckDB SQL feature generation module (`SharedFeatureGenerator`) as the training pipeline to eliminate training-serving skew.
* **Schema Validation:** Dynamically compares the materialized T-1 feature matrix against the serialized JSON schema of the champion model. Fails gracefully if required features are missing.
* **Output:** Generates a financially-prioritized business report (sorting by Churn Probability × Customer LTV) and a Hive-partitioned MLOps telemetry log.

### 4. Monitoring Pipeline (Drift & ROI Evaluation)

Evaluates statistical drift and business performance to conditionally trigger retraining.

* **SHAP-Guided Monitoring:** Extracts the top *N* features from the champion model's SHAP baseline and restricts covariate shift monitoring to these highly predictive features, minimizing false alarms.
* **Statistical Drift (PSI):** Computes a physical-type-aware Population Stability Index (PSI). Applies epsilon-clipping to algorithmically handle zero-bin edge cases.
* **Decoupled Trigger Engine:** Outputs an immutable `need_update.json` token to S3 if critical drift or performance degradation (relative Brier Score decay) is detected. The master DAG polls this token to conditionally trigger the training pipeline.

## Infrastructure & Security

### Containerization

Pipelines are containerized independently (`python:3.12.1-slim-bookworm`) to isolate dependencies and minimize the attack surface.

* **Multi-Stage Builds:** C-linked dependencies are compiled in a builder stage. Only the `/opt/venv` environment is copied to the final runtime image, entirely removing compilers and reducing image size.
* **Least Privilege (PoLP):** Containers execute under a non-root `pipelineuser` initialized with a disabled login shell (`/sbin/nologin`).
* **Memory Safety:** Explicitly provisions a `/tmp/data_pipeline_spill` directory, allowing DuckDB to spill to disk without triggering OS permission errors during massive joins.

### CI/CD & Orchestration

* **CI/CD:** GitHub Actions executes Ruff linting and Pytest suites on pull requests. Successful merges to `main` trigger a CD workflow that builds and pushes ECR images tagged with the Git commit hash (`${{ github.sha }}`).
* **OIDC Authentication:** AWS authentication is handled via OpenID Connect (OIDC), removing long-lived IAM credentials from GitHub.
* **Orchestration:** A cron-triggered DAG initializes temporal context (`run_id`, lookback windows) injected via environment variables. Execution is strictly sequential with concurrency locks to prevent Data Lake corruption.

## Local Setup & Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/thesahilmandal/E-Commerce-Customer-Retention-MLOps-System.git
cd E-Commerce-Customer-Retention-MLOps-System

python3.12 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

```

### 2. AWS Prerequisites

1. Configure your local AWS CLI:

```bash
aws configure

```

2. Create two S3 buckets and define them in a local `.env` file at the repository root (use bucket names, not URIs):

```env
S3_CUSTOMER_DATABASE="<YOUR_CUSTOMER_DATABASE_BUCKET_NAME>"
S3_PIPELINE_RUN_ARTIFACTS="<YOUR_ARTIFACTS_BUCKET_NAME>"

```

3. Load the environment variables:

```bash
set -a && source .env && set +a

```

### 3. Data Lake Bootstrap

Populate the S3 Data Lake with the simulated historical dataset.

```bash
# 1. Ingest raw dataset into artifacts bucket
python -m tools.raw_data_loader

# 2. Transform to Hive-partitioned format and move to Customer Database bucket
python -m tools.hive_partitioned_generator

```

### 4. Local End-to-End Execution

Run the core pipelines sequentially via their module runners. You can inspect the `logs/` directory for operational telemetry during and after execution.

```bash
# 1. Data Pipeline: Materialize feature matrix
python -m pipelines.data_pipeline.src.runner \
  --run-id="testing_01" \
  --start-date="2016-09-01" \
  --end-date="2018-03-01"

# 2. Training Pipeline: Optimize, calibrate, and register Champion
# Replace bucket placeholder below with your artifacts bucket name
python -m pipelines.training_pipeline.src.runner \
  --run-id="testing_01" \
  --dataset-uri="s3://<YOUR_ARTIFACTS_BUCKET_NAME>/feature_store/testing_01/dataset.parquet"

# 3. Inference Simulation: Generate synthetic 'yesterday' footprints
python -m tools.synthetic_data_generator

# 4. Inference Pipeline: Score the active customer base
python -m pipelines.inference_pipeline.src.runner \
  --run-id="testing_01"

# 5. Monitoring Pipeline: Evaluate statistical drift and ROI
python -m pipelines.monitoring_pipeline.src.runner \
  --run-id="testing_01" \
  --execution-date="$(date +%Y-%m-%d)"

```

### 5. Cloud Deployment (GitHub Actions)

1. **AWS Infrastructure:** Create an IAM OIDC Identity Provider connected to your GitHub repository. Provision four Amazon ECR repositories to host the pipeline images.
2. **GitHub Variables:** Configure the following repository variables (not secrets) in GitHub:
* `AWS_REGION`
* `AWS_ROLE_ARN` *(IAM role assumable via OIDC)*
* `ECR_DATA_PIPELINE`, `ECR_TRAINING_PIPELINE`, `ECR_INFERENCE_PIPELINE`, `ECR_MONITORING_PIPELINE` *(Repository names, not URIs)*
* `S3_PIPELINE_RUN_ARTIFACTS`, `S3_CUSTOMER_DATABASE` *(Bucket names, not URIs)*


3. **Workflow Execution:** Push changes to `main`. Manually trigger the deployment workflows in this strict order for the initial run:
* **First:** Run **Continuous Deployment** (`cd.yml`) to build and push images to ECR.
* **Second:** Run **Master Orchestrator** (`master_orchestrator.yml`) to execute the nightly batch DAG.



## Repository Structure

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml                        # Code quality & testing
│       ├── cd.yml                        # Build & push to ECR
│       └── master_orchestrator.yml       # Production ML orchestration DAG
├── pipelines/
│   ├── data_pipeline/
│   │   ├── configs/                      # YAML definition files
│   │   ├── src/                          # Discovery, validation, materialization
│   │   ├── tests/
│   │   └── Dockerfile                    # Multi-stage definition
│   ├── training_pipeline/
│   │   ├── configs/
│   │   ├── src/                          # Model tuning, evaluation, WORM registry
│   │   ├── tests/
│   │   └── Dockerfile
│   ├── inference_pipeline/
│   │   ├── configs/
│   │   ├── src/                          # Matrix builder, contract validation, scoring
│   │   ├── tests/
│   │   └── Dockerfile
│   └── monitoring_pipeline/
│       ├── configs/
│       ├── src/                          # Drift, performance, and rule engine
│       ├── tests/
│       └── Dockerfile
├── shared_core/
│   ├── cloud/                            # Idempotent S3 operations
│   ├── exceptions/                       # Centralized error handling
│   ├── features/                         # Shared temporal SQL logic
│   ├── logging/                          # Standardized JSON log formatting
│   └── utils/
├── tools/                                # Synthetic data generation & loaders
├── .gitignore
├── requirements.txt
└── README.md

```