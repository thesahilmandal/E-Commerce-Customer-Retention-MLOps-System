# E-commerce Customer Retention MLOps System

An end-to-end, automated MLOps system for customer churn prediction, production monitoring, business-driven model evaluation, and conditional retraining.

## Overview

Customer retention is generally more cost-effective than customer acquisition. This project implements an end-to-end machine learning system that predicts customer churn, produces actionable customer-risk reports, continuously monitors the deployed model, and automatically retrains and promotes a new model when measurable degradation is detected.

The system is organized into four decoupled ML pipelines:

* **Data Pipeline** — materializes point-in-time training features from the S3 data lake.
* **Training Pipeline** — trains, calibrates, evaluates, and conditionally promotes churn models.
* **Inference Pipeline** — performs scheduled batch scoring and publishes business and telemetry outputs.
* **Monitoring Pipeline** — detects statistical drift and performance degradation and produces a deterministic retraining decision.

These pipelines are independently containerized, validated through CI, deployed to Amazon ECR, and coordinated by a GitHub Actions-based Master Orchestrator.

The project also includes supporting data-engineering utilities for bootstrapping the initial Bronze Data Lake and generating synthetic T-1 transactional data for manually testing the inference workflow.

After the initial model deployment, the intended workflow operates without manual intervention: nightly inference is followed by monitoring, and retraining is initiated only when the monitoring decision indicates that the production model requires an update.

---

## System at a Glance

| Area                            | Implementation                                       |
| ------------------------------- | ---------------------------------------------------- |
| **ML problem**                  | Customer churn prediction                            |
| **Processing paradigm**         | Batch / scheduled ML workloads                       |
| **Data lake**                   | Amazon S3                                            |
| **Query / analytical engine**   | DuckDB                                               |
| **Feature format**              | Parquet                                              |
| **Model**                       | XGBoost classifier                                   |
| **Probability calibration**     | Isotonic Regression                                  |
| **Hyperparameter optimization** | Optuna                                               |
| **Explainability**              | SHAP                                                 |
| **Model registry**              | Versioned S3 artifacts + mutable model-state pointer |
| **Monitoring**                  | PSI, Brier Score, Log Loss, Realized ROI             |
| **Containers**                  | Docker                                               |
| **CI/CD**                       | GitHub Actions                                       |
| **Container registry**          | Amazon ECR                                           |
| **Cloud authentication**        | AWS OIDC                                             |
| **Orchestration**               | GitHub Actions                                       |
| **Configuration**               | YAML + immutable Python dataclasses                  |
| **Data partitioning**           | Hive-style S3 partitions                             |
| **Execution model**             | Ephemeral batch containers                           |
| **Test data generation**        | Deterministic, Hive-partitioned synthetic T-1 data   |

---

## Architecture

The system separates data preparation, model lifecycle management, production inference, and production monitoring into independently executable components.

```text
                         ┌──────────────────────────┐
                         │      Amazon S3 Data      │
                         │       Lake / Bronze      │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │      Data Pipeline       │
                         │                          │
                         │ Discovery → Validation   │
                         │ → Feature Materialize    │
                         │ → Metadata Registry      │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                              Feature Matrix
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │     Training Pipeline    │
                         │                          │
                         │ Split → Train → Evaluate │
                         │ → Champion/Challenger    │
                         └────────────┬─────────────┘
                                      │
                             Model + State Pointer
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    Inference Pipeline    │
                         │                          │
                         │ Resolve Model → Build    │
                         │ Features → Validate →    │
                         │ Score → Publish          │
                         └────────────┬─────────────┘
                                      │
                             Business Report
                               + Telemetry
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   Monitoring Pipeline    │
                         │                          │
                         │ Drift → Performance →    │
                         │ ROI → Rule Engine        │
                         └────────────┬─────────────┘
                                      │
                               need_update.json
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │    Master Orchestrator   │
                         │                          │
                         │ Decision Gate             │
                         │       │                  │
                         │       ▼                  │
                         │ Data → Training           │
                         └──────────────────────────┘
```

The four production pipelines share common feature-generation and configuration infrastructure while remaining independently executable.

Supporting utilities provide the initial data-lake bootstrap and controlled synthetic data for inference testing.

---

# End-to-End Workflow

## Initial Data-Lake Bootstrap

Before the automated ML lifecycle can begin, the initial data foundation and first production model must be created manually.

The bootstrap sequence is:

```text
Kaggle E-commerce Dataset
          ↓
     Raw Data Loader
          ↓
Raw Parquet Files in S3
          ↓
  Hive Partition Generator
          ↓
      Bronze Data Lake
          ↓
       Data Pipeline
          ↓
   Initial Feature Matrix
          ↓
    Training Pipeline
          ↓
 Initial Production Model
```

The raw-data loader downloads the `olistbr/brazilian-ecommerce` dataset through `kagglehub`, discovers CSV files, reads them in chunks, converts the chunks to Parquet, and uploads the resulting files to S3. The default CSV chunk size is `100,000` rows.

The Hive partition generator then downloads the required raw Parquet datasets, derives `year`, `month`, and `day` from `order_purchase_timestamp`, and writes day-level Snappy-compressed Parquet partitions for orders, customers, and payments. Customer and payment partitions inherit their temporal partitioning from the associated order records.

The resulting partitions are synchronized into the configured Bronze Data Lake S3 locations. The generator also uses a 4-thread, 4 GB DuckDB configuration and removes its temporary workspace and closes the DuckDB connection after execution.

This bootstrap process is separate from the recurring ML lifecycle.

## Synthetic T-1 Data for Inference Testing

The project also includes a synthetic data generator that simulates an upstream daily data-engineering batch.

Its purpose is to populate the Bronze/customer database with data for the previous day so that the Inference Pipeline can be manually tested against a realistic T-1 data state.

By default, the generator:

* Targets the previous UTC day.
* Generates `1,000` customers.
* Generates `1,500` orders.
* Produces customer, order, and payment records.
* Writes Snappy-compressed Parquet directly to S3.
* Uses Hive-style `year/month/day` partitioning.
* Creates `_SUCCESS` markers after successful completion.
* Skips an already-completed target partition.
* Allows an explicit target date through `--target-date`.

The synthetic customer pool uses a fixed random seed so the generated global customer identifiers remain stable across executions. This preserves customer continuity across daily test runs and allows rolling-window feature engineering to observe the same customers over time.

The generator also cleans the target partition before writing it, which prevents duplicate records if a run needs to be repeated after a partial failure.

## 1. Nightly Inference

The Master Orchestrator starts the inference pipeline using the currently active Champion model.

The Inference Pipeline:

1. Resolves the active model from the S3 model-state pointer.
2. Retrieves the corresponding model bundle.
3. Builds the current point-in-time feature matrix.
4. Validates the generated schema against the model's schema contract.
5. Performs batch predictions.
6. Generates a business-facing churn report.
7. Generates a complete MLOps telemetry dataset.
8. Publishes the outputs to partitioned S3 locations.

## 2. Production Monitoring

The Monitoring Pipeline then:

1. Resolves the active model's monitoring baselines.
2. Loads current inference telemetry.
3. Retrieves matured ground-truth labels for the configured lookback period.
4. Calculates feature and prediction drift.
5. Evaluates predictive performance.
6. Calculates realized financial impact.
7. Applies configured monitoring rules.
8. Publishes a `need_update.json` decision artifact.

## 3. Conditional Retraining

The Master Orchestrator reads the monitoring decision artifact.

If no retraining is required, the workflow ends.

If `need_update == true`:

```text
Monitoring Decision
        │
        ▼
   Data Pipeline
        │
        ▼
   Training Dataset
        │
        ▼
 Training Pipeline
        │
        ▼
Challenger vs Champion
        │
        ├── Challenger loses → retain Champion
        │
        └── Challenger wins  → register new Champion
```

The next inference execution resolves the newly promoted Champion through the model-state pointer without requiring an inference code change or container rebuild.

---

# 1. Data Pipeline

## Purpose

The Data Pipeline transforms raw transactional data from the S3 Bronze Data Lake into a point-in-time feature matrix suitable for model training and evaluation.

It also provides the feature artifact and metadata contracts consumed by downstream components.

The pipeline is intentionally implemented as a single-node, out-of-core workload using DuckDB rather than introducing a distributed processing framework.

## Workflow

The pipeline is organized into four sequential stages:

```text
Data Discovery
      ↓
Data Validation
      ↓
Feature Materialization
      ↓
Metadata Registry
```

### Data Discovery

The pipeline first calculates the expected temporal bounds and performs lightweight S3 partition checks.

It uses `MaxKeys=1` list operations to determine whether required Hive partitions exist before performing expensive processing.

Missing required partitions cause the pipeline to terminate early.

### Data Validation

The validation stage performs out-of-core queries against S3 to verify:

* Expected schema
* Record counts
* Required identifiers
* Non-null constraints

Critical columns such as `order_id` and `customer_unique_id` are explicitly validated.

Validation behavior is controlled through configurable policies such as `fail_fast` and `strict_mode`.

### Feature Materialization

The pipeline joins customer, order, and payment data to produce historical features and forward-looking churn labels.

The resulting Parquet dataset is streamed directly to the Feature Store S3 prefix rather than being retained as a large local artifact.

### Metadata Registry

After feature materialization, the pipeline generates deterministic metadata including:

* Row counts
* MD5 schema hashes
* Standardized JSON metadata

This metadata acts as a downstream data contract.

## Point-in-Time Feature Generation

The feature-generation process explicitly enforces temporal boundaries.

Transactions occurring on or after the execution `end_date` are excluded from historical feature generation.

The shared feature-generation logic also masks future information. For example, when delivery timestamps occur after the point-in-time cutoff, they are nullified and the corresponding order state is reverted to `processing`.

This prevents future information from entering historical features.

## Shared Feature Generation

Feature engineering is centralized in a `SharedFeatureGenerator`.

The same feature-generation implementation is consumed by training and inference.

This establishes a single source of truth for feature construction and prevents the training and serving environments from implementing different feature logic.

## Compute Architecture

The pipeline uses DuckDB with S3 access through `httpfs`.

Temporal filters are translated into Hive partition predicates such as:

```text
year
month
day
```

DuckDB can therefore push the relevant filters toward the S3 data source instead of scanning the complete data lake.

The pipeline uses explicit resource limits, including:

* 4 DuckDB threads
* 8 GB memory limit
* Disk spilling for intermediate operations exceeding available memory

This provides an out-of-core execution model without requiring distributed infrastructure.

## Configuration and Resource Management

Pipeline configuration is defined through YAML and resolved through environment variables.

Configuration is mapped into immutable nested Python `dataclasses` using `frozen=True`.

A `PipelineContext` manages:

* `run_id`
* Execution dates
* DuckDB connection
* S3 client
* Resource lifecycle
* Cleanup

The context manager guarantees resource teardown through `__exit__`.

S3 synchronization uses thread-local `boto3` clients through a custom `S3Sync` utility and supports native botocore credential resolution and STS credential rotation.

---

# 2. Training Pipeline

## Purpose

The Training Pipeline trains candidate churn models and determines whether a newly trained Challenger should replace the current production Champion.

Model promotion is based not only on predictive metrics but also on expected financial return.

## Workflow

```text
Feature Matrix
      ↓
Data Processor
      ↓
Train / Validation / Test
      ↓
Model Trainer
      ↓
Optuna + XGBoost
      ↓
Probability Calibration
      ↓
Model Evaluator
      ↓
Challenger vs Champion
      ↓
Model Registry
```

## Data Processing

The pipeline loads the master feature matrix from S3 and performs an out-of-core randomized Train/Validation/Test split using DuckDB.

A schema enforcer is fitted during training to preserve data-type consistency between training and inference.

The split is performed directly against the Parquet dataset using DuckDB SQL rather than loading the complete dataset into memory.

## Model Training

The model-training stage uses:

* XGBoost
* Optuna for hyperparameter optimization
* Isotonic Regression for probability calibration
* SHAP for global feature importance

The optimized XGBoost classifier is calibrated using `CalibratedClassifierCV` with Isotonic Regression and five-fold cross-validation.

Calibration is particularly relevant because downstream business evaluation depends on the predicted probability values rather than ranking alone.

## Expected ROI Evaluation

The pipeline evaluates models using business-oriented Expected ROI.

The decision logic uses:

```text
EROI =
(True Positives × Customer LTV × Save Rate)
−
(Total Interventions × Campaign Cost)
```

Prediction thresholds from `0.01` through `0.99` are evaluated to determine the threshold that maximizes EROI.

The Challenger is then evaluated against the active Champion using the same holdout Test set.

## Champion–Challenger Hysteresis

The system does not replace the production model merely because a new model performs slightly better.

The Challenger must exceed the Champion's EROI by the configured `eroi_hysteresis_margin`.

For example, a configured 2% hysteresis margin requires a material improvement before promotion.

This creates an explicit business gate against unnecessary model changes.

## Categorical Schema Enforcement

XGBoost's native categorical handling requires consistent Pandas categorical types and category definitions.

A custom `CategoricalSchemaEnforcer` is fitted on the training data and bundled with the model.

It ensures consistency in:

* Column ordering
* Categorical types
* Category definitions
* Memory layout

The enforcer is packaged with the estimator as part of a deployment-ready pipeline.

## Model Registry

Successful model promotion uses a two-phase deployment process.

### Phase 1 — Immutable Artifacts

The complete model bundle is stored under a versioned, WORM-oriented S3 path identified by `run_id`.

The bundle includes:

* Model pipeline
* Categorical schema
* SHAP artifacts
* Monitoring baselines
* Metadata
* Curated inference `requirements.txt`

### Phase 2 — Mutable State Pointer

A single `model_state.json` pointer is updated to identify the active model version.

Inference consumes the state pointer rather than hardcoded model paths.

This separates immutable model artifacts from the mutable notion of which model is currently active.

## Monitoring Contracts

The Training Pipeline generates monitoring reference data alongside the model.

This includes:

* Feature distribution statistics
* Means
* Standard deviations
* Categorical frequencies
* Baseline Log Loss
* Baseline Brier Score

These artifacts are stored with the model and consumed by the Monitoring Pipeline.

---

# 3. Inference Pipeline

## Purpose

The Inference Pipeline performs scheduled batch scoring of the customer population and converts model predictions into both business-facing outputs and production telemetry.

## Workflow

```text
Model Loader
      ↓
Feature Matrix Builder
      ↓
Inference Validator
      ↓
Report Generator
      ↓
Report Publisher
```

## Dynamic Model Resolution

The pipeline reads the active `model_state.json` pointer from the S3 Model Registry.

It uses the referenced `run_id` to retrieve the corresponding model bundle, schemas, and monitoring artifacts.

No model version or S3 model URI is hardcoded into the inference implementation.

This allows model deployment to occur independently of inference code releases.

## Feature Generation

Inference reuses the same `SharedFeatureGenerator` used by the training workflow.

DuckDB generates the T-1 point-in-time feature matrix directly against the Bronze Data Lake.

The resulting feature matrix is stored as local Parquet for batch processing.

## Schema Validation

Before scoring, the `InferenceValidator` compares the generated feature schema with the schema required by the active model.

The validation checks for missing predictive features and required system-level mapping columns such as `customer_unique_id`.

A validation failure stops inference before model scoring occurs.

## Dual-Output Design

The inference system deliberately produces two different outputs.

### Business Report

The business-facing CSV contains actionable customer-level information such as:

* `customer_unique_id`
* Churn probability
* Binary risk flag
* `revenue_at_risk`

`revenue_at_risk` is calculated using churn probability and customer monetary value.

The report is intended to support customer-retention decisions without exposing unnecessary ML implementation details.

### MLOps Telemetry

The telemetry output is a compressed Parquet dataset containing the complete feature matrix together with:

* `inference_run_id`
* Raw prediction probabilities

This dataset provides the information required for subsequent production monitoring and drift analysis.

## Master Inference Ledger

The Report Publisher aggregates stage-level execution metadata into a Master Inference Ledger.

The ledger records information including:

* Dataset sizes
* Row counts
* Validation outcomes
* Execution times
* Stage metadata

Business reports, telemetry, and ledger artifacts are uploaded using Hive-partitioned S3 paths.

---

# 4. Monitoring Pipeline

## Purpose

The Monitoring Pipeline evaluates whether the active production model remains suitable for continued use.

It combines:

* Label-independent statistical drift
* Label-dependent predictive performance
* Business-level realized ROI

The output is a deterministic retraining decision rather than a direct invocation of the Training Pipeline.

## Workflow

```text
Baseline & Telemetry Resolver
            ↓
Statistical Drift Calculator
            ↓
Performance Evaluator
            ↓
Rule Engine
            ↓
Artifact Publisher
```

## Baseline and Telemetry Resolution

The pipeline retrieves the active model's:

* Reference distributions
* Performance baselines
* SHAP feature importance
* Monitoring metadata

It also retrieves current inference telemetry and joins historical predictions with matured labels from the Bronze Data Lake.

The lookback period is configured around a T-30 maturity window.

## Statistical Drift

The Monitoring Pipeline calculates Population Stability Index (PSI) for:

* Selected predictive features
* Prediction probability distribution

The most important predictive features are selected using SHAP importance generated during model training.

Categorical PSI calculations preserve the training category definitions so that production categories remain aligned with the reference distributions.

## Performance Monitoring

When ground-truth labels have matured, the pipeline calculates:

* Brier Score
* Log Loss
* Realized ROI

The system evaluates prediction quality and financial performance rather than relying solely on feature drift.

## Realized ROI

The monitoring system translates observed model outcomes into financial impact using configured business parameters such as:

* Campaign cost
* Customer LTV
* Intervention save rate

The result is a realized ROI value for the matured cohort.

## Rule Engine

The Rule Engine compares monitoring results with configured thresholds.

Retraining can be triggered by:

1. Critical prediction drift
2. Critical feature drift
3. Severe performance degradation

The thresholds are defined in YAML configuration.

Examples include:

```yaml
prediction_drift_threshold_psi: 0.20
brier_degradation_threshold_factor: 1.05
```

The final decision is published as:

```text
need_update.json
```

This artifact acts as the interface between monitoring and orchestration.

## Handling Operational Edge Cases

### Zero Inference Traffic

If current-day telemetry is unavailable, the system constructs an empty strongly typed Parquet schema rather than treating the absence of traffic as a fatal pipeline error.

### Immature Monitoring Window

When insufficient historical data exists to perform the T-30 performance evaluation, the system marks the state as:

```text
INSUFFICIENT_LOOKBACK_MATURITY
```

Label-independent drift monitoring can still proceed.

### Numerical Stability

PSI and Log Loss calculations use configurable epsilon values to avoid undefined operations involving zero probability or zero-frequency bins.

## Monitoring Lineage

Each stage writes execution metadata containing information such as:

* Execution times
* Row counts
* Applied thresholds

The Artifact Publisher combines these records into a Master Execution Ledger before publishing the monitoring artifacts.

---

# 5. Dockerization and Unit Testing

## Container Architecture

The Data, Training, Inference, and Monitoring pipelines are packaged as independent Docker images.

Each image uses a multi-stage build based on:

```text
python:3.12.1-slim-bookworm
```

### Builder Stage

The builder:

* Installs required compilation dependencies.
* Creates `/opt/venv`.
* Installs pipeline-specific Python requirements.

### Runtime Stage

The runtime image receives only the compiled virtual environment and required runtime dependencies.

Build toolchains and caches are excluded.

Each container executes as a non-root user.

## Container Isolation

Docker build contexts explicitly copy:

```text
shared_core/
pipelines/<pipeline_name>/
```

rather than copying the entire repository.

This prevents unrelated pipeline code, local caches, and artifacts from being incorporated into individual images.

Each pipeline therefore retains its own dependency and deployment boundary.

## Security

Containers execute as a dedicated non-root `pipelineuser` with a disabled login shell.

File permissions are scoped to the required application directories.

The images omit exposed ports because these components are designed as short-lived batch jobs rather than persistent web services.

## Runtime Dependencies

Pipeline-specific operating-system dependencies are installed only where required.

For example:

* Training and Inference include `libgomp1` for OpenMP support.
* Pipelines include `ca-certificates` for secure TLS communication with AWS services.

The Data Pipeline also provisions a writable DuckDB spill directory for out-of-core processing.

## Runtime Configuration

Containers use:

```text
PYTHONPATH=/app
PYTHONDONTWRITEBYTECODE=1
PYTHONUNBUFFERED=1
```

Pipelines are launched through module-based entrypoints such as:

```bash
python -m pipelines.<pipeline_name>.src.runner
```

This keeps execution consistent with the repository's Python package structure.

## Testing

The project's CI workflow executes the available `pytest` unit and integration test suite as part of the quality gate.

Testing is therefore integrated into the same automated workflow that validates changes before they reach the deployment path.

---

# 6. CI/CD

## Purpose

The CI/CD system automates code validation and container deployment using GitHub Actions.

It establishes a repeatable path from source-code changes to versioned container artifacts in Amazon ECR.

## Continuous Integration

The CI workflow is triggered by Pull Requests targeting `main`.

The workflow:

1. Creates an ephemeral Ubuntu runner.
2. Caches Python dependencies.
3. Runs `ruff`.
4. Runs the `pytest` unit and integration test suite.

The test step uses `if: always()` so that tests still execute when the linting step fails, providing both structural and functional feedback in a single workflow run.

## Continuous Deployment

The CD workflow runs on pushes and merged changes to `main`.

It:

1. Authenticates to AWS.
2. Authenticates with Amazon ECR.
3. Builds the four pipeline images.
4. Tags each image with the Git commit SHA.
5. Pushes the images to their corresponding ECR repositories.

The four pipelines therefore have independent container registry destinations.

## AWS OIDC Authentication

The workflows use GitHub Actions OIDC authentication rather than storing long-lived AWS access keys in GitHub Secrets.

The workflow requests:

```text
id-token: write
```

and dynamically assumes a configured AWS IAM role.

This removes the need to manage static AWS access credentials in the CI/CD environment.

## Immutable Image Tagging

Images are tagged with:

```text
${{ github.sha }}
```

rather than a mutable `latest` tag.

The tag provides a direct relationship between:

```text
Git commit → Docker image → ECR artifact → execution
```

This establishes deterministic artifact lineage.

---

# 7. Master Orchestrator

## Purpose

The Master Orchestrator coordinates the complete nightly ML lifecycle.

It is implemented as a GitHub Actions dependency graph running on ephemeral `ubuntu-latest` runners.

## Workflow

```text
Global Initialization
        ↓
    Inference
        ↓
   Monitoring
        ↓
  Decision Gate
        ↓
   ┌────┴────┐
   │         │
 false      true
   │         │
   ▼         ▼
  End    Data Pipeline
             ↓
      Training Pipeline
```

## Global Initialization

The initialization stage generates the execution context shared across the workflow.

This includes:

* Globally unique `RUN_ID`
* Logical execution dates
* Temporal data windows
* Hive-partitioned S3 paths

The generated values are exported as workflow outputs and injected into downstream container executions.

## Inference and Monitoring

Inference and monitoring are mandatory stages of the nightly workflow.

If inference fails, the dependency graph prevents monitoring from proceeding.

Monitoring evaluates the production model and creates the `need_update.json` decision artifact.

## Decision Gate

The Decision Gate is an infrastructure-level interface between monitoring and retraining.

It:

1. Retrieves the decision artifact from S3.
2. Verifies that the artifact exists.
3. Parses the boolean decision using `jq`.
4. Determines whether the retraining branch should execute.

If the decision artifact is missing, the job fails instead of assuming that retraining is unnecessary.

## Conditional Retraining

Only when:

```text
need_update == true
```

does the orchestrator execute:

```text
Data Pipeline → Training Pipeline
```

This prevents unnecessary data processing and model training when the deployed model remains within its configured monitoring boundaries.

## Concurrency Control

The workflow uses GitHub Actions concurrency controls:

```yaml
group: production-mlops-orchestrator
cancel-in-progress: false
```

This prevents overlapping nightly lifecycle executions from simultaneously modifying or consuming shared S3 state.

## Zero-Trust AWS Access

Each orchestrator job independently authenticates to AWS using OIDC and dynamically assumes the configured AWS IAM role.

No long-lived IAM user credentials are required by the orchestration workflow.

## Cross-Pipeline Traceability

The initialization stage creates a global `RUN_ID`.

That identifier and the shared execution dates are injected into every container invocation.

As a result, artifacts generated by Data, Training, Inference, and Monitoring can be associated with the same orchestration execution.

## Immutable Container Execution

The orchestrator executes Docker images using the Git commit SHA as the image tag.

This ensures that scheduled workloads execute a specific immutable container artifact rather than a mutable `latest` image.

---

# Model Lifecycle

The system separates model artifacts from the identity of the active production model.

```text
                         Training
                            │
                            ▼
                     Challenger Model
                            │
                            ▼
                      EROI Evaluation
                            │
                     ┌──────┴──────┐
                     │             │
                  Reject        Promote
                     │             │
                     ▼             ▼
               Keep Champion   Immutable Model
                                     │
                                     ▼
                              model_state.json
                                     │
                                     ▼
                                  Inference
```

A promoted model is stored as an immutable versioned artifact.

The mutable `model_state.json` pointer identifies which version is currently active.

Inference therefore resolves the current Champion dynamically, while historical model bundles remain versioned separately.

---

# Production Monitoring and Retraining Loop

The complete lifecycle can be summarized as:

```text
                   ┌───────────────────┐
                   │  Current Champion │
                   └─────────┬─────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Nightly Scoring │
                    └────────┬────────┘
                             │
                    Business + Telemetry
                             │
                             ▼
                    ┌─────────────────┐
                    │    Monitoring   │
                    └────────┬────────┘
                             │
                Drift / Performance / ROI
                             │
                             ▼
                    ┌─────────────────┐
                    │   Rule Engine   │
                    └────────┬────────┘
                             │
                       need_update?
                        /          \
                      No            Yes
                      │              │
                      ▼              ▼
                     End      ┌─────────────┐
                              │ Data        │
                              │ Pipeline    │
                              └──────┬──────┘
                                     │
                                     ▼
                              ┌─────────────┐
                              │ Training    │
                              │ Pipeline    │
                              └──────┬──────┘
                                     │
                              Champion/Challenger
                                     │
                                     ▼
                              Model Promotion
                                     │
                                     └───────→ Next
                                              Inference
```

The system therefore combines predictive ML, business-based model evaluation, production monitoring, and conditional model lifecycle management into a single automated workflow.

---

# Engineering Principles Demonstrated

## Single Source of Truth for Features

Training and inference reuse the same feature-generation implementation, reducing the risk of inconsistent feature logic between development and serving.

## Point-in-Time Correctness

Temporal boundaries and future-state masking are enforced during feature generation to prevent information leakage.

## Out-of-Core Processing

DuckDB provides a practical single-node execution model with memory limits and disk spilling rather than introducing distributed infrastructure where it is not required by the implementation.

## Explicit Data Contracts

The system uses schemas, metadata artifacts, model contracts, monitoring baselines, and decision tokens to establish explicit interfaces between independently executed components.

## Immutable Artifacts with Mutable State

Model bundles and container images are versioned immutably, while small state pointers determine which artifact is currently active.

## Business-Aware Model Selection

Model promotion is tied to Expected ROI rather than treating statistical model performance as the sole deployment criterion.

## Fail-Fast Validation

Data availability, schema compatibility, decision artifacts, and model contracts are validated before dependent processing proceeds.

## Decoupled Continual Learning

Monitoring produces a decision artifact rather than directly invoking retraining, allowing the orchestration layer to control the lifecycle independently.

## Ephemeral Execution

The pipelines are designed as short-lived batch workloads running in isolated containers with explicit resource management and cleanup.

## Zero-Trust Cloud Authentication

CI/CD and orchestration use AWS OIDC role assumption instead of long-lived cloud credentials.

## Traceability

Run IDs, Git commit-based image tags, S3 artifacts, execution metadata, and master ledgers provide cross-component lineage.

---

# Key Technical Capabilities

This project demonstrates the integration of:

* End-to-end ML pipeline design
* Feature engineering and point-in-time data preparation
* Data validation and contracts
* Out-of-core analytical processing
* XGBoost model training
* Hyperparameter optimization with Optuna
* Probability calibration
* SHAP-based model explainability
* Business-driven model evaluation
* Champion–Challenger model management
* Model artifact versioning
* Batch inference
* Production telemetry
* Statistical drift monitoring
* Predictive performance monitoring
* Automated retraining decisions
* Docker multi-stage builds
* Non-root container execution
* Unit and integration testing
* GitHub Actions CI/CD
* Amazon ECR deployment
* AWS OIDC authentication
* Automated workflow orchestration
* Synthetic-data-driven pipeline testing
* Cross-pipeline execution lineage

---

# Repository Architecture

The implementation is organized around shared functionality and independently executable pipeline boundaries.

```text
.
├── .github/
│   └── workflows/
│       ├── ci.yml                         # Code quality & testing
│       ├── cd.yml                         # Build & push to ECR
│       └── master_orchestrator.yml        # Production ML orchestration DAG
│
├── pipelines/
│   ├── data_pipeline/
│   │   ├── configs/                       # YAML definition files
│   │   ├── src/                           # Discovery, validation, materialization
│   │   ├── tests/
│   │   └── Dockerfile                     # Multi-stage definition
│   │
│   ├── training_pipeline/
│   │   ├── configs/
│   │   ├── src/                           # Model tuning, evaluation, WORM registry
│   │   ├── tests/
│   │   └── Dockerfile
│   │
│   ├── inference_pipeline/
│   │   ├── configs/
│   │   ├── src/                           # Matrix builder, contract validation, scoring
│   │   ├── tests/
│   │   └── Dockerfile
│   │
│   └── monitoring_pipeline/
│       ├── configs/
│       ├── src/                           # Drift, performance, and rule engine
│       ├── tests/
│       └── Dockerfile
│
├── shared_core/
│   ├── cloud/                             # Idempotent S3 operations
│   ├── exceptions/                        # Centralized error handling
│   ├── features/                          # Shared temporal SQL logic
│   ├── logging/                           # Standardized JSON log formatting
│   └── utils/
│
├── tools/
│   ├── raw_data_loader.py
│   ├── hive_partitioned_generator.py
│   └── synthetic_data_generator.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

Each production pipeline has its own execution boundary and Docker image while consuming shared functionality where consistency is required.

The supporting `tools/` scripts handle initial data ingestion, Bronze Data Lake preparation, and synthetic data generation for inference testing.

---

# Business and Operational Outputs

The system produces artifacts for both business and engineering use cases.

| Output                      | Purpose                                    |
| --------------------------- | ------------------------------------------ |
| Customer churn report       | Supports customer-retention decisions      |
| Inference telemetry         | Enables production monitoring              |
| Feature matrix              | Supports model training                    |
| Model bundle                | Provides a versioned production model      |
| Monitoring baselines        | Establishes reference behavior             |
| `need_update.json`          | Communicates retraining decisions          |
| Inference ledger            | Provides batch execution traceability      |
| Monitoring ledger           | Provides monitoring execution traceability |
| Model state pointer         | Identifies the active Champion             |
| Bronze Data Lake partitions | Provide temporally organized source data   |
| Synthetic T-1 partitions    | Enable controlled inference testing        |

---

# System Lifecycle

The system requires an initial data-lake bootstrap and model deployment to establish the first production Champion.

After that initial setup, the automated lifecycle is:

```text
Nightly Trigger
      ↓
Inference
      ↓
Monitoring
      ↓
Decision
      ↓
┌─────┴─────┐
│           │
Healthy   Degraded
│           │
▼           ▼
End       Retrain
            ↓
      Evaluate Challenger
            ↓
     Promote if EROI improves
            ↓
      Next nightly inference
```

For manual inference testing, the synthetic data generator can populate the previous day's Bronze partition with deterministic, schema-compatible customer, order, and payment records before the Inference Pipeline is executed.

The architecture therefore separates routine prediction from model maintenance and only enters the more expensive retraining branch when the monitoring system produces the configured decision to do so.

---

# Technology Stack

## Machine Learning

* XGBoost
* Scikit-learn
* Optuna
* SHAP
* Pandas
* NumPy

## Data Engineering

* DuckDB
* PyArrow
* Parquet
* Amazon S3
* Hive-style partitioning
* `s3fs`

## MLOps

* Model artifact registry on S3
* Champion–Challenger evaluation
* PSI drift detection
* Brier Score
* Log Loss
* Expected ROI / Realized ROI
* Automated retraining decisioning

## Infrastructure

* Docker
* GitHub Actions
* Amazon ECR
* AWS OIDC / IAM role assumption

## Development and Testing

* pytest
* ruff
* Deterministic synthetic data generation
* Immutable configuration using Python dataclasses

---

# Local Setup & Reproducibility

## 1. Environment Setup

```bash
git clone https://github.com/thesahilmandal/E-Commerce-Customer-Retention-MLOps-System.git

cd E-Commerce-Customer-Retention-MLOps-System

python3.12 -m venv venv

source venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt
```

## 2. AWS Prerequisites

### Configure the AWS CLI

```bash
aws configure
```

### Configure S3 Buckets

Create two S3 buckets and define them in a local `.env` file at the repository root.

Use bucket names, not S3 URIs:

```env
S3_CUSTOMER_DATABASE="<YOUR_CUSTOMER_DATABASE_BUCKET_NAME>"
S3_PIPELINE_RUN_ARTIFACTS="<YOUR_ARTIFACTS_BUCKET_NAME>"
```

Load the environment variables:

```bash
set -a && source .env && set +a
```

## 3. Data Lake Bootstrap

Populate the S3 Data Lake with the simulated historical dataset.

```bash
# 1. Ingest raw dataset into artifacts bucket
python -m tools.raw_data_loader

# 2. Transform to Hive-partitioned format and move to Customer Database bucket
python -m tools.hive_partitioned_generator
```

## 4. Local End-to-End Execution

Run the core pipelines sequentially through their module runners. You can inspect the `logs/` directory for operational telemetry during and after execution.

### 1. Data Pipeline — Materialize Feature Matrix

```bash
python -m pipelines.data_pipeline.src.runner \
  --run-id="testing_01" \
  --start-date="2016-09-01" \
  --end-date="2018-03-01"
```

### 2. Training Pipeline — Optimize, Calibrate, and Register Champion

Replace the bucket placeholder with your artifacts bucket name:

```bash
python -m pipelines.training_pipeline.src.runner \
  --run-id="testing_01" \
  --dataset-uri="s3://<YOUR_ARTIFACTS_BUCKET_NAME>/feature_store/testing_01/dataset.parquet"
```

### 3. Inference Simulation — Generate Synthetic "Yesterday" Footprints

```bash
python -m tools.synthetic_data_generator
```

### 4. Inference Pipeline — Score the Active Customer Base

```bash
python -m pipelines.inference_pipeline.src.runner \
  --run-id="testing_01"
```

### 5. Monitoring Pipeline — Evaluate Statistical Drift and ROI

```bash
python -m pipelines.monitoring_pipeline.src.runner \
  --run-id="testing_01" \
  --execution-date="$(date +%Y-%m-%d)"
```

---

# Cloud Deployment

The cloud deployment path uses GitHub Actions, Amazon ECR, and AWS OIDC authentication.

## 1. AWS Infrastructure

Create an IAM OIDC Identity Provider connected to your GitHub repository.

Provision four Amazon ECR repositories to host the pipeline images.

## 2. GitHub Variables

Configure the following repository variables, not secrets, in GitHub:

* `AWS_REGION`
* `AWS_ROLE_ARN` — IAM role assumable via OIDC
* `ECR_DATA_PIPELINE`
* `ECR_TRAINING_PIPELINE`
* `ECR_INFERENCE_PIPELINE`
* `ECR_MONITORING_PIPELINE` — repository names, not URIs
* `S3_PIPELINE_RUN_ARTIFACTS`
* `S3_CUSTOMER_DATABASE` — bucket names, not URIs

## 3. Workflow Execution

Push changes to `main`.

For the initial cloud run, manually trigger the deployment workflows in this order:

1. **Continuous Deployment** (`cd.yml`) — build and push the pipeline images to ECR.
2. **Master Orchestrator** (`master_orchestrator.yml`) — execute the production batch DAG.

---

# Summary

This project implements a complete batch-oriented ML lifecycle rather than stopping at model training.

The system connects:

```text
Raw Data
   ↓
Bronze Data Lake
   ↓
Validated Features
   ↓
Model Training
   ↓
Business-Based Evaluation
   ↓
Versioned Model
   ↓
Production Inference
   ↓
Business Report + Telemetry
   ↓
Drift + Performance Monitoring
   ↓
Deterministic Retraining Decision
   ↓
Conditional Retraining
   ↓
New Champion
   ↓
Next Inference Run
```

The architecture emphasizes explicit interfaces, point-in-time correctness, shared feature logic, resource-aware processing, business-driven model evaluation, immutable artifacts, automated monitoring, container isolation, CI/CD, cloud-native authentication, controlled synthetic-data testing, and traceable orchestration across the complete ML lifecycle.