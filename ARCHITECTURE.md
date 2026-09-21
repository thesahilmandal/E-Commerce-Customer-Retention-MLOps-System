# Architecture Decisions and System Design

## 1. System Philosophy and Constraints

This project was built with a strict focus on pragmatic engineering, cost efficiency, and raw ML/software engineering fundamentals. Because I developed this system with zero budget for cloud infrastructure, I deliberately avoided expensive, heavy-duty tools like Kubernetes, Apache Airflow, or Databricks.

Instead of over-engineering the infrastructure, I focused on solving the core MLOps challenges—point-in-time correctness, data contracts, model calibration, drift monitoring, and conditional retraining—using lightweight, cost-effective technologies. The result is a fully decoupled, batch-oriented system orchestrated entirely via GitHub Actions and executed in ephemeral Docker containers, minimizing cloud compute costs to near-zero while maintaining enterprise-grade design patterns.

## 2. Compute and Orchestration Architecture

Instead of maintaining an always-on cluster, the system relies on ephemeral batch execution.

* **Master Orchestrator (GitHub Actions):** Rather than deploying Apache Airflow or Prefect, I utilized GitHub Actions as a DAG orchestrator. The `master_orchestrator.yml` defines the execution order (Inference → Monitoring → Decision Gate → Optional Retraining) and manages concurrency (`cancel-in-progress: false` to prevent race conditions on S3).
* **Containerization (Docker):** The four pipelines (Data, Training, Inference, Monitoring) are isolated in independent Docker images using multi-stage builds. They execute as non-root users and only install required OS dependencies (e.g., `libgomp1` for XGBoost).
* **Zero-Trust Authentication (AWS OIDC):** To maintain security without managing long-lived static credentials, the GitHub Actions runners authenticate dynamically to AWS using IAM OIDC role assumption.

## 3. Data Architecture and Processing

The system uses a Data Lakehouse pattern backed by Amazon S3, avoiding the need for an expensive relational data warehouse like Snowflake or Redshift.

* **Bronze Data Lake (S3 & Parquet):** Raw transactional data is stored in S3 using Hive-style partitioning (year/month/day) and Snappy-compressed Parquet files. This allows downstream queries to leverage partition pruning and predicate pushdown.
* **Out-of-Core Processing (DuckDB):** Instead of using Apache Spark—which introduces distributed computing overhead and JVM management—I used DuckDB. DuckDB runs as a single-node, out-of-core analytical engine inside the container. By applying an 8 GB memory limit, 4 execution threads, and disk spilling, the system can process datasets larger than available RAM locally and efficiently without cluster costs.
* **Fail-Fast Validation:** Before expensive processing begins, DuckDB performs lightweight `MaxKeys=1` S3 list operations and schema/null-constraint checks. If data is missing or corrupted, the pipeline terminates immediately.

## 4. Feature Engineering and Point-in-Time Correctness

Data leakage is a critical failure point in ML systems. To prevent this, I centralized feature generation.

* **Shared Feature Generator:** Both the Data (Training) and Inference pipelines consume the exact same `SharedFeatureGenerator` implementation. This guarantees that serving logic never drifts from training logic.
* **Temporal Masking:** The SQL queries executed by DuckDB strictly enforce temporal boundaries. If an order delivery timestamp occurs after the arbitrary `end_date` (the point-in-time cutoff), the system nullifies that timestamp and reverts the order state to "processing." This absolutely prevents future information from bleeding into historical training features.

## 5. Model Training and Business-Driven Evaluation

The ML approach treats churn prediction as a business problem, not just a statistical one.

* **Algorithm & Tuning:** I chose XGBoost due to its superior performance on tabular data and reasonable training latency on single nodes. Hyperparameters are optimized using Optuna.
* **Probability Calibration:** Because downstream business logic relies on the actual probability of churn (not just rank order), I wrapped the XGBoost estimator in `CalibratedClassifierCV` using Isotonic Regression and 5-fold cross-validation.
* **Expected ROI Evaluation:** Instead of promoting models based solely on F1-score or AUC, the system calculates Expected Return on Investment (EROI). It evaluates thresholds (0.01 to 0.99) using a business formula: `(True Positives × Customer LTV × Save Rate) − (Total Interventions × Campaign Cost)`.
* **Champion–Challenger Hysteresis:** A new model is only promoted if its EROI exceeds the current Champion's EROI by a configured `eroi_hysteresis_margin`. This acts as a business gate, preventing unnecessary production updates for statistically insignificant gains.

## 6. Model Registry and Deployment Strategy

I implemented a two-phase deployment strategy to separate immutable ML artifacts from mutable deployment state.

* **Phase 1 (WORM Storage):** When a model is promoted, its complete bundle (the pipeline object, categorical schema enforcer, SHAP artifacts, monitoring baselines, and `requirements.txt`) is saved to a versioned, immutable S3 path keyed by a unique `run_id`.
* **Phase 2 (State Pointer):** I use a lightweight `model_state.json` file in S3 as a mutable pointer. It simply holds the `run_id` of the active Champion.
* **Dynamic Resolution:** The Inference and Monitoring pipelines read `model_state.json` at runtime to pull the correct artifacts. This means model updates happen instantly without requiring a container rebuild or code deployment.

## 7. Inference and Monitoring Pipelines

Because this is a batch system, the inference pipeline runs on a scheduled basis (simulated nightly) rather than via real-time API requests.

* **Dual-Output Inference:** The Inference pipeline generates two distinct artifacts. First, a business-facing CSV containing customer IDs, churn probabilities, and `revenue_at_risk` (for retention teams). Second, a complete telemetry Parquet dataset containing the raw feature matrix and predictions (for MLOps tracking).
* **Label-Independent Drift (PSI):** The Monitoring pipeline calculates the Population Stability Index (PSI) for the most critical features (identified via SHAP during training) and the prediction probability distribution. Categorical schemas are explicitly enforced to ensure PSI bins match training baselines.
* **Label-Dependent Performance:** Using a T-30 maturity window, the system joins historical predictions with matured ground-truth labels from the Bronze Data Lake to calculate Brier Score, Log Loss, and Realized ROI.
* **Deterministic Retraining (The Rule Engine):** Instead of calling the training code directly, the monitoring pipeline applies YAML-defined thresholds (e.g., `prediction_drift_threshold_psi: 0.20`) and outputs a `need_update.json` file. The GitHub Actions master orchestrator reads this artifact using `jq` to determine if the retraining pipeline should execute.

## 8. Limitations and Trade-offs

To maintain zero cloud costs and architectural simplicity, I made deliberate trade-offs:

* **No Real-Time Serving:** The system currently lacks a low-latency API serving layer (e.g., FastAPI). It is strictly a batch-scoring architecture.
* **Vertical Scaling Limits:** While DuckDB handles out-of-core processing excellently, it is still bound by single-node compute limits. If data volume scales to the petabyte range, the processing layer would need to be migrated to Apache Spark or Ray.
* **No Managed Experiment Tracking:** Rather than paying for or hosting MLflow, I rely on deterministic S3 file paths and generated metadata JSON files for experiment tracking and lineage.