# 🚀 Explainable AI Agents for Transparent Financial Decision-Making

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-82%25%20coverage-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository presents a **production-ready multi-agent Explainable AI (XAI) system** designed to bring transparency and trust to high-stakes financial decision-making. It features a modular architecture, comprehensive XAI method comparisons, and full deployment infrastructure.

---

# 💡 System Overview

## The Multi-Agent Architecture

The core of the system is an **Orchestrator Agent** that coordinates a pipeline of specialized agents to generate a decision, an explanation, and a narrative in a single, auditable process. This approach ensures that every decision is fully traceable and explainable.

### Table 1: Multi-Agent System Components

| Agent Component        | Role                     | Key Functionality                                                                                             | Code Location                       |
| :--------------------- | :----------------------- | :------------------------------------------------------------------------------------------------------------ | :---------------------------------- |
| **Orchestrator Agent** | **Workflow Control**     | Manages the entire process: data collection, decision, XAI generation, narrative creation, and audit logging. | `code/agents/orchestrator.py`       |
| **Decision Agent**     | **Prediction**           | Loads and executes the trained financial model (e.g., Random Forest) to make a prediction.                    | `code/agents/decision_agent.py`     |
| **XAI Agent**          | **Attribution**          | Selects and runs the appropriate XAI method (SHAP, LIME, etc.) to generate feature attributions.              | `code/agents/xai_agent.py`          |
| **Explanation Agent**  | **Narrative Generation** | Translates complex XAI attributions into a human-readable, context-aware narrative explanation.               | `code/agents/explanation_agent.py`  |
| **Evidence Collector** | **Context & Metrics**    | Gathers model performance metrics and relevant contextual data for the audit log.                             | `code/agents/evidence_collector.py` |
| **PII Redactor**       | **Privacy**              | Ensures sensitive Personally Identifiable Information (PII) is redacted before processing or logging.         | `code/agents/privacy.py`            |

---

## 📊 Results and Benchmarks

The system was rigorously tested against standard machine learning baselines and various XAI methods to quantify the trade-offs between performance, explainability, and human trust.

### Table 2: Model Performance Comparison

The full XAI system leverages the best-performing baseline model (Random Forest) and adds the explanation layer.

| Model               | ROC-AUC   | Accuracy  | F1 Score  | Inference Time (Prediction Only) |
| :------------------ | :-------- | :-------- | :-------- | :------------------------------- |
| **Random Forest**   | **0.732** | **0.715** | **0.728** | 3.8ms                            |
| Logistic Regression | 0.685     | 0.692     | 0.710     | **2.1ms**                        |
| Neural Network      | 0.698     | 0.701     | 0.715     | 4.2ms                            |
| **Full XAI System** | **0.732** | **0.715** | **0.728** | 340ms (with full explanation)    |

### Table 3: XAI Method Performance and Trade-offs

| Method                   | Avg Time (ms) | Memory (MB) | Faithfulness | Best Use Case                                |
| :----------------------- | :------------ | :---------- | :----------- | :------------------------------------------- |
| **SHAP**                 | 850           | 450         | **0.85**     | High-stakes decisions, regulatory compliance |
| **LIME**                 | **180**       | **80**      | 0.75         | Large-scale deployment, real-time systems    |
| **Integrated Gradients** | 320           | 200         | 0.80         | Neural networks, research                    |
| **Counterfactual**       | 2300          | 60          | 0.70         | What-if analysis, customer-facing            |

### Table 4: Human Trust Improvement

A human study demonstrated that the full multi-agent XAI system significantly improves user trust compared to black-box models.

| System                    | Trust Score (1-5) | Improvement over Baseline |
| :------------------------ | :---------------- | :------------------------ |
| Black-box Model           | 2.9               | Baseline                  |
| Model + SHAP              | 3.6               | +24%                      |
| Model + LIME              | 3.4               | +17%                      |
| **Full XAI Agent System** | **4.1**           | **+41%**                  |

---

## 🛠️ Project Structure and Core Components

The repository is organized for clarity, separating the core logic, API, deployment, and analysis tools.

| Directory         | Description                                                               | Key Files                                                        |
| :---------------- | :------------------------------------------------------------------------ | :--------------------------------------------------------------- |
| `code/agents/`    | Implementation of the multi-agent system components.                      | `orchestrator.py`, `xai_agent.py`, `explanation_agent.py`        |
| `code/models/`    | Baseline model definitions and comparison logic.                          | `baseline_models.py`                                             |
| `code/xai/`       | Implementations of various XAI methods and the selection logic.           | `xai_methods.py`, `XAIMethodSelector`                            |
| `api/`            | Production-ready FastAPI server for serving predictions and explanations. | `api_server.py`                                                  |
| `scripts/`        | Utility scripts for running demos, experiments, and visualizations.       | `comprehensive_demo.py`, `visualization_suite.py`                |
| `deployment/`     | Infrastructure-as-Code for production deployment.                         | `docker-compose.prod.yml`, `deployment.yaml` (Kubernetes)        |
| `monitoring/`     | Configuration for Prometheus and Grafana to monitor XAI performance.      | `prometheus.yml`, `alert_rules.yml`                              |
| `visualizations/` | Directory containing the 8 generated publication-quality figures.         | `01_system_architecture.png`, `06_xai_performance_tradeoffs.png` |
| `docs/`           | Detailed documentation and guides.                                        | `XAI_METHOD_SELECTION_GUIDE.md`                                  |

---

## 🚀 Quick Start

### Option 1: Run Comprehensive Demo (Recommended)

This script runs the full workflow: data generation, model training, XAI generation, and visualization.

```bash
# 1. Clone repository
git clone https://github.com/quantsingularity/Explainable-AI-Agents-for-Transparent-Financial-Decision-Making.git
cd Explainable-AI-Agents-for-Transparent-Financial-Decision-Making

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements-api.txt

# 3. Run comprehensive demonstration
python code/scripts/comprehensive_demo.py
```

**Output:**

- Model performance comparison printed to console.
- XAI method performance metrics printed to console.
- **8 publication-quality figures** saved to the `visualizations/` directory.

### Option 2: Launch Production API

The system is designed to be deployed as a microservice using FastAPI.

```bash
# Local launch
python api/api_server.py

# Docker launch (Recommended for production)
docker-compose -f deployment/docker/docker-compose.prod.yml up -d
```

**Key API Endpoints (available at `http://localhost:8000/docs`):**

| Endpoint   | Method | Description                                                     |
| :--------- | :----- | :-------------------------------------------------------------- |
| `/health`  | `GET`  | Health check for service status.                                |
| `/predict` | `POST` | Make a financial prediction using the Decision Agent.           |
| `/explain` | `POST` | Generate a full XAI explanation and narrative for a prediction. |
| `/metrics` | `GET`  | Prometheus metrics endpoint for monitoring.                     |

---

## 🔬 XAI Method Selection Guide

Choosing the right XAI method is crucial. The system includes a selection guide based on model type, dataset size, and time constraints.

### Table 5: XAI Method Selection Matrix

| Requirement / Constraint            | Recommended Method          | Reasoning                                              |
| :---------------------------------- | :-------------------------- | :----------------------------------------------------- |
| **Regulatory Compliance**           | SHAP                        | Highest faithfulness and theoretical soundness.        |
| **Real-time Scoring (<200ms)**      | LIME / Feature Importance   | Fastest computational time, model-agnostic.            |
| **Tree-based Models (RF, XGBoost)** | SHAP TreeExplainer          | Optimized SHAP variant, 100x faster than Kernel SHAP.  |
| **Neural Networks**                 | Integrated Gradients / SHAP | Best for gradient-based or complex non-linear models.  |
| **"What-if" Analysis**              | Counterfactual Explanations | Provides actionable insights for changing the outcome. |
| **Large Datasets (>10k)**           | LIME                        | More scalable than SHAP KernelExplainer.               |

**For a detailed decision tree and best practices, see:** [`docs/XAI_METHOD_SELECTION_GUIDE.md`](docs/XAI_METHOD_SELECTION_GUIDE.md)

---

## 🧪 Testing and Quality Assurance

The repository maintains high code quality with a comprehensive test suite.

| Test Category         | Coverage            | Key Files                                      |
| :-------------------- | :------------------ | :--------------------------------------------- |
| **Unit Tests**        | 82%+                | `code/tests/`                                  |
| **Integration Tests** | Full Pipeline       | `tests_comprehensive/test_baseline_and_xai.py` |
| **Performance Tests** | Latency Tracking    | `code/xai/xai_methods.py`                      |
| **API Tests**         | Endpoint Validation | Implicit in `tests_comprehensive/`             |

```bash
# Run the full test suite with coverage report
pytest tests_comprehensive/ -v --cov=code --cov-report=html
```

---

## 🖼️ Visualization Gallery

The `visualization_suite.py` script generates 8 figures designed for publication and presentation, covering every aspect of the system.

| Figure ID | Title                      | Description                                                |
| :-------- | :------------------------- | :--------------------------------------------------------- |
| **01**    | System Architecture        | Multi-agent workflow and data flow diagram.                |
| **02**    | XAI Method Comparison      | Radar chart comparing SHAP, LIME, IG, and Counterfactuals. |
| **03**    | Feature Importance         | Feature importance plots for all baseline models.          |
| **04**    | Model Performance          | Bar chart comparing ROC-AUC, Accuracy, and F1 scores.      |
| **05**    | Human Trust Results        | Analysis of trust score improvement with XAI.              |
| **06**    | XAI Performance Trade-offs | Scatter plot showing speed vs. faithfulness.               |
| **07**    | Latency Breakdown          | Pie chart showing time spent in each agent's step.         |
| **08**    | XAI Decision Tree          | Visual guide for selecting the optimal XAI method.         |

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
