# Explainable AI Agents for Transparent Financial Decision-Making

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-82%25%20coverage-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready multi-agent Explainable AI system for transparent, auditable financial decision-making. An Orchestrator Agent coordinates a pipeline of specialized agents to generate predictions, feature attributions, and human-readable narratives in a single traceable process.

---

## Table of Contents

- [Architecture](#architecture)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [XAI Method Selection](#xai-method-selection)
- [Testing](#testing)
- [Visualizations](#visualizations)
- [License](#license)

---

## Architecture

| Agent                  | Role                 | Key Function                                                                         |
| :--------------------- | :------------------- | :----------------------------------------------------------------------------------- |
| **Orchestrator**       | Workflow Control     | Manages end-to-end process: data collection, decision, XAI, narrative, audit logging |
| **Decision Agent**     | Prediction           | Executes the trained financial model to produce a prediction                         |
| **XAI Agent**          | Attribution          | Selects and runs the appropriate XAI method (SHAP, LIME, etc.)                       |
| **Explanation Agent**  | Narrative Generation | Translates XAI attributions into human-readable, context-aware narratives            |
| **Evidence Collector** | Context and Metrics  | Gathers model performance metrics and contextual data for the audit log              |
| **PII Redactor**       | Privacy              | Redacts sensitive PII before processing or logging                                   |

---

## Results

### Model Performance

| Model               | ROC-AUC   | Accuracy  | F1 Score  | Inference Time           |
| :------------------ | :-------- | :-------- | :-------- | :----------------------- |
| **Random Forest**   | **0.732** | **0.715** | **0.728** | 3.8ms                    |
| Logistic Regression | 0.685     | 0.692     | 0.710     | 2.1ms                    |
| Neural Network      | 0.698     | 0.701     | 0.715     | 4.2ms                    |
| **Full XAI System** | **0.732** | **0.715** | **0.728** | 340ms (with explanation) |

### XAI Method Comparison

| Method                   | Avg Time (ms) | Memory (MB) | Faithfulness | Best Use Case                                |
| :----------------------- | :------------ | :---------- | :----------- | :------------------------------------------- |
| **SHAP**                 | 850           | 450         | **0.85**     | Regulatory compliance, high-stakes decisions |
| **LIME**                 | **180**       | **80**      | 0.75         | Real-time systems, large-scale deployment    |
| **Integrated Gradients** | 320           | 200         | 0.80         | Neural networks, research                    |
| **Counterfactual**       | 2300          | 60          | 0.70         | What-if analysis, customer-facing            |

### Human Trust Study

| System                    | Trust Score (1-5) | vs Baseline |
| :------------------------ | :---------------- | :---------- |
| Black-box Model           | 2.9               | baseline    |
| Model + SHAP              | 3.6               | +24%        |
| Model + LIME              | 3.4               | +17%        |
| **Full XAI Agent System** | **4.1**           | **+41%**    |

---

## Repository Structure

| Path              | Description                                                  |
| :---------------- | :----------------------------------------------------------- |
| `code/agents/`    | Orchestrator, XAI agent, explanation agent, privacy redactor |
| `code/models/`    | Baseline model definitions and comparison logic              |
| `code/xai/`       | XAI method implementations and selection logic               |
| `api/`            | FastAPI server for predictions and explanations              |
| `scripts/`        | Demo runner, experiments, visualization suite                |
| `deployment/`     | Docker Compose and Kubernetes configuration                  |
| `monitoring/`     | Prometheus and Grafana configuration                         |
| `visualizations/` | 8 generated publication-quality figures                      |
| `docs/`           | XAI method selection guide and documentation                 |

---

## Quick Start

### Comprehensive Demo

```bash
git clone https://github.com/quantsingularity/Explainable-AI-Agents-for-Transparent-Financial-Decision-Making.git
cd Explainable-AI-Agents-for-Transparent-Financial-Decision-Making

pip install -r requirements.txt -r requirements-api.txt

python code/scripts/comprehensive_demo.py
```

Outputs model and XAI performance metrics to console and saves 8 figures to `visualizations/`.

### Production API

```bash
# Local
python api/api_server.py

# Docker
docker-compose -f deployment/docker/docker-compose.prod.yml up -d
```

| Endpoint   | Method | Description                             |
| :--------- | :----- | :-------------------------------------- |
| `/health`  | GET    | Service health check                    |
| `/predict` | POST   | Financial prediction via Decision Agent |
| `/explain` | POST   | Full XAI explanation and narrative      |
| `/metrics` | GET    | Prometheus metrics endpoint             |

API docs available at `http://localhost:8000/docs`.

---

## XAI Method Selection

| Requirement                     | Recommended Method        | Reasoning                                      |
| :------------------------------ | :------------------------ | :--------------------------------------------- |
| Regulatory compliance           | SHAP                      | Highest faithfulness and theoretical soundness |
| Real-time scoring under 200ms   | LIME / Feature Importance | Fastest, model-agnostic                        |
| Tree-based models (RF, XGBoost) | SHAP TreeExplainer        | 100x faster than Kernel SHAP                   |
| Neural networks                 | Integrated Gradients      | Best for gradient-based non-linear models      |
| What-if analysis                | Counterfactual            | Actionable insights for outcome changes        |
| Large datasets over 10k samples | LIME                      | More scalable than SHAP KernelExplainer        |

Full decision tree and best practices: [`docs/XAI_METHOD_SELECTION_GUIDE.md`](docs/XAI_METHOD_SELECTION_GUIDE.md)

---

## Testing

```bash
pytest code/tests/ -v --cov=code --cov-report=html
```

| Type              | Coverage         | Location                              |
| :---------------- | :--------------- | :------------------------------------ |
| Unit Tests        | 82%+             | `code/tests/`                         |
| Integration Tests | Full pipeline    | `code/tests/test_baseline_and_xai.py` |
| Performance Tests | Latency tracking | `code/xai/xai_methods.py`             |

---

## Visualizations

Running `visualization_suite.py` generates 8 publication-quality figures saved to `visualizations/`.

| Figure | Title                      |
| :----- | :------------------------- |
| 01     | System Architecture        |
| 02     | XAI Method Comparison      |
| 03     | Feature Importance         |
| 04     | Model Performance          |
| 05     | Human Trust Results        |
| 06     | XAI Performance Trade-offs |
| 07     | Latency Breakdown          |
| 08     | XAI Decision Tree          |

---

## License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
