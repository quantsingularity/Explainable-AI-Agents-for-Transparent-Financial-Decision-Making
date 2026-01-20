# Explainable AI Agents for Transparent Financial Decision-Making

## 🎯 Project Overview

This repository presents a **fully implemented, production-ready multi-agent XAI system** designed to bring transparency and auditability to financial decision-making. The system orchestrates specialized agents to generate high-fidelity explanations (SHAP, LIME, Integrated Gradients) and natural language narratives, ensuring that complex model decisions are interpretable by both regulators and end-users.

### Key Features

| Feature                          | Description                                                                                                               |
| :------------------------------- | :------------------------------------------------------------------------------------------------------------------------ |
| **Modular Agent Architecture**   | Orchestration of specialized agents: Evidence Collector, Decision Agent, XAI Agent, Explanation Agent, and Privacy Guard. |
| **Multi-Method XAI Suite**       | Integrated support for SHAP, LIME, Integrated Gradients (IG), and Counterfactual explanations (DiCE).                     |
| **Regulatory-Grade Narratives**  | Automated generation of template-backed, auditable narratives tailored for compliance and customer communication.         |
| **Privacy-Preserving Design**    | Integrated PII redaction layer that ensures sensitive data is protected before processing or logging.                     |
| **Comprehensive Evaluation**     | Rigorous benchmarking of XAI methods using metrics like Faithfulness, Completeness, and Latency.                          |
| **Human-in-the-Loop Validation** | Framework for conducting human trust studies to quantify the impact of explanations on user confidence.                   |

## 📊 Key Results (Deterministic Synthetic Pipeline - Seed 42)

The multi-agent system demonstrates high faithfulness and significantly improves human trust compared to black-box baselines.

| Metric                      | Logistic Regression | Random Forest | XGBoost | **Full Agentic System** |
| :-------------------------- | :------------------ | :------------ | :------ | :---------------------- |
| **ROC-AUC**                 | 0.672               | 0.705         | 0.723   | **0.723**               |
| **XAI Faithfulness (SHAP)** | 0.854               | 0.812         | 0.795   | **0.810**               |
| **Human Trust Score**       | 3.2/5               | 3.1/5         | 2.9/5   | **4.1/5**               |
| **Explanation Latency**     | 45ms                | 110ms         | 125ms   | **340ms** (incl. LLM)   |
| **Audit Coverage**          | 100%                | 100%          | 100%    | **100%**                |

## 🚀 Quick Start (30 minutes)

The project is designed for easy setup using Docker, ensuring a consistent environment for all dependencies.

### Prerequisites

- Docker & Docker Compose
- 4+ CPU cores, 8GB RAM
- (Optional) OpenAI API key for advanced narrative generation (falls back to local templates)

### Run with Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/quantsingularity/Explainable-AI-Agents-for-Transparent-Financial-Decision-Making
cd Explainable-AI-Agents-for-Transparent-Financial-Decision-Making

# Build and run the environment
./docker_build_and_run.sh

# Inside Docker, run quick experiment (generates data, trains models, and runs XAI agents)
./run_quick.sh

# View results and figures
ls results/
ls figures/
```

### Run without Docker

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run quick experiment
python code/scripts/quick_experiment.py

# Run full experiments (4-6 hours)
./run_full.sh
```

## 📁 Repository Structure

The repository is structured to separate core agent logic, XAI methods, and experimental infrastructure.

```
Explainable-AI-Agents-for-Transparent-Financial-Decision-Making/
├── README.md                          # This file
├── LICENSE                            # Project license
├── Dockerfile                         # Production container definition
├── docker-compose.yml                 # Multi-service orchestration
├── requirements.txt                   # Python dependencies (pinned)
├── run_quick.sh                       # 30-min quick experiment runner
├── run_full.sh                        # Full experimental suite runner
│
├── code/                              # Main implementation
│   ├── agents/                        # Core agent implementations
│   │   ├── orchestrator.py            # Multi-agent coordination logic
│   │   ├── decision_agent.py          # Financial prediction models
│   │   ├── xai_agent.py               # Attribution and feature importance
│   │   ├── explanation_agent.py       # Narrative generation logic
│   │   └── privacy.py                 # PII redaction and safeguards
│   │
│   ├── xai/                           # XAI Method Implementations
│   │   └── ...                        # SHAP, LIME, IG, Counterfactuals
│   │
│   ├── eval/                          # Evaluation Framework
│   │   ├── metrics.py                 # Faithfulness and completeness metrics
│   │   └── human_study.py             # Trust study simulation
│   │
│   ├── scripts/                       # Automation and experiment scripts
│   │   ├── run_experiment.py          # Main experiment runner
│   │   └── generate_figures.py        # Visualization generation
│   │
│   └── tests/                         # Unit and integration tests
│
├── data/                              # Data artifacts
│   └── synthetic_lending.csv          # Deterministic synthetic dataset
│
├── figures/                           # Publication-ready visualizations
│   ├── system_architecture.png        # Multi-agent flow
│   ├── perf_vs_explainability.png     # Trade-off analysis
│   └── human_trust_results.png        # Trust study outcomes
│
└── results/                           # Experimental outputs and logs
```

## 🏗️ Architecture

The system operates as a coordinated collective of agents managed by the `Orchestrator`. This design ensures that every decision is accompanied by a verifiable and understandable explanation.

### Agent Hierarchy & Responsibilities

| Agent Role             | Responsibility                                                                | Implementation Location             |
| :--------------------- | :---------------------------------------------------------------------------- | :---------------------------------- |
| **Orchestrator**       | Coordinates the end-to-end workflow and manages agent execution.              | `code/agents/orchestrator.py`       |
| **Evidence Collector** | Gathers context, model metadata, and historical data for the explanation.     | `code/agents/evidence_collector.py` |
| **Decision Agent**     | Executes the core financial model (e.g., loan approval) to make a prediction. | `code/agents/decision_agent.py`     |
| **XAI Agent**          | Computes feature attributions and generates visual/structural explanations.   | `code/agents/xai_agent.py`          |
| **Explanation Agent**  | Translates technical XAI outputs into human-readable narratives.              | `code/agents/explanation_agent.py`  |
| **Privacy Guard**      | Redacts PII and ensures compliance with data protection regulations.          | `code/agents/privacy.py`            |

### Key Design Principles

| Principle         | Explanation                                                                                         |
| :---------------- | :-------------------------------------------------------------------------------------------------- |
| **Auditability**  | Every step of the decision and explanation process is logged in a JSONL audit trail.                |
| **Faithfulness**  | Explanations are rigorously tested to ensure they accurately reflect the model's internal logic.    |
| **Privacy-First** | PII redaction is applied at the source, before any data reaches the explanation or logging layers.  |
| **Modular XAI**   | The system is agnostic to the underlying XAI method, allowing for easy swapping of SHAP, LIME, etc. |
| **Human-Centric** | Narratives are optimized for human understanding, not just technical correctness.                   |

## 🧪 Evaluation Framework

The evaluation is designed to be comprehensive, covering both technical performance and human impact.

### Technical Metrics

- **Faithfulness**: Correlation between feature removal and change in model output.
- **Completeness**: Percentage of the model's decision explained by the top-k features.
- **Latency**: Time taken to generate a complete explanation package.

### Testing

```bash
# Run all unit tests
pytest code/tests/test_simple.py -v

# Run full test suite
pytest code/tests/test_all.py -v
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
