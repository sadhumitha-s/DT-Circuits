# DT-Circuits: Mechanistic Interpretability for Decision Transformers

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

DT-Circuits is a research framework for mechanistic interpretability of Decision Transformers, focused on causal analysis, sparse feature decomposition, and circuit-level understanding of sequential decision-making agents.

---

## Motivation

Mechanistic interpretability has primarily focused on language models, while reinforcement learning agents remain comparatively underexplored.

Decision Transformers provide a uniquely analyzable architecture because trajectories, rewards, and actions are represented in a unified autoregressive sequence.

DT-Circuits aims to make RL agents inspectable at the circuit level rather than only through behavioral evaluation.

---

## Table of Contents
- [Features](#features)
- [Technical Architecture](#technical-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

--- 

## Documentation
- [Circuit Discovery](./docs/circuit_discovery.md)
- [Activation Patching](./docs/activation_patching.md)
- [SAEs & Steering](./docs/sae_steering.md)

--- 

## Features

### 1. Neural Mapping
- **Hooked-DT**: Access any internal activation or weight during the agent's run.
- **Logit Attribution**: See which attention heads or MLP layers drive specific actions.
- **Induction Scan**: Find heads that recognize temporal patterns and past states.

### 2. Testing Causality
- **Activation Patching**: Swap internal states to see what actually changes the agent's move.
- **Behavior Steering**: Add vectors to activations to push the agent toward specific goals without retraining.

### 3. Concept Discovery
- **TopK SAEs**: Decompose complex activations into a few active "concepts" for easier reading.
- **Auto-Labeling (NLA)**: Use an LLM to automatically describe what each discovered neuron feature does.
- **Cross-Model Probes**: Check if different agents (like DQNs) learn the same internal concepts as the DT.

### 4. Circuit Analysis
- **ACDC**: Automatically strip the model down to the minimal circuit needed for a task.
- **Path Patching**: Trace how a signal flows from a specific input token to the final action.
- **Evolutionary Scan**: Watch how decision-making circuits form during training.

--- 

## Technical Architecture

- **Data**: Collects expert paths using a PPO harvester.
- **Model**: Custom Decision Transformer compatible with TransformerLens.
- **Tools**: Dedicated modules for attribution, patching, SAEs, and steering.
- **Dashboard**: Streamlit UI for real-time model analysis.

---

## Project Structure

```text
DT-Circuits/
├── scripts/                
│   ├── train_dt.py         # Decision Transformer training pipeline
│   └── train_sae.py        # Sparse Autoencoder (SAE) training script
├── src/                    
│   ├── dashboard/          
│   │   └── app.py          # Streamlit-based visualization UI
│   ├── data/               
│   │   └── harvester.py    # PPO-based expert trajectory harvester
│   ├── interpretability/   
│   │   ├── acdc.py         # Automated Circuit Discovery logic
│   │   ├── attribution.py  # Direct Logit Attribution (DLA)
│   │   ├── evolution.py    # Training Dynamics Analysis
│   │   ├── induction_scan.py # Induction head detection logic
│   │   ├── nla.py          # Natural Language Autoencoder Explainer
│   │   ├── patching.py     # Causal activation patching tools
│   │   ├── path_patching.py # Path-based causal intervention engine
│   │   ├── sae_manager.py  # SAE deployment and anomaly detection
│   │   ├── steering.py     # Steering vector generation and injection
│   │   └── universality.py # Cross-architecture feature mapping
│   ├── models/             
│   │   └── hooked_dt.py    # TransformerLens-wrapped Decision Transformer
│   └── utils/              
├── tests/                  # Unit tests for all modules
├── config.yaml             
└── requirements.txt        
```

--- 

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 2.x
- TransformerLens
- SAE-Lens

### Quick Start

Follow these steps to initialize the environment and verify the installation.

1. **Environment Setup**
   ```bash
   python -m venv venv
   source venv/bin/activate  
   pip install -r requirements.txt
   ```

2. **Verification**
   Run the component tests to ensure all dependencies and hooks are correctly configured.
   ```bash
   PYTHONPATH=. pytest tests/test_components.py
   ```

3. **Dashboard Execution**
   Launch the `DT-Explorer` dashboard. The dashboard will initialize with a random model if no trained weights are detected.
   ```bash
   streamlit run src/dashboard/app.py
   ```

### Workflow

1. **Data Harvesting & Model Training**
   ```bash
   python scripts/train_dt.py
   ```

2. **Interpretability Analysis**
   ```bash
   streamlit run src/dashboard/app.py
   ```
