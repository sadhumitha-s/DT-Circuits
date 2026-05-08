# DT-Circuits: Mechanistic Interpretability for Decision Transformers

DT-Circuits is a framework for mechanistic interpretability of Decision Transformers (DT). Using TransformerLens, it enables mapping neural circuits, decomposing activations with Sparse Autoencoders (SAEs), and performing causal interventions on agent decision-making.

The goal is to understand how Reward-to-Go, State, and Action tokens are processed within the residual stream, moving beyond basic behavioral observation.

## Table of Contents
- [Core Capabilities](#core-capabilities)
- [Technical Architecture](#technical-architecture)
- [Getting Started](#getting-started)
- [Project Documentation](#project-documentation)
- [Testing](#testing)
- [Project Structure](#project-structure)

## Project Documentation
Detailed explanations of the mechanistic interpretability techniques used in this project:
- [Circuit Discovery](./docs/circuit_discovery.md)
- [Activation Patching](./docs/activation_patching.md)
- [SAEs & Steering](./docs/sae_steering.md)



## Core Capabilities

### 1. Circuit Foundation
- **Hooked-DT**: A Decision Transformer implementation wrapped in TransformerLens for access to internal activations and weights.
- **Direct Logit Attribution (DLA)**: Quantifies the contribution of individual heads and MLP layers to action logits.
- **Induction Head Discovery**: Tools to identify heads responsible for temporal pattern recognition.

### 2. Causal Interventions
- **Activation Patching**: Replaces activations between clean and corrupted runs to identify causal paths.
- **Steering**: Generates and applies steering vectors (e.g., via Contrastive Activation Addition) to manipulate agent behavior at inference time.

### 3. SAEs & Safety
- **SAE Integration**: Tools to train and deploy SAEs on the residual stream to find monosemantic latents.
- **Anomaly Detection**: Uses SAE reconstruction error to detect out-of-distribution (OOD) states.

### 4. Path-Causal Microscope
- **ACDC (Automated Circuit Discovery)**: Prunes the DT into a minimal sufficient subgraph for specific behaviors.
- **Path Patching**: High-fidelity causal tracing between specific internal nodes (e.g., Goal Token → Induction Head → Action Logit).
- **Evolutionary Scan**: Analyzes how decision-making circuits form and stabilize across training checkpoints.

## Technical Architecture

The platform consists of:
- **Data Layer**: PPO Trajectory Harvester for collecting expert demonstrations (e.g., MiniGrid).
- **Model Layer**: HookedDT implementation.
- **Interpretability Layer**: Modules for attribution, patching, SAE management, and steering.
- **Visualization Layer**: Streamlit dashboard for real-time monitoring and intervention.

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 2.x
- TransformerLens
- SAE-Lens

### Installation
```bash
pip install -r requirements.txt
```

### Basic Workflow
1. **Generate Trajectories**:
   Use the harvester to collect teacher data for model training or SAE feature extraction.
   ```bash
   python scripts/train_dt.py
   ```

2. **Run Interpretability Dashboard**:
   Launch the interactive UI to perform real-time patching and steering interventions.
   ```bash
   streamlit run src/dashboard/app.py
   ```

## Testing

```bash
PYTHONPATH=. pytest tests/
```

## Project Structure

```text
DT-Circuits/
├── scripts/                # Training and harvesting entry points
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
│   │   ├── evolution.py    # Developmental/Evolutionary MI scan
│   │   ├── induction_scan.py # Induction head detection logic
│   │   ├── patching.py     # Causal activation patching tools
│   │   ├── path_patching.py # Path-based causal intervention engine
│   │   ├── sae_manager.py  # SAE deployment and anomaly detection
│   │   └── steering.py     # Steering vector generation and injection
│   ├── models/             
│   │   └── hooked_dt.py    # TransformerLens-wrapped Decision Transformer
│   └── utils/              
├── tests/                  # Unit and integration test suite
│   ├── test_components.py  
│   ├── test_path_causal_microscope.py # Phase 4 Path-Causal tests
│   └── test_sae_and_steering.py 
├── config.yaml             # Experiment and environment configuration
└── requirements.txt        # Environment dependencies
```
