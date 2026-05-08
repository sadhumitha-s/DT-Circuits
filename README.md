# DT-Circuits: Mechanistic Interpretability for Decision Transformers

DT-Circuits is a research-grade framework designed for the rigorous mechanistic interpretability of Decision Transformers (DT). By leveraging the TransformerLens paradigm, this platform enables researchers to map internal neural circuits, decompose activations using Sparse Autoencoders, and perform causal interventions on agent decision-making.

The primary objective is to move beyond behavioral observation and saliency maps toward a quantitative understanding of how Reward-to-Go, State, and Action tokens are processed within the residual stream.

## Core Capabilities

### 1. Circuit Foundation
- **Hooked-DT Architecture**: A custom Decision Transformer implementation wrapped in TransformerLens, providing full access to internal activations, weights, and the residual stream.
- **Direct Logit Attribution (DLA)**: Quantitative mapping of individual attention heads and MLP layers to the final action logits.
- **Induction Head Discovery**: Automated scanning tools to identify heads responsible for temporal pattern recognition and "memory" in RL tasks.

### 2. Causal Interventions
- **Activation Patching**: Surgical replacement of activations between "clean" and "corrupted" runs to identify bottleneck features and causal paths.
- **Contrastive Activation Addition (CAA)**: Generation of steering vectors by calculating the mean difference between positive and negative activation sets.
- **Steering Library**: A persistent library of pre-calculated vectors (e.g., success_vector, exploration_vector) that can be injected at inference time to manipulate agent behavior without retraining.

### 3. Deep Discovery & Safety
- **Sparse Autoencoder (SAE) Integration**: Tools to train and deploy SAEs on the residual stream, decomposing polysemantic neurons into monosemantic latents.
- **Mechanistic Anomaly Detection**: Utilizing SAE reconstruction error as a high-fidelity proxy for detecting out-of-distribution (OOD) states.

## Technical Architecture

The platform is divided into four primary layers:
- **Data Layer**: PPO Trajectory Harvester for generating high-quality expert demonstrations in Gymnasium environments (e.g., MiniGrid).
- **Model Layer**: The HookedDT implementation which maintains compatibility with standard DT architectures while adding hook-based visibility.
- **Interpretability Layer**: A suite of modules for attribution, patching, SAE management, and steering.
- **Visualization Layer**: A Streamlit-based dashboard for real-time activation monitoring and interactive steering.

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
│   │   ├── attribution.py  # Direct Logit Attribution (DLA)
│   │   ├── induction_scan.py # Induction head detection logic
│   │   ├── patching.py     # Causal activation patching tools
│   │   ├── sae_manager.py  # SAE deployment and anomaly detection
│   │   └── steering.py     # Steering vector generation and injection
│   ├── models/             
│   │   └── hooked_dt.py    # TransformerLens-wrapped Decision Transformer
│   └── utils/              
├── tests/                  # Unit and integration test suite
│   ├── test_components.py  
│   └── test_sae_and_steering.py 
├── config.yaml             # Experiment and environment configuration
└── requirements.txt        # Environment dependencies
```
