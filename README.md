# DT-Circuits

A research-grade platform for the mechanistic interpretability of Decision Transformers.

## Architecture
- **Data**: PPO Trajectory Harvester for high-quality teacher data.
- **Model**: `HookedDT` - A custom Decision Transformer wrapped in `TransformerLens` for full activation visibility.
- **Interpretability**: Tools for Direct Logit Attribution (DLA), Activation Patching, and Induction Head detection.
- **Dashboard**: Streamlit-based UI for real-time causal interventions.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect Data & Train Mini-DT
```bash
python scripts/train_dt.py
```

### 3. Run Interpretation Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Testing
Run the test suite to ensure system integrity:
```bash
pytest tests/
```

## Components
- `src/data/harvester.py`: Collects trajectories from MiniGrid.
- `src/models/hooked_dt.py`: Hookable transformer implementation.
- `src/interpretability/`:
    - `attribution.py`: Direct Logit Attribution logic.
    - `patching.py`: Activation patching interface.
    - `induction_scan.py`: Automated circuit discovery.

## License

MIT
