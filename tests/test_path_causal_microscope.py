import pytest
import torch
from src.models.hooked_dt import HookedDT
from src.interpretability.acdc import ACDCDiscovery
from src.interpretability.path_patching import PathPatchingEngine
from src.interpretability.evolution import EvolutionaryScanner
import os
import json

@pytest.fixture
def model():
    return HookedDT.from_config(state_dim=10, action_dim=3, n_layers=2, n_heads=2, d_model=32)

@pytest.fixture
def sample_inputs():
    batch_size = 1
    seq_len = 5
    state_dim = 10
    action_dim = 3
    return {
        "states": torch.randn(batch_size, seq_len, state_dim),
        "actions": torch.zeros(batch_size, seq_len, action_dim),
        "returns_to_go": torch.ones(batch_size, seq_len, 1),
        "timesteps": torch.arange(seq_len).unsqueeze(0)
    }

def test_acdc_discovery(model, sample_inputs):
    # Ensure model is in eval mode
    model.eval()
    
    target_action = 1
    acdc = ACDCDiscovery(model, threshold=0.5) # High threshold for quick test
    circuit = acdc.run(sample_inputs, target_action)
    
    assert "active_heads" in circuit
    assert "initial_perf" in circuit
    assert "final_perf" in circuit
    
    # Save manifest check
    manifest_path = "circuit_manifest.json"
    acdc.save_manifest(manifest_path)
    assert os.path.exists(manifest_path)
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
        assert "active_heads" in data
    
    os.remove(manifest_path)

def test_path_patching_ablation(model, sample_inputs):
    engine = PathPatchingEngine(model)
    
    # Run original
    orig_output, _, _ = model(**sample_inputs)
    
    # Ablate L0 H0
    ablated_output, _, _ = engine.perform_edge_ablation(
        sample_inputs, layer=0, head_index=0, ablation_type="zero"
    )
    
    # Check if they differ - using a very small tolerance or direct check
    diff = (orig_output - ablated_output).abs().max().item()
    assert diff > 0, "Ablation should have some effect on output"

def test_evolutionary_scanner_mock(model, sample_inputs, tmp_path):
    # Create dummy checkpoints
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    
    torch.save(model.state_dict(), checkpoint_dir / "step_100.pt")
    torch.save(model.state_dict(), checkpoint_dir / "step_200.pt")
    
    scanner = EvolutionaryScanner(HookedDT, state_dim=10, action_dim=3)
    # Pass d_model and n_heads to match the fixture model
    results = scanner.scan_checkpoints(
        str(checkpoint_dir), 
        sample_inputs, 
        target_action=1,
        d_model=32,
        n_heads=2
    )
    
    assert len(results) == 2
    assert "checkpoint" in results[0]
    assert "active_heads" in results[0]

