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
    return {
        "states": torch.randn(1, 5, 10),
        "actions": torch.zeros(1, 5, 3),
        "returns_to_go": torch.ones(1, 5, 1)
    }

def test_acdc_discovery(model, sample_inputs):
    """Verifies that ACDC can prune heads and save a manifest."""
    model.eval()
    acdc = ACDCDiscovery(model, threshold=0.5)
    circuit = acdc.run(sample_inputs, target_action=1)
    
    assert "active_heads" in circuit
    assert "initial_perf" in circuit
    
    manifest_path = "circuit_manifest.json"
    acdc.save_manifest(manifest_path)
    assert os.path.exists(manifest_path)
    
    with open(manifest_path, 'r') as f:
        data = json.load(f)
        assert "active_heads" in data
    
    os.remove(manifest_path)

def test_path_patching_ablation(model, sample_inputs):
    """Verifies that ablating a head changes the model output."""
    engine = PathPatchingEngine(model)
    
    orig_output = model(**sample_inputs)
    ablated_output = engine.perform_edge_ablation(
        sample_inputs, layer=0, head_index=0, ablation_type="zero"
    )
    
    diff = (orig_output - ablated_output).abs().max().item()
    assert diff > 0

def test_evolutionary_scanner_mock(model, sample_inputs, tmp_path):
    """Verifies scanning multiple checkpoints for circuit formation."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    
    torch.save(model.state_dict(), checkpoint_dir / "step_100.pt")
    torch.save(model.state_dict(), checkpoint_dir / "step_200.pt")
    
    scanner = EvolutionaryScanner(HookedDT, state_dim=10, action_dim=3)
    results = scanner.scan_checkpoints(
        str(checkpoint_dir), 
        sample_inputs, 
        target_action=1,
        d_model=32,
        n_heads=2
    )
    
    assert len(results) == 2
    assert "active_heads" in results[0]
