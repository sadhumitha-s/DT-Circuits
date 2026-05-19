import pytest
import torch
import os
import json
from src.models.hooked_dt import HookedDT
from src.interpretability.circuit_surgeon import CircuitSurgeon
from src.interpretability.neuronpedia import NeuronpediaExporter

@pytest.fixture
def test_model():
    """Initializes a small, reproducible HookedDT model for testing."""
    return HookedDT.from_config(
        state_dim=3,
        action_dim=7,
        n_layers=2,
        n_heads=2,
        d_model=16,
        max_length=5
    )

def test_circuit_surgeon_registration(test_model):
    """Verifies registration and parsing of component node/edge surgical targets."""
    surgeon = CircuitSurgeon(test_model)
    
    # 1. Component node registration
    surgeon.add_node_ablation("L0H0")
    surgeon.add_node_ablation("L1MLP")
    assert "L0H0" in surgeon.ablated_nodes
    assert "L1MLP" in surgeon.ablated_nodes
    
    surgeon.remove_node_ablation("L0H0")
    assert "L0H0" not in surgeon.ablated_nodes
    assert "L1MLP" in surgeon.ablated_nodes
    
    # 2. Communication path registration
    surgeon.add_edge_ablation("L0H1", "L1H0")
    assert ("L0H1", "L1H0") in surgeon.ablated_edges
    
    surgeon.remove_edge_ablation("L0H1", "L1H0")
    assert ("L0H1", "L1H0") not in surgeon.ablated_edges
    
    surgeon.clear_ablations()
    assert len(surgeon.ablated_nodes) == 0
    assert len(surgeon.ablated_edges) == 0

def test_node_parsing(test_model):
    """Verifies string components are mapped into exact layers and attention heads."""
    surgeon = CircuitSurgeon(test_model)
    
    layer, head = surgeon.parse_node("L0H1")
    assert layer == 0
    assert head == 1
    
    layer_mlp, head_mlp = surgeon.parse_node("L2MLP")
    assert layer_mlp == 2
    assert head_mlp is None

def test_surgeon_hooks_compilation(test_model):
    """Tests the correct generation and layering of PyTorch forward hook configurations."""
    surgeon = CircuitSurgeon(test_model)
    
    # Empty surgeon should yield no hooks
    baseline_cache = {}
    hooks = surgeon.get_ablation_hooks(baseline_cache)
    assert len(hooks) == 0
    
    # Register attention head node and MLP node
    surgeon.add_node_ablation("L0H1")
    surgeon.add_node_ablation("L1MLP")
    hooks = surgeon.get_ablation_hooks(baseline_cache)
    
    # We expect 2 hooks (one result hook for L0H1, one mlp_out hook for L1MLP)
    assert len(hooks) == 2
    hook_names = [h[0] for h in hooks]
    assert "blocks.0.attn.hook_result" in hook_names
    assert "blocks.1.hook_mlp_out" in hook_names

def test_circuit_surgeon_forward_pass(test_model):
    """Performs end-to-end ablated forward passes ensuring clean executions without shape mismatches."""
    surgeon = CircuitSurgeon(test_model)
    
    states = torch.randn(1, 3, 3)
    actions = torch.randn(1, 3, 7)
    returns = torch.randn(1, 3, 1)
    
    # 1. Run baseline
    preds_clean = test_model(states, actions, returns)
    assert preds_clean.shape == (1, 3, 7)
    
    # 2. Add MLP layer ablation and run
    surgeon.add_node_ablation("L0MLP")
    preds_ablated = surgeon.compute_ablated_forward(states, actions, returns)
    assert preds_ablated.shape == (1, 3, 7)
    # The output logit values should differ since we severed an entire layer
    assert not torch.allclose(preds_clean, preds_ablated, atol=1e-4)

    # 3. Add path/edge ablation and run
    surgeon.clear_ablations()
    surgeon.add_edge_ablation("L0H0", "L1H1")
    preds_edge_ablated = surgeon.compute_ablated_forward(states, actions, returns)
    assert preds_edge_ablated.shape == (1, 3, 7)

def test_neuronpedia_exporter_local(tmp_path):
    """Verifies formatting of circuit blueprints and local serialization backups."""
    exporter = NeuronpediaExporter()
    local_file = os.path.join(tmp_path, "test_export.json")
    
    manifest = {
        "active_heads": ["L0H0", "L1H1"],
        "pruned_count": 2,
        "initial_perf": 0.95,
        "final_perf": 0.88,
        "ablated_paths": ["L0H1 -> L1H0"],
        "ablated_nodes": ["L0H1", "L1MLP"],
        "state_dim": 3,
        "action_dim": 7,
        "n_layers": 2,
        "n_heads": 2
    }
    
    res = exporter.export_circuit("mini_dt", manifest, local_path=local_file)
    assert res["status"] == "success_local"
    assert os.path.exists(local_file)
    
    # Verify exact JSON schema
    with open(local_file, "r") as f:
        data = json.load(f)
        assert data["model_id"] == "mini_dt"
        assert data["source"] == "DT-Circuits"
        assert "circuit" in data
        assert data["circuit"]["pruned_count"] == 2
        assert "L0H0" in data["circuit"]["active_heads"]
