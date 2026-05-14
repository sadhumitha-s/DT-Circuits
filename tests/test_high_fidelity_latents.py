import torch
import torch.nn as nn
import pytest
import os
from src.interpretability.sae_manager import SAEManager
from src.interpretability.nla import NLAExplainer
from src.interpretability.universality import UniversalityProbe

class MockModel(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
        self.d_model = d_model

def test_topk_sae_setup_and_training():
    model = MockModel()
    manager = SAEManager(model, sae_dir="tests/artifacts/saes")
    
    hook_point = "blocks.0.hook_resid_post"
    d_model = 128
    
    # Setup TopK SAE
    sae = manager.setup_sae(hook_point, d_model, architecture="topk", k=10)
    
    assert sae.cfg.k == 10
    assert sae.cfg.d_in == d_model
    
    # Mock activations
    activations = torch.randn(100, d_model)
    
    # Test training (short run)
    manager.train_on_trajectories(hook_point, activations, epochs=1, batch_size=10)
    
    # Test feature extraction
    features = manager.get_feature_activations(hook_point, activations)
    assert features.shape[0] == 100
    # TopK should have exactly k active features per sample (or less if some are zero, but usually k)
    l0 = (features > 0).float().sum(dim=-1)
    assert torch.all(l0 <= 10)
    
    # Test save/load
    manager.save_all_saes()
    
    new_manager = SAEManager(model, sae_dir="tests/artifacts/saes")
    loaded_sae = new_manager.load_sae(hook_point)
    assert loaded_sae.cfg.k == 10

def test_nla_explainer():
    explainer = NLAExplainer()
    
    feature_id = 42
    top_acts = [
        {"state": "near_wall", "value": 0.9},
        {"state": "facing_wall", "value": 0.85}
    ]
    
    label = explainer.generate_label(feature_id, top_acts, context_description="Wall avoidance")
    assert "Mock Feature 42" in label
    assert explainer.get_label(feature_id) == label

def test_universality_probe():
    dt_model = MockModel(d_model=128)
    dqn_model = MockModel(d_model=64)
    probe = UniversalityProbe(dt_model, dqn_model)
    
    # Mock data
    dt_features = torch.randn(100, 32)
    dqn_activations = torch.randn(100, 16)
    
    # Force a high correlation for testing
    dt_features[:, 0] = dqn_activations[:, 0] * 2 + 0.1
    
    corr_matrix = probe.compute_cross_correlation(dt_features, dqn_activations)
    assert corr_matrix.shape == (32, 16)
    
    universal = probe.identify_universal_features(corr_matrix, threshold=0.9)
    assert len(universal) > 0
    assert universal[0]["dt_feature_idx"] == 0
    assert universal[0]["dqn_neuron_idx"] == 0

if __name__ == "__main__":
    pytest.main([__file__])
