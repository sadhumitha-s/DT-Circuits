import torch
import numpy as np
import pytest
from src.models.hooked_dt import HookedDT
from src.interpretability.attribution import LogitAttributionEngine
from src.interpretability.patching import ActivationPatcher

@pytest.fixture
def model():
    return HookedDT.from_config(state_dim=10, action_dim=7, n_layers=1, n_heads=2, d_model=32)

@pytest.fixture
def mock_data():
    batch_size = 1
    seq_len = 5
    state_dim = 10
    action_dim = 7
    
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.nn.functional.one_hot(torch.randint(0, action_dim, (batch_size, seq_len)), num_classes=action_dim).float()
    returns = torch.randn(batch_size, seq_len, 1)
    
    return {"states": states, "actions": actions, "returns_to_go": returns}

def test_logit_attribution(model, mock_data):
    engine = LogitAttributionEngine(model)
    preds, cache = model(**mock_data, return_cache=True)
    target_action = preds[0, -1].argmax().item()
    
    dla = engine.calculate_dla(cache, target_logit_index=target_action, token_index=-2)
    
    assert dla.shape == (model.cfg.n_layers, model.cfg.n_heads)
    assert not torch.isnan(dla).any()

def test_activation_patching(model, mock_data):
    patcher = ActivationPatcher(model)
    
    # Clean run
    clean_preds, clean_cache = model(**mock_data, return_cache=True)
    clean_probs = torch.softmax(clean_preds, dim=-1)
    target_action = clean_preds[0, -1].argmax().item()
    
    # Create corrupted run (zeroed states)
    corrupted_data = mock_data.copy()
    corrupted_data["states"] = torch.zeros_like(mock_data["states"])
    _, corrupted_cache = model(**corrupted_data, return_cache=True)
    
    # Patch head 0 of layer 0
    patched_logits = patcher.patch_head(
        mock_data, 
        corrupted_cache, 
        layer=0, 
        head_index=0, 
        target_token_index=-2
    )
    patched_probs = torch.softmax(patched_logits, dim=-1)
    
    drop = patcher.calculate_probability_drop(clean_probs, patched_probs, target_action)
    
    assert isinstance(drop, float)
    # Patching with corrupted (zeros) should generally decrease performance/probability
    # but at minimum we check it returns a valid number.
    assert not np.isnan(drop)
