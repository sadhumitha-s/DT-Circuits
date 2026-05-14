import pytest
import torch
from src.models.hooked_dt import HookedDT
from src.interpretability.attribution import LogitAttributionEngine
from transformer_lens import HookedTransformerConfig

def test_hooked_dt_forward():
    """Verifies basic forward pass of HookedDT."""
    state_dim, action_dim, seq_len, batch_size = 10, 5, 5, 2
    model = HookedDT.from_config(state_dim, action_dim, n_layers=1, n_heads=2, d_model=32)
    
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    returns = torch.randn(batch_size, seq_len, 1)
    
    action_preds = model(states, actions, returns)
    assert action_preds.shape == (batch_size, seq_len, action_dim)

def test_hooked_dt_with_cache():
    """Verifies that cache is returned correctly."""
    state_dim, action_dim, seq_len, batch_size = 10, 5, 5, 1
    model = HookedDT.from_config(state_dim, action_dim, n_layers=1, n_heads=2, d_model=32)
    
    states = torch.randn(batch_size, seq_len, state_dim)
    actions = torch.randn(batch_size, seq_len, action_dim)
    returns = torch.randn(batch_size, seq_len, 1)
    
    preds, cache = model(states, actions, returns, return_cache=True)
    assert "blocks.0.attn.hook_result" in cache
    assert preds.shape == (batch_size, seq_len, action_dim)

def test_logit_attribution_shape():
    """Checks that DLA engine produces the correct result matrix."""
    state_dim, action_dim = 10, 5
    model = HookedDT.from_config(state_dim, action_dim, n_layers=2, n_heads=4, d_model=32)
    engine = LogitAttributionEngine(model)
    
    cache = {f"blocks.{l}.attn.hook_result": torch.randn(1, 15, 4, 32) for l in range(2)}
    dla = engine.calculate_dla(cache, target_logit_index=0, token_index=-1)
    assert dla.shape == (2, 4)

if __name__ == "__main__":
    pytest.main([__file__])
