import pytest
import torch
from src.models.hooked_dt import HookedDT
from src.interpretability.sae_manager import SAEManager
from src.interpretability.steering import RTGSteerer, SteeringLibrary

@pytest.fixture
def model():
    return HookedDT.from_config(state_dim=10, action_dim=5, n_layers=1, n_heads=2, d_model=32)

@pytest.fixture
def sae_manager(model):
    return SAEManager(model)

def test_sae_lifecycle(sae_manager):
    hook_point = "blocks.0.hook_resid_post"
    d_model = 32
    
    # 1. Setup SAE
    sae = sae_manager.setup_sae(hook_point, d_model, expansion_factor=2)
    assert hook_point in sae_manager.saes
    assert sae.cfg.d_sae == 64
    
    # 2. Mock activations
    activations = torch.randn(100, d_model)
    
    # 3. Test reconstruction shape
    reconstructed = sae_manager.reconstruct(hook_point, activations)
    assert reconstructed.shape == activations.shape
    
    # 4. Test feature activations shape
    latents = sae_manager.get_feature_activations(hook_point, activations)
    assert latents.shape == (100, 64)
    
    # 5. Test anomaly score
    scores = sae_manager.compute_anomaly_score(hook_point, activations)
    assert scores.shape == (100,)
    assert (scores >= 0).all()

def test_steering_library():
    lib = SteeringLibrary(d_model=32)
    vec = torch.randn(32)
    lib.add_vector("test_vec", vec)
    
    assert "test_vec" in lib.list_vectors()
    assert torch.equal(lib.get_vector("test_vec"), vec)
    
    with pytest.raises(ValueError):
        lib.add_vector("wrong_dim", torch.randn(16))

def test_caa_generation(model):
    steerer = RTGSteerer(model)
    pos_acts = torch.randn(10, 32) + 1.0
    neg_acts = torch.randn(10, 32) - 1.0
    
    vector = steerer.generate_caa_vector(pos_acts, neg_acts)
    assert vector.shape == (32,)
    # Mean difference should be around 2.0 for each dimension
    assert vector.mean() > 0.0

def test_steering_hook(model):
    lib = SteeringLibrary(d_model=32)
    vec = torch.ones(32)
    lib.add_vector("boost", vec)
    
    steerer = RTGSteerer(model, library=lib)
    hook = steerer.apply_steering_hook("blocks.0.hook_resid_post", "boost", alpha=2.0)
    
    input_acts = torch.zeros(1, 5, 32)
    output_acts = hook(input_acts, None)
    
    assert torch.allclose(output_acts, torch.ones(1, 5, 32) * 2.0)

if __name__ == "__main__":
    pytest.main([__file__])
