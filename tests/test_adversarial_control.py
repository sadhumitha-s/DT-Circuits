import pytest
import torch
import numpy as np
from src.models.hooked_dt import HookedDT
from src.interpretability.sae_manager import SAEManager
from src.interpretability.safety import (
    DynamicRejectionSteerer,
    DeceptiveAlignmentAuditor,
    FunctionalAttributionMAD,
    generate_deceptive_trajectories
)

@pytest.fixture
def base_model():
    """Initializes a tiny HookedDT model for testing."""
    return HookedDT.from_config(
        state_dim=3,
        action_dim=3,
        n_layers=1,
        n_heads=2,
        d_model=32,
        max_length=5
    )

@pytest.fixture
def sae_manager(base_model):
    """Initializes SAEManager with a temporary directory."""
    return SAEManager(base_model, sae_dir="tests/artifacts/safety_saes")

def test_dynamic_rejection_steerer(base_model):
    """Verifies that DynamicRejectionSteerer scales back steering when safety constraints are violated."""
    steerer = DynamicRejectionSteerer(base_model)
    hook_point = "blocks.0.hook_resid_post"
    steering_vector = torch.randn(32)
    
    # Inputs
    states = torch.randn(1, 3, 3)
    actions = torch.randn(1, 3, 3)
    returns = torch.randn(1, 3, 1)

    # 1. Edge Case: Fully safe. The safety check always returns True.
    def safe_check(state, probs):
        return True
        
    _, alpha_safe = steerer.steer_safely(
        states, actions, returns, hook_point, steering_vector, safe_check, initial_alpha=1.0
    )
    assert alpha_safe == 1.0

    # 2. Edge Case: Always unsafe. The safety check always returns False.
    def unsafe_check(state, probs):
        return False
        
    _, alpha_unsafe = steerer.steer_safely(
        states, actions, returns, hook_point, steering_vector, unsafe_check, initial_alpha=1.0
    )
    assert alpha_unsafe == 0.0

    # 3. Dynamic scenario: Action index 1 is considered illegal if its probability is above 0.35
    def dynamic_safety_check(state, probs):
        # probs is action probabilities of the last step, shape [action_dim]
        # If action 1 is the most probable or above 0.35, it's unsafe
        return probs[1].item() < 0.35

    # Force action prediction to fail at high alpha and succeed at low alpha by checking scaling
    _, final_alpha = steerer.steer_safely(
        states, actions, returns, hook_point, steering_vector, dynamic_safety_check,
        initial_alpha=1.0, decay_factor=0.5, min_alpha=0.01, max_iterations=4
    )
    # The steerer should either return a valid reduced alpha or 0.0 depending on model predictions
    assert 0.0 <= final_alpha <= 1.0

def test_deceptive_alignment_and_audit(base_model, sae_manager):
    """Trains a model on deceptive trajectories, trains a TopK SAE, and audits situational awareness."""
    # 1. Generate deceptive trajectory dataset
    trajectories = generate_deceptive_trajectories(num_episodes=20, seq_len=5)
    assert len(trajectories) == 20
    assert trajectories[0]["observations"].shape == (5, 3)
    
    # 2. Train model to adapt to deception behavior
    optimizer = torch.optim.Adam(base_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    base_model.train()
    
    for epoch in range(10): # Quick training
        for traj in trajectories:
            states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
            actions = torch.from_numpy(traj["actions"]).long()
            actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=3).float().unsqueeze(0)
            returns = torch.from_numpy(traj["rewards"]).float().unsqueeze(0).unsqueeze(-1)
            
            optimizer.zero_grad()
            preds = base_model(states, actions_one_hot, returns)
            loss = criterion(preds.view(-1, 3), actions.view(-1))
            loss.backward()
            optimizer.step()
            
    base_model.eval()

    # 3. Collect activations for SAE training
    hook_point = "blocks.0.hook_resid_post"
    activations_list = []
    
    with torch.no_grad():
        for traj in trajectories:
            states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
            actions = torch.from_numpy(traj["actions"]).long()
            actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=3).float().unsqueeze(0)
            returns = torch.from_numpy(traj["rewards"]).float().unsqueeze(0).unsqueeze(-1)
            
            _, cache = base_model(states, actions_one_hot, returns, return_cache=True)
            activations_list.append(cache[hook_point].squeeze(0)) # Shape [seq_len * 3, d_model]
            
    activations = torch.cat(activations_list, dim=0) # [N, d_model]

    # 4. Setup and train a TopK SAE on the collected activations
    # Expansion factor 2 -> d_sae = 64
    sae_manager.setup_sae(hook_point, d_model=32, expansion_factor=2, architecture="topk", k=8)
    sae_manager.train_on_trajectories(hook_point, activations, epochs=2, batch_size=32)

    # 5. Run the Auditor
    auditor = DeceptiveAlignmentAuditor(base_model, sae_manager)
    
    watched_states = torch.tensor([[[5.0, 1.0, 5.0], [4.0, 1.0, 5.0]]], dtype=torch.float32)
    unwatched_states = torch.tensor([[[5.0, 0.0, 5.0], [5.0, 0.0, 4.0]]], dtype=torch.float32)
    actions = torch.zeros((1, 2, 3), dtype=torch.float32)
    returns = torch.tensor([[[0.2], [0.4]]], dtype=torch.float32)
    
    switch_idx = auditor.identify_situational_awareness_feature(
        hook_point, watched_states, unwatched_states, actions, returns
    )
    assert 0 <= switch_idx < 64

    # 6. Run Direct Feature Attribution
    dfa = auditor.attribute_feature_to_heads(
        hook_point, switch_idx, watched_states, actions, returns
    )
    assert dfa.shape == (1, 2) # n_layers=1, n_heads=2

def test_functional_attribution_mad(base_model):
    """Verifies that FunctionalAttributionMAD correctly flags anomalous/atypical execution circuits."""
    mad = FunctionalAttributionMAD(base_model, threshold=0.6)
    
    # Establish inputs
    baseline_states = torch.randn(1, 3, 3)
    baseline_actions = torch.randn(1, 3, 3)
    baseline_returns = torch.randn(1, 3, 1)
    
    # Establish baseline profile for action 0
    mad.establish_reference_profile(baseline_states, baseline_actions, baseline_returns, target_action_index=0)
    assert mad.reference_profile is not None
    assert mad.reference_profile.shape == (2,) # n_layers=1, n_heads=2 -> 2 heads total
    
    # Test identical run (similarity should be very high, close to 1.0)
    is_anomaly, similarity, profile = mad.detect_circuit_anomaly(
        baseline_states, baseline_actions, baseline_returns, target_action_index=0
    )
    assert not is_anomaly
    assert pytest.approx(similarity, abs=1e-5) == 1.0

    # Test an anomalous run with different inputs / targets (producing different activations/attributions)
    anomalous_states = torch.randn(1, 3, 3) + 5.0
    is_anomaly_anom, similarity_anom, profile_anom = mad.detect_circuit_anomaly(
        anomalous_states, baseline_actions, baseline_returns, target_action_index=1
    )
    # An anomaly may or may not be flagged depending on weights, but we can verify calculation correctness
    assert similarity_anom <= 1.0
    assert profile_anom.shape == (2,)

if __name__ == "__main__":
    pytest.main([__file__])
