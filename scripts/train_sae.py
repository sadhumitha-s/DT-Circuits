import torch
from sae_lens import SAEConfig, SAE
from src.models.hooked_dt import HookedDT

def train_sae():
    # Load DT
    state_dim = 2739
    action_dim = 7
    model = HookedDT.from_config(state_dim, action_dim)
    # model.load_state_dict(torch.load("models/mini_dt.pt"))

    # Configure SAE
    cfg = SAEConfig(
        d_in=128, # d_model
        d_sae=128 * 8, # Expansion factor
        hook_point="blocks.0.hook_resid_post",
        hook_point_layer=0,
        architecture="standard",
        activation_fn="relu",
        expansion_factor=8,
        l1_coefficient=5e-4,
        lr=3e-4,
        train_batch_size=4096,
        context_size=30, # Sequence length
    )

    sae = SAE(cfg)
    
    # Training logic would go here, using activations from the DT
    print("SAE Configured for DT-Explorer.")
    print(f"Hooking into: {cfg.hook_point}")

if __name__ == "__main__":
    train_sae()
