import sys
from pathlib import Path
import torch
from sae_lens import TopKSAEConfig, TopKSAE

# Add project root to path for absolute imports
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import random
import numpy as np
from src.models.hooked_dt import HookedDT
from src.interpretability.sae_manager import SAEManager
from src.config import cfg

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_sae():
    # 0. Set seed for reproducibility
    set_seed(cfg.train.seed)

    # 1. Load Trajectories to get dimensions
    traj_path = "data/trajectories.pt"
    if not Path(traj_path).exists():
        print(f"Error: {traj_path} not found. Please run scripts/train_dt.py first.")
        return
    
    trajectories = torch.load(traj_path, weights_only=False)
    print(f"Loaded {len(trajectories)} trajectories.")

    # 2. Initialize Model
    state_dim = trajectories[0]["observations"].shape[1]
    action_dim = cfg.model.action_dim
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = HookedDT.from_config(
        state_dim=state_dim, 
        action_dim=action_dim,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        d_model=cfg.model.d_model
    )
    model.to(device)
    
    # Check for trained DT checkpoint
    checkpoint_path = "models/mini_dt.pt"
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded DT weights from {checkpoint_path}")
    else:
        print(f"Warning: {checkpoint_path} not found. Training SAE on random weights.")

    # 3. & 4. Train SAEs for ALL layers
    manager = SAEManager(model, sae_dir="artifacts/saes")
    
    for layer in range(model.cfg.n_layers):
        hook_point = f"blocks.{layer}.hook_resid_post"
        all_activations = []
        
        print(f"\n--- Processing Layer {layer} ({hook_point}) ---")
        
        # Extract Activations
        model.eval()
        print(f"Extracting activations...")

        # Number of trajectories from config
        num_trajs_to_use = min(len(trajectories), cfg.sae.num_episodes)
        
        with torch.no_grad():
            for traj in trajectories[:num_trajs_to_use]:
                states = torch.from_numpy(traj["observations"]).float().to(device).unsqueeze(0)
                actions = torch.from_numpy(traj["actions"]).long().to(device)
                actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=action_dim).float().unsqueeze(0)
                returns = torch.from_numpy(traj["rewards"]).float().to(device).unsqueeze(0).unsqueeze(-1)

                _, cache = model(states, actions_one_hot, returns, return_cache=True)
                all_activations.append(cache[hook_point].squeeze(0).cpu())
        
        activations = torch.cat(all_activations, dim=0)
        print(f"Collected {activations.shape[0]} activation vectors.")

        # Setup and Train
        print(f"Starting TopK SAE training...")
        manager.setup_sae(
            hook_point=hook_point,
            d_model=cfg.model.d_model,
            architecture="topk",
            k=cfg.sae.k
        )
        
        manager.train_on_trajectories(
            hook_point=hook_point,
            activations=activations,
            epochs=cfg.sae.epochs,
            batch_size=cfg.sae.batch_size
        )

    # Save all SAEs once training is complete for all layers
    manager.save_all_saes()

    print(f"\nSAE Training Complete for all {model.cfg.n_layers} layers. Results saved to artifacts/saes/")

if __name__ == "__main__":
    train_sae()
