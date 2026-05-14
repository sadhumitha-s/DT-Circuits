import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Add project root to path for absolute imports
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from src.models.hooked_dt import HookedDT
from src.data.harvester import PPOHarvester

def train():
    """Main training loop for Decision Transformer."""
    # Step 1: Collect data from expert PPO teacher
    harvester = PPOHarvester(model_path="models/ppo_teacher.zip")
    trajectories = harvester.collect_trajectories(num_episodes=100)
    
    # Save trajectories for the dashboard to use later
    harvester.save_trajectories(trajectories, "data/trajectories.pt")
    
    state_dim = trajectories[0]["observations"].shape[1]
    action_dim = 7 # MiniGrid standard actions
    
    model = HookedDT.from_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_layers=2,
        n_heads=4,
        d_model=128
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Step 2: Train the DT
    model.train()
    for epoch in range(10):
        total_loss = 0
        for traj in tqdm(trajectories, desc=f"Epoch {epoch}"):
            states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
            actions = torch.from_numpy(traj["actions"]).long()
            actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=action_dim).float().unsqueeze(0)
            returns = torch.from_numpy(traj["rewards"]).float().unsqueeze(0).unsqueeze(-1)

            # Predict actions based on State tokens
            action_preds = model(states, actions_one_hot, returns)
            
            # Cross entropy loss on predicted actions
            loss = criterion(action_preds.view(-1, action_dim), actions.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch} Loss: {total_loss / len(trajectories)}")

    # Step 3: Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/mini_dt.pt")
    print("Model saved to models/mini_dt.pt")

if __name__ == "__main__":
    train()
