import torch
import torch.nn as nn
import torch.optim as optim
from src.models.hooked_dt import HookedDT
from src.data.harvester import PPOHarvester
import numpy as np
from tqdm import tqdm

def train():
    harvester = PPOHarvester(model_path="ppo_minigrid_teacher.zip")
    trajectories = harvester.collect_trajectories(num_episodes=100)
    
    state_dim = trajectories[0]["observations"].shape[1]
    action_dim = 7 # MiniGrid 
    
    model = HookedDT.from_config(
        state_dim=state_dim,
        action_dim=action_dim,
        n_layers=1,
        n_heads=4,
        d_model=128
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0
        for traj in tqdm(trajectories, desc=f"Epoch {epoch}"):
            states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
            actions = torch.from_numpy(traj["actions"]).long().unsqueeze(0)
            actions_one_hot = torch.nn.functional.one_hot(actions, num_classes=action_dim).float()
            returns = torch.from_numpy(traj["rewards"]).float().unsqueeze(0).unsqueeze(-1)
            timesteps = torch.arange(states.shape[1]).unsqueeze(0)

            action_preds, _, _ = model(states, actions_one_hot, returns, timesteps)
            
            loss = criterion(action_preds.view(-1, action_dim), actions.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch} Loss: {total_loss / len(trajectories)}")

    torch.save(model.state_dict(), "models/mini_dt.pt")
    print("Model saved to models/mini_dt.pt")

if __name__ == "__main__":
    train()
