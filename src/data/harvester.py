import os
import gymnasium as gym
import torch
import numpy as np
from minigrid.wrappers import FlatObsWrapper
from stable_baselines3 import PPO
from tqdm import tqdm

class PPOHarvester:
    """
    Utility to run a 'Teacher' PPO agent to collect high-quality state-action-reward triplets.
    """
    def __init__(self, env_id="MiniGrid-Empty-8x8-v0", model_path=None):
        self.env_id = env_id
        self.env = FlatObsWrapper(gym.make(env_id, render_mode="rgb_array"))
        if model_path and os.path.exists(model_path):
            self.model = PPO.load(model_path, env=self.env)
        else:
            print(f"No model found at {model_path}. Training a new one for collection...")
            self.model = PPO("MlpPolicy", self.env, verbose=1)
            self.model.learn(total_timesteps=20000)
            if model_path:
                self.model.save(model_path)

    def collect_trajectories(self, num_episodes=100):
        trajectories = []
        for i in tqdm(range(num_episodes), desc="Collecting trajectories"):
            obs, _ = self.env.reset(seed=42 + i)
            done = False
            truncated = False
            episode = {
                "observations": [],
                "actions": [],
                "rewards": [],
                "dones": []
            }
            while not (done or truncated):
                action, _states = self.model.predict(obs, deterministic=False)
                next_obs, reward, done, truncated, info = self.env.step(action)
                
                episode["observations"].append(obs)
                episode["actions"].append(action)
                episode["rewards"].append(reward)
                episode["dones"].append(done)
                
                obs = next_obs
            
            # Convert to numpy arrays
            for key in episode:
                episode[key] = np.array(episode[key])
            
            trajectories.append(episode)
        
        return trajectories

    def save_trajectories(self, trajectories, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(trajectories, file_path)
        print(f"Saved {len(trajectories)} trajectories to {file_path}")

if __name__ == "__main__":
    harvester = PPOHarvester(model_path="ppo_minigrid_teacher.zip")
    trajs = harvester.collect_trajectories(num_episodes=50)
    harvester.save_trajectories(trajs, "data/trajectories.pt")
