from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import yaml
from pathlib import Path

@dataclass
class ModelConfig:
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    max_length: int = 30
    state_dim: Optional[int] = None
    action_dim: int = 7

@dataclass
class DataConfig:
    env_id: str = "MiniGrid-Empty-8x8-v0"
    num_episodes: int = 1000
    collection_method: str = "PPO-Teacher"

@dataclass
class TrainConfig:
    lr: float = 1e-4
    epochs: int = 10
    seed: int = 42

@dataclass
class SAEConfig:
    expansion_factor: int = 8
    k: int = 32
    l1_coeff: float = 0.0005
    lr: float = 3e-4
    epochs: int = 5
    batch_size: int = 1024
    num_episodes: int = 100

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sae: SAEConfig = field(default_factory=SAEConfig)

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "config.yaml") -> "Config":
        """Loads configuration from a YAML file, overriding defaults."""
        path = Path(yaml_path)
        if not path.exists():
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Helper to safely update dataclass from dict
        def update_dataclass(dc_obj, dc_dict):
            for key, value in dc_dict.items():
                if hasattr(dc_obj, key):
                    setattr(dc_obj, key, value)

        config = cls()
        if "model" in data:
            update_dataclass(config.model, data["model"])
        if "data" in data:
            update_dataclass(config.data, data["data"])
        if "train" in data:
            update_dataclass(config.train, data["train"])
        if "sae" in data:
            update_dataclass(config.sae, data["sae"])
            
        return config

# Global config instance for easy access
cfg = Config.load_from_yaml()
