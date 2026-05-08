import torch
import torch.nn as nn
from typing import Dict, List, Optional
class SteeringLibrary:
    """
    A persistent library of pre-calculated steering vectors (CAA).
    Includes vectors for exploration, safety, and goal-directedness.
    """
    def __init__(self, d_model: int):
        self.d_model = d_model
        self.vectors: Dict[str, torch.Tensor] = {}

    def add_vector(self, name: str, vector: torch.Tensor):
        if vector.shape[-1] != self.d_model:
            raise ValueError(f"Vector dimension {vector.shape[-1]} does not match d_model {self.d_model}")
        self.vectors[name] = vector

    def get_vector(self, name: str) -> torch.Tensor:
        if name not in self.vectors:
            raise KeyError(f"Vector '{name}' not found in library.")
        return self.vectors[name]

    def list_vectors(self) -> List[str]:
        return list(self.vectors.keys())

class RTGSteerer:
    """
    Enables 'Behavioral Steering' by manipulating Reward-to-Go (RTG) tokens or internal activations.
    Supports Contrastive Activation Addition (CAA).
    """
    def __init__(self, model, library: Optional[SteeringLibrary] = None):
        self.model = model
        self.library = library or SteeringLibrary(model.cfg.d_model)

    def steer_rtg(
        self,
        base_rtg: torch.Tensor,
        vector_name: Optional[str] = None,
        custom_vector: Optional[torch.Tensor] = None,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Adds a steering vector to the RTG embeddings.
        """
        vector = custom_vector if custom_vector is not None else self.library.get_vector(vector_name)
        
        with torch.no_grad():
            rtg_emb = self.model.embed_return(base_rtg)
            return rtg_emb + alpha * vector

    def generate_caa_vector(
        self,
        positive_activations: torch.Tensor,
        negative_activations: torch.Tensor,
        method: str = "mean_diff"
    ) -> torch.Tensor:
        """
        Generates a steering vector using Contrastive Activation Addition.
        'mean_diff' calculates the difference between the means of positive and negative sets.
        """
        if method == "mean_diff":
            pos_mean = positive_activations.mean(dim=0)
            neg_mean = negative_activations.mean(dim=0)
            return pos_mean - neg_mean
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

    def apply_steering_hook(self, hook_point: str, vector_name: str, alpha: float = 1.0):
        """
        Returns a HookedTransformer compatible hook function that applies steering.
        """
        vector = self.library.get_vector(vector_name)
        
        def steering_hook(activations, hook):
            # activations: [batch, pos, d_model]
            return activations + alpha * vector
            
        return steering_hook
