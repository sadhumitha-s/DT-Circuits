import torch
import torch.nn as nn
from typing import Optional

class RTGSteerer:
    """
    Enables 'Behavioral Steering' by manipulating Reward-to-Go (RTG) tokens.
    """
    def __init__(self, model):
        self.model = model

    def steer(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        base_rtg: torch.Tensor,
        steering_vector: torch.Tensor,
        alpha: float = 1.0
    ):
        """
        Adds a steering vector to the RTG embeddings.
        RTG_new = RTG_base + alpha * steering_vector
        """
        # Embed base RTG
        with torch.no_grad():
            rtg_emb = self.model.embed_return(base_rtg)
            
            # Apply steering
            steered_rtg_emb = rtg_emb + alpha * steering_vector
            
            # Hook the model to use the steered RTG
            # This requires a slightly more complex hook in HookedDT
            # For now, we returns the steered embedding to be used in a custom forward pass
            return steered_rtg_emb

    def find_success_vector(self, high_reward_cache, low_reward_cache):
        """
        Identifies the 'Success Vector' by comparing high vs low reward activations.
        Vector = Mean(High Reward Residual) - Mean(Low Reward Residual)
        """
        high_res = high_reward_cache["blocks.0.hook_resid_post"].mean(dim=(0, 1))
        low_res = low_reward_cache["blocks.0.hook_resid_post"].mean(dim=(0, 1))
        return high_res - low_res
