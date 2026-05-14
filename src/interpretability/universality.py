import torch
import torch.nn as nn
from typing import Dict, List, Any
import numpy as np

class UniversalityProbe:
    """
    Probes for universal feature representations across different architectures (e.g., DT vs DQN).
    """
    def __init__(self, dt_model: nn.Module, dqn_model: nn.Module):
        self.dt_model = dt_model
        self.dqn_model = dqn_model

    def collect_paired_activations(
        self, 
        env_states: torch.Tensor,
        dt_hook_point: str,
        dqn_layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Collects activations from both models on the same set of environmental states.
        """
        # DT activations (assuming cache is handled or provided)
        # This is a simplified placeholder
        dt_acts = torch.randn(env_states.shape[0], 128) # Mock
        
        # DQN activations
        # dqn_acts = self.dqn_model.get_layer_activations(env_states, dqn_layer_idx)
        dqn_acts = torch.randn(env_states.shape[0], 64) # Mock
        
        return {
            "dt": dt_acts,
            "dqn": dqn_acts
        }

    def compute_cross_correlation(
        self, 
        dt_sae_features: torch.Tensor, 
        dqn_activations: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the correlation matrix between DT SAE features and DQN activations.
        High correlation suggests a 'Universal Concept'.
        """
        # Normalize
        dt_feat_norm = (dt_sae_features - dt_sae_features.mean(dim=0)) / (dt_sae_features.std(dim=0) + 1e-8)
        dqn_act_norm = (dqn_activations - dqn_activations.mean(dim=0)) / (dqn_activations.std(dim=0) + 1e-8)
        
        # Correlation matrix
        correlation = torch.matmul(dt_feat_norm.t(), dqn_act_norm) / dt_feat_norm.shape[0]
        return correlation

    def identify_universal_features(
        self, 
        correlation_matrix: torch.Tensor, 
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Identifies pairs of (DT Feature, DQN Neuron) that represent the same concept.
        """
        universal_pairs = []
        matches = (correlation_matrix.abs() > threshold).nonzero()
        
        for i, j in matches:
            universal_pairs.append({
                "dt_feature_idx": i.item(),
                "dqn_neuron_idx": j.item(),
                "correlation": correlation_matrix[i, j].item()
            })
            
        return universal_pairs
