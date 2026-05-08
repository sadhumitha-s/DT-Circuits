import torch
import torch.nn as nn
import os
from typing import Dict, List, Optional, Tuple, Union
from sae_lens import StandardSAE, StandardSAEConfig
from jaxtyping import Float

class SAEManager:
    """
    Research-grade manager for Sparse Autoencoders (SAEs) integrated with Decision Transformers.
    Handles training, decomposition into monosemantic latents, and mechanistic anomaly detection.
    """
    def __init__(self, model: nn.Module, sae_dir: str = "artifacts/saes"):
        self.model = model
        self.sae_dir = sae_dir
        self.saes: Dict[str, StandardSAE] = {}
        os.makedirs(sae_dir, exist_ok=True)

    def setup_sae(
        self,
        hook_point: str,
        d_model: int,
        expansion_factor: int = 8,
    ) -> StandardSAE:
        """
        Initializes an SAE for a specific hook point in the transformer.
        """
        cfg = StandardSAEConfig(
            d_in=d_model,
            d_sae=d_model * expansion_factor,
            device=str(next(self.model.parameters()).device)
        )
        sae = StandardSAE(cfg)
        self.saes[hook_point] = sae
        return sae

    def train_on_trajectories(
        self,
        hook_point: str,
        activations: Float[torch.Tensor, "n_samples d_model"],
        l1_coefficient: float = 0.0001,
        batch_size: int = 1024,
        epochs: int = 10,
    ):
        """
        Trains the SAE on collected trajectory activations.
        """
        if hook_point not in self.saes:
            self.setup_sae(hook_point, activations.shape[-1])
        
        sae = self.saes[hook_point]
        optimizer = torch.optim.Adam(sae.parameters(), lr=0.0004)
        
        sae.train()
        n_samples = activations.shape[0]
        
        for epoch in range(epochs):
            permutation = torch.randperm(n_samples)
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                indices = permutation[i:i+batch_size]
                batch_acts = activations[indices].to(sae.device)
                
                optimizer.zero_grad()
                
                # Manual forward pass for training
                feature_acts = sae.encode(batch_acts)
                sae_out = sae.decode(feature_acts)
                
                mse_loss = torch.nn.functional.mse_loss(sae_out, batch_acts)
                l1_loss = l1_coefficient * feature_acts.abs().sum()
                loss = mse_loss + l1_loss
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / (n_samples / batch_size):.4f}")

    def get_feature_activations(
        self,
        hook_point: str,
        activations: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_sae"]:
        """
        Decomposes activations into monosemantic features.
        """
        if hook_point not in self.saes:
            raise ValueError(f"SAE for {hook_point} not found. Train or load it first.")
        
        sae = self.saes[hook_point]
        sae.eval()
        with torch.no_grad():
            feature_acts = sae.encode(activations.to(sae.device))
        return feature_acts

    def reconstruct(
        self,
        hook_point: str,
        activations: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "... d_model"]:
        """
        Reconstructs the original activations using the SAE.
        """
        if hook_point not in self.saes:
            raise ValueError(f"SAE for {hook_point} not found.")
        
        sae = self.saes[hook_point]
        sae.eval()
        with torch.no_grad():
            feature_acts = sae.encode(activations.to(sae.device))
            sae_out = sae.decode(feature_acts)
        return sae_out

    def compute_anomaly_score(
        self,
        hook_point: str,
        activations: Float[torch.Tensor, "... d_model"]
    ) -> Float[torch.Tensor, "..."]:
        """
        Calculates reconstruction error as a proxy for mechanistic anomaly detection.
        Formula: ||x - x_hat|| / ||x||
        """
        if hook_point not in self.saes:
            raise ValueError(f"SAE for {hook_point} not found.")
        
        sae = self.saes[hook_point]
        sae.eval()
        with torch.no_grad():
            x = activations.to(sae.device)
            feature_acts = sae.encode(x)
            x_hat = sae.decode(feature_acts)
            
            error = torch.norm(x - x_hat, dim=-1) / (torch.norm(x, dim=-1) + 1e-8)
        return error

    def save_all_saes(self):
        for hook, sae in self.saes.items():
            path = os.path.join(self.sae_dir, f"{hook.replace('.', '_')}_sae.pt")
            torch.save({
                'state_dict': sae.state_dict(),
                'cfg': sae.cfg
            }, path)
            print(f"Saved SAE for {hook} to {path}")

    def load_sae(self, hook_point: str):
        path = os.path.join(self.sae_dir, f"{hook_point.replace('.', '_')}_sae.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved SAE found at {path}")
        
        checkpoint = torch.load(path, map_location=str(next(self.model.parameters()).device))
        sae = StandardSAE(checkpoint['cfg'])
        sae.load_state_dict(checkpoint['state_dict'])
        self.saes[hook_point] = sae
        return sae
