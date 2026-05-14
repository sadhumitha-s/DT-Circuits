import torch
from typing import Dict, Optional, Tuple

class PathPatchingEngine:
    """
    Engine for path-based causal interventions.
    Helps isolate which internal paths are necessary for a decision.
    """
    def __init__(self, model):
        self.model = model

    def perform_edge_ablation(
        self,
        inputs: dict,
        layer: int,
        head_index: int,
        ablation_type: str = "zero"
    ) -> torch.Tensor:
        """Zeroes out a specific head's output to check its causal necessity."""
        def ablation_hook(value, hook):
            if ablation_type == "zero":
                value[:, :, head_index, :] = 0.0
            return value

        hook_name = f"blocks.{layer}.attn.hook_result"
        with self.model.transformer.hooks(fwd_hooks=[(hook_name, ablation_hook)]):
            preds = self.model(**inputs)
        return preds

    def patch_path(
        self,
        clean_inputs: dict,
        corrupted_cache: dict,
        layer: int,
        head: int
    ) -> torch.Tensor:
        """Patches a specific head's output with activations from a corrupted run."""
        def patch_hook(value, hook):
            # value: [batch, pos, head, d_model]
            corrupted_val = corrupted_cache[hook.name]
            value[:, :, head, :] = corrupted_val[:, :, head, :]
            return value

        hook_name = f"blocks.{layer}.attn.hook_result"
        with self.model.transformer.hooks(fwd_hooks=[(hook_name, patch_hook)]):
            preds = self.model(**clean_inputs)
        return preds
