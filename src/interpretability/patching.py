import torch
from typing import Callable, List, Optional
from transformer_lens import HookedTransformer

class ActivationPatcher:
    """
    Interface for causal interventions via activation patching.
    """
    def __init__(self, model):
        self.model = model

    def patch_head(
        self,
        clean_inputs: dict,
        corrupted_cache: dict,
        layer: int,
        head_index: int,
        target_token_index: int = -1
    ):
        """Patches head output with values from a corrupted run."""
        def patch_hook(value, hook):
            # value: [batch, pos, head, d_model]
            corrupted_value = corrupted_cache[hook.name]
            value[:, target_token_index, head_index, :] = corrupted_value[:, target_token_index, head_index, :]
            return value

        hook_name = f"blocks.{layer}.attn.hook_result"
        
        with self.model.transformer.hooks(fwd_hooks=[(hook_name, patch_hook)]):
            patched_outputs = self.model(**clean_inputs)
        
        return patched_outputs

    def calculate_probability_drop(
        self,
        clean_probs: torch.Tensor,
        patched_probs: torch.Tensor,
        correct_action_index: int
    ) -> float:
        """Calculates impact of patching on target action probability."""
        clean_val = clean_probs[0, -1, correct_action_index].item()
        patched_val = patched_probs[0, -1, correct_action_index].item()
        return clean_val - patched_val
