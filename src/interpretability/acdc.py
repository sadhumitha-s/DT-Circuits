import torch
import json
from typing import Dict, List, Callable, Optional, Tuple
from tqdm import tqdm

class ACDCDiscovery:
    """
    Automated Circuit Discovery and Click-through (ACDC).
    Finds the minimal set of heads needed to maintain model performance.
    """
    def __init__(self, model, threshold: float = 0.1):
        self.model = model
        self.threshold = threshold
        self.current_circuit = {}

    def get_metric(self, action_preds: torch.Tensor, target_action: int) -> float:
        """Calculates logit of the target action at the last timestep."""
        return action_preds[0, -1, target_action].item()

    def run(self, inputs: dict, target_action: int) -> dict:
        """Greedily prunes heads while keeping performance above threshold."""
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        
        # Get baseline performance
        initial_preds = self.model(**inputs)
        initial_perf = self.get_metric(initial_preds, target_action)
        
        all_heads = [(l, h) for l in range(n_layers) for h in range(n_heads)]
        pruned_heads = []
        
        pbar = tqdm(all_heads, desc="ACDC Pruning")
        for layer, head in pbar:
            # Try pruning this head + already pruned heads
            trial_pruned = pruned_heads + [(layer, head)]
            perf = self._eval_with_pruning(inputs, trial_pruned, target_action)
            
            # If performance is still good, keep it pruned
            if abs(perf - initial_perf) < self.threshold:
                pruned_heads.append((layer, head))
                pbar.set_postfix({"pruned": len(pruned_heads)})
        
        active_heads = [h for h in all_heads if h not in pruned_heads]
        self.current_circuit = {
            "active_heads": active_heads,
            "pruned_count": len(pruned_heads),
            "initial_perf": initial_perf,
            "final_perf": self._eval_with_pruning(inputs, pruned_heads, target_action)
        }
        return self.current_circuit

    def _eval_with_pruning(self, inputs: dict, pruned_heads: list, target_action: int) -> float:
        """Evaluates model with specified heads zeroed out."""
        def pruning_hook(value, hook):
            layer_idx = int(hook.name.split(".")[1])
            for p_layer, p_head in pruned_heads:
                if p_layer == layer_idx:
                    value[:, :, p_head, :] = 0.0
            return value

        hooks = [(f"blocks.{l}.attn.hook_result", pruning_hook) for l in range(self.model.cfg.n_layers)]
        
        with self.model.transformer.hooks(fwd_hooks=hooks):
            preds = self.model(**inputs)
            
        return self.get_metric(preds, target_action)

    def save_manifest(self, path: str):
        """Saves discovered circuit to a JSON file."""
        with open(path, 'w') as f:
            data = self.current_circuit.copy()
            data["active_heads"] = [f"L{l}H{h}" for l, h in data["active_heads"]]
            json.dump(data, f, indent=4)
