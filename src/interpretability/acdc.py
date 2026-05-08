import torch
import json
from typing import Dict, List, Callable, Optional, Tuple
from tqdm import tqdm

class ACDCDiscovery:
    """
    Automated Circuit Discovery and Click-through (ACDC).
    Prunes a model to find the minimal sufficient subgraph for a specific behavior.
    """
    def __init__(
        self, 
        model, 
        threshold: float = 0.1,
        metric_fn: Optional[Callable] = None
    ):
        self.model = model
        self.threshold = threshold
        self.metric_fn = metric_fn
        self.current_circuit = {
            "layers": [],
            "heads": [],
            "mlps": []
        }

    def default_metric(self, model_outputs: Tuple, target_action: int) -> float:
        """
        Default metric: Logit of the target action.
        """
        action_preds = model_outputs[0] # [batch, seq, action_dim]
        return action_preds[0, -1, target_action].item()

    def run(
        self, 
        inputs: Dict[str, torch.Tensor], 
        target_action: int
    ) -> Dict:
        """
        Runs the ACDC algorithm to prune heads.
        """
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        
        # Baseline performance
        initial_outputs = self.model(**inputs)
        initial_perf = self.default_metric(initial_outputs, target_action)
        
        active_heads = []
        for l in range(n_layers):
            for h in range(n_heads):
                active_heads.append((l, h))
        
        pruned_heads = []
        
        # Greedy pruning (backward selection)
        pbar = tqdm(active_heads, desc="ACDC Pruning")
        for layer, head in pbar:
            # Try removing this head
            current_pruned = pruned_heads + [(layer, head)]
            
            perf = self._eval_with_pruning(inputs, current_pruned, target_action)
            
            # Retain pruning if performance remains within threshold
            if abs(perf - initial_perf) < self.threshold:
                pruned_heads.append((layer, head))
                pbar.set_postfix({"pruned": len(pruned_heads)})
        
        final_circuit = {
            "active_heads": [h for h in active_heads if h not in pruned_heads],
            "pruned_count": len(pruned_heads),
            "initial_perf": initial_perf,
            "final_perf": self._eval_with_pruning(inputs, pruned_heads, target_action)
        }
        
        self.current_circuit = final_circuit
        return final_circuit

    def _eval_with_pruning(
        self, 
        inputs: Dict[str, torch.Tensor], 
        pruned_heads: List[Tuple[int, int]],
        target_action: int
    ) -> float:
        
        def pruning_hook(value, hook):
            # hook.name format: "blocks.L.attn.hook_result"
            layer_idx = int(hook.name.split(".")[1])
            for p_layer, p_head in pruned_heads:
                if p_layer == layer_idx:
                    value[:, :, p_head, :] = 0.0
            return value

        hook_names = [f"blocks.{l}.attn.hook_result" for l in range(self.model.cfg.n_layers)]
        
        with self.model.transformer.hooks(fwd_hooks=[(name, pruning_hook) for name in hook_names]):
            outputs = self.model(**inputs)
            
        return self.default_metric(outputs, target_action)

    def save_manifest(self, path: str):
        """Saves the circuit manifest to a JSON file."""
        with open(path, 'w') as f:
            # Convert tuples to strings for JSON
            serializable_circuit = self.current_circuit.copy()
            serializable_circuit["active_heads"] = [f"L{l}H{h}" for l, h in serializable_circuit["active_heads"]]
            json.dump(serializable_circuit, f, indent=4)
