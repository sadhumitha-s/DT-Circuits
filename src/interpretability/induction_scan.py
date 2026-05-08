import torch
from typing import List, Tuple

class InductionScanner:
    """
    Identifies induction heads that attend to tokens following a previous occurrence.
    """
    def __init__(self, model):
        self.model = model

    def scan(self, cache, sequence: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Scans heads for induction behavior.
        """
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        
        induction_heads = []

        for layer in range(n_layers):
            # [batch, head, query_pos, key_pos]
            attn_pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
            
            for head in range(n_heads):
                score = self._calculate_induction_score(attn_pattern[0, head])
                if score > 0.5:
                    induction_heads.append((layer, head))
        
        return induction_heads

    def _calculate_induction_score(self, pattern: torch.Tensor) -> float:
        """
        Heuristic check for shifted diagonal attention.
        """
        # Checks if attention is shifted by 1 relative to diagonal.
        return torch.diagonal(pattern, offset=-1).mean().item()
