import torch
from jaxtyping import Float
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns

class LogitAttributionEngine:
    """
    Calculates the Direct Logit Attribution (DLA) of transformer components.
    """
    def __init__(self, model):
        self.model = model

    def calculate_dla(
        self, 
        cache, 
        target_logit_index: int,
        token_index: int = -1
    ) -> Dict[str, Float[torch.Tensor, "layer head"]]:
        """
        Computes DLA for each head: Activation @ W_O @ W_U [target_logit]
        """
        n_layers = self.model.cfg.n_layers
        n_heads = self.model.cfg.n_heads
        
        # Action prediction unembedding
        W_U = self.model.predict_action[0].weight[target_logit_index] 

        dla_results = torch.zeros((n_layers, n_heads))

        for layer in range(n_layers):
            # [batch, pos, head, d_model]
            head_outputs = cache[f"blocks.{layer}.attn.hook_result"] 
            
            # S_t is at 3t + 1 in interleaved (R, S, A)
            last_token_output = head_outputs[0, token_index] 
            
            dla_results[layer] = torch.matmul(last_token_output, W_U)

        return dla_results

    def plot_dla(self, dla_results: torch.Tensor, title="Direct Logit Attribution"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(dla_results.detach().cpu().numpy(), annot=True, fmt=".2f", cmap="RdBu_r", center=0)
        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.title(title)
        plt.show()
