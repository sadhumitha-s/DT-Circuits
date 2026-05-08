import torch
import torch.nn as nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from jaxtyping import Float, Int
from typing import Optional, Union, List

class HookedDT(nn.Module):
    """
    A Decision Transformer implementation wrapped in TransformerLens logic.
    Supports State, Action, and Reward-to-Go (RTG) tokens.
    """
    def __init__(
        self,
        cfg: HookedTransformerConfig,
        state_dim: int,
        action_dim: int,
        max_length: int = 30,
        max_ep_len: int = 1000,
    ):
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        # TransformerLens core blocks
        self.transformer = HookedTransformer(cfg)

        # DT-specific embeddings
        self.embed_return = nn.Linear(1, cfg.d_model)
        self.embed_state = nn.Linear(state_dim, cfg.d_model)
        self.embed_action = nn.Linear(action_dim, cfg.d_model)

        self.embed_ln = nn.LayerNorm(cfg.d_model)

        # Prediction heads
        self.predict_action = nn.Sequential(
            nn.Linear(cfg.d_model, action_dim)
        )
        self.predict_return = nn.Sequential(
            nn.Linear(cfg.d_model, 1)
        )
        self.predict_state = nn.Sequential(
            nn.Linear(cfg.d_model, state_dim)
        )

    def forward(
        self,
        states: Float[torch.Tensor, "batch seq state_dim"],
        actions: Float[torch.Tensor, "batch seq action_dim"],
        returns_to_go: Float[torch.Tensor, "batch seq 1"],
        timesteps: Int[torch.Tensor, "batch seq"],
        attention_mask: Optional[Float[torch.Tensor, "batch seq"]] = None,
    ):
        batch_size, seq_len, _ = states.shape

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        
        # Interleave (Return, State, Action)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(batch_size, 3 * seq_len, self.cfg.d_model)
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        def embed_hook(value, hook):
            return stacked_inputs

        # Inject interleaved embeddings via hook
        dummy_input = torch.zeros((batch_size, 3 * seq_len), dtype=torch.long, device=stacked_inputs.device)
        
        last_block_hook = f"blocks.{self.cfg.n_layers - 1}.hook_resid_post"
        
        with self.transformer.hooks(fwd_hooks=[("hook_embed", embed_hook)]):
            _, cache = self.transformer.run_with_cache(
                dummy_input,
                names_filter=lambda name: name == last_block_hook
            )
        
        transformer_outputs = cache[last_block_hook]
        x = transformer_outputs.reshape(batch_size, seq_len, 3, self.cfg.d_model)

        # Compute predictions
        action_preds = self.predict_action(x[:, :, 1]) 
        return_preds = self.predict_return(x[:, :, 2]) 
        state_preds = self.predict_state(x[:, :, 2])   

        return action_preds, state_preds, return_preds

    @classmethod
    def from_config(cls, state_dim, action_dim, n_layers=2, n_heads=4, d_model=128):
        cfg = HookedTransformerConfig(
            n_layers=n_layers,
            d_model=d_model,
            n_ctx=300, 
            d_head=d_model // n_heads,
            n_heads=n_heads,
            d_vocab=10, 
            act_fn="relu", 
            d_mlp=d_model * 4,
            normalization_type="LN",
            use_attn_result=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return cls(cfg, state_dim, action_dim)
