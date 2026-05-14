import torch
import torch.nn as nn
from transformer_lens import HookedTransformer, HookedTransformerConfig
from jaxtyping import Float, Int
from typing import Optional, Union, List

class HookedDT(nn.Module):
    """
    Decision Transformer wrapped in TransformerLens logic.
    Supports State, Action, and Reward-to-Go (RTG) tokens.
    """
    def __init__(
        self,
        cfg: HookedTransformerConfig,
        state_dim: int,
        action_dim: int,
        max_length: int = 30,
    ):
        super().__init__()
        self.cfg = cfg
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_length = max_length

        # Core transformer blocks from TransformerLens
        self.transformer = HookedTransformer(cfg)

        # DT-specific embeddings
        self.embed_return = nn.Linear(1, cfg.d_model)
        self.embed_state = nn.Linear(state_dim, cfg.d_model)
        self.embed_action = nn.Linear(action_dim, cfg.d_model)
        self.embed_ln = nn.LayerNorm(cfg.d_model)

        # Prediction heads
        self.predict_action = nn.Sequential(nn.Linear(cfg.d_model, action_dim))
        self.predict_return = nn.Sequential(nn.Linear(cfg.d_model, 1))
        self.predict_state = nn.Sequential(nn.Linear(cfg.d_model, state_dim))

    def get_embeddings(self, states, actions, returns_to_go):
        """Interleaves RTG, State, and Action embeddings."""
        batch_size, seq_len, _ = states.shape
        
        ret_emb = self.embed_return(returns_to_go)
        state_emb = self.embed_state(states)
        act_emb = self.embed_action(actions)
        
        # Interleave: [R1, S1, A1, R2, S2, A2, ...]
        stacked = torch.stack((ret_emb, state_emb, act_emb), dim=2)
        stacked = stacked.reshape(batch_size, 3 * seq_len, self.cfg.d_model)
        return self.embed_ln(stacked)

    def forward(self, states, actions, returns_to_go, timesteps=None, return_cache=False):
        """Forward pass through DT."""
        embeddings = self.get_embeddings(states, actions, returns_to_go)
        dummy_tokens = torch.zeros((embeddings.shape[0], embeddings.shape[1]), 
                                 dtype=torch.long, device=embeddings.device)
        
        def inject_embeddings(value, hook):
            return embeddings

        # We need the residual stream post-processing from the last block
        last_resid_hook = f"blocks.{self.cfg.n_layers-1}.hook_resid_post"
        
        if return_cache:
            with self.transformer.hooks(fwd_hooks=[("hook_embed", inject_embeddings)]):
                _, cache = self.transformer.run_with_cache(dummy_tokens)
            
            last_resid = cache[last_resid_hook]
            x = last_resid.reshape(states.shape[0], states.shape[1], 3, self.cfg.d_model)
            action_preds = self.predict_action(x[:, :, 1]) # State token predicts action
            return action_preds, cache
        else:
            with self.transformer.hooks(fwd_hooks=[("hook_embed", inject_embeddings)]):
                # run_with_cache is safer to ensure we can grab the specific hook output
                _, cache = self.transformer.run_with_cache(dummy_tokens, names_filter=lambda n: n == last_resid_hook)
            
            last_resid = cache[last_resid_hook]
            x = last_resid.reshape(states.shape[0], states.shape[1], 3, self.cfg.d_model)
            action_preds = self.predict_action(x[:, :, 1])
            return action_preds

    @classmethod
    def from_config(cls, state_dim, action_dim, n_layers=2, n_heads=4, d_model=128):
        cfg = HookedTransformerConfig(
            n_layers=n_layers,
            d_model=d_model,
            n_ctx=300, 
            d_head=d_model // n_heads,
            n_heads=n_heads,
            d_vocab=10, # Dummy vocab size
            act_fn="relu", 
            d_mlp=d_model * 4,
            normalization_type="LN",
            use_attn_result=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return cls(cfg, state_dim, action_dim)

