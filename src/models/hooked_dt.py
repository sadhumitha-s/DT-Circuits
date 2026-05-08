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

        # HookedTransformer for the core transformer blocks
        self.transformer = HookedTransformer(cfg)

        # Custom embeddings for DT
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

        # Embed tokens
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        
        # In DT, we interleave (R, S, A)
        # Sequence: (R1, S1, A1, R2, S2, A2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=2
        ).reshape(batch_size, 3 * seq_len, self.cfg.d_model)
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Add positional embeddings manually or via HookedTransformer
        # DT usually uses learned positional embeddings for timesteps
        # HookedTransformer usually handles this via its own embed_pos
        # We'll use the timestep info to get positional embeddings
        
        # For simplicity, let's assume we can use HookedTransformer's forward
        # but we need to handle the interleaved nature.
        
        # We pass the stacked_inputs directly to the transformer blocks
        # We use run_with_cache or standard forward based on whether we need the cache
        # For TransformerLens, we need to specify that we are passing embeddings
        
        # Note: HookedTransformer expects [batch, pos, d_model] if input is embeddings
        # We need to set use_local_embeddings=True or similar if we want to bypass default embeds
        
        # A better way is to use model.blocks directly or use the hook_embed to inject
        
        def embed_hook(value, hook):
            return stacked_inputs

        # We inject our interleaved embeddings into the 'hook_embed'
        # and pass a dummy tensor of the right shape to the transformer
        dummy_input = torch.zeros((batch_size, 3 * seq_len), dtype=torch.long, device=stacked_inputs.device)
        
        # We want the residual stream after the last block
        # HookedTransformer.run_with_cache returns (output, cache)
        # We can also use return_type="residual" or similar in some versions, 
        # but let's just use the cache or the direct output if we set it up correctly.
        
        # In TransformerLens, the output of the forward pass is usually the logits.
        # We want the 'hook_resid_post' of the last block.
        
        last_block_hook = f"blocks.{self.cfg.n_layers - 1}.hook_resid_post"
        
        with self.transformer.hooks(fwd_hooks=[("hook_embed", embed_hook)]):
            _, cache = self.transformer.run_with_cache(
                dummy_input,
                names_filter=lambda name: name == last_block_hook
            )
        
        transformer_outputs = cache[last_block_hook]

        # Reshape back to (batch, seq, 3, d_model)
        x = transformer_outputs.reshape(batch_size, seq_len, 3, self.cfg.d_model)

        # Predict (A from S, S from A, R from S?)
        # Standard DT: Action is predicted from State token
        action_preds = self.predict_action(x[:, :, 1]) # predict next action from state
        return_preds = self.predict_return(x[:, :, 2]) # predict next return from action
        state_preds = self.predict_state(x[:, :, 2])   # predict next state from action

        return action_preds, state_preds, return_preds

    @classmethod
    def from_config(cls, state_dim, action_dim, n_layers=2, n_heads=4, d_model=128):
        cfg = HookedTransformerConfig(
            n_layers=n_layers,
            d_model=d_model,
            n_ctx=300, # Max sequence length * 3
            d_head=d_model // n_heads,
            n_heads=n_heads,
            d_vocab=10, # Dummy value, we use custom embeddings
            act_fn="relu", # DT original uses ReLU or GeLU
            d_mlp=d_model * 4,
            normalization_type="LN",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return cls(cfg, state_dim, action_dim)
