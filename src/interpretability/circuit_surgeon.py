import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Union, Optional

class CircuitSurgeon:
    """
    Manages interactive node and path ablations on the Decision Transformer.
    Enables zeroing out specific components (Heads, MLPs) or cutting
    specific communication paths between components.
    """
    def __init__(self, model):
        self.model = model
        self.ablated_nodes: Set[str] = set()  # format: "L<layer>H<head>" or "L<layer>MLP"
        self.ablated_edges: Set[Tuple[str, str]] = set()  # format: ("L<l1>H<h1>", "L<l2>H<h2>")

    def add_node_ablation(self, node: str):
        """Adds a component node (e.g., 'L0H1' or 'L1MLP') to the ablated set."""
        self.ablated_nodes.add(node)

    def remove_node_ablation(self, node: str):
        """Removes a component node from the ablated set."""
        self.ablated_nodes.discard(node)

    def add_edge_ablation(self, from_node: str, to_node: str):
        """Adds a communication path (edge) to the ablated set."""
        self.ablated_edges.add((from_node, to_node))

    def remove_edge_ablation(self, from_node: str, to_node: str):
        """Removes a communication path from the ablated set."""
        self.ablated_edges.discard((from_node, to_node))

    def clear_ablations(self):
        """Clears all registered ablations."""
        self.ablated_nodes.clear()
        self.ablated_edges.clear()

    def parse_node(self, node: str) -> Tuple[int, Optional[int]]:
        """Parses a node name into layer and head index (None for MLP)."""
        if "MLP" in node:
            layer = int(node.replace("MLP", "").replace("L", ""))
            return layer, None
        else:
            parts = node.split("H")
            layer = int(parts[0].replace("L", ""))
            head = int(parts[1])
            return layer, head

    def get_ablation_hooks(self, baseline_cache: Dict[str, torch.Tensor]) -> List[Tuple[str, callable]]:
        """
        Generates PyTorch forward hook functions for registered node and edge ablations.
        """
        hooks = []

        # 1. Node Ablations
        # Group by layer for efficiency
        attn_nodes_by_layer = {}
        mlp_layers = set()

        for node in self.ablated_nodes:
            layer, head = self.parse_node(node)
            if head is None:
                mlp_layers.add(layer)
            else:
                if layer not in attn_nodes_by_layer:
                    attn_nodes_by_layer[layer] = []
                attn_nodes_by_layer[layer].append(head)

        # Attention Node Hooks
        for layer, heads in attn_nodes_by_layer.items():
            def make_attn_hook(l, hs):
                def attn_hook(value, hook):
                    for h in hs:
                        value[:, :, h, :] = 0.0
                    return value
                return attn_hook

            hook_name = f"blocks.{layer}.attn.hook_result"
            hooks.append((hook_name, make_attn_hook(layer, heads)))

        # MLP Node Hooks
        for layer in mlp_layers:
            def make_mlp_hook(l):
                def mlp_hook(value, hook):
                    value[:, :, :] = 0.0
                    return value
                return mlp_hook

            hook_name = f"blocks.{layer}.hook_mlp_out"
            hooks.append((hook_name, make_mlp_hook(layer)))

        # 2. Path/Edge Ablations
        # Group edges by their destination node to avoid redundant hooks
        edges_by_dest = {}
        for from_node, to_node in self.ablated_edges:
            # Skip if either endpoint is already node-ablated (subsumed by node ablation)
            if from_node in self.ablated_nodes or to_node in self.ablated_nodes:
                continue
            if to_node not in edges_by_dest:
                edges_by_dest[to_node] = []
            edges_by_dest[to_node].append(from_node)

        for to_node, from_nodes in edges_by_dest.items():
            to_layer, to_head = self.parse_node(to_node)

            if to_head is not None:
                # Target is an attention head (L2H2)
                # Hook Q, K, V activations of that layer
                def make_path_attn_hooks(tl, th, fns):
                    def q_hook(value, hook):
                        for fn in fns:
                            fl, fh = self.parse_node(fn)
                            if fh is None:
                                # Source is MLP (L1MLP)
                                src_key = f"blocks.{fl}.hook_mlp_out"
                            else:
                                # Source is Head (L1H1)
                                src_key = f"blocks.{fl}.attn.hook_result"

                            if src_key in baseline_cache:
                                src_out = baseline_cache[src_key]
                                if fh is not None:
                                    src_out = src_out[:, :, fh, :]
                                
                                # Apply downstream layer's first layernorm
                                ln_out = self.model.transformer.blocks[tl].ln1(src_out)
                                # Project to Query
                                W_Q = self.model.transformer.blocks[tl].attn.W_Q[th]
                                q_contrib = ln_out @ W_Q
                                value[:, :, th, :] -= q_contrib
                        return value

                    def k_hook(value, hook):
                        for fn in fns:
                            fl, fh = self.parse_node(fn)
                            if fh is None:
                                src_key = f"blocks.{fl}.hook_mlp_out"
                            else:
                                src_key = f"blocks.{fl}.attn.hook_result"

                            if src_key in baseline_cache:
                                src_out = baseline_cache[src_key]
                                if fh is not None:
                                    src_out = src_out[:, :, fh, :]
                                
                                ln_out = self.model.transformer.blocks[tl].ln1(src_out)
                                W_K = self.model.transformer.blocks[tl].attn.W_K[th]
                                k_contrib = ln_out @ W_K
                                value[:, :, th, :] -= k_contrib
                        return value

                    def v_hook(value, hook):
                        for fn in fns:
                            fl, fh = self.parse_node(fn)
                            if fh is None:
                                src_key = f"blocks.{fl}.hook_mlp_out"
                            else:
                                src_key = f"blocks.{fl}.attn.hook_result"

                            if src_key in baseline_cache:
                                src_out = baseline_cache[src_key]
                                if fh is not None:
                                    src_out = src_out[:, :, fh, :]
                                
                                ln_out = self.model.transformer.blocks[tl].ln1(src_out)
                                W_V = self.model.transformer.blocks[tl].attn.W_V[th]
                                v_contrib = ln_out @ W_V
                                value[:, :, th, :] -= v_contrib
                        return value

                    return q_hook, k_hook, v_hook

                    
                qh, kh, vh = make_path_attn_hooks(to_layer, to_head, from_nodes)
                hooks.append((f"blocks.{to_layer}.attn.hook_q", qh))
                hooks.append((f"blocks.{to_layer}.attn.hook_k", kh))
                hooks.append((f"blocks.{to_layer}.attn.hook_v", vh))

            else:
                # Target is an MLP layer (L2MLP)
                # Hook the input to the MLP
                def make_path_mlp_hook(tl, fns):
                    def mlp_in_hook(value, hook):
                        for fn in fns:
                            fl, fh = self.parse_node(fn)
                            if fh is None:
                                src_key = f"blocks.{fl}.hook_mlp_out"
                            else:
                                src_key = f"blocks.{fl}.attn.hook_result"

                            if src_key in baseline_cache:
                                src_out = baseline_cache[src_key]
                                if fh is not None:
                                    src_out = src_out[:, :, fh, :]
                                
                                # Apply downstream layer's second layernorm
                                ln_out = self.model.transformer.blocks[tl].ln2(src_out)
                                value -= ln_out
                        return value
                    return mlp_in_hook

                hooks.append((f"blocks.{to_layer}.hook_mlp_in", make_path_mlp_hook(to_layer, from_nodes)))

        return hooks

    def compute_ablated_forward(
        self, 
        states: torch.Tensor, 
        actions: torch.Tensor, 
        returns_to_go: torch.Tensor,
        return_cache: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Executes a baseline pass to cache source activations, compiles the
        necessary ablation hooks, and performs the ablated forward pass.
        """
        # Step 1: Run baseline pass to populate cache
        _, baseline_cache = self.model(states, actions, returns_to_go, return_cache=True)

        # Step 2: Compile hooks
        hooks = self.get_ablation_hooks(baseline_cache)

        # Step 3: Run ablated forward pass
        if len(hooks) == 0:
            if return_cache:
                return self.model(states, actions, returns_to_go, return_cache=True)
            return self.model(states, actions, returns_to_go)

        # Register hooks using the model's transformer context manager
        if return_cache:
            with self.model.transformer.hooks(fwd_hooks=hooks):
                preds, cache = self.model(states, actions, returns_to_go, return_cache=True)
            return preds, cache
        else:
            with self.model.transformer.hooks(fwd_hooks=hooks):
                preds = self.model(states, actions, returns_to_go)
            return preds
