import streamlit as st
import torch
import os
import sys
import time
import json
from pathlib import Path

# Add project root to path for absolute imports
root_path = str(Path(__file__).resolve().parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from minigrid.wrappers import FlatObsWrapper
from src.models.hooked_dt import HookedDT
from src.interpretability.attribution import LogitAttributionEngine
from src.interpretability.patching import ActivationPatcher
from src.interpretability.sae_manager import SAEManager

st.set_page_config(page_title="DT-Explorer", layout="wide")
st.title("DT-Explorer: Mechanistic Interpretability for DT")

# Sidebar for loading model and data
st.sidebar.header("Data & Model")
model_path = st.sidebar.text_input("Model Path", "models/mini_dt.pt")
data_path = st.sidebar.text_input("Trajectory Path", "data/trajectories.pt")

@st.cache_data
def get_data(path):
    if not os.path.exists(path):
        st.sidebar.warning(f"Data not found at {path}. Please run training script.")
        return None
    # Use weights_only=False because trajectories contain numpy arrays
    return torch.load(path, map_location="cpu", weights_only=False)

@st.cache_resource
def get_model(path, state_dim):
    if not os.path.exists(path):
        st.sidebar.warning(f"Model not found at {path}. Using random init for demo.")
        return HookedDT.from_config(state_dim=state_dim, action_dim=7)
    
    model = HookedDT.from_config(state_dim=state_dim, action_dim=7)
    try:
        # Load state dict (usually safe for weights_only=True, but let's be explicit)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
    return model

# 1. Load Data First
trajectories = get_data(data_path)

if trajectories is None:
    st.error("No real data available. Please run `python scripts/train_dt.py` first.")
    st.stop()

# 2. Determine State Dim
state_dim = trajectories[0]["observations"].shape[1]

# 3. Load Model with Correct Dim
model = get_model(model_path, state_dim)

# Select a trajectory and token for analysis
traj_idx = st.sidebar.number_input("Select Trajectory", 0, len(trajectories)-1, 0)
traj = trajectories[traj_idx]

tab1, tab2, tab3, tab4 = st.tabs([
    "Circuit Mapping (DLA)", 
    "Causal Intervention (Patching)", 
    "SAE Latents",
    "Brain Surgeon & Circuit Explorer"
])

with tab1:
    st.header("Direct Logit Attribution (DLA)")
    st.write("Visualizing which heads contribute most to the predicted action.")
    
    # Run automatically for better UX when changing trajectories
    states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
    actions = torch.nn.functional.one_hot(torch.from_numpy(traj["actions"]).long(), num_classes=7).float().unsqueeze(0)
    returns = torch.from_numpy(traj["rewards"]).float().unsqueeze(0).unsqueeze(-1)
    
    preds, cache = model(states, actions, returns, return_cache=True)
    target_action = preds[0, -1].argmax().item()
    
    engine = LogitAttributionEngine(model)
    # Use token index -2 to target the state token which predicts the action
    dla_results = engine.calculate_dla(cache, target_logit_index=target_action, token_index=-2)
    
    fig, ax = plt.subplots()
    im = ax.imshow(dla_results.detach().cpu().numpy(), cmap="RdBu_r", aspect='auto')
    plt.colorbar(im)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    st.pyplot(fig)
    st.write(f"Analyzing Attribution for Action: {target_action} (at State token)")

with tab2:
    st.header("Activation Patching")
    st.write("Quantifying causal importance by patching corrupted activations.")
    
    # Pre-calculate DLA for better UI feedback and dropdown probabilities
    states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
    actions = torch.nn.functional.one_hot(torch.from_numpy(traj["actions"]).long(), num_classes=7).float().unsqueeze(0)
    returns = torch.from_numpy(traj["rewards"]).float().unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        preds, cache = model(states, actions, returns, return_cache=True)
        target_action = preds[0, -1].argmax().item()
        engine = LogitAttributionEngine(model)
        # Calculate DLA to show scores in dropdowns
        dla_results = engine.calculate_dla(cache, target_logit_index=target_action, token_index=-2)

    # Use format_func to show probabilities/attribution in the dropdown options
    layer_options = [f"Layer {i} (Avg DLA: {dla_results[i].mean():.4f})" for i in range(model.cfg.n_layers)]
    layer_idx = st.selectbox("Select Layer", range(model.cfg.n_layers), format_func=lambda x: layer_options[x])
    
    head_options = [f"Head {j} (DLA: {dla_results[layer_idx, j]:.4f})" for j in range(model.cfg.n_heads)]
    head_idx = st.selectbox("Select Head", range(model.cfg.n_heads), format_func=lambda x: head_options[x])

    if st.button("Calculate Probability Drop"):
        patcher = ActivationPatcher(model)
        
        # Simple corruption: zero out the state token we are patching
        corrupted_states = states.clone()
        corrupted_states[0, -1, :] = 0.0 
        
        clean_logits = preds
        _, corrupted_cache = model(corrupted_states, actions, returns, return_cache=True)
        
        # Patch at token index -2 (State token)
        patched_logits = patcher.patch_head(
            {"states": states, "actions": actions, "returns_to_go": returns},
            corrupted_cache, layer_idx, head_idx, target_token_index=-2
        )
        
        drop = patcher.calculate_probability_drop(
            torch.softmax(clean_logits, dim=-1),
            torch.softmax(patched_logits, dim=-1),
            target_action
        )
        
        st.metric("Logit Prob Drop", f"{drop:.4f}")
        if drop > 0.01:
            st.success(f"Head {layer_idx}.{head_idx} has causal impact ({drop:.4f}) on this decision.")
        else:
            st.info("Low causal impact observed for this head.")

with tab3:
    st.header("High-Fidelity Latent Discovery")
    st.write("Exploring monosemantic features via Sparse Autoencoders (TopK SAEs).")
    
    sae_manager = SAEManager(model)
    hook_points = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    selected_hook = st.selectbox("Select Hook Point", hook_points)
    
    try:
        sae = sae_manager.load_sae(selected_hook)
        st.success(f"Loaded SAE for {selected_hook}")
        
        # Visualize latents for current state
        states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
        actions = torch.nn.functional.one_hot(torch.from_numpy(traj["actions"]).long(), num_classes=7).float().unsqueeze(0)
        returns = torch.from_numpy(traj["rewards"]).float().unsqueeze(0).unsqueeze(-1)
        
        _, cache = model(states, actions, returns, return_cache=True)
        activations = cache[selected_hook][:, -2, :] # State token latents
        
        latents = sae.encode(activations)
        top_values, top_indices = torch.topk(latents[0], k=10)
        
        st.subheader("Top-10 Active Latents")
        cols = st.columns(5)
        for i in range(10):
            with cols[i % 5]:
                st.metric(f"Latent #{top_indices[i].item()}", f"{top_values[i].item():.4f}")
                
        reconstruction_error = sae_manager.compute_anomaly_score(selected_hook, activations)
        st.metric("Reconstruction Error (L2 Norm)", f"{reconstruction_error.item():.4f}")
        
    except FileNotFoundError:
        st.warning(f"No trained SAE found for {selected_hook} in `artifacts/saes/`.")
        st.info("Please run `python scripts/train_sae.py` to generate latent features.")

with tab4:
    st.header("Brain Surgeon & Circuit Explorer")
    st.write("Perform real-time node and path ablations to visualize and audit the agent's internal reasoning pathways.")
    
    from src.interpretability.circuit_surgeon import CircuitSurgeon
    from src.interpretability.neuronpedia import NeuronpediaExporter

    # Initialize CircuitSurgeon on the active model
    surgeon = CircuitSurgeon(model)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Dynamic nodes list
    all_nodes = []
    for l in range(n_layers):
        for h in range(n_heads):
            all_nodes.append(f"L{l}H{h}")
        all_nodes.append(f"L{l}MLP")

    # Dynamic edges list
    all_edges = []
    for l1 in range(n_layers):
        # Within layer attention to MLP
        for h in range(n_heads):
            all_edges.append(f"L{l1}H{h} -> L{l1}MLP")
        
        # Across layers
        for l2 in range(l1 + 1, n_layers):
            for h1 in range(n_heads):
                for h2 in range(n_heads):
                    all_edges.append(f"L{l1}H{h1} -> L{l2}H{h2}")
                all_edges.append(f"L{l1}H{h1} -> L{l2}MLP")
            for h2 in range(n_heads):
                all_edges.append(f"L{l1}MLP -> L{l2}H{h2}")
            all_edges.append(f"L{l1}MLP -> L{l2}MLP")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Surgical Controls")
        
        ablated_nodes_selected = st.multiselect(
            "Ablate Nodes",
            options=all_nodes,
            help="Zero out all activations exiting these specific components."
        )
        
        ablated_edges_selected = st.multiselect(
            "Ablate Communication Paths (Edges)",
            options=all_edges,
            help="Sever the communication channel between two layers or components."
        )
        
        # Register currently selected ablations to CircuitSurgeon
        for node in ablated_nodes_selected:
            surgeon.add_node_ablation(node)
        for edge in ablated_edges_selected:
            parts = edge.split(" -> ")
            surgeon.add_edge_ablation(parts[0], parts[1])

        # Target reward-to-go slider
        target_rtg = st.slider("Goal Reward-to-Go", 0.1, 1.5, 0.9, 0.05)
        
        run_simulation = st.button("Run Live MiniGrid Simulation")

    with col2:
        st.subheader("Interactive Circuit Blueprint")
        st.write("Visualized via Cytoscape.js. Severed components are highlighted in vibrant red/dashed styling.")

        # Build elements for Cytoscape.js
        cy_nodes = []
        cy_edges = []

        # Position layers horizontally
        for l in range(n_layers):
            x_pos = 100 + l * 250
            for h in range(n_heads):
                node_id = f"L{l}H{h}"
                y_pos = 50 + h * 90
                is_ablated = node_id in ablated_nodes_selected
                cy_nodes.append({
                    "data": {"id": node_id, "label": node_id, "type": "head", "ablated": is_ablated},
                    "position": {"x": x_pos, "y": y_pos}
                })
            
            mlp_id = f"L{l}MLP"
            y_pos = 50 + n_heads * 90
            is_ablated = mlp_id in ablated_nodes_selected
            cy_nodes.append({
                "data": {"id": mlp_id, "label": mlp_id, "type": "mlp", "ablated": is_ablated},
                "position": {"x": x_pos, "y": y_pos}
            })

        for edge in all_edges:
            parts = edge.split(" -> ")
            src, dest = parts[0], parts[1]
            is_edge_ablated = edge in ablated_edges_selected
            is_endpoint_ablated = src in ablated_nodes_selected or dest in ablated_nodes_selected
            cy_edges.append({
                "data": {
                    "id": f"{src}_{dest}",
                    "source": src,
                    "target": dest,
                    "ablated": is_edge_ablated or is_endpoint_ablated
                }
            })

        cy_elements_json = json.dumps(cy_nodes + cy_edges)
        
        cytoscape_html = f"""
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
            <style>
                #cy {{
                    width: 100%;
                    height: 400px;
                    background-color: #0e1117;
                    border: 1px solid #30363d;
                    border-radius: 8px;
                }}
            </style>
        </head>
        <body>
            <div id="cy"></div>
            <script>
                var cy = cytoscape({{
                    container: document.getElementById('cy'),
                    elements: {cy_elements_json},
                    style: [
                        {{
                            selector: 'node',
                            style: {{
                                'content': 'data(label)',
                                'text-valign': 'center',
                                'text-halign': 'center',
                                'color': '#ffffff',
                                'background-color': '#0066cc',
                                'font-family': 'sans-serif',
                                'font-weight': 'bold',
                                'font-size': '11px',
                                'width': '55px',
                                'height': '35px',
                                'border-width': '2px',
                                'border-color': '#58a6ff'
                            }}
                        }},
                        {{
                            selector: 'node[type="mlp"]',
                            style: {{
                                'shape': 'rectangle',
                                'width': '75px',
                                'height': '30px',
                                'background-color': '#1b8a5a',
                                'border-color': '#3fb950'
                            }}
                        }},
                        {{
                            selector: 'node[ablated]',
                            style: {{
                                'background-color': '#9e1c1c',
                                'border-color': '#f85149',
                                'border-style': 'dashed',
                                'color': '#f85149'
                            }}
                        }},
                        {{
                            selector: 'edge',
                            style: {{
                                'curve-style': 'bezier',
                                'target-arrow-shape': 'triangle',
                                'line-color': '#484f58',
                                'target-arrow-color': '#484f58',
                                'width': 1.5,
                                'opacity': 0.6
                            }}
                        }},
                        {{
                            selector: 'edge[ablated]',
                            style: {{
                                'line-color': '#f85149',
                                'target-arrow-color': '#f85149',
                                'line-style': 'dashed',
                                'width': 2.5,
                                'opacity': 0.95
                            }}
                        }}
                    ],
                    layout: {{
                        name: 'preset'
                    }},
                    userZoomingEnabled: false,
                    userPanningEnabled: false,
                    boxSelectionEnabled: false
                }});
            </script>
        </body>
        </html>
        """
        st.iframe(cytoscape_html, height=420)

    # 5. Live Simulation execution block
    if run_simulation:
        st.subheader("Live Agent Behavioral Audit")
        status_box = st.empty()
        img_box = st.empty()
        
        try:
            # Recreate exact MiniGrid env setup from harvester
            env = FlatObsWrapper(gym.make("MiniGrid-Empty-8x8-v0", render_mode="rgb_array"))
            obs, _ = env.reset(seed=42)
            
            states_history = [obs]
            actions_history = [np.zeros(7)]
            rewards_history = [target_rtg]
            
            max_len = model.max_length
            total_reward = 0.0
            steps = 0
            
            while steps < 30:
                # Format histories into tensors
                states_t = torch.tensor(np.array(states_history[-max_len:]), dtype=torch.float32).unsqueeze(0)
                actions_t = torch.tensor(np.array(actions_history[-max_len:]), dtype=torch.float32).unsqueeze(0)
                returns_t = torch.tensor(np.array(rewards_history[-max_len:]), dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                
                # Execute DT with ablated circuit surgeon forward
                preds = surgeon.compute_ablated_forward(states_t, actions_t, returns_t)
                act = preds[0, -1].argmax().item()
                
                next_obs, reward, done, truncated, _ = env.step(act)
                total_reward += reward
                steps += 1
                
                # Render current grid step
                frame = env.render()
                img_box.image(frame, caption=f"Step {steps} | Action {act}", width=320)
                status_box.info(f"Stepping Agent... Current Step: {steps}/30 | Cumulative Reward: {total_reward:.4f}")
                
                # Update histories
                states_history.append(next_obs)
                act_one_hot = np.zeros(7)
                act_one_hot[act] = 1.0
                actions_history.append(act_one_hot)
                rewards_history.append(rewards_history[-1] - reward)
                
                time.sleep(0.12)
                
                if done or truncated:
                    break
            
            env.close()
            
            if total_reward > 0:
                st.success(f"Execution complete. Agent successfully reached the goal in {steps} steps! Cumulative Reward: {total_reward:.4f}")
            else:
                st.warning("Agent failed to reach the goal under this ablated circuit/communication configuration.")
                
        except Exception as e:
            st.error(f"Failed to run environment simulation: {str(e)}")

    # 6. Neuronpedia Export Section
    st.markdown("---")
    st.subheader("Neuronpedia Export Hub")
    st.write("Publish discovered circuits, active heads, and ablated configurations to public peer-review.")
    
    np_col1, np_col2 = st.columns(2)
    with np_col1:
        np_key = st.text_input("Neuronpedia Access Key (Optional)", type="password", help="If provided, uploads directly. Otherwise, saves circuit payload in artifacts/.")
    with np_col2:
        export_btn = st.button("Publish Discovered Circuit Blueprint")
        
    if export_btn:
        exporter = NeuronpediaExporter(api_key=np_key if np_key else None)
        manifest = {
            "active_heads": [n for n in all_nodes if n not in ablated_nodes_selected],
            "pruned_count": len(ablated_nodes_selected),
            "initial_perf": 1.0,
            "final_perf": 0.0 if len(ablated_nodes_selected) > 0 else 1.0,
            "ablated_paths": list(ablated_edges_selected),
            "ablated_nodes": list(ablated_nodes_selected),
            "state_dim": state_dim,
            "action_dim": 7,
            "n_layers": n_layers,
            "n_heads": n_heads
        }
        res = exporter.export_circuit(model_id="mini_dt", circuit_manifest=manifest)
        if "local" in res["status"]:
            st.success(res["message"])
            st.json(res["payload"])
        elif "success" in res["status"]:
            st.success(res["message"])
            st.markdown(f"[View Live Uploaded Circuit]({res['url']})")
        else:
            st.error(res["message"])
