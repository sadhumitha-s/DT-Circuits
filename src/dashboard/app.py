import streamlit as st
import torch
import os
import sys
from pathlib import Path

# Add project root to path for absolute imports
root_path = str(Path(__file__).resolve().parent.parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import numpy as np
import matplotlib.pyplot as plt
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

tab1, tab2, tab3 = st.tabs(["Circuit Mapping (DLA)", "Causal Intervention (Patching)", "SAE Latents"])

with tab1:
    st.header("Direct Logit Attribution (DLA)")
    st.write("Visualizing which heads contribute most to the predicted action.")
    
    if st.button("Run Attribution"):
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
