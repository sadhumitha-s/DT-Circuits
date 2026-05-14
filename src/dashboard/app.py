import streamlit as st
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from src.models.hooked_dt import HookedDT
from src.interpretability.attribution import LogitAttributionEngine
from src.interpretability.patching import ActivationPatcher

st.set_page_config(page_title="DT-Explorer", layout="wide")
st.title("DT-Explorer: Mechanistic Interpretability for DT")

# Sidebar for loading model and data
st.sidebar.header("Data & Model")
model_path = st.sidebar.text_input("Model Path", "models/mini_dt.pt")
data_path = st.sidebar.text_input("Trajectory Path", "data/trajectories.pt")

@st.cache_resource
def get_model(path):
    if not os.path.exists(path):
        st.sidebar.warning(f"Model not found at {path}. Using random init for demo.")
        return HookedDT.from_config(state_dim=2739, action_dim=7)
    
    model = HookedDT.from_config(state_dim=2739, action_dim=7)
    try:
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
    return model

@st.cache_data
def get_data(path):
    if not os.path.exists(path):
        st.sidebar.warning(f"Data not found at {path}. Please run training script.")
        return None
    return torch.load(path)

model = get_model(model_path)
trajectories = get_data(data_path)

if trajectories is None:
    st.error("No real data available. Please run `python scripts/train_dt.py` first.")
    st.stop()

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
        dla_results = engine.calculate_dla(cache, target_logit_index=target_action)
        
        fig, ax = plt.subplots()
        im = ax.imshow(dla_results.detach().cpu().numpy(), cmap="RdBu_r", aspect='auto')
        plt.colorbar(im)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        st.pyplot(fig)
        st.write(f"Analyzing Attribution for Action: {target_action}")

with tab2:
    st.header("Activation Patching")
    st.write("Quantifying causal importance by patching corrupted activations.")
    
    # Simple corruption: zero out the last observation
    corrupted_states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
    corrupted_states[0, -1, :] = 0.0 
    
    states = torch.from_numpy(traj["observations"]).float().unsqueeze(0)
    actions = torch.nn.functional.one_hot(torch.from_numpy(traj["actions"]).long(), num_classes=7).float().unsqueeze(0)
    returns = torch.from_numpy(traj["rewards"]).float().unsqueeze(0).unsqueeze(-1)

    layer = st.selectbox("Layer to Patch", range(model.cfg.n_layers))
    head = st.selectbox("Head to Patch", range(model.cfg.n_heads))

    if st.button("Calculate Probability Drop"):
        patcher = ActivationPatcher(model)
        
        clean_logits = model(states, actions, returns)
        _, corrupted_cache = model(corrupted_states, actions, returns, return_cache=True)
        
        patched_logits = patcher.patch_head(
            {"states": states, "actions": actions, "returns_to_go": returns},
            corrupted_cache, layer, head
        )
        
        target_idx = clean_logits[0, -1].argmax().item()
        drop = patcher.calculate_probability_drop(
            torch.softmax(clean_logits, dim=-1),
            torch.softmax(patched_logits, dim=-1),
            target_idx
        )
        
        st.metric("Logit Prob Drop", f"{drop:.4f}")
        if drop > 0.05:
            st.success(f"Head {layer}.{head} has causal impact on this decision.")
        else:
            st.info("Low causal impact observed for this head.")

with tab3:
    st.header("SAE Feature Exploration")
    st.info("SAE Integration ready for Phase 3. Latents will be mapped to trajectories here.")
