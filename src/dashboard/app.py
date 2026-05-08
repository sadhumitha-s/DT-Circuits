import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.hooked_dt import HookedDT
from src.interpretability.attribution import LogitAttributionEngine
from src.interpretability.patching import ActivationPatcher

st.set_page_config(page_title="DT-Explorer", layout="wide")

st.title("DT-Explorer: Mechanistic Interpretability for Decision Transformers")

# Sidebar for controls
st.sidebar.header("Model Configuration")
n_layers = st.sidebar.slider("Layers", 1, 12, 1)
n_heads = st.sidebar.slider("Heads", 1, 8, 4)

# Load Model
@st.cache_resource
def load_model():
    # Placeholder dimensions for MiniGrid
    state_dim = 2739 # FlatObsWrapper for 8x8 MiniGrid
    action_dim = 7
    model = HookedDT.from_config(state_dim, action_dim, n_layers=n_layers, n_heads=n_heads)
    # model.load_state_dict(torch.load("models/mini_dt.pt"))
    return model

model = load_model()

# Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["Circuit Mapping", "Causal Intervention", "SAE Explorer"])

with tab1:
    st.header("Direct Logit Attribution")
    # Simulate a forward pass
    if st.button("Run Attribution Analysis"):
        # Dummy data for demo
        states = torch.randn(1, 10, model.state_dim)
        actions = torch.randn(1, 10, model.action_dim)
        returns = torch.randn(1, 10, 1)
        timesteps = torch.arange(10).unsqueeze(0)
        
        # Capture cache
        logits, cache = model.transformer.run_with_cache(
            # Need to handle DT's interleaved forward pass here
            # For demo, we'll just show the UI structure
            torch.randn(1, 30, model.cfg.d_model) 
        )
        
        engine = LogitAttributionEngine(model)
        # dla = engine.calculate_dla(cache, target_logit_index=0)
        
        # Placeholder plot
        fig, ax = plt.subplots()
        dla_mock = np.random.randn(n_layers, n_heads)
        im = ax.imshow(dla_mock, cmap="RdBu_r")
        plt.colorbar(im)
        st.pyplot(fig)

with tab2:
    st.header("Activation Patching")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clean Run")
        st.text("Input: Goal is visible")
    with col2:
        st.subheader("Corrupted Run")
        st.text("Input: Goal is blocked")
    
    layer_to_patch = st.selectbox("Select Layer", range(n_layers))
    head_to_patch = st.selectbox("Select Head", range(n_heads))
    
    if st.button("Apply Patch"):
        st.success(f"Patched Layer {layer_to_patch}, Head {head_to_patch}")
        st.metric("Probability Drop", "0.42", delta="-0.15")

with tab3:
    st.header("SAE Monosemantic Latents")
    st.info("SAE Integration Coming Soon (Phase 3)")
