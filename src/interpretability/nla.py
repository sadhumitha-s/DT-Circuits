import torch
from typing import List, Dict, Optional
import requests

class NLAExplainer:
    """
    Natural Language Autoencoder (NLA) Explainer.
    Uses an LLM to auto-label SAE features based on activation patterns.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self.feature_labels: Dict[int, str] = {}

    def generate_label(
        self, 
        feature_id: int, 
        top_activations: List[Dict], 
        context_description: str = "MiniGrid environment agent state"
    ) -> str:
        """
        Generates a natural language label for a specific SAE feature.
        In a real scenario, this would call an LLM API.
        """
        if not self.api_key:
            # Mock labeling for demonstration if no API key is provided
            label = f"Mock Feature {feature_id}: Activates on {context_description} pattern"
            self.feature_labels[feature_id] = label
            return label

        prompt = self._build_prompt(feature_id, top_activations, context_description)
        
        # This is a placeholder for a real API call (e.g., OpenAI, Anthropic, or custom)
        # label = self._call_llm_api(prompt)
        label = f"Auto-labeled Feature {feature_id}" 
        
        self.feature_labels[feature_id] = label
        return label

    def _build_prompt(self, feature_id: int, top_activations: List[Dict], context: str) -> str:
        """Constructs the prompt for the LLM explainer."""
        examples = "\n".join([f"- State: {a['state']}, Activation: {a['value']:.4f}" for a in top_activations])
        return (
            f"I have a Sparse Autoencoder feature (ID: {feature_id}) trained on a Decision Transformer. "
            f"The context is: {context}.\n"
            f"Here are the top activations for this feature:\n{examples}\n"
            "What is the most likely semantic meaning of this feature? Provide a concise label."
        )

    def get_label(self, feature_id: int) -> str:
        return self.feature_labels.get(feature_id, f"Unlabeled Feature {feature_id}")

    def bulk_label(self, feature_ids: List[int], activation_data: Dict[int, List[Dict]]):
        """Labels multiple features in sequence."""
        for fid in feature_ids:
            if fid in activation_data:
                self.generate_label(fid, activation_data[fid])
