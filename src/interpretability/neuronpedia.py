import os
import json
import requests
from typing import Dict, List, Any, Optional

class NeuronpediaExporter:
    """
    Handles exporting discovered circuits, SAE feature activations, and natural language
    labels to the Neuronpedia platform for public sharing and peer review.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://www.neuronpedia.org/api"

    def format_circuit_payload(self, model_id: str, circuit_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """
        Formats a circuit manifest (active heads, ablated paths, performance scores)
        into a standardized Neuronpedia-compatible schema.
        """
        payload = {
            "model_id": model_id,
            "source": "DT-Circuits",
            "schema_version": "1.0.0",
            "circuit": {
                "active_heads": circuit_manifest.get("active_heads", []),
                "pruned_count": circuit_manifest.get("pruned_count", 0),
                "initial_performance": circuit_manifest.get("initial_perf", 0.0),
                "final_performance": circuit_manifest.get("final_perf", 0.0),
                "ablated_paths": list(circuit_manifest.get("ablated_paths", [])),
                "ablated_nodes": list(circuit_manifest.get("ablated_nodes", []))
            },
            "metadata": {
                "state_dim": circuit_manifest.get("state_dim"),
                "action_dim": circuit_manifest.get("action_dim"),
                "n_layers": circuit_manifest.get("n_layers"),
                "n_heads": circuit_manifest.get("n_heads"),
            }
        }
        return payload

    def export_circuit(
        self, 
        model_id: str, 
        circuit_manifest: Dict[str, Any], 
        local_path: str = "artifacts/neuronpedia_export.json"
    ) -> Dict[str, Any]:
        """
        Exports the discovered circuit. If an API key is present, uploads it
        directly to Neuronpedia; otherwise, serializes it to a local JSON file.
        """
        payload = self.format_circuit_payload(model_id, circuit_manifest)
        
        # Ensure target directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Save locally as backup or primary artifact
        with open(local_path, "w") as f:
            json.dump(payload, f, indent=4)

        if not self.api_key:
            return {
                "status": "success_local",
                "message": f"Circuit exported locally to {local_path}. Provide a Neuronpedia API Key for online sharing.",
                "path": local_path,
                "payload": payload
            }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/circuits/upload", 
                json=payload, 
                headers=headers,
                timeout=10
            )
            if response.status_code == 200 or response.status_code == 201:
                return {
                    "status": "success_api",
                    "message": "Circuit successfully uploaded to Neuronpedia!",
                    "url": response.json().get("url", "https://www.neuronpedia.org"),
                    "payload": payload
                }
            else:
                return {
                    "status": "error_api",
                    "message": f"Neuronpedia API rejected upload: {response.status_code} - {response.text}",
                    "path": local_path,
                    "payload": payload
                }
        except Exception as e:
            return {
                "status": "error_exception",
                "message": f"Network exception during upload: {str(e)}. Saved to {local_path}.",
                "path": local_path,
                "payload": payload
            }
