import torch
import os
from typing import List, Dict
from src.interpretability.acdc import ACDCDiscovery

class EvolutionaryScanner:
    """
    Analyzes how circuits evolve across different training checkpoints.
    """
    def __init__(self, model_class, state_dim: int, action_dim: int):
        self.model_class = model_class
        self.state_dim = state_dim
        self.action_dim = action_dim

    def scan_checkpoints(
        self, 
        checkpoint_dir: str, 
        inputs: Dict[str, torch.Tensor],
        target_action: int,
        threshold: float = 0.1,
        **model_kwargs
    ) -> List[Dict]:
        """
        Runs ACDC on checkpoints and returns the results.
        """
        results = []
        ckpt_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".pt") or f.endswith(".pth")])
        
        for ckpt in ckpt_files:
            ckpt_path = os.path.join(checkpoint_dir, ckpt)
            print(f"Analyzing checkpoint: {ckpt}")
            
            # Load model
            model = self.model_class.from_config(self.state_dim, self.action_dim, **model_kwargs)
            model.load_state_dict(torch.load(ckpt_path, map_location=model.transformer.cfg.device))
            model.eval()
            
            # Run ACDC
            acdc = ACDCDiscovery(model, threshold=threshold)
            circuit = acdc.run(inputs, target_action)
            circuit["checkpoint"] = ckpt
            
            results.append(circuit)
            
        return results

    def detect_phase_transition(self, scan_results: List[Dict]) -> int:
        """
        Identifies the step where a major jump in circuit stability or performance occurred.
        """
        # Identifies checkpoint where performance > 0.5 and circuit stabilizes.
        for i, res in enumerate(scan_results):
            if res["final_perf"] > 0.5 and len(res["active_heads"]) > 0:
                return i
        return -1
