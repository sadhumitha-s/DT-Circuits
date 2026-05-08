# Causal Interventions: Activation Patching

Activation patching (or Resample Ablation) is a technique used to localize where information is processed in a model by swapping activations between a "clean" run and a "corrupted" run.

## Patching Workflow

1. **Clean Run**: Run the model on a standard input (e.g., a high-reward trajectory).
2. **Corrupted Run**: Run the model on a modified input (e.g., a zero-reward trajectory).
3. **Patch**: Replace a specific activation (head, residual stream, etc.) in the corrupted run with the corresponding activation from the clean run.
4. **Measure**: Observe the change in output (logits). If the output recovers toward the clean run, the patched component is causally significant.

```mermaid
flowchart LR
    subgraph Clean Run
    C1[Input A] --> C2[Layer X] --> C3[Output A]
    end
    
    subgraph Corrupted Run
    D1[Input B] --> D2[Layer X] --> D3[Output B]
    end
    
    C2 -.->|Patch Activation| D2
    D2 --> D4[Output B']
    
    style D4 fill:#f96,stroke:#333,stroke-width:4px
```

## Path Patching

Path patching is a more granular version of activation patching. Instead of patching a whole layer, it patches the information flow between two specific nodes (e.g., from an Attention Head to the Final Logits).

### Example: Goal Token → Action Logit

```mermaid
graph TD
    RTG[Reward-to-Go] --> Head1[Attention Head L0H5]
    State[Current State] --> Head1
    Head1 --> Res[Residual Stream]
    Res --> Logits[Action Logits]
    
    subgraph Path Patching
    Head1 -->|Causal Link| Logits
    end
```
