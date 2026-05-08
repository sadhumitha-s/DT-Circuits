# Circuit Discovery in Decision Transformers

Circuit discovery is the process of identifying the minimal set of neural components (heads, neurons, paths) that are responsible for a specific behavior in a Decision Transformer.

## Automated Circuit Discovery (ACDC)

ACDC is used to prune the full model into a task-specific subgraph. It works by iteratively removing edges that do not significantly contribute to the model's performance on a specific metric (e.g., action prediction).

### ACDC Workflow

```mermaid
graph TD
    A[Full Model Graph] --> B{Edge Importance Check}
    B -- Significant --> C[Keep Edge]
    B -- Insignificant --> D[Prune Edge]
    C --> E[New Subgraph]
    D --> E
    E --> F{Converged?}
    F -- No --> B
    F -- Yes --> G[Final Circuit]
```

## Induction Head Discovery

Induction heads are key components in Transformers that perform temporal pattern recognition. In DTs, these are often responsible for matching current states to past experiences to determine the next action.

### The Induction Mechanism
Induction heads typically follow a two-step pattern:
1. **Search**: Look for previous occurrences of the current token.
2. **Retrieve**: Extract the token that followed the previous occurrence.

```mermaid
sequenceDiagram
    participant S as State Token (T)
    participant P as Previous State (T-k)
    participant N as Next Action (T-k+1)
    participant O as Output Action (T+1)
    
    S->>P: Key-Query Match
    P->>N: Value Retrieval
    N->>O: Contribution to Logits
```
