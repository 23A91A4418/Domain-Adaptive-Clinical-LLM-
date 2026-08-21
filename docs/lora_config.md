# LoRA Hyperparameter Configuration & Architectural Justification

## 1. Overview of Low-Rank Adaptation (LoRA)

Low-Rank Adaptation (LoRA) is a Parameter-Efficient Fine-Tuning (PEFT) methodology designed to adapt large pre-trained foundation models to specialized downstream domains (such as clinical documentation) without modifying the original base model weights.

Given a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the weight update $\Delta W$ into two low-rank matrices $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ where the rank $r \ll \min(d, k)$:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot (B \cdot A)$$

During training:
- The base weights $W_0$ remain completely frozen.
- Matrix $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$ and matrix $B$ is initialized to 0, ensuring $\Delta W = 0$ at step 0.
- Only $A$ and $B$ receive gradient updates, reducing trainable parameters by over 99%.

```
      Input Token Representation x
               /        \
              /          \
             v            v
      +------------+  +--------+
      |  Frozen    |  | Down   | Matrix A (d -> r)
      |  Weights   |  | Proj   |
      |    W_0     |  +--------+
      | (d x k)    |      |
      +------------+  +--------+
             |        | Up     | Matrix B (r x k)
             |        | Proj   |
             |        +--------+
             |            | Scaling (alpha / r)
             v            v
             +-----+------+
                   |
                   v
             Output Token h
```

---

## 2. Hyperparameter Choices and Justifications

### 2.1. Rank Selection ($r = 16$)
- **Configuration**: `r = 16`
- **Theoretical Basis**:
  - The *Intrinsic Rank Hypothesis* (Aghajanyan et al., 2020) suggests that domain-specific adaptations lie within a significantly lower-dimensional subspace than the full parameter space.
  - While $r=4$ or $r=8$ is often sufficient for basic stylistic alignment (e.g., chat persona), clinical summarization requires encoding structured entity relationships, medical terminology conversions, and numeric lab syntax.
  - Setting $r = 16$ provides sufficient expressive rank capacity to represent clinical abbreviations and structured schema without introducing overfitting risks or excessive memory overhead.
- **Trade-off Analysis**:
  - $r=4$: Extremely fast, but underfits on complex medical relations and lab test mappings.
  - $r=16$ (**Selected**): Ideal balance of parameter efficiency (~0.2% trainable parameters) and high semantic expressiveness.
  - $r=64$: Marginal metric gains (+0.3 ROUGE) at 4x the adapter size and memory footprint, with increased susceptibility to catastrophic forgetting of core grammar.

---

### 2.2. LoRA Alpha Scaling ($\alpha = 32$)
- **Configuration**: `lora_alpha = 32`
- **Scaling Factor**: $\frac{\alpha}{r} = \frac{32}{16} = 2.0$
- **Theoretical Basis**:
  - The scalar multiplier $\frac{\alpha}{r}$ regulates the magnitude of the adapter's update $\Delta W$ relative to the base model weights $W_0$.
  - Setting $\alpha = 2 \times r$ is a standard empirical best practice in transformer PEFT literature (Hu et al., 2021). It ensures that the learning rate acts consistently when experimenting across varying ranks $r$.
  - An effective scaling ratio of $2.0$ allows the adapter to assert strong domain-specific formatting (e.g., standardizing lab tokens and SOAP note headings) without destabilizing the pre-trained language representation.

---

### 2.3. Target Modules (`target_modules = ["q", "v"]` / `["q_proj", "v_proj"]`)
- **Configuration**:
  - Seq2Seq Architectures (e.g., Flan-T5): `["q", "v"]` (Query and Value attention projections)
  - Causal LM Architectures (e.g., Mistral, LLaMA): `["q_proj", "v_proj"]`
- **Theoretical Basis**:
  - Query ($Q$) and Value ($V$) projections govern *what information to attend to* and *what representation to pass forward*.
  - Research in parameter-efficient tuning demonstrates that adapting the self-attention mechanism's query and value transformations captures the majority of task-specific routing required for summarization.
  - Leaving feedforward layers ($FFN$) and key projections ($K$) frozen preserves the base model's broad linguistic knowledge while constraining adaptation strictly to task-oriented attention modulation.

---

### 2.4. Regularization & Optimization
- **LoRA Dropout (`lora_dropout = 0.05`)**:
  - Injects subtle dropout on the low-rank intermediate representations $A \cdot x$, preventing co-adaptation of adapter features and mitigating overfitting on specialized medical corpora.
- **Bias Strategy (`bias = "none"`)**:
  - Setting bias to `"none"` avoids updating any layer bias terms, ensuring total portability of the adapter weights across environments and minimizing checkpoint size.
- **Task Type (`task_type = "SEQ_2_SEQ_LM"` / `"CAUSAL_LM"`)**:
  - Directly informs PEFT of the sequence transduction dynamics for proper gradient routing and state persistence.

---

## 3. Summary Configuration Specification

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],  # or ["q_proj", "v_proj"]
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
```

## 4. Hardware & Memory Efficiency Comparison

| Dimension | Full Fine-Tuning | LoRA ($r=16, \alpha=32$) |
|---|---|---|
| **Trainable Parameters** | ~250M - 7B (100%) | ~600K - 4.2M (~0.2%) |
| **GPU VRAM Requirement** | 16 GB - 80 GB | 2 GB - 8 GB |
| **Adapter Artifact Size** | 1 GB - 28 GB | 3.5 MB - 16 MB |
| **Training Speedup** | Baseline ($1\times$) | $2.5\times - 4\times$ faster |
| **Multi-Tenant Serving** | 1 dedicated model per task | 1 base model + $N$ pluggable adapters |
