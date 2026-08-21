# Video Walkthrough Script: Domain-Adaptive Clinical LLM Summarization

**Duration**: ~3:00 Minutes  
**Target Audience**: ML Engineers, Clinical AI Researchers, Technical Recruiters  
**Topic**: Parameter-Efficient Fine-Tuning (LoRA) for Medical Note Summarization with Docker & Custom Heuristics

---

## Visual & Audio Storyboard

### [0:00 - 0:35] Scene 1: Introduction & The Medical Text Challenge
- **Visual**: Screen recording showing raw, unformatted clinical note containing messy abbreviations (`pt`, `s/p CABG`, `h/o CHF`, `K+ 5.2`, `BP 162/94`).
- **Speaker**:
  > *"Hi everyone! Welcome to this project walkthrough where we build a production-grade, containerized machine learning pipeline to fine-tune a Large Language Model for clinical note summarization.*
  >
  > *Medical notes are notoriously difficult for standard NLP models. They are packed with complex clinical shorthand, vital signs, and lab values. A misinterpretation can have critical consequences in patient care.*
  >
  > *In this project, we adapt an LLM to master medical summarization using Parameter-Efficient Fine-Tuning—specifically LoRA—enabling fast, low-cost training while preserving model safety."*

---

### [0:35 - 1:15] Scene 2: Custom Preprocessing Heuristics
- **Visual**: Code walkthrough of `src/preprocess.py` showing `expand_clinical_abbreviations` and `normalize_lab_values`. Showing side-by-side terminal diff of raw vs. cleaned note.
- **Speaker**:
  > *"First, we tackle clinical data normalization at the foundational level without relying on heavy third-party NLP packages.*
  >
  > *In `src/preprocess.py`, we implemented custom regex heuristics:
  > 1. Abbreviation expansion that turns cryptic medical shorthand into standard medical terminology.
  > 2. Lab value normalization, transforming unstructured metrics like `K+ 5.2` and `BP 162/94` into structured, machine-readable tokens like `LAB(name=K+, value=5.2)` and `VITALS(BP=162/94)`.*
  >
  > *This structure provides unambiguous semantic signals for the language model during training."*

---

### [1:15 - 1:55] Scene 3: Parameter-Efficient Fine-Tuning with LoRA & Stability Analysis
- **Visual**: Diagram of LoRA decomposition ($W = W_0 + B \cdot A$). Showing `src/train.py`, `docs/lora_config.md`, and the `results/training_analysis.json` schema.
- **Speaker**:
  > *"Next is the training pipeline. Full fine-tuning of multi-billion parameter models is computationally heavy and risks catastrophic forgetting. Instead, we use LoRA from Hugging Face's PEFT library.*
  >
  > *We freeze the base model and inject low-rank update matrices with rank $r=16$, alpha $\alpha=32$, and target attention modules `q` and `v`. This trains less than 0.5% of the total parameters, reducing VRAM footprint dramatically.*
  >
  > *To test pipeline resilience, we orchestrate two runs: an unstable run with extreme learning rates that demonstrates loss explosion and NaN handling, followed by a stable run with healthy gradient descent and convergence."*

---

### [1:55 - 2:30] Scene 4: Domain-Specific Evaluation & Clinical Hallucination Analysis
- **Visual**: Showing `results/evaluation_metrics.json` table and `results/hallucination_analysis.md`.
- **Speaker**:
  > *"For evaluation, standard ROUGE scores alone aren't enough because medical summaries require strict factual fidelity. We implemented a custom human-proxy metric: **Clinical Entity Recall**.*
  >
  > *This measures whether key clinical diagnoses, medications, and lab values from the reference summary are factually retained in the generated output. Our fine-tuned adapter significantly outperformed the zero-shot base model in both ROUGE-L and entity recall.*
  >
  > *We also conducted an in-depth hallucination audit in `hallucination_analysis.md`, diagnosing cases of extrinsic medication fabrication and defining guardrails like entity verification and constrained decoding."*

---

### [2:30 - 3:00] Scene 5: Docker Containerization & Conclusion
- **Visual**: Terminal executing `docker-compose up`. Showing generated artifacts in `output/final_adapter/` and `results/`.
- **Speaker**:
  > *"Finally, the entire system is fully containerized. With a single `docker-compose up`, Docker mounts our volumes, executes `main.sh`, runs healthchecks, and produces all adapters and evaluation reports in minutes.*
  >
  > *The result is a portable, reproducible, and safe clinical AI solution. Check out the complete codebase and documentation on GitHub. Thank you for watching!"*

---

## Presentation Checklist & Tips
- [x] Clear audio with background noise suppression.
- [x] Highlight key code lines in `src/preprocess.py` and `src/train.py`.
- [x] Show the `results/evaluation_metrics.json` output clearly on screen.
- [x] Conclude with the reproducibility of `docker-compose up`.
