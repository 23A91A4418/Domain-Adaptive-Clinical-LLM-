#!/usr/bin/env python3
"""
Pipeline Orchestrator for Domain-Adaptive Clinical LLM.

Coordinates:
1. Data Preprocessing (custom heuristics)
2. Training Instability Simulation (unstable run with NaN loss)
3. Stable LoRA Fine-Tuning (saving adapter weights)
4. Training Analysis JSON Generation
5. Comprehensive Evaluation (ROUGE & Clinical Entity Recall)
"""

import os
import sys
import json
import argparse
from preprocess import preprocess_dataset
from train import train_model
from evaluate import evaluate_models


def run_full_pipeline(
    raw_data: str = "data/raw_data.jsonl",
    train_data: str = "data/cleaned_train.jsonl",
    test_data: str = "data/cleaned_test.jsonl",
    base_model_name: str = None,
    output_adapter_dir: str = "output/final_adapter",
    results_dir: str = "results",
    hf_token: str = None
):
    if base_model_name is None:
        base_model_name = os.getenv("BASE_MODEL_NAME", "google/flan-t5-base")
        
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(output_adapter_dir, exist_ok=True)
    
    print("\n========================================================")
    print("  DOMAIN-ADAPTIVE CLINICAL LLM SUMMARIZATION PIPELINE  ")
    print("========================================================\n")
    
    # ---------------------------------------------------------
    # Step 1: Preprocessing Pipeline
    # ---------------------------------------------------------
    print(">>> [STAGE 1/4] Running Custom Clinical Preprocessing Pipeline...")
    n_train, n_test = preprocess_dataset(
        input_file=raw_data,
        output_train=train_data,
        output_test=test_data
    )
    print(f"[STAGE 1/4] Preprocessing complete. Train samples: {n_train}, Test samples: {n_test}\n")
    
    # ---------------------------------------------------------
    # Step 2: Training Instability Simulation (Unstable Run)
    # ---------------------------------------------------------
    print(">>> [STAGE 2/4] Simulating Training Instability (Unstable Run: lr=1e-1)...")
    unstable_log = os.path.join(results_dir, "unstable_train.log")
    unstable_metrics = train_model(
        model_name=base_model_name,
        train_file=train_data,
        eval_file=test_data,
        output_dir="output/unstable_checkpoint",
        log_file=unstable_log,
        run_type="unstable",
        learning_rate=float(os.getenv("UNSTABLE_LEARNING_RATE", 1e-1)),
        lora_rank=int(os.getenv("LORA_RANK", 16)),
        lora_alpha=int(os.getenv("LORA_ALPHA", 32)),
        lora_dropout=float(os.getenv("LORA_DROPOUT", 0.05)),
        num_epochs=1,
        batch_size=int(os.getenv("BATCH_SIZE", 2)),
        hf_token=hf_token
    )
    print(f"[STAGE 2/4] Unstable Run Complete. Metrics: {unstable_metrics}\n")
    
    # ---------------------------------------------------------
    # Step 3: Stable LoRA Fine-Tuning Run
    # ---------------------------------------------------------
    print(">>> [STAGE 3/4] Running Stable LoRA Fine-Tuning (Stable Run: lr=3e-4)...")
    stable_log = os.path.join(results_dir, "stable_train.log")
    stable_metrics = train_model(
        model_name=base_model_name,
        train_file=train_data,
        eval_file=test_data,
        output_dir=output_adapter_dir,
        log_file=stable_log,
        run_type="stable",
        learning_rate=float(os.getenv("LEARNING_RATE", 3e-4)),
        lora_rank=int(os.getenv("LORA_RANK", 16)),
        lora_alpha=int(os.getenv("LORA_ALPHA", 32)),
        lora_dropout=float(os.getenv("LORA_DROPOUT", 0.05)),
        num_epochs=int(os.getenv("NUM_EPOCHS", 3)),
        batch_size=int(os.getenv("BATCH_SIZE", 2)),
        hf_token=hf_token
    )
    print(f"[STAGE 3/4] Stable Fine-Tuning Complete. Metrics: {stable_metrics}\n")
    
    # Save training_analysis.json with required schema
    training_analysis = {
        "stable_run": {
            "log_path": unstable_log.replace("unstable_train.log", "stable_train.log"),
            "final_metrics": {
                "train_loss": stable_metrics["train_loss"],
                "eval_loss": stable_metrics["eval_loss"]
            }
        },
        "unstable_run": {
            "log_path": unstable_log,
            "final_metrics": {
                "train_loss": unstable_metrics["train_loss"],
                "eval_loss": unstable_metrics["eval_loss"]
            }
        }
    }
    
    analysis_path = os.path.join(results_dir, "training_analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(training_analysis, f, indent=2)
    print(f"[Analysis] Saved training analysis report to: {analysis_path}\n")
    
    # ---------------------------------------------------------
    # Step 4: Comprehensive Model Evaluation
    # ---------------------------------------------------------
    print(">>> [STAGE 4/4] Evaluating Base Model vs Fine-Tuned Model...")
    eval_metrics_file = os.path.join(results_dir, "evaluation_metrics.json")
    eval_results = evaluate_models(
        model_name=base_model_name,
        adapter_path=output_adapter_dir,
        test_file=test_data,
        output_metrics_file=eval_metrics_file,
        hf_token=hf_token
    )
    
    print("\n========================================================")
    print("          PIPELINE EXECUTION COMPLETED SUCCESSFULLY      ")
    print("========================================================")
    print(f"1. LoRA Adapter Saved to:    {output_adapter_dir}")
    print(f"2. Training Analysis:        {analysis_path}")
    print(f"3. Evaluation Metrics:       {eval_metrics_file}")
    print("========================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Clinical LLM Pipeline Orchestrator")
    parser.add_argument("--raw_data", default="data/raw_data.jsonl")
    parser.add_argument("--train_data", default="data/cleaned_train.jsonl")
    parser.add_argument("--test_data", default="data/cleaned_test.jsonl")
    parser.add_argument("--model_name", default=os.getenv("BASE_MODEL_NAME", "google/flan-t5-base"))
    parser.add_argument("--output_adapter", default="output/final_adapter")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN", None))
    args = parser.parse_args()
    
    run_full_pipeline(
        raw_data=args.raw_data,
        train_data=args.train_data,
        test_data=args.test_data,
        base_model_name=args.model_name,
        output_adapter_dir=args.output_adapter,
        results_dir=args.results_dir,
        hf_token=args.hf_token
    )


if __name__ == "__main__":
    main()
