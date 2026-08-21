#!/usr/bin/env python3
"""
Domain-Adaptive Fine-Tuning Script using LoRA (PEFT) for Clinical LLM Summarization.

This script fine-tunes a base language model using Parameter-Efficient Fine-Tuning (LoRA)
from Hugging Face's PEFT library. It supports both Seq2Seq (e.g. Flan-T5, BART) and Causal LM
(e.g. LLaMA, Mistral, Qwen) architectures.

It also supports simulating training instability (e.g., exploding gradients resulting in NaN loss)
to evaluate training stability dynamics.
"""

import os
import sys
import math
import json
import logging
import argparse
from typing import Dict, Any, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)


class ClinicalDataset(Dataset):
    """PyTorch Dataset for Clinical Notes and Reference Summaries."""
    
    def __init__(self, data_file: str, tokenizer, max_source_len: int = 512, max_target_len: int = 128, is_causal: bool = False):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.is_causal = is_causal
        self.samples = []
        
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        source_text = item.get("text", "")
        target_text = item.get("summary", "")
        
        prompt = f"Summarize the following clinical note:\n{source_text}"
        
        if self.is_causal:
            # For Causal LM formatting
            full_text = f"{prompt}\n\nSummary:\n{target_text}"
            encodings = self.tokenizer(
                full_text,
                max_length=self.max_source_len + self.max_target_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"].squeeze(0)
            attention_mask = encodings["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
        else:
            # For Seq2Seq LM formatting
            model_inputs = self.tokenizer(
                prompt,
                max_length=self.max_source_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            with self.tokenizer.as_target_tokenizer():
                labels_encoding = self.tokenizer(
                    target_text,
                    max_length=self.max_target_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
            
            labels = labels_encoding["input_ids"].squeeze(0)
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                "input_ids": model_inputs["input_ids"].squeeze(0),
                "attention_mask": model_inputs["attention_mask"].squeeze(0),
                "labels": labels
            }


def setup_logger(log_file: str) -> logging.Logger:
    """Configures a file and console logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger


def determine_model_and_task(model_name: str, hf_token: Optional[str] = None):
    """
    Loads config to determine whether model is Seq2Seq or Causal LM,
    and returns appropriate default target modules.
    """
    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    is_seq2seq = config.is_encoder_decoder
    
    if is_seq2seq:
        task_type = TaskType.SEQ_2_SEQ_LM
        # Common Seq2Seq modules (e.g. T5 has 'q', 'v', BART has 'q_proj', 'v_proj')
        target_modules = ["q", "v"] if "t5" in model_name.lower() else ["q_proj", "v_proj"]
    else:
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "v_proj"]
        
    return is_seq2seq, task_type, target_modules


def train_model(
    model_name: str = "google/flan-t5-base",
    train_file: str = "data/cleaned_train.jsonl",
    eval_file: str = "data/cleaned_test.jsonl",
    output_dir: str = "output/final_adapter",
    log_file: str = "results/stable_train.log",
    run_type: str = "stable",
    learning_rate: float = 3e-4,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    batch_size: int = 2,
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main training execution function for LoRA fine-tuning.
    """
    logger = setup_logger(log_file)
    logger.info(f"--- Starting {run_type.upper()} LoRA Fine-Tuning Run ---")
    logger.info(f"Base Model: {model_name}")
    logger.info(f"Hyperparameters: lr={learning_rate}, rank={lora_rank}, alpha={lora_alpha}, epochs={num_epochs}, batch_size={batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} (CUDA available: {torch.cuda.is_available()})")
    
    # 1. Load Tokenizer & Model Architecture Info
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    is_seq2seq, task_type, default_target_modules = determine_model_and_task(model_name, hf_token)
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Architecture: {'Seq2Seq LM' if is_seq2seq else 'Causal LM'}, Task Type: {task_type}")
    
    # 2. Load Base Model
    if is_seq2seq:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=hf_token)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
        
    base_model.to(device)
    
    # 3. Configure LoRA (PEFT)
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=default_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=task_type
    )
    
    logger.info(f"Instantiated LoRA Config: rank={lora_config.r}, alpha={lora_config.lora_alpha}, target_modules={lora_config.target_modules}")
    
    # 4. Wrap Model with PEFT
    peft_model = get_peft_model(base_model, lora_config)
    trainable_params, all_params = peft_model.get_nb_trainable_parameters()
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_params:,} || "
        f"Trainable ratio: {100 * trainable_params / all_params:.4f}%"
    )
    
    # 5. Prepare DataLoaders
    train_dataset = ClinicalDataset(train_file, tokenizer, is_causal=(not is_seq2seq))
    eval_dataset = ClinicalDataset(eval_file, tokenizer, is_causal=(not is_seq2seq))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
    
    # 6. Optimizer and Scheduler
    optimizer = AdamW(peft_model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps
    )
    
    # 7. Training Loop
    peft_model.train()
    final_train_loss = 0.0
    final_eval_loss = 0.0
    is_nan_encountered = False
    
    logger.info("Beginning training loop...")
    global_step = 0
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        batches_in_epoch = 0
        
        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # For unstable run simulation with exploding gradients
            if run_type == "unstable" and epoch >= 1 and batch_idx >= 1:
                # Deliberately inject unstable gradient explosion to trigger NaN dynamics
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss * 1e8  # Force loss explosion / NaN
            else:
                outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
            current_loss_val = loss.item()
            
            if math.isnan(current_loss_val) or math.isinf(current_loss_val):
                logger.warning(f"Epoch {epoch} | Step {global_step}: Loss exploded to NaN/Inf!")
                is_nan_encountered = True
                final_train_loss = "NaN"
                final_eval_loss = "NaN"
                break
                
            loss.backward()
            
            if run_type == "stable":
                torch.nn.utils.clip_grad_norm_(peft_model.parameters(), 1.0)
                
            optimizer.step()
            scheduler.step()
            
            epoch_loss += current_loss_val
            batches_in_epoch += 1
            
            logger.info(f"Epoch {epoch}/{num_epochs} | Step {global_step}/{total_steps} | Loss: {current_loss_val:.4f}")
            
        if is_nan_encountered:
            logger.error("Training aborted due to numerical instability (loss = NaN).")
            break
            
        if batches_in_epoch > 0:
            avg_epoch_loss = epoch_loss / batches_in_epoch
            final_train_loss = round(avg_epoch_loss, 4)
            logger.info(f"Epoch {epoch} Completed | Average Train Loss: {final_train_loss}")
            
    # 8. Evaluation
    if not is_nan_encountered:
        peft_model.eval()
        eval_loss_accum = 0.0
        eval_batches = 0
        
        with torch.no_grad():
            for eval_batch in eval_loader:
                input_ids = eval_batch["input_ids"].to(device)
                attention_mask = eval_batch["attention_mask"].to(device)
                labels = eval_batch["labels"].to(device)
                
                eval_out = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                eval_loss_accum += eval_out.loss.item()
                eval_batches += 1
                
        if eval_batches > 0:
            final_eval_loss = round(eval_loss_accum / eval_batches, 4)
            logger.info(f"Evaluation Loss: {final_eval_loss}")
            
        # 9. Save final adapter if stable
        if run_type == "stable":
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving fine-tuned LoRA adapter to: {output_dir}")
            peft_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"Successfully saved adapter files to {output_dir}")
    else:
        final_train_loss = "NaN"
        final_eval_loss = "NaN"
        
    metrics = {
        "train_loss": final_train_loss,
        "eval_loss": final_eval_loss
    }
    
    logger.info(f"--- Finished {run_type.upper()} Run with Metrics: {metrics} ---")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="LoRA Clinical LLM Training")
    parser.add_argument("--model_name", default=os.getenv("BASE_MODEL_NAME", "google/flan-t5-base"))
    parser.add_argument("--train_file", default="data/cleaned_train.jsonl")
    parser.add_argument("--eval_file", default="data/cleaned_test.jsonl")
    parser.add_argument("--output_dir", default="output/final_adapter")
    parser.add_argument("--log_file", default="results/stable_train.log")
    parser.add_argument("--run_type", choices=["stable", "unstable"], default="stable")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_rank", type=int, default=int(os.getenv("LORA_RANK", 16)))
    parser.add_argument("--lora_alpha", type=int, default=int(os.getenv("LORA_ALPHA", 32)))
    parser.add_argument("--lora_dropout", type=float, default=float(os.getenv("LORA_DROPOUT", 0.05)))
    parser.add_argument("--num_epochs", type=int, default=int(os.getenv("NUM_EPOCHS", 3)))
    parser.add_argument("--batch_size", type=int, default=int(os.getenv("BATCH_SIZE", 2)))
    parser.add_argument("--hf_token", default=os.getenv("HF_TOKEN", None))
    args = parser.parse_args()
    
    # Default learning rate depending on run_type
    if args.learning_rate is None:
        if args.run_type == "unstable":
            lr = float(os.getenv("UNSTABLE_LEARNING_RATE", 1e-1))
        else:
            lr = float(os.getenv("LEARNING_RATE", 3e-4))
    else:
        lr = args.learning_rate
        
    # Auto-adjust log file if default
    if args.run_type == "unstable" and args.log_file == "results/stable_train.log":
        args.log_file = "results/unstable_train.log"
        
    metrics = train_model(
        model_name=args.model_name,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        log_file=args.log_file,
        run_type=args.run_type,
        learning_rate=lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        hf_token=args.hf_token
    )
    
    print(f"[Training Result] Run: {args.run_type} | Train Loss: {metrics['train_loss']} | Eval Loss: {metrics['eval_loss']}")


if __name__ == "__main__":
    main()
