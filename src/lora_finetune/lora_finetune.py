#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA Fine-tuning script for Nüshu Character Recognition

This script fine-tunes a DeepSeek-R1-Distill-Qwen-1.5B model 
using LoRA (Low-Rank Adaptation) with the Nüshu character dataset.
The training format has been optimized to match the inference format
used in deploy_rag_with_deepseek.py for better consistency.

Usage:
    python lora_finetune.py
"""

import os
import json
import logging
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    pipeline
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from datasets import Dataset as HFDataset
from dotenv import load_dotenv
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check for GPU
assert torch.cuda.is_available(), "GPU is required for training!"
device = torch.device("cuda")
logger.info(f"Using device: {device}")
logger.info(f"CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}")
logger.info(f"GPU count: {torch.cuda.device_count()}, GPU name: {torch.cuda.get_device_name(0)}")

# Configuration
load_dotenv()
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logger.info(f"Current directory: {current_dir}")

# Create training log directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.join(current_dir, "logs", f"lora_finetune_{timestamp}")
os.makedirs(log_dir, exist_ok=True)
logger.info(f"Created log directory: {log_dir}")

# Model path setup
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
local_model_path = os.path.join(current_dir, "model", model_id)
output_path = os.path.join(current_dir, "model", f"{model_id}-lora-finetuned")
cache_dir = os.path.join(current_dir, "model")

# Dataset path
dataset_path = os.path.join(current_dir, "data/processed/lora_dataset/nushu_lora_dataset.json")


# Custom callback to track training loss
class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if "loss" in logs:
            self.losses.append(logs["loss"])
            if len(self.losses) % 10 == 0:
                logger.info(f"Current loss: {logs['loss']}")


def load_and_prepare_dataset(tokenizer, dataset_path: str):
    """Load and preprocess the Nüshu dataset for training"""
    
    # Load the dataset from JSON
    logger.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Convert to the format expected by Hugging Face's Dataset class
    data_dict = {
        "Question": [],
        "Context": [],
        "Response": [],
    }
    
    # Extract the data from the JSON structure
    for item in raw_data["data"]:
        data_dict["Question"].append(item["Question"])
        data_dict["Context"].append(item["Context"])
        data_dict["Response"].append(item["Response"])
    
    # Create a Hugging Face Dataset
    dataset = HFDataset.from_dict(data_dict)
    
    logger.info(f"Dataset loaded with {len(dataset)} examples")
    
    # Tokenize and format the dataset
    def preprocess_function(example):
        # Updated prompt template to match the format used in deploy_rag_with_deepseek.py
        prompt = f"""<system>
You are a knowledgeable assistant specializing in Nüshu, the women-only writing system from China.
When responding about Nüshu characters:
1. Provide a SINGLE, CONCISE, and NON-REPETITIVE response
2. Format each Nüshu character entry EXACTLY as follows:
   - Nüshu character: [actual character]
   - Chinese: [corresponding Chinese character(s)]
   - Meaning: [meaning]
   - Pronunciation: [pronunciation]
</system>

<context>
Retrieved information about Nüshu characters and Chinese characters:
{example['Context']}
</context>

<question>{example['Question']}</question>

Provide a single, clear, well-formatted answer based on the retrieved information. Include relevant Nüshu characters with their complete details.
"""
        response = example['Response']
        
        # Combine for the full text
        full_text = prompt + response
        
        # Debug information to track token counts
        # logger.info(f"Sample input shape: prompt:{len(tokenizer.encode(prompt))}, response:{len(tokenizer.encode(response))}")
        
        # Tokenize with appropriate padding (right side)
        tokenized = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        
        # Calculate the length of the prompt part (everything before the assistant's response)
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_length = len(prompt_tokens)
        
        # Create labels: -100 for prompt (not used in loss), actual tokens for response
        labels = tokenized["input_ids"].clone()[0]  # Remove batch dimension
        labels[:prompt_length] = -100  # Mask the prompt part
        
        # Debug information to ensure we have valid labels
        non_masked_labels = (labels != -100).sum().item()
        # logger.info(f"Non-masked labels count: {non_masked_labels}")
        if non_masked_labels == 0:
            logger.warning(f"WARNING: All labels are masked for: {example['Question'][:30]}...")
        
        return {
            "input_ids": tokenized["input_ids"][0],  # Remove batch dimension
            "attention_mask": tokenized["attention_mask"][0],
            "labels": labels
        }
    
    processed_dataset = dataset.map(preprocess_function, remove_columns=dataset.column_names)
    logger.info("Dataset preprocessing complete")
    
    # Split into training and validation sets (90% train, 10% validation)
    split_dataset = processed_dataset.train_test_split(test_size=0.1, seed=42)
    
    return split_dataset


def main():
    """Main training function"""
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if model exists locally, otherwise download
    model_config_path = os.path.join(local_model_path, "config.json")
    if os.path.exists(model_config_path):
        logger.info(f"Loading model from local path: {local_model_path}")
        model_path_to_use = local_model_path
    else:
        logger.info(f"Model not found locally, will download from {model_id}")
        model_path_to_use = model_id
    
    # Load model with parameters specifically adjusted to resolve the zero-loss issue
    logger.info("Loading base model (this may take a few minutes)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path_to_use,
        torch_dtype=torch.float16,  # Use FP16 for efficiency
        device_map="auto",  # Automatically decide which parts go on which devices
        trust_remote_code=True,
        cache_dir=cache_dir,
        use_cache=False,  # Critical for training to disable KV cache
    )
    
    # Double-check we don't have model issues
    logger.info(f"Model is loaded with architecture: {model.config.model_type}")
    logger.info(f"Model hidden size: {model.config.hidden_size}")
    
    # Prepare model for training - This is critical for correct gradient flow
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA with simpler settings to ensure it works
    logger.info("Setting up LoRA configuration")
    lora_config = LoraConfig(
        r=16,                     # Increased rank for better capacity
        lora_alpha=32,            # Higher alpha for stronger updates
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Focus on attention modules first
        lora_dropout=0.05,        # Lower dropout to ensure gradient flow
        bias="none",              # Don't train bias
        task_type="CAUSAL_LM"     # Task type (causal language modeling)
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Log trainable vs total parameters
    model.print_trainable_parameters()
    
    # Load and preprocess dataset
    dataset = load_and_prepare_dataset(tokenizer, dataset_path)
    
    # Create training arguments with settings to fix the zero-loss problem
    training_args = TrainingArguments(
        output_dir=output_path,
        per_device_train_batch_size=1,     # Keep batch size small for stability
        gradient_accumulation_steps=1,     # Start with no accumulation to isolate issues
        learning_rate=1e-4,                # Higher learning rate to get gradients flowing
        num_train_epochs=3,                # Reduce epochs for quicker debugging cycles
        max_steps=-1,                      # Use epochs
        warmup_ratio=0.03,                 # Short warmup
        logging_steps=1,                   # Log every step to diagnose issues
        save_strategy="steps",             # Save more frequently
        save_steps=50,                     # Save checkpoints frequently
        eval_strategy="steps",       # Evaluate frequently
        eval_steps=50,                     # Check evaluation frequently
        fp16=True,                         # Use mixed precision
        fp16_full_eval=False,              # Don't use full fp16 in eval
        report_to="none",                  # Don't report to any platform
        remove_unused_columns=False,       # Keep all columns
        push_to_hub=False,                 # Don't push to the hub
        load_best_model_at_end=True,       # Load the best model at the end of training
        save_total_limit=3,                # Save more checkpoints
        metric_for_best_model="eval_loss", # Use eval loss for best model selection
        greater_is_better=False,           # Lower loss is better
        optim="adamw_torch",               # Use PyTorch's AdamW optimizer
        weight_decay=0.0,                  # Turn off weight decay for debugging
        gradient_checkpointing=True,       # Enable gradient checkpointing for memory efficiency
        logging_first_step=True,           # Log the first step
        logging_dir=log_dir,               # Specify logging directory
        ddp_find_unused_parameters=False,  # Don't look for unused parameters
        dataloader_drop_last=False,        # Don't drop the last batch
    )
    
    # Initialize the loss callback
    loss_callback = LossCallback()
    
    # Add a data validator callback to check inputs/outputs during training
    class DataValidationCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                logger.info("Validating data for the first batch...")
                try:
                    # Get the first batch
                    batch = next(iter(trainer.get_train_dataloader()))
                    logger.info(f"Batch keys: {batch.keys()}")
                    logger.info(f"input_ids shape: {batch['input_ids'].shape}")
                    logger.info(f"attention_mask shape: {batch['attention_mask'].shape}")
                    logger.info(f"labels shape: {batch['labels'].shape}")
                    
                    # Check if we have valid labels (not all -100)
                    valid_label_count = (batch['labels'] != -100).sum().item()
                    logger.info(f"Valid label tokens: {valid_label_count}")
                    
                    if valid_label_count == 0:
                        logger.error("ERROR: All labels are masked! Training will fail.")
                except Exception as e:
                    logger.error(f"Error in data validation: {e}")

    # Initialize the Trainer with additional logging and validation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        callbacks=[loss_callback, DataValidationCallback()]
    )
    
    # Add a simple example to verify model can generate outputs before training
    logger.info("Testing model before training to verify basic functionality...")
    test_prompt = "<User>\nWhat is Nüshu?\n</User>\n\n<Assistant>\n"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    
    try:
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_new_tokens=50)
            logger.info(f"Pre-training test output: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    except Exception as e:
        logger.error(f"Error in pre-training test: {e}")
    
    # Start training with error handling
    logger.info("Starting LoRA fine-tuning...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        # Try to save what we have even if training failed
        try:
            logger.info("Attempting to save partial model...")
            trainer.model.save_pretrained(os.path.join(output_path, "partial_model"))
        except Exception as save_error:
            logger.error(f"Error saving partial model: {save_error}")
        raise e
    
    # Run a final evaluation on the test set
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Final evaluation results: {eval_results}")
    
    # Save evaluation results to file
    with open(os.path.join(output_path, "eval_results.json"), "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Save the final model (LoRA weights only)
    logger.info(f"Saving LoRA weights to {output_path}")
    trainer.model.save_pretrained(output_path)
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_path)
    
    # Plot the loss curve
    plt.figure(figsize=(12, 8))
    plt.plot(loss_callback.losses)
    plt.title("LoRA Fine-tuning Loss Curve for Nüshu Character Model")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, "loss_curve.png"))
    logger.info(f"Loss curve saved to {os.path.join(output_path, 'loss_curve.png')}")
    
    # Test a few examples to verify model quality
    logger.info("Testing model with a few examples...")
    test_examples = [
        "What is the pronunciation of the Nüshu character 𛅰?",
        "What Chinese character corresponds to the Nüshu character with ID 3?",
        "How many strokes does the Nüshu character 𛆋 have?"
    ]
    
    # Create a simple inference pipeline
    model.config.use_cache = True  # Re-enable KV cache for inference
    test_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
        repetition_penalty=1.2,
        top_k=50,
        top_p=0.95
    )
    
    # Run test examples and log results
    test_results = {}
    for i, example in enumerate(test_examples):
        system_prompt = "<system>You are an expert on Nvshu script (女书), a syllabic script used exclusively by women in Jiangyong County, Hunan, China.</system>"
        question = f"<question>{example}</question>"
        full_prompt = f"{system_prompt}\n\n{question}\n\nProvide a focused answer:"
        
        result = test_pipeline(full_prompt)[0]['generated_text']
        answer = result.split("Provide a focused answer:")[-1].strip()
        test_results[f"example_{i+1}"] = {
            "question": example,
            "response": answer
        }
        logger.info(f"Test example {i+1}:\nQ: {example}\nA: {answer}\n")
    
    # Save test results
    with open(os.path.join(output_path, "test_examples.json"), "w", encoding="utf-8") as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    logger.info("Fine-tuning complete!")


def debug_dataset(dataset_path):
    """Debug the dataset structure to identify potential issues"""
    logger.info("=== Debugging dataset structure ===")
    try:
        # Load the dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        logger.info(f"Dataset keys: {raw_data.keys()}")
        logger.info(f"Dataset item count: {len(raw_data['data'])}")
        
        # Check first few examples
        for i, example in enumerate(raw_data['data'][:2]):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"Question: {example['Question']}")
            logger.info(f"Context length: {len(example['Context'])} chars")
            logger.info(f"Response length: {len(example['Response'])} chars")
            
            # Check if there's any actual content to learn from
            if len(example['Response']) < 5:
                logger.warning(f"Example {i+1} has very short response: {example['Response']}")
                
    except Exception as e:
        logger.error(f"Error debugging dataset: {e}")

if __name__ == "__main__":
    # Clear GPU memory before starting
    torch.cuda.empty_cache()
    
    # Check the dataset first
    dataset_path = os.path.join(current_dir, "data/processed/lora_dataset/nushu_lora_dataset.json")
    debug_dataset(dataset_path)
    
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        # Print full traceback
        import traceback
        logger.error(traceback.format_exc())
