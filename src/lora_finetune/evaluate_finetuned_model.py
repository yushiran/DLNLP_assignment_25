#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation script for the LoRA fine-tuned model

This script evaluates the performance of the fine-tuned model 
on a test set of Nüshu character questions.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import accuracy_score
import rouge
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import GPT4oClient
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.gpt4o_search import GPT4oClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Set up paths
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
base_model_path = os.path.join(current_dir, "model", model_id)
lora_model_path = os.path.join(current_dir, "model", f"{model_id}-lora-finetuned")
cache_dir = os.path.join(current_dir, "model")
dataset_path = os.path.join(current_dir, "data/processed/lora_dataset/nushu_lora_dataset.json")
results_dir = os.path.join(current_dir, "evaluation_results")
os.makedirs(results_dir, exist_ok=True)


def load_models():
    """Load both the base model and the fine-tuned model for comparison"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # Load the base model without LoRA
    logger.info("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    
    # Load the fine-tuned model with LoRA
    logger.info(f"Loading LoRA model...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    return tokenizer, base_model, lora_model


def prepare_test_dataset(dataset_path, test_size=0.01):
    """Prepare a test dataset from the full dataset"""
    
    # Load the dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to a list for easier manipulation
    examples = data['data']
    
    # Take 20% for testing
    split_idx = int(len(examples) * (1 - test_size))
    test_examples = examples[split_idx:]
    
    logger.info(f"Using {len(test_examples)} examples for evaluation")
    
    return test_examples


def generate_response(model, tokenizer, question, context):
    """Generate a response from the model for a given question and context"""
    
    # Create input prompt using the same format as in lora_finetune.py
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
{context}
</context>

<question>{question}</question>

Provide a single, clear, well-formatted answer based on the retrieved information."""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the answer part - the model output will follow the prompt
    answer = result.replace(prompt, "").strip()
    
    return answer


def generate_gpt4o_response(client, question, context):
    """Generate a response using GPT-4o Mini for the same question and context"""
    
    # Create the same input prompt format for consistency
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
{context}
</context>

<question>{question}</question>

Provide a single, clear, well-formatted answer based on the retrieved information. Include relevant Nüshu characters with their complete details."""
    
    # Use the client to generate a response
    try:
        response = client.chat(
            query=prompt,
            system_prompt="You are a helpful assistant that answers questions about Nüshu characters based on the provided context.",
            temperature=0.7
        )
        return response
    except Exception as e:
        logger.error(f"Error generating GPT-4o response: {e}")
        return f"Error: {str(e)}"


def compute_metrics(references, predictions):
    """Compute evaluation metrics for generated responses"""
    
    # Initialize ROUGE scorer
    rouge_scorer = rouge.Rouge()
    
    # Character accuracy - exact match
    character_matches = 0
    
    # Process each response to check for correct character identification
    prediction_ids = []
    reference_ids = []
    
    for ref, pred in zip(references, predictions):
        # Try to extract character ID from reference and prediction
        ref_id = extract_character_id(ref)
        pred_id = extract_character_id(pred)
        
        if ref_id is not None and pred_id is not None:
            reference_ids.append(ref_id)
            prediction_ids.append(pred_id)
            
            if ref_id == pred_id:
                character_matches += 1
    
    # Calculate character accuracy if any IDs were extracted
    character_accuracy = character_matches / len(references) if references else 0
    
    # Calculate ROUGE scores
    try:
        rouge_scores = rouge_scorer.get_scores(predictions, references, avg=True)
    except Exception as e:
        logger.error(f"Error calculating ROUGE scores: {e}")
        rouge_scores = {
            'rouge-1': {'f': 0, 'p': 0, 'r': 0},
            'rouge-2': {'f': 0, 'p': 0, 'r': 0},
            'rouge-l': {'f': 0, 'p': 0, 'r': 0}
        }
    
    metrics = {
        'character_accuracy': character_accuracy,
        'rouge1_f1': rouge_scores['rouge-1']['f'],
        'rouge2_f1': rouge_scores['rouge-2']['f'],
        'rougeL_f1': rouge_scores['rouge-l']['f']
    }
    
    return metrics


def extract_character_id(text):
    """Extract Nüshu character ID or the character itself if present"""
    import re
    
    # First try to extract Nüshu characters (Unicode range U+1B170 to U+1B2FF)
    nushu_chars = re.findall(r'[\U0001B170-\U0001B2FF]', text)
    
    # If we found any Nüshu characters, return the first one as the identifier
    if nushu_chars:
        return nushu_chars[0]
    
    # If no Nüshu character found, try to find ID patterns
    id_patterns = [
        r'ID[:\s]+(\d+)',
        r'character[:\s]+(\d+)',
        r'ID number[:\s]+(\d+)',
        r'number[:\s]+(\d+)'
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def evaluate_models():
    """Evaluate and compare base model, fine-tuned model, and GPT-4o Mini"""
    
    # Load models
    tokenizer, base_model, lora_model = load_models()
    
    # Initialize GPT-4o Mini client
    gpt4o_client = GPT4oClient()
    
    # Prepare test dataset
    test_examples = prepare_test_dataset(dataset_path)
    
    # Results to collect
    results = {
        'question': [],
        'reference': [],
        'base_prediction': [],
        'lora_prediction': [],
        'gpt4o_prediction': []
    }
    
    # Process each test example
    logger.info("Generating predictions...")
    for example in tqdm(test_examples):
        question = example['Question']
        context = example['Context']
        reference = example['Response']
        
        # Generate responses
        base_prediction = generate_response(base_model, tokenizer, question, context)
        lora_prediction = generate_response(lora_model, tokenizer, question, context)
        
        # Generate response from GPT-4o Mini
        gpt4o_prediction = generate_gpt4o_response(gpt4o_client, question, context)
        
        # logger.info(f"base_prediction: {base_prediction}")
        # logger.info(f"lora_prediction: {lora_prediction}")
        # logger.info(f"gpt4o_prediction: {gpt4o_prediction}")

        # Store results
        results['question'].append(question)
        results['reference'].append(reference)
        results['base_prediction'].append(base_prediction)
        results['lora_prediction'].append(lora_prediction)
        results['gpt4o_prediction'].append(gpt4o_prediction)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    base_metrics = compute_metrics(results['reference'], results['base_prediction'])
    lora_metrics = compute_metrics(results['reference'], results['lora_prediction'])
    gpt4o_metrics = compute_metrics(results['reference'], results['gpt4o_prediction'])
    
    # Create results dataframe
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(results_dir, 'prediction_results.csv'), index=False)
    
    # Print metrics
    print("\n===== Evaluation Results =====")
    print("\nBase Model Metrics:")
    for metric, value in base_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nLoRA Fine-tuned Model Metrics:")
    for metric, value in lora_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nGPT-4o Mini Model Metrics:")
    for metric, value in gpt4o_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Create metrics summary
    metrics_summary = {
        'Metric': list(base_metrics.keys()),
        'Base Model': [base_metrics[k] for k in base_metrics.keys()],
        'LoRA Fine-tuned Model': [lora_metrics[k] for k in base_metrics.keys()],
        'GPT-4o Mini': [gpt4o_metrics[k] for k in gpt4o_metrics.keys()]
    }
    
    df_metrics = pd.DataFrame(metrics_summary)
    df_metrics.to_csv(os.path.join(results_dir, 'metrics_summary.csv'), index=False)
    
    # Plot comparison chart with all three models
    plot_metrics_comparison(base_metrics, lora_metrics, gpt4o_metrics)
    
    return df_results, df_metrics


def plot_metrics_comparison(base_metrics, lora_metrics, gpt4o_metrics):
    """Create a bar chart comparing model metrics for all three models"""
    metrics = list(base_metrics.keys())
    base_values = [base_metrics[k] for k in metrics]
    lora_values = [lora_metrics[k] for k in metrics]
    gpt4o_values = [gpt4o_metrics[k] for k in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25  # Narrower bars to fit three models
    
    fig, ax = plt.subplots(figsize=(14, 8))  # Larger figure to accommodate three bars
    ax.bar(x - width, base_values, width, label='Base Model', color='#1f77b4')
    ax.bar(x, lora_values, width, label='LoRA Fine-tuned Model', color='#ff7f0e')
    ax.bar(x + width, gpt4o_values, width, label='GPT-4o Mini', color='#2ca02c')
    
    ax.set_title('Model Performance Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    
    # Add value labels on top of bars
    for i, v in enumerate(base_values):
        ax.text(i - width, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
        
    for i, v in enumerate(lora_values):
        ax.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
        
    for i, v in enumerate(gpt4o_values):
        ax.text(i + width, v + 0.01, f'{v:.4f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_comparison.png'), dpi=300)
    logger.info(f"Metrics comparison chart saved to {os.path.join(results_dir, 'metrics_comparison.png')}")


if __name__ == "__main__":
    evaluate_models()
