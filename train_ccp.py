#!/usr/bin/env python3
"""
Phoenix Context Collapse Parser (CCP) — Training Script

Fine-tunes phi-2 with LoRA on Mac MPS for GTM intent parsing.
Run: python train_ccp.py
"""

import json
import os
import torch
import platform

print("=" * 60)
print("CCP TRAINING SCRIPT")
print("=" * 60)
print(f"Python: {platform.python_version()}")
print(f"Torch: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
print("=" * 60)

# ===== CONFIG =====
BASE_MODEL_PATH = "./models/phi-2"
DATASET_PATH = "./data/ccp_training.jsonl"
OUTPUT_DIR = "./ccp-adapter"
IR_SCHEMA_VERSION = "1.0.0"

NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LENGTH = 512  # Reduced for memory

# GTM Intent IR Schema
GTM_INTENT_IR_SCHEMA = {
    "intent_type": [
        "account_discovery", "pipeline_analysis", "expansion_identification",
        "churn_risk_assessment", "lead_prioritization", "territory_planning",
        "forecast_review", "competitive_analysis", "engagement_summary"
    ],
    "motion": ["outbound", "inbound", "expansion", "renewal", "churn_prevention"],
    "role_assumption": ["sales_rep", "sales_manager", "revops", "marketing", "cs", "exec", "sdr"],
    "account_scope": ["net_new", "existing", "churned", "all"],
    "time_horizon": ["immediate", "this_week", "this_month", "this_quarter", "this_year", "custom"],
    "output_format": ["list", "summary", "detailed", "export", "visualization"],
}

CCP_SYSTEM_PROMPT = """You are the Phoenix Context Collapse Parser (CCP). Your job is to transform ambiguous GTM (Go-To-Market) prompts into structured GTM Intent IR.

Given a user's GTM request, output a JSON object with:
- intent_type: The primary intent category
- motion: The GTM motion (outbound, expansion, renewal, etc.)
- role_assumption: Inferred user role
- account_scope: Which accounts (net_new, existing, all)
- icp_selector: Which ICP to apply (default, or specific product/segment)
- icp_resolution_required: true if ICP needs downstream resolution
- geography_scope: Geographic filter if mentioned (null if global)
- time_horizon: Time scope for the request
- output_format: How results should be presented
- confidence_scores: 0.0-1.0 confidence for each inferred field
- assumptions_applied: List of assumptions made
- clarification_needed: true if request is too ambiguous

Output ONLY valid JSON. No explanation."""

# ===== IMPORTS =====
print("\n[1/8] Importing libraries...")
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ===== TOKENIZER =====
print("\n[2/8] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  Tokenizer loaded: {tokenizer.name_or_path}")

# ===== MODEL (No quantization for MPS) =====
print("\n[3/8] Loading model (float32 for MPS)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    device_map={"": "mps"},
    local_files_only=True,
    trust_remote_code=True,
)
model.config.use_cache = False

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
print(f"  Model loaded: {BASE_MODEL_PATH}")
print(f"  Model dtype: {model.dtype}")

# ===== LORA =====
print("\n[4/8] Configuring LoRA adapter...")
lora_config = LoraConfig(
    r=8,  # Reduced rank for memory
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===== DATASET =====
print("\n[5/8] Loading and formatting dataset...")
dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train"
)
print(f"  Dataset size: {len(dataset)} examples")

def format_ccp_example(example):
    """Format training example for CCP: GTM prompt -> Intent IR JSON"""
    prompt = example["gtm_prompt"].strip()

    if "intent_ir" in example:
        ir_json = example["intent_ir"]
        if isinstance(ir_json, str):
            ir_output = ir_json
        else:
            ir_output = json.dumps(ir_json)  # Compact JSON to save tokens
    else:
        ir_output = example.get("response", "{}").strip()

    # Shorter format to fit in context
    text = f"### GTM Prompt:\n{prompt}\n\n### Intent IR:\n{ir_output}"
    return {"text": text}

dataset = dataset.map(format_ccp_example)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )

tokenized_dataset = dataset.map(
    tokenize,
    remove_columns=dataset.column_names
)
print(f"  Tokenized: {len(tokenized_dataset)} examples")

# ===== TRAINING =====
print("\n[6/8] Setting up trainer...")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    optim="adamw_torch",
    bf16=False,
    fp16=False,
    dataloader_pin_memory=False,  # Important for MPS
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print("\n[7/8] Starting training...")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient accumulation: {GRAD_ACCUM_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Total steps: {len(tokenized_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRAD_ACCUM_STEPS)}")
print("-" * 60)

trainer.train()

# ===== SAVE =====
print("\n[8/8] Saving CCP adapter...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

ccp_metadata = {
    "schema_version": IR_SCHEMA_VERSION,
    "base_model": BASE_MODEL_PATH,
    "intent_types": GTM_INTENT_IR_SCHEMA["intent_type"],
    "motions": GTM_INTENT_IR_SCHEMA["motion"],
    "role_assumptions": GTM_INTENT_IR_SCHEMA["role_assumption"],
}

with open(os.path.join(OUTPUT_DIR, "ccp_metadata.json"), "w") as f:
    json.dump(ccp_metadata, f, indent=2)

print(f"\nCCP adapter saved to {OUTPUT_DIR}")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
