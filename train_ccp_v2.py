#!/usr/bin/env python3
"""
CCP v2 Training Script - Chain-of-Thought + JSON

This script runs the fine-tuning process for the Context Collapse Parser v2,
which uses Chain-of-Thought training to ensure the model learns GTM semantics.
"""

import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

# ===== CCP CONFIG =====
BASE_MODEL_PATH = "./models/phi-2"
DATASET_PATH = "./data/ccp_training_with_reasoning.jsonl"
OUTPUT_DIR = "./ccp-adapter"
IR_SCHEMA_VERSION = "2.0.0"

# ===== TRAINING PARAMS =====
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 16  # Increased to compensate for smaller batch
MAX_SEQ_LENGTH = 1024  # Reduced for memory efficiency on MPS

# ===== SYSTEM PROMPT =====
CCP_SYSTEM_PROMPT = """You are the Phoenix Context Collapse Parser (CCP). Your job is to transform ambiguous GTM (Go-To-Market) prompts into structured GTM Intent IR.

IMPORTANT: You must REASON about the request before producing structured output. This ensures you understand the GTM semantics, not just the JSON format.

Given a user's GTM request:
1. First, analyze the prompt in a <reasoning> section:
   - Identify explicit signals (keywords, jargon, metrics mentioned)
   - Infer implicit context (role, motion, time horizon)
   - Note any ambiguities that affect confidence
   - Explain WHY you chose each field value

2. Then, output the structured intent in an <intent_ir> section as JSON with:
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

Format your response EXACTLY as:
<reasoning>
[Your analysis here]
</reasoning>

<intent_ir>
[Valid JSON here]
</intent_ir>"""


def main():
    print("=" * 60)
    print("CCP v2 Training - Chain-of-Thought + JSON")
    print("=" * 60)

    # Check environment
    print(f"\nPython torch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Load tokenizer
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Load model (without quantization for MPS compatibility)
    print("\n[2/7] Loading model for MPS...")
    print("  Note: Using float32 without quantization (MPS doesn't support bitsandbytes 4-bit yet)")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float32,
        device_map={"": "mps"},
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    print(f"  Model loaded: {model.__class__.__name__}")

    # Apply LoRA
    print("\n[3/7] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("\n[4/7] Loading training dataset...")
    dataset = load_dataset(
        "json",
        data_files=DATASET_PATH,
        split="train"
    )
    print(f"  Loaded {len(dataset)} examples")

    # Format examples with Chain-of-Thought template
    print("\n[5/7] Formatting examples with CoT template...")
    def format_ccp_example(example):
        prompt = example["gtm_prompt"].strip()
        reasoning = example.get("reasoning", "").strip()
        if not reasoning:
            reasoning = "- Analyzing prompt for GTM signals\n- Inferring context from keywords"

        ir_json = example["intent_ir"]
        if isinstance(ir_json, str):
            ir_output = ir_json
        else:
            ir_output = json.dumps(ir_json, indent=2)

        text = f"""<s>[INST] {CCP_SYSTEM_PROMPT}

User request: {prompt} [/INST]
<reasoning>
{reasoning}
</reasoning>

<intent_ir>
{ir_output}
</intent_ir></s>"""

        return {"text": text}

    dataset = dataset.map(format_ccp_example)

    # Tokenize
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
    print(f"  Tokenized {len(tokenized_dataset)} examples")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments
    print("\n[6/7] Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        optim="adamw_torch",
        bf16=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n[7/7] Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRAD_ACCUM_STEPS})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
    print()

    # Clear MPS cache before training to free memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        print("  MPS cache cleared before training")

    trainer.train()

    # Save
    print("\n" + "=" * 60)
    print("Training complete! Saving adapter...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save metadata
    ccp_metadata = {
        "schema_version": IR_SCHEMA_VERSION,
        "base_model": BASE_MODEL_PATH,
        "training_approach": "chain_of_thought",
        "output_format": {
            "reasoning": "<reasoning>...</reasoning>",
            "intent_ir": "<intent_ir>{JSON}</intent_ir>"
        },
    }

    with open(os.path.join(OUTPUT_DIR, "ccp_metadata.json"), "w") as f:
        json.dump(ccp_metadata, f, indent=2)

    print(f"\nCCP v2 adapter saved to {OUTPUT_DIR}")
    print(f"Schema version: {IR_SCHEMA_VERSION}")
    print("Training approach: Chain-of-Thought + JSON")
    print("=" * 60)


if __name__ == "__main__":
    main()
