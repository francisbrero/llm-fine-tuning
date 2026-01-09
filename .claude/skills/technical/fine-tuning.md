---
description: LLM fine-tuning patterns with QLoRA, PEFT, and Transformers
globs:
  - "*.ipynb"
  - "**/*fine_tune*.py"
alwaysApply: false
---

# Fine-Tuning Skill

## Overview
Use this skill when working on model fine-tuning, LoRA adapters, or training configuration.

## Key Concepts

### QLoRA Configuration
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                       # LoRA rank
    lora_alpha=32,              # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Arguments (Mac MPS)
```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./ccp-adapter",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Small for memory
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=False,                      # No fp16 on macOS
    logging_steps=10,
    save_steps=100,
    use_mps_device=True,             # Metal acceleration
)
```

### Chain-of-Thought Format
CCP uses reasoning before output:
```
<s>[INST] {system_prompt}

User request: {gtm_prompt} [/INST]
<reasoning>
{step-by-step reasoning}
</reasoning>

<intent_ir>
{json_output}
</intent_ir></s>
```

## Common Issues

### Memory (OOM/Exit 137)
- Reduce batch_size to 1
- Use gradient_accumulation_steps instead
- Consider gradient_checkpointing=True

### MPS Compatibility
- No fp16 on macOS - use float32
- Some operations fall back to CPU (expected)

## Key Files
- `fine_tune_llm_mac_mps.ipynb` - Main training notebook
- `ccp-adapter/` - Saved LoRA adapter
- `models/phi-2/` - Base model

## Commands
```bash
# Run training notebook
jupyter notebook fine_tune_llm_mac_mps.ipynb

# Test inference
python simple_test.py
```

## Resources
- PEFT docs: https://huggingface.co/docs/peft
- Transformers docs: https://huggingface.co/docs/transformers
- PRD.md - Architecture overview
