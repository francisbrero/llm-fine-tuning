---
description: Steps for running CCP fine-tuning
globs:
  - "**/*fine_tune*.ipynb"
  - "**/*fine_tune*.py"
alwaysApply: false
---

# Runbook: Run Training

## Overview
Use this runbook to fine-tune CCP on updated training data.

## Prerequisites
- [ ] macOS with M-series chip (MPS support)
- [ ] 16GB+ RAM (24GB recommended)
- [ ] ~10GB disk space
- [ ] Virtual environment activated

## Pre-Training Checklist

### 1. Activate Environment
```bash
source .venv/bin/activate
```

### 2. Verify Dependencies
```bash
pip list | grep -E "torch|transformers|peft|bitsandbytes"
```

### 3. Verify Base Model
```bash
ls -la models/phi-2/
# Should contain: config.json, model.safetensors, tokenizer files
```

### 4. Verify Training Data
```bash
wc -l data/ccp_training_with_reasoning.jsonl
# Should be 1000+ examples for good results

# Validate JSON
python -c "
import json
with open('data/ccp_training_with_reasoning.jsonl') as f:
    for i, line in enumerate(f, 1):
        json.loads(line)
print(f'Validated {i} examples')
"
```

### 5. Check System Resources
```bash
# Memory
vm_stat | head -5

# Disk space
df -h .
```

## Training Steps

### 1. Open Notebook
```bash
jupyter notebook fine_tune_llm_mac_mps.ipynb
```

### 2. Run Cells in Order
1. **Imports** - Load libraries
2. **Configuration** - Set hyperparameters
3. **Load Data** - Load and format training data
4. **Load Model** - Load base model with quantization
5. **Apply LoRA** - Add LoRA adapters
6. **Train** - Run training loop
7. **Save** - Save adapter to `ccp-adapter/`

### 3. Monitor Training
Watch for:
- Loss decreasing over time
- No OOM errors
- Reasonable training speed

### 4. Expected Output
```
Training loss: 2.5 → 0.8 over epochs
Saved adapter to: ./ccp-adapter/
```

## Troubleshooting

### OOM / Exit 137
Reduce batch size in training config:
```python
per_device_train_batch_size=1
gradient_accumulation_steps=16  # Increase this
```

### MPS Errors
Some operations fall back to CPU (normal):
```
UserWarning: MPS: no support for int64 type, using int32
```

### Slow Training
- Ensure MPS is being used (not CPU-only)
- Check Activity Monitor for GPU usage

### Loss Not Decreasing
- Learning rate may be too low/high
- Training data may have issues
- Try different hyperparameters

## Post-Training

### 1. Verify Adapter Saved
```bash
ls -la ccp-adapter/
# Should contain: adapter_config.json, adapter_model.safetensors
```

### 2. Quick Test
```bash
python simple_test.py
```

### 3. Run Evaluation
```bash
python run_evals.py --quick
```

### 4. Compare to Previous
```bash
python eval_history.py --compare 2
```

## Training Configuration Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base Model | phi-2 | 2.7B params |
| LoRA r | 16 | Rank |
| LoRA alpha | 32 | Scaling |
| Target modules | q_proj, v_proj | Attention |
| Batch size | 1 | Memory constrained |
| Gradient accum | 8-16 | Effective batch size |
| Learning rate | 2e-4 | Standard for LoRA |
| Epochs | 3 | May need more |

## Resources
- `fine_tune_llm_mac_mps.ipynb` - Training notebook
- `.claude/skills/technical/fine-tuning.md` - Technical details
- PRD.md - Architecture overview
