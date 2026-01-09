---
description: Common issues and solutions for CCP development
globs:
  - "**/*.py"
  - "**/*.ipynb"
alwaysApply: false
---

# Runbook: Troubleshooting

## Overview
Common issues and solutions when working with CCP.

## Memory Issues

### Exit Code 137 (OOM Killed)

**Symptoms:**
- Process killed during training or inference
- Exit code 137
- System becomes unresponsive

**Solutions:**

1. **Reduce batch size**
   ```python
   per_device_train_batch_size=1
   gradient_accumulation_steps=16  # Compensate
   ```

2. **Use gradient checkpointing**
   ```python
   model.gradient_checkpointing_enable()
   ```

3. **Close other applications**
   - Browsers (Chrome uses lots of RAM)
   - Docker containers
   - Other Python processes

4. **Run one model at a time**
   ```bash
   # Instead of --model both
   python run_evals.py --model finetuned
   python run_evals.py --model base
   ```

5. **Use quick eval**
   ```bash
   python run_evals.py --quick  # 5 examples vs 30
   ```

### MPS Memory Fragmentation

**Symptoms:**
- "MPS backend out of memory" errors
- Memory usage keeps growing

**Solutions:**
```python
import torch
torch.mps.empty_cache()
```

Or restart Python process between runs.

## Training Issues

### Loss Not Decreasing

**Possible Causes:**
1. Learning rate too low/high
2. Data quality issues
3. Insufficient training data

**Solutions:**
1. Try learning rate: 1e-4, 2e-4, 5e-4
2. Validate training data format
3. Add more diverse examples

### Training Very Slow

**Symptoms:**
- Hours per epoch
- Low GPU utilization

**Solutions:**
1. Verify MPS is being used:
   ```python
   print(torch.backends.mps.is_available())  # Should be True
   ```

2. Check device assignment:
   ```python
   model.to("mps")
   ```

### Model Not Saving

**Symptoms:**
- No files in `ccp-adapter/`
- Save step errors

**Solutions:**
```bash
# Check permissions
ls -la ccp-adapter/

# Create directory if missing
mkdir -p ccp-adapter
```

## Inference Issues

### Invalid JSON Output

**Symptoms:**
- JSONDecodeError
- Empty predictions
- Truncated output

**Possible Causes:**
1. Output truncated by max_length
2. Model not following format
3. Training data format issues

**Solutions:**
1. Increase max_new_tokens:
   ```python
   model.generate(..., max_new_tokens=500)
   ```

2. Check training data format consistency

3. Add more format examples to training

### Wrong Schema Fields

**Symptoms:**
- `role_scope` instead of `role_assumption`
- Missing required fields
- Extra unexpected fields

**Root Cause:**
Training data schema inconsistency (see GitHub Issue #5)

**Solution:**
Audit and fix training data:
```bash
# Find non-standard fields
grep -n '"role_scope"' data/ccp_training_with_reasoning.jsonl
grep -n '"time_scope"' data/ccp_training_with_reasoning.jsonl
```

### Slow Inference (~77s/example)

**Context:**
This is expected on CPU/MPS without optimization.

**Future Improvements:**
- Use GPU (CUDA)
- Quantize for inference
- Use llama.cpp or similar

## Evaluation Issues

### No Results File Created

**Check:**
```bash
ls -la eval_results/
```

**Solutions:**
```bash
# Create directory
mkdir -p eval_results

# Check permissions
chmod 755 eval_results
```

### Eval Hangs

**Symptoms:**
- No progress for minutes
- No output

**Solutions:**
1. Add verbose flag:
   ```bash
   python run_evals.py --verbose
   ```

2. Check for infinite loops in model output

3. Add timeout to inference

## Data Issues

### Invalid JSONL

**Symptoms:**
- JSONDecodeError on load
- Training fails to start

**Diagnosis:**
```bash
python -c "
import json
with open('data/ccp_training_with_reasoning.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"
```

**Fix:**
Edit the problematic line or regenerate.

### Missing Fields

**Symptoms:**
- KeyError during training
- Incomplete outputs

**Diagnosis:**
```bash
python -c "
import json
required = {'gtm_prompt', 'reasoning', 'intent_ir'}
with open('data/ccp_training_with_reasoning.jsonl') as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        missing = required - set(data.keys())
        if missing:
            print(f'Line {i}: missing {missing}')
"
```

## Environment Issues

### Import Errors

**Symptoms:**
- ModuleNotFoundError
- ImportError

**Solutions:**
```bash
# Activate venv
source .venv/bin/activate

# Reinstall dependencies
pip install torch transformers peft bitsandbytes datasets accelerate
```

### MPS Not Available

**Check:**
```python
import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
```

**If False:**
- Requires macOS 12.3+
- Requires M1/M2/M3 chip
- Update PyTorch: `pip install --upgrade torch`

## Quick Diagnostic Commands

```bash
# System info
uname -a
python --version
pip show torch transformers peft

# Memory status
vm_stat | head -5

# GPU status (macOS)
system_profiler SPDisplaysDataType | grep -A5 "Chipset Model"

# Disk space
df -h .

# Process status
ps aux | grep python
```

## Resources
- GitHub Issues - Known problems
- EVAL_RESULTS.md - Performance tracking
- PRD.md - Architecture constraints
