---
description: Steps for running CCP evaluation
globs:
  - "**/run_evals.py"
  - "**/eval*.py"
  - "eval_results/**"
alwaysApply: false
---

# Runbook: Run Evaluation

## Overview
Use this runbook to evaluate CCP model performance and track progress.

## Prerequisites
- [ ] Virtual environment activated
- [ ] Fine-tuned model available at `ccp-adapter/`
- [ ] Eval data at `data/ccp_eval.jsonl`

## Quick Evaluation (5 examples)

```bash
python run_evals.py --quick
```

Output:
- JSON valid rate
- Field accuracy
- Per-category breakdown

## Full Evaluation (30 examples)

```bash
python run_evals.py --model finetuned
```

Options:
- `--model finetuned` - Test fine-tuned model only
- `--model base` - Test base model only
- `--model both` - Compare both models
- `--verbose` - Show per-example details
- `--quick` - Run only 5 examples

## View Historical Results

### Summary of All Runs
```bash
python eval_history.py
```

### Compare Last N Runs
```bash
python eval_history.py --compare 2
```

### Detailed View of Latest
```bash
python eval_history.py --detail latest
```

### Export to CSV
```bash
python eval_history.py --export csv
```

## Understanding Results

### JSON Valid Rate
- **95%+**: Good - model follows format
- **80-95%**: Acceptable - some format issues
- **<80%**: Problem - review training format

### Field Accuracy
- **70%+**: Target - model understands schema
- **50-70%**: Needs work - schema inconsistencies
- **<50%**: Critical - likely training data issue

### By Category
- `standard`: Basic GTM requests
- `slang`: B2B colloquialisms (harder)
- `ambiguous`: Unclear requests
- `adversarial`: Edge cases

### By Field
Shows which fields need improvement:
- Low `intent_type` accuracy = core issue
- Low other fields = schema mismatch

## Result Files

```
eval_results/
├── eval_finetuned_20260109_082956.json  # Timestamped
├── eval_finetuned_latest.json           # Latest link
└── eval_base_*.json                     # Base model runs
```

## Troubleshooting

### OOM / Exit 137
Eval loads model into memory. Try:
```bash
# Run with fewer examples
python run_evals.py --quick

# Or close other applications
```

### Slow Inference
CPU inference is slow (~77s/example). Normal for unoptimized setup.

### Missing Eval Data
```bash
# Check eval file exists
ls -la data/ccp_eval.jsonl
wc -l data/ccp_eval.jsonl  # Should be ~30 lines
```

### Invalid JSON in Results
Model output may be malformed. Check:
```bash
python eval_history.py --detail latest
```

Look at "WORST PERFORMING EXAMPLES" section.

## After Evaluation

### If Results Improved
1. Document changes made
2. Consider full eval if was quick run
3. Update EVAL_RESULTS.md

### If Results Degraded
1. Check recent training data changes
2. Review eval_history for regression
3. May need to rollback adapter

### If Results Unchanged
1. Changes may not have been significant
2. May need more training data
3. Consider different approach

## Evaluation Categories Explained

| Category | Count | Purpose |
|----------|-------|---------|
| `standard` | 10 | Basic competency |
| `slang` | 12 | B2B vocabulary |
| `ambiguous` | 3 | Clarification handling |
| `adversarial` | 5 | Edge case robustness |

## Resources
- `run_evals.py` - Evaluation runner
- `eval_history.py` - History viewer
- `EVAL_RESULTS.md` - Latest analysis
- `.claude/skills/technical/evaluation.md` - Technical details
