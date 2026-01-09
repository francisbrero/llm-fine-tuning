---
description: Model evaluation and metrics tracking for CCP
globs:
  - "**/eval*.py"
  - "**/eval_results/**"
  - "data/ccp_eval.jsonl"
alwaysApply: false
---

# Evaluation Skill

## Overview
Use this skill when running evaluations, analyzing results, or tracking model performance over time.

## Key Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **JSON Valid Rate** | % of outputs that are valid JSON | 95%+ |
| **Field Accuracy** | % of fields that exactly match expected | 70%+ |
| **Intent Type Accuracy** | % of correct intent_type values | 80%+ |

## Running Evaluations

### Quick Eval (5 examples)
```bash
python run_evals.py --quick
```

### Full Eval (30 examples)
```bash
python run_evals.py --model finetuned
```

### Compare Base vs Fine-tuned
```bash
python run_evals.py --model both
```

### View History
```bash
python eval_history.py              # Summary
python eval_history.py --compare 2  # Compare last 2 runs
python eval_history.py --detail latest  # Detailed view
python eval_history.py --export csv # Export to CSV
```

## Evaluation Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `standard` | Clear GTM requests | "Show me my pipeline" |
| `slang` | B2B colloquialisms | "Who's ghosting us?" |
| `ambiguous` | Needs clarification | "How are we doing?" |
| `adversarial` | Non-GTM or edge cases | "What's the weather?" |

## Difficulty Levels

| Difficulty | Description |
|------------|-------------|
| `easy` | Direct, unambiguous |
| `medium` | Some inference needed |
| `hard` | Heavy context collapse required |

## Result Files

```
eval_results/
├── eval_finetuned_YYYYMMDD_HHMMSS.json  # Timestamped results
├── eval_base_YYYYMMDD_HHMMSS.json       # Base model results
└── eval_finetuned_latest.json           # Symlink to latest
```

## Result Format
```json
{
  "model_type": "finetuned",
  "timestamp": "2026-01-09T08:23:31",
  "aggregate": {
    "json_valid_rate": 0.8,
    "field_accuracy": 0.04,
    "by_category": {...},
    "by_difficulty": {...},
    "by_field": {...}
  },
  "examples": [...]
}
```

## Interpreting Results

### Low JSON Valid Rate (<80%)
- Model not following output format
- Check training data format consistency
- May need more training examples

### Low Field Accuracy (<50%)
- Schema mismatch between training and eval
- Audit `ccp_training_with_reasoning.jsonl` for field name consistency
- See GitHub issue #5

### High Intent Accuracy, Low Field Accuracy
- Model understands intent but outputs wrong schema
- Focus on standardizing field names in training data

## Key Files
- `run_evals.py` - Main evaluation runner
- `eval_history.py` - Historical results viewer
- `data/ccp_eval.jsonl` - Test cases (30 examples)
- `eval_results/` - Result storage
- `EVAL_RESULTS.md` - Latest analysis

## Resources
- GitHub Issue #5 - Schema standardization task
