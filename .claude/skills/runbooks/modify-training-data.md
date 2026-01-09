---
description: Steps for modifying CCP training data
globs:
  - "data/**/*.jsonl"
alwaysApply: false
---

# Runbook: Modify Training Data

## Overview
Use this runbook when adding, editing, or fixing training examples for CCP.

## Prerequisites
- Virtual environment activated: `source .venv/bin/activate`
- Understanding of GTM Intent IR schema (see `.claude/skills/reference/ccp-architecture.md`)

## Training Data Location
```
data/ccp_training_with_reasoning.jsonl
```

## Example Format

Each line is a JSON object:
```json
{
  "gtm_prompt": "Help me prepare for my QBR",
  "reasoning": "- This is a forecast review request\n- QBR = Quarterly Business Review\n- User needs performance summary",
  "intent_ir": {
    "intent_type": "forecast_review",
    "motion": "expansion",
    "role_assumption": "sales_rep",
    "account_scope": "existing",
    "time_horizon": "this_quarter",
    "output_format": "summary",
    "confidence_scores": {
      "intent_type": 0.9,
      "motion": 0.8,
      "role_assumption": 0.7,
      "account_scope": 0.8,
      "time_horizon": 0.9
    },
    "clarification_needed": false
  }
}
```

## Step-by-Step: Add New Examples

### 1. Identify Gap
Check eval results for missed cases:
```bash
python eval_history.py --detail latest
```

### 2. Write Example
Create properly formatted JSON with:
- [ ] `gtm_prompt` - Natural user input
- [ ] `reasoning` - Step-by-step inference logic
- [ ] `intent_ir` - Exact schema fields

### 3. Validate JSON
```bash
# Check single example
echo '{"gtm_prompt": "test", ...}' | python -m json.tool

# Validate entire file
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

### 4. Check Schema Consistency
All examples MUST use these exact field names:
- `intent_type` (NOT `intent`)
- `motion` (NOT `inferred_motion`)
- `role_assumption` (NOT `role_scope`, `inferred_role`)
- `account_scope` (NOT `churn_risk_scope`, `view_scope`)
- `time_horizon` (NOT `time_scope`, `renewal_time_horizon`)

### 5. Add to File
Append to training file:
```bash
echo '{"gtm_prompt": "...", ...}' >> data/ccp_training_with_reasoning.jsonl
```

### 6. Count Examples
```bash
wc -l data/ccp_training_with_reasoning.jsonl
```

## Step-by-Step: Fix Schema Issues

### 1. Audit Current Schema
```bash
# Find unique field names in intent_ir
python -c "
import json
fields = set()
with open('data/ccp_training_with_reasoning.jsonl') as f:
    for line in f:
        data = json.loads(line)
        fields.update(data.get('intent_ir', {}).keys())
print(sorted(fields))
"
```

### 2. Find Non-Standard Fields
```bash
grep -n '"role_scope"' data/ccp_training_with_reasoning.jsonl
grep -n '"time_scope"' data/ccp_training_with_reasoning.jsonl
grep -n '"inferred_' data/ccp_training_with_reasoning.jsonl
```

### 3. Replace Field Names
Use sed or Python script to standardize.

### 4. Validate After Changes
```bash
python -c "
import json
required = {'intent_type', 'motion', 'role_assumption', 'account_scope', 'time_horizon'}
with open('data/ccp_training_with_reasoning.jsonl') as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        ir = data.get('intent_ir', {})
        missing = required - set(ir.keys())
        if missing:
            print(f'Line {i}: missing {missing}')
"
```

## After Modifying

1. **Retrain Model**
   See `.claude/skills/runbooks/run-training.md`

2. **Run Evaluation**
   ```bash
   python run_evals.py --quick
   ```

3. **Compare Results**
   ```bash
   python eval_history.py --compare 2
   ```

## Common Mistakes

| Mistake | Correct |
|---------|---------|
| `"intent": "..."` | `"intent_type": "..."` |
| `"role_scope": "..."` | `"role_assumption": "..."` |
| Missing confidence_scores | Add `"confidence_scores": {...}` |
| Inconsistent casing | Use snake_case for all fields |

## Resources
- GitHub Issue #5 - Schema standardization task
- `.claude/skills/technical/gtm-domain.md` - Valid field values
