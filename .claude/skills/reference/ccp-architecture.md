---
description: CCP architecture and GTM Intent IR schema reference
globs:
  - "**/*.py"
  - "**/*.ipynb"
alwaysApply: false
---

# CCP Architecture Reference

## Overview
CCP (Context Collapse Parser) is a small (~3B) language model that translates ambiguous GTM prompts into structured GTM Intent IR.

## System Architecture

```
User Prompt → CCP → GTM Intent IR → Phoenix MCP Tools → Response
     ↓           ↓
  "QBR prep"   { intent_type: "forecast_review", ... }
```

## GTM Intent IR Schema

### Required Fields

```json
{
  "intent_type": "forecast_review",
  "motion": "expansion",
  "role_assumption": "sales_rep",
  "account_scope": "existing",
  "time_horizon": "this_quarter",
  "output_format": "summary",
  "confidence_scores": {
    "intent_type": 0.9,
    "motion": 0.8,
    "role_assumption": 0.7
  },
  "clarification_needed": false
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `intent_type` | string | Primary intent category |
| `motion` | string | GTM motion context |
| `role_assumption` | string | Inferred user role |
| `account_scope` | string | Account filter |
| `time_horizon` | string | Time range |
| `output_format` | string | Preferred output style |
| `confidence_scores` | object | Per-field confidence (0-1) |
| `clarification_needed` | boolean | Whether to ask for more info |

### Valid Values

**intent_type**: `forecast_review`, `pipeline_analysis`, `lead_prioritization`, `churn_risk_assessment`, `account_discovery`, `deal_coaching`, `campaign_performance`, `territory_mapping`

**motion**: `outbound`, `inbound`, `expansion`, `renewal`, `churn_prevention`

**role_assumption**: `sales_rep`, `sales_manager`, `sdr`, `cs`, `revops`, `marketing`, `exec`

**account_scope**: `net_new`, `existing`, `churned`, `all`

**time_horizon**: `this_week`, `this_month`, `this_quarter`, `next_quarter`, `this_year`

**output_format**: `summary`, `detailed`, `list`, `comparison`

## Chain-of-Thought Format

CCP outputs reasoning before JSON:

```
<reasoning>
- This is a forecast review request
- User mentions QBR = Quarterly Business Review
- Likely a sales rep or manager role
- Time horizon is this quarter
</reasoning>

<intent_ir>
{
  "intent_type": "forecast_review",
  ...
}
</intent_ir>
```

## Model Stack

| Component | Value |
|-----------|-------|
| Base Model | microsoft/phi-2 (~2.7B params) |
| Fine-tuning | QLoRA (r=16, alpha=32) |
| Targets | q_proj, v_proj |
| Quantization | 4-bit (bitsandbytes) |
| Compute | MPS (Mac Metal) |

## File Locations

| File | Purpose |
|------|---------|
| `models/phi-2/` | Base model |
| `ccp-adapter/` | LoRA adapter weights |
| `data/ccp_training_with_reasoning.jsonl` | Training data |
| `data/ccp_eval.jsonl` | Evaluation data |
| `fine_tune_llm_mac_mps.ipynb` | Training notebook |

## Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + adapter
base_model = AutoModelForCausalLM.from_pretrained("./models/phi-2")
model = PeftModel.from_pretrained(base_model, "./ccp-adapter")
tokenizer = AutoTokenizer.from_pretrained("./models/phi-2")

# Format prompt
prompt = f"<s>[INST] {system_prompt}\n\nUser request: {user_input} [/INST]"
```

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Inference latency | <100ms | ~77s (unoptimized) |
| JSON valid rate | 95%+ | 80% |
| Field accuracy | 70%+ | 4% |

## Resources
- PRD.md - Full product requirements
- gtm_domain_knowledge.md - Domain terminology
- EVAL_RESULTS.md - Latest evaluation results
