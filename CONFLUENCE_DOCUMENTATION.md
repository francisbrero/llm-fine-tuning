# Phoenix Context Collapse Parser (CCP)

> **A domain-specific language model for parsing ambiguous GTM prompts into structured, executable intent**

---

## Executive Summary

The Phoenix Context Collapse Parser (CCP) is a fine-tuned ~3B parameter language model that serves as the intelligent front-end for GTM (Go-To-Market) tooling. It transforms natural language requests from sales, marketing, and customer success teams into structured, machine-readable intent that downstream systems can execute reliably.

**Key Value Proposition**: Enterprise users communicate in shorthand with implicit context. CCP bridges the gap between human ambiguity and machine precision.

| Attribute | Value |
|-----------|-------|
| Model Size | ~2.7B parameters (Phi-2 base) |
| Fine-tuning Method | LoRA (Low-Rank Adaptation) |
| Training Examples | 1,278 with Chain-of-Thought reasoning |
| Inference Target | Sub-100ms |
| Hardware Requirements | Runs on commodity laptops (16GB+ RAM) |

---

## Problem Statement

### The Challenge

When enterprise users interact with GTM tools, they use natural language filled with:

- **Implicit role context**: "Show me my pipeline" (Am I a rep? A manager? RevOps?)
- **Ambiguous scope**: "Best accounts" (Best fit? Highest ARR? Most engaged?)
- **Industry jargon**: "Who's likely to churn?" (Which time horizon? Which segment?)
- **Assumed knowledge**: "Hit the number this quarter" (What quota? What metric?)

### Why Existing Solutions Fall Short

| Approach | Limitation |
|----------|------------|
| **Static system prompts** | Cannot handle situational, probabilistic context |
| **Deterministic rules** | Brittle; fail on edge cases and novel phrasing |
| **General-purpose LLMs** | Lack GTM domain expertise; hallucinate tool calls |
| **Keyword matching** | Miss semantic meaning; can't infer intent |

### CCP's Solution

CCP acts as a **probabilistic parsing layer** that:

1. Interprets ambiguous GTM prompts
2. Infers missing context based on domain knowledge
3. Outputs structured intent with confidence scores
4. Flags when clarification is needed

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                               │
│              "Show me accounts likely to churn"                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CCP (Fine-tuned Phi-2)                        │
│                                                                  │
│  1. Parse natural language                                       │
│  2. Infer role, motion, scope from context                       │
│  3. Apply GTM domain knowledge                                   │
│  4. Generate structured GTM Intent IR                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GTM Intent IR (JSON)                       │
│                                                                  │
│  {                                                               │
│    "intent_type": "churn_risk_assessment",                       │
│    "motion": "churn_prevention",                                 │
│    "role_assumption": "cs",                                      │
│    "account_scope": "existing",                                  │
│    "confidence_scores": { ... },                                 │
│    "assumptions_applied": [ ... ]                                │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               Phoenix MCP Tools / Downstream LLM                 │
│                                                                  │
│  Execute structured intent with appropriate tool calls           │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principle

**CCP outputs selectors, not definitions.** ICP (Ideal Customer Profile) definitions do NOT live in the model. CCP emits selectors like `icp_selector: "segment_enterprise"`, and downstream Phoenix tools resolve these to customer-specific definitions.

---

## GTM Intent IR Specification

The GTM Intent IR (Intermediate Representation) is the structured output format from CCP.

### Core Fields

| Field | Type | Description | Example Values |
|-------|------|-------------|----------------|
| `intent_type` | string | Primary intent category | `account_discovery`, `pipeline_analysis`, `churn_risk_assessment`, `forecast_review`, `expansion_identification` |
| `motion` | string | GTM motion | `outbound`, `inbound`, `expansion`, `renewal`, `churn_prevention` |
| `role_assumption` | string | Inferred user role | `sales_rep`, `sales_manager`, `sdr`, `cs`, `revops`, `exec` |
| `account_scope` | string | Which accounts | `net_new`, `existing`, `churned`, `all` |
| `time_horizon` | string | Time scope | `immediate`, `this_week`, `this_month`, `this_quarter`, `this_year` |
| `geography_scope` | string/null | Geographic filter | `EMEA`, `NA`, `APAC`, or `null` for global |
| `output_format` | string | Desired output | `list`, `summary`, `detailed`, `export`, `visualization` |
| `icp_selector` | string | ICP reference | `default`, `product_analytics`, `segment_enterprise` |
| `icp_resolution_required` | boolean | Needs ICP lookup | `true` / `false` |

### Confidence & Context Fields

| Field | Type | Description |
|-------|------|-------------|
| `confidence_scores` | object | 0.0-1.0 confidence per field |
| `assumptions_applied` | array | Human-readable list of inferences made |
| `clarification_needed` | boolean | Flag if request is too ambiguous |

### Confidence Score Interpretation

| Range | Meaning |
|-------|---------|
| 0.9 - 1.0 | Explicitly stated in prompt |
| 0.7 - 0.9 | Strong contextual signal |
| 0.5 - 0.7 | Reasonable inference |
| 0.3 - 0.5 | Weak signal, may need confirmation |
| < 0.3 | Guessing; clarification recommended |

### Example Output

**Input**: "Show me my best accounts"

**Output**:
```json
{
  "intent_type": "account_discovery",
  "motion": "expansion",
  "role_assumption": "sales_rep",
  "account_scope": "existing",
  "icp_selector": "default",
  "icp_resolution_required": false,
  "geography_scope": null,
  "time_horizon": "this_quarter",
  "output_format": "list",
  "confidence_scores": {
    "intent_type": 0.85,
    "motion": 0.7,
    "role_assumption": 0.6,
    "account_scope": 0.8,
    "time_horizon": 0.5
  },
  "assumptions_applied": [
    "Assumed 'best' means high-fit existing accounts",
    "Assumed sales rep role from 'my accounts'",
    "Assumed expansion motion for existing accounts"
  ],
  "clarification_needed": false
}
```

---

## Supported Intent Types

CCP recognizes 25 distinct intent types across GTM workflows:

### Discovery & Research
- `account_discovery` - Find accounts matching criteria
- `contact_discovery` - Find contacts/personas
- `competitor_analysis` - Analyze competitive landscape

### Pipeline & Forecasting
- `pipeline_analysis` - Analyze deal pipeline
- `forecast_review` - Review revenue forecasts
- `deal_inspection` - Deep-dive on specific deals
- `pipeline_generation` - Identify pipeline sources

### Customer Success
- `churn_risk_assessment` - Identify at-risk accounts
- `expansion_identification` - Find upsell/cross-sell opportunities
- `health_scoring` - Assess customer health
- `renewal_management` - Track upcoming renewals

### Performance & Analytics
- `performance_tracking` - Track team/individual metrics
- `territory_analysis` - Analyze territory coverage
- `conversion_analysis` - Analyze funnel conversion
- `engagement_tracking` - Track customer engagement

### Operations
- `data_enrichment` - Enrich account/contact data
- `lead_routing` - Route leads to appropriate owners
- `prioritization` - Prioritize accounts/tasks
- `reporting` - Generate reports

---

## GTM Domain Knowledge

### GTM Motions

| Motion | Description | Typical Roles |
|--------|-------------|---------------|
| **Outbound** | Prospecting net-new accounts | SDR, BDR, AE |
| **Inbound** | Responding to inbound demand | SDR, AE |
| **Expansion** | Growing existing customers | AE, CSM |
| **Renewal** | Retaining existing ARR | CSM, AM |
| **Churn Prevention** | Saving at-risk accounts | CSM, AM |

### Role Inference Signals

CCP uses contextual signals to infer user roles:

| Signal | Inferred Role |
|--------|---------------|
| "my quota", "my number", "my pipeline" | `sales_rep` |
| "team forecast", "rep performance" | `sales_manager` |
| "leads", "MQLs", "campaigns" | `marketing` or `sdr` |
| "health score", "renewal", "NPS" | `cs` |
| "funnel metrics", "attribution" | `revops` |
| "board", "investors", "company metrics" | `exec` |

### Common Shorthand & Jargon

CCP understands B2B/SaaS vocabulary:

| Term | Meaning |
|------|---------|
| "pipe" / "pipeline" | Active opportunities |
| "hit the number" | Achieve quota/target |
| "book of business" | Assigned accounts |
| "whitespace" | Untapped opportunity in existing account |
| "land and expand" | Small initial deal, then grow |
| "ARR", "MRR", "NRR" | Revenue metrics |
| "ACV", "TCV" | Contract value metrics |
| "MQL", "SQL", "SAL" | Lead qualification stages |
| "ICP" | Ideal Customer Profile |
| "POC" | Proof of Concept |

---

## Technical Implementation

### Model Stack

| Component | Technology |
|-----------|------------|
| Base Model | Microsoft Phi-2 (~2.7B parameters) |
| Fine-tuning | LoRA via PEFT library |
| Training Framework | Hugging Face Transformers |
| Hardware | macOS M-series with Metal/MPS |
| Compute Type | Float32 (MPS requirement) |

### LoRA Configuration

```python
LoraConfig(
    r=16,                          # Rank - balance between capacity and efficiency
    lora_alpha=32,                 # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Why these settings?**
- `r=16`: Sufficient capacity for domain adaptation without overfitting
- `target_modules`: Query and Value projections are where semantic information flows
- Only ~0.09% of parameters are trainable, preventing catastrophic forgetting

### Chain-of-Thought Training (v2)

CCP v2 uses Chain-of-Thought training to ensure semantic understanding over pattern matching.

**Training Format**:
```
<s>[INST] You are CCP, a GTM intent parser...

User request: Show me my best accounts [/INST]
<reasoning>
- 'my accounts' indicates personally assigned accounts, suggesting sales rep role
- 'best' is ambiguous: could mean highest ARR, best fit, or most engaged
- No time horizon mentioned, defaulting to current quarter
- Expansion motion inferred for existing customer accounts
</reasoning>
<intent_ir>
{
  "intent_type": "account_discovery",
  ...
}
</intent_ir></s>
```

**Why Chain-of-Thought?**
- v1 trained directly on JSON output, risking pattern-matching without understanding
- v2 forces the model to reason first, then output
- Improves interpretability and validates semantic learning
- Enables debugging by inspecting reasoning chains

### Training Data

| Metric | Value |
|--------|-------|
| Total Examples | 1,278 |
| With Reasoning | 100% (v2) |
| Intent Types | 25 categories |
| Average Tokens | ~500 per example |

**Distribution by Motion**:
- Expansion: 436 (34%)
- Outbound: 434 (34%)
- Inbound: 238 (19%)
- Renewal: 115 (9%)
- Churn Prevention: 55 (4%)

**Distribution by Role**:
- Sales Rep: 521 (41%)
- Sales Manager: 315 (25%)
- SDR: 137 (11%)
- CS: 83 (6%)
- RevOps: 54 (4%)
- Other: 168 (13%)

---

## Repository Structure

```
local-llm/
├── models/
│   └── phi-2/                    # Base model (~5.5GB)
├── ccp-adapter/                  # Trained LoRA weights (~20MB)
│   └── adapter_model.safetensors
├── data/
│   ├── ccp_training_with_reasoning.jsonl  # 1,278 training examples
│   └── ccp_eval.jsonl            # 30 evaluation examples
├── train_ccp_v2.py               # Training script
├── test_ccp_v2.py                # Inference testing
├── run_evals.py                  # Evaluation framework
├── eval_history.py               # Historical tracking
├── simple_test.py                # Quick A/B comparison
├── fine_tune_llm_mac_mps.ipynb   # Jupyter notebook version
├── PRD.md                        # Product requirements
├── gtm_domain_knowledge.md       # Domain reference
├── FINE_TUNING_EXPLAINED.md      # Technical explanation
└── CLAUDE.md                     # Development guidelines
```

---

## Evaluation & Testing

### Evaluation Framework

CCP includes a comprehensive evaluation system (`run_evals.py`):

**Metrics Tracked**:
- JSON validity rate
- Field accuracy (exact match per field)
- Category breakdown (standard, slang, adversarial, ambiguous)
- Difficulty breakdown (easy, medium, hard)
- Inference latency

**Evaluation Fields** (8 total):
- `intent_type`, `motion`, `role_assumption`, `account_scope`
- `time_horizon`, `geography_scope`, `output_format`, `clarification_needed`

### Test Categories

| Category | Description | Example |
|----------|-------------|---------|
| Standard | Clear GTM requests | "Show me my pipeline" |
| Slang | B2B jargon | "Who's crushing their number?" |
| Ambiguous | Multiple interpretations | "Best accounts" |
| Adversarial | Edge cases, non-GTM | "What's the weather?" |

### Running Evaluations

```bash
# Full evaluation
python run_evals.py

# Quick A/B test (base vs fine-tuned)
python simple_test.py

# Interactive inference testing
python test_ccp_v2.py
```

---

## Known Limitations & Future Work

### Current Limitations

| Issue | Description | Impact |
|-------|-------------|--------|
| **Over-confidence on adversarial inputs** | Model responds confidently to nonsense or non-GTM requests | May execute incorrect tool calls |
| **No "unknown" intent type** | Cannot explicitly flag out-of-domain requests | Forced to classify everything |
| **Limited adversarial training data** | Only ~30 adversarial examples in training | Poor uncertainty calibration |

### Planned Improvements

1. **Adversarial Training Examples** (Issue #3)
   - Add 50-100 examples covering:
     - Made-up jargon ("Show me the zorbax metrics")
     - Non-GTM requests ("What's the weather?")
     - Minimal/ambiguous prompts ("accounts")
   - New intent types: `unknown`, `out_of_domain`

2. **Confidence Calibration**
   - Ensure adversarial inputs receive low confidence (<0.3)
   - Set `clarification_needed: true` for ambiguous cases

3. **Expanded Intent Types**
   - Additional pipeline stages
   - Marketing-specific intents
   - Cross-functional workflows

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Downstream tool call failures | < 5% | Monitor Phoenix execution errors |
| First-pass execution accuracy | > 90% | Correct intent on first try |
| Unnecessary clarifications | < 10% | Avoid over-asking for context |
| Semantic consistency | > 95% | Same intent for equivalent prompts |
| Inference latency | < 100ms | Production timing |

---

## Getting Started

### Prerequisites

- Python 3.11+
- macOS M-series (for MPS) or CUDA GPU
- 16GB+ RAM (24GB recommended)
- ~10GB disk space

### Setup

```bash
# Clone repository
git clone <repo-url>
cd local-llm

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch transformers peft bitsandbytes datasets accelerate

# Download base model
huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2
```

### Running Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("./models/phi-2")
tokenizer = AutoTokenizer.from_pretrained("./models/phi-2")

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "./ccp-adapter")

# Run inference
prompt = "Show me accounts likely to churn this quarter"
# ... (see test_ccp_v2.py for full example)
```

### Training

```bash
# Run training (~7 hours on M-series Mac)
python train_ccp_v2.py
```

---

## Contributing

### Development Guidelines

1. **Read CLAUDE.md** for codebase conventions
2. **Use Chain-of-Thought format** for new training examples
3. **Include reasoning chains** that explain inference logic
4. **Test with adversarial prompts** before merging
5. **Update evaluation dataset** when adding new intent types

### Adding Training Examples

New examples should follow this format:

```json
{
  "gtm_prompt": "User's natural language request",
  "reasoning": "- Bullet points explaining inference logic\n- Each assumption documented\n- Confidence rationale included",
  "intent_ir": {
    "intent_type": "...",
    "motion": "...",
    "role_assumption": "...",
    "account_scope": "...",
    "confidence_scores": { ... },
    "assumptions_applied": [ ... ]
  }
}
```

---

## Appendix: Intent Type Reference

| Intent Type | Description | Typical Use Case |
|-------------|-------------|------------------|
| `account_discovery` | Find accounts matching criteria | "Show me enterprise accounts in EMEA" |
| `pipeline_analysis` | Analyze deal pipeline | "What's my pipeline coverage?" |
| `churn_risk_assessment` | Identify at-risk accounts | "Who's likely to churn?" |
| `forecast_review` | Review revenue forecasts | "What's our Q4 forecast?" |
| `expansion_identification` | Find growth opportunities | "Which accounts have whitespace?" |
| `performance_tracking` | Track metrics | "How's the team performing?" |
| `deal_inspection` | Analyze specific deals | "Tell me about the Acme deal" |
| `contact_discovery` | Find contacts | "Who should I reach out to?" |
| `health_scoring` | Assess customer health | "What's the health score for X?" |
| `renewal_management` | Track renewals | "Upcoming renewals this quarter" |
| `territory_analysis` | Analyze territories | "How's the West region doing?" |
| `prioritization` | Prioritize work | "Which accounts should I focus on?" |
| `engagement_tracking` | Track engagement | "Who's been active this week?" |
| `conversion_analysis` | Analyze funnels | "What's our MQL to SQL rate?" |
| `reporting` | Generate reports | "I need a pipeline report" |

---

*Last updated: January 2026*
*Version: 2.0 (Chain-of-Thought)*
