# Semantic Evaluation Framework for CCP

## Problem with Current Eval

**Current approach (mechanical):**
- Tests exact field matching: `predicted["motion"] == "expansion"` ✓/✗
- Treats all field errors equally
- Doesn't test understanding of GTM context
- Doesn't validate business logic

**Why this is wrong:**
- We fine-tuned the model to output exact field names
- High scores on this eval just mean the model memorized the schema
- Doesn't test if the model actually understands GTM semantics
- Misses the real business value: **tool selection accuracy**

## Real Success Metrics (from PRD Section 12)

1. **Reduction in downstream tool call failures**
2. **Improvement in first-pass execution accuracy**
3. **Reduction in unnecessary clarification questions**
4. **Consistency of intent interpretation across semantically similar prompts**

## Proposed Semantic Evaluation Framework

### Level 1: Context Understanding Tests

Test if the model understands **implicit GTM context**:

#### Test: Account Scope Understanding
```
Prompt: "Show me my best accounts"
Expected understanding: ✓ Existing accounts (not net_new)
Bad interpretation: Suggesting net_new/prospecting accounts

Prompt: "Help me with prospecting"
Expected understanding: ✓ Net new accounts
Bad interpretation: Existing account expansion
```

#### Test: Motion Understanding
```
Prompt: "QBR prep"
Expected understanding: ✓ Review existing accounts (expansion/renewal)
Bad interpretation: Outbound prospecting motion

Prompt: "Pipeline from last webinar"
Expected understanding: ✓ Inbound leads
Bad interpretation: Existing account motion
```

#### Test: Role Context Understanding
```
Prompt: "Which accounts haven't been touched in 30 days?"
Expected understanding: ✓ CS/AM role concerned with churn
Bad interpretation: Sales rep prospecting

Prompt: "Show me SQLs"
Expected understanding: ✓ SDR/Sales role looking at leads
Bad interpretation: CS looking at existing accounts
```

#### Test: Slang/Jargon Understanding
```
Prompt: "Who's ghosting us?"
Expected understanding: ✓ Accounts going silent = churn risk
Bad interpretation: Pipeline/leads status

Prompt: "Show me SQLs from the webinar"
Expected understanding: ✓ Sales Qualified Leads (not database queries)
Bad interpretation: Literal SQL queries
```

### Level 2: Tool Selection Validation

Test if the Intent IR would lead to **correct tool calls**:

```
Prompt: "Show me accounts at risk"

Expected tool selection path:
✓ churn_risk_assessment → ICP filter → risk scoring tools
✗ account_discovery → prospecting tools
✗ pipeline_analysis → deal stage tools

Scoring:
- Would this IR select the RIGHT category of tools? (pass/fail)
- Would it avoid selecting WRONG categories? (pass/fail)
```

### Level 3: Semantic Similarity (not exact match)

Instead of exact field matching, use **semantic equivalence**:

```python
# Current (wrong)
intent_type_match = (predicted == "churn_risk_assessment")

# Proposed (semantic)
intent_type_acceptable = predicted in [
    "churn_risk_assessment",
    "churn_risk_analysis",
    "at_risk_identification"
] or semantic_similarity(predicted, expected) > 0.85
```

Categories of acceptable matches:
- **Exact match**: `churn_risk_assessment` == `churn_risk_assessment`
- **Semantic equivalent**: `churn_risk_assessment` ≈ `churn_risk_analysis`
- **Subcategory**: `expansion_identification` is valid for `account_discovery`
- **Supercategory**: `pipeline_analysis` is valid for `deal_stage_review`

### Level 4: Critical vs Non-Critical Field Errors

Weight errors by **business impact**:

**Critical errors (breaks tool selection):**
- Wrong intent_type: `churn_risk` vs `account_discovery` → Wrong tools
- Wrong account_scope: `net_new` vs `existing` → Wrong data filter
- Wrong motion: `outbound` vs `churn_prevention` → Wrong workflow

**Minor errors (degraded but functional):**
- Wrong time_horizon: `this_month` vs `this_quarter` → Slightly wrong filter
- Wrong output_format: `list` vs `summary` → UI preference
- Missing confidence_scores → Observability loss but functional

**Scoring:**
```
Critical field accuracy: 80%+ (must get tool selection right)
Overall semantic accuracy: 70%+ (including minor fields)
```

### Level 5: Human Business Logic Validation

For each eval example, define **success criteria** in business terms:

```json
{
  "gtm_prompt": "Who's ghosting us?",
  "success_criteria": {
    "must_understand": [
      "This is about EXISTING accounts (not prospects)",
      "Ghosting = going silent = churn risk signal",
      "User wants to identify at-risk accounts"
    ],
    "must_avoid": [
      "Treating as pipeline/leads question",
      "Treating as literal 'ghost' reference",
      "Suggesting prospecting tools"
    ],
    "acceptable_outputs": {
      "intent_type": ["churn_risk_assessment", "engagement_analysis", "account_health_check"],
      "motion": ["churn_prevention", "retention"],
      "account_scope": ["existing"]
    }
  }
}
```

## Implementation Plan

### Phase 1: Enhanced Eval Data Format
Add semantic annotations to eval dataset:

```json
{
  "gtm_prompt": "Show me my best accounts",
  "expected": {
    "intent_type": "account_discovery",
    "motion": "expansion",
    "account_scope": "existing"
  },
  "semantic_validation": {
    "critical_fields": ["intent_type", "motion", "account_scope"],
    "intent_type_acceptable": ["account_discovery", "account_ranking", "top_accounts"],
    "motion_acceptable": ["expansion", "upsell"],
    "must_understand": [
      "Existing accounts (not prospecting)",
      "Looking for expansion opportunities"
    ],
    "tool_categories_correct": ["account_scoring", "icp_filter", "engagement_data"],
    "tool_categories_wrong": ["lead_gen", "prospecting", "outbound"]
  }
}
```

### Phase 2: Semantic Scoring Functions

```python
def score_semantic_understanding(predicted_ir, eval_example):
    """Score based on GTM understanding, not exact field match."""

    score = {
        "critical_fields_correct": 0.0,  # Must get these right
        "semantic_match": 0.0,            # Acceptable variations
        "tool_selection_valid": False,    # Would select right tools
        "context_understood": False,      # Got the implicit meaning
    }

    # Test critical field understanding
    critical_correct = 0
    for field in eval_example["semantic_validation"]["critical_fields"]:
        if is_semantically_acceptable(
            predicted_ir[field],
            eval_example["semantic_validation"][f"{field}_acceptable"]
        ):
            critical_correct += 1

    score["critical_fields_correct"] = critical_correct / len(critical_fields)

    # Test tool selection
    predicted_tools = infer_tools_from_ir(predicted_ir)
    correct_tools = eval_example["semantic_validation"]["tool_categories_correct"]
    wrong_tools = eval_example["semantic_validation"]["tool_categories_wrong"]

    score["tool_selection_valid"] = (
        any(t in predicted_tools for t in correct_tools) and
        not any(t in predicted_tools for t in wrong_tools)
    )

    return score
```

### Phase 3: Comparative Analysis

Run both evals side-by-side:

```
MECHANICAL EVAL (syntax):
  JSON Valid: 100%
  Field Accuracy: 41% (exact match)

SEMANTIC EVAL (understanding):
  Critical Field Accuracy: 65%
  Tool Selection Accuracy: 78%
  Context Understanding: 82%
  Overall Business Value: 75%
```

## Success Criteria

The model is **production-ready** when:

1. **Critical field accuracy > 80%** (tool selection works)
2. **Tool selection accuracy > 85%** (picks right tool categories)
3. **Context understanding > 75%** (gets implicit GTM meaning)
4. **Zero catastrophic failures** (never picks completely wrong tool category)

## Appendix: Example Mappings

### Tool Category Mapping

```python
INTENT_TO_TOOLS = {
    "churn_risk_assessment": ["risk_scoring", "engagement_analysis", "health_metrics"],
    "account_discovery": ["account_search", "icp_filter", "firmographic_data"],
    "pipeline_analysis": ["deal_stage", "forecast", "pipeline_health"],
    "lead_prioritization": ["lead_scoring", "intent_signals", "engagement_tracking"],
}

MOTION_TO_TOOLS = {
    "churn_prevention": ["retention_tools", "engagement_tools", "health_monitoring"],
    "expansion": ["upsell_signals", "expansion_opportunities", "account_scoring"],
    "outbound": ["prospecting", "lead_gen", "cold_outreach"],
    "inbound": ["lead_routing", "response_management", "qualification"],
}
```

This ensures we test what matters: **Does the IR lead to the right actions?**
