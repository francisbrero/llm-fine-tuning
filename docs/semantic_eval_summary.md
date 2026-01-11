# Semantic Evaluation Framework - Implementation Summary

## What We Built

A new evaluation system that tests **GTM context understanding and business value** rather than exact syntactic field matching.

## The Problem We Solved

### Before (Mechanical Eval):
```python
# Tests exact string matching
predicted["motion"] == "expansion"  # ✓ or ✗
```

**Issues:**
- Only tests if model memorized exact field names
- Treats "expansion" ≠ "upsell" as wrong (they're semantically equivalent)
- Doesn't test if model understands GTM context
- Doesn't validate tool selection (the real business value)
- Missing the forest for the trees

### After (Semantic Eval):
```python
# Tests understanding
is_semantically_equivalent("expansion", ["expansion", "upsell", "cross_sell"])  # ✓
would_select_correct_tools(predicted_ir, expected_tools)  # ✓
understands_context("best accounts" → existing, not net_new)  # ✓
```

## Implementation

### 1. New Semantic Scoring Module

**File:** `ccp/eval/semantic_scoring.py`

**Key Functions:**
- `score_semantic_understanding()` - Main scoring function
- `score_tool_selection()` - Tests if IR leads to correct tool categories
- `is_semantically_equivalent()` - Allows acceptable variations
- `compare_semantic_vs_exact()` - Shows difference between approaches

**Semantic Equivalence Mappings:**
```python
INTENT_TYPE_EQUIVALENTS = {
    "churn_risk_assessment": ["churn_risk_analysis", "at_risk_identification", ...],
    "account_discovery": ["account_search", "account_ranking", "top_accounts", ...],
}

MOTION_EQUIVALENTS = {
    "expansion": ["upsell", "cross_sell", "growth"],
    "churn_prevention": ["retention", "save"],
}
```

**Tool Category Mappings:**
```python
INTENT_TO_TOOL_CATEGORIES = {
    "churn_risk_assessment": {
        "correct": ["risk_scoring", "engagement_analysis", "health_metrics"],
        "wrong": ["prospecting", "lead_gen", "outbound_tools"]
    },
}
```

### 2. New CLI Command

**Command:** `python ccp_cli.py eval-semantic`

**Options:**
```bash
--quick                 # 5 examples
--compare-with-exact    # Show semantic vs exact comparison
--model both            # Test base + fine-tuned
--verbose               # Show per-example details
```

**Usage:**
```bash
# Quick semantic evaluation
python ccp_cli.py eval-semantic --quick

# Compare approaches
python ccp_cli.py eval-semantic --quick --compare-with-exact

# Full evaluation
python ccp_cli.py eval-semantic --model finetuned
```

### 3. New Metrics

**Critical Field Accuracy** (Target: 80%+)
- Fields that determine tool selection: `intent_type`, `motion`, `account_scope`
- Must get these right for system to work
- Uses semantic equivalence, not exact match

**Tool Selection Accuracy** (Target: 85%+)
- Would the Intent IR select the RIGHT tool categories?
- Would it avoid selecting WRONG categories?
- This is the KEY business value metric

**Overall Semantic Score** (Target: 75%+)
- Weighted score: tool selection counts 2x
- Formula: `(critical_accuracy * 1.0 + tool_selection * 2.0) / 3.0`
- Emphasizes what matters for downstream execution

## Key Differences

| Metric | Exact Eval | Semantic Eval |
|--------|------------|---------------|
| **What it tests** | Syntax | Understanding |
| **Field matching** | Exact string match | Semantic equivalence |
| **"expansion" vs "upsell"** | Wrong (0%) | Correct (100%) |
| **Tool selection** | Not tested | Core metric |
| **Business value** | Indirect | Direct |
| **Expected score** | Lower (~41%) | Higher (TBD) |

## Examples of Semantic Understanding

### Example 1: Account Scope
```
Prompt: "Show me my best accounts"

Exact eval:
  account_scope == "existing" → ✓/✗

Semantic eval:
  ✓ Understands "my accounts" = existing (not prospects)
  ✓ Would filter to account_data, not lead_gen tools
  ✓ Gets implicit context
```

### Example 2: Slang Handling
```
Prompt: "Who's ghosting us?"

Exact eval:
  intent_type == "churn_risk_assessment" → ✗ (model says "engagement_analysis")

Semantic eval:
  ✓ engagement_analysis is semantically equivalent
  ✓ Would select risk_scoring + engagement_analysis tools (correct!)
  ✓ Understands "ghosting" = accounts going silent
```

### Example 3: Motion Understanding
```
Prompt: "Help me prepare for my QBR"

Exact eval:
  motion == "expansion" → ✗ (model says "renewal")

Semantic eval:
  ✓ renewal is valid for QBR context (existing accounts)
  ✓ Would NOT select prospecting tools (correct!)
  ✓ Understands QBR = review existing accounts
```

## Success Criteria (PRD Section 12)

The semantic eval directly tests the real success metrics:

1. **Reduction in downstream tool call failures**
   - → Tool Selection Accuracy metric

2. **Improvement in first-pass execution accuracy**
   - → Critical Field Accuracy metric

3. **Reduction in unnecessary clarification questions**
   - → Tests if model understands implicit context

4. **Consistency across semantically similar prompts**
   - → Semantic equivalence mappings

## Next Steps

1. **Run semantic eval on current model** - Get baseline semantic scores
2. **Compare with exact eval** - Show the improvement
3. **Identify weak areas** - Where is GTM understanding failing?
4. **Enhance training data** - Focus on semantic understanding, not just syntax
5. **Iterate** - Use semantic scores to guide model improvements

## Files Changed

**New files:**
- `ccp/eval/__init__.py` - Module init
- `ccp/eval/semantic_scoring.py` - Scoring logic
- `ccp/commands/eval_semantic.py` - CLI command
- `docs/semantic_eval_design.md` - Design doc
- `docs/semantic_eval_summary.md` - This file

**Modified files:**
- `ccp/cli.py` - Registered new command
- `CLAUDE.md` - Added semantic eval documentation
- `README.md` - Added semantic eval section

## How to Use

### For Development
```bash
# Quick test during development
python ccp_cli.py eval-semantic --quick --verbose

# Compare approaches to see improvement
python ccp_cli.py eval-semantic --quick --compare-with-exact
```

### For Model Evaluation
```bash
# Full semantic evaluation
python ccp_cli.py eval-semantic --model finetuned

# Compare base vs fine-tuned
python ccp_cli.py eval-semantic --model both
```

### For Training Iteration
```bash
# After training, run both evals
python ccp_cli.py eval --model finetuned --quick
python ccp_cli.py eval-semantic --model finetuned --quick

# Compare to see if semantic understanding improved
```

## Expected Impact

With semantic evaluation, we can:
- **Measure real business value** - Tool selection accuracy
- **Give credit for understanding** - "expansion" ≈ "upsell"
- **Focus training** - Identify where GTM understanding fails
- **Validate readiness** - Is the model production-ready?
- **Track progress** - Semantic scores trend up with better training

The model may already understand GTM better than exact eval suggests!
