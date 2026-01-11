# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the **Phoenix Context Collapse Parser (CCP)** — a small language model (~3B parameters) fine-tuned to translate ambiguous GTM (Go-To-Market) prompts into structured, executable intent (GTM Intent IR). CCP is a domain-specific parsing layer that collapses implied enterprise GTM context before passing to a downstream general-purpose LLM.

## Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if needed)
pip install torch transformers peft bitsandbytes datasets accelerate

# Download base model (~5.5GB)
huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2

# Run training notebook
jupyter notebook fine_tune_llm_mac_mps.ipynb
```

- Python 3.11, macOS M-series with Metal/MPS acceleration
- Requires ~16GB RAM (24GB recommended), ~10GB disk

## Key Files

- `fine_tune_llm_mac_mps.ipynb` — QLoRA fine-tuning notebook for CCP on Mac MPS
- `PRD.md` — Product requirements document defining CCP architecture and goals
- `gtm_domain_knowledge.md` — Domain knowledge reference for GTM context collapse (motions, roles, ICP handling, shorthand)

## Fine-Tuning Configuration

The notebook expects:
- Base model at `./models/phi-2` (or other ~3B param model)
- Training data at `./data/ccp_training.jsonl` with `{gtm_prompt, intent_ir}` format
- Output adapter saved to `./ccp-adapter/`

Training uses instruction format:
```
<s>[INST] {system_prompt}\n\nUser request: {gtm_prompt} [/INST]\n{intent_ir_json}</s>
```

## Architecture Notes

- CCP outputs a **GTM Intent IR** — a structured representation with fields like `intent_type`, `motion`, `role_assumption`, `icp_selector`, `confidence_scores`
- ICP definitions are NOT encoded in the model; CCP emits selectors, downstream Phoenix MCP tools resolve them
- Target: sub-100ms inference, offline-capable, runs on commodity hardware

## Model Training Stack

- Transformers + PEFT (QLoRA)
- 4-bit quantization via bitsandbytes
- LoRA config: r=16, lora_alpha=32, targets q_proj/v_proj
- No fp16 on macOS — uses float32 compute dtype

## GTM Intent IR Fields

Required output fields from CCP:
- `intent_type`: account_discovery, pipeline_analysis, churn_risk_assessment, etc.
- `motion`: outbound, inbound, expansion, renewal, churn_prevention
- `role_assumption`: sales_rep, sales_manager, revops, marketing, cs, exec
- `account_scope`: net_new, existing, churned, all
- `confidence_scores`: 0.0-1.0 per inferred field
- `assumptions_applied`: human-readable list of inferences made

## Evaluation

### Comparative Evaluation Workflow

**Always compare base model vs fine-tuned model** to measure the impact of fine-tuning:

```bash
# 1. Quick comparison (5 examples) - recommended for rapid iteration
python ccp_cli.py eval --model both --quick

# 2. View comparison
python ccp_cli.py history --compare 2

# 3. Full evaluation (30 examples) - for comprehensive assessment
python ccp_cli.py eval --model both

# 4. View all results side-by-side
python ccp_cli.py history --model all
```

### Semantic Evaluation (Recommended)

**Semantic evaluation tests GTM understanding and business value**, not just syntactic field matching:

```bash
# Quick semantic evaluation (tests context understanding)
python ccp_cli.py eval-semantic --quick

# Compare semantic vs exact field matching
python ccp_cli.py eval-semantic --quick --compare-with-exact

# Full semantic evaluation
python ccp_cli.py eval-semantic --model finetuned

# Evaluate both models with semantic scoring
python ccp_cli.py eval-semantic --model both
```

**Why semantic evaluation matters:**
- Tests if the model understands **implicit GTM context** (e.g., "best accounts" = existing, not prospects)
- Validates **tool selection accuracy** (would the IR lead to correct tools?)
- Measures **business value**, not just syntax correctness
- Allows semantic equivalents (e.g., "churn_risk_assessment" ≈ "churn_risk_analysis")

**Key semantic metrics:**
- **Critical Field Accuracy**: Fields that determine tool selection (target: 80%+)
- **Tool Selection Accuracy**: Would select correct tool categories (target: 85%+)
- **Overall Semantic Score**: Weighted score emphasizing tool selection (target: 75%+)

### Individual Model Evaluation

```bash
# Evaluate base model only
python ccp_cli.py eval --model base --quick

# Evaluate fine-tuned model only
python ccp_cli.py eval --model finetuned

# Filter by specific categories
python ccp_cli.py eval --category adversarial
python ccp_cli.py eval --category slang
python ccp_cli.py eval --category standard

# View detailed results
python ccp_cli.py history --detail latest
```

### Evaluation Results

Key metrics:
- **JSON Valid Rate**: Target 95%+ (measures output format validity)
- **Field Accuracy**: Target 70%+ (measures correct field extraction)
- **Intent Type Accuracy**: Target 80%+ (measures intent classification)

**Baseline Performance** (as of Jan 2026):

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| JSON Valid Rate | 0% | 100% | +100% |
| Field Accuracy | 0% | 41% | +41% |

Fine-tuning is **essential** - the base Phi-2 model has zero capability for GTM intent parsing without domain-specific training.

Results are saved to `eval_results/` and can be exported:

```bash
# Export to CSV
python ccp_cli.py history --export results.csv
```

## Skill Auto-Activation System

This project includes automatic skill activation based on prompt context.

**How it works:**
1. When you submit a prompt, the `skill-activation-prompt` hook analyzes it
2. It matches keywords and patterns against skill definitions in `.claude/hooks/skill-rules.json`
3. Matching skills are displayed with priority levels (critical/high/medium/low)
4. The hook instructs Claude to read referenced skill files before responding

**Skill types:**
- **Technical skills** (`.claude/skills/technical/`): Framework patterns, best practices
- **Runbooks** (`.claude/skills/runbooks/`): Step-by-step procedures
- **References** (`.claude/skills/reference/`): Architecture documentation

**Setup (if hooks not working):**
```bash
cd .claude/hooks
npm install   # Install tsx dependency
chmod +x *.sh # Make scripts executable
```

## Dev Docs System

For multi-session tasks, use the dev docs system for context persistence:

```
dev/
├── active/           # In-progress tasks (gitignored)
│   └── issue-123/
│       ├── plan.md       # Implementation plan
│       ├── context.md    # Current state, key files
│       └── tasks.md      # Checklist with status
├── completed/        # Archived tasks (gitignored)
└── templates/        # Templates (tracked in git)
```

**Workflow:**
1. Create task folder: `mkdir -p dev/active/issue-123`
2. Copy templates: `cp dev/templates/*.md dev/active/issue-123/`
3. Update context frequently during work
4. Move to `completed/` when done

## Slash Commands

- `/fix-issue <number>` - Fetch GitHub issue and implement with full workflow

## CCP CLI

The project includes a CLI for all fine-tuning and evaluation operations:

```bash
# Show all commands
python ccp_cli.py --help

# Validate training data
python ccp_cli.py validate-data
python ccp_cli.py validate-data --file data/ccp_training_with_reasoning.jsonl --verbose

# Train the model
python ccp_cli.py train --dry-run              # Validate setup without training
python ccp_cli.py train                         # Run training with defaults
python ccp_cli.py train --epochs 5 --lr 1e-4   # Custom parameters
python ccp_cli.py train --resume-from latest   # Resume from checkpoint

# Evaluate model performance
python ccp_cli.py eval --model both --quick        # Compare base vs fine-tuned (5 examples)
python ccp_cli.py eval --model both                # Full comparison (30 examples)
python ccp_cli.py eval --model finetuned           # Evaluate fine-tuned only
python ccp_cli.py eval --category adversarial      # Filter by category

# Semantic evaluation (recommended for business value)
python ccp_cli.py eval-semantic --quick --compare-with-exact  # Compare approaches
python ccp_cli.py eval-semantic --model finetuned             # Test GTM understanding

# Run inference on a single prompt
python ccp_cli.py inference "Show me my best accounts"
python ccp_cli.py inference "Accounts at risk" --json-only | jq .

# View evaluation history
python ccp_cli.py history
python ccp_cli.py history --compare 2          # Compare last 2 runs
python ccp_cli.py history --detail latest      # Detailed view
python ccp_cli.py history --export results.csv # Export to CSV
```

### CLI Options

| Command | Description |
|---------|-------------|
| `validate-data` | Validate training data schema and content |
| `train` | Fine-tune the CCP model with checkpoint resume support |
| `eval` | Evaluate model performance against test set |
| `inference` | Run single inference on a GTM prompt |
| `history` | View and analyze evaluation history |

## Legacy Commands

The following standalone scripts are still available:

```bash
# Training (notebook)
jupyter notebook fine_tune_llm_mac_mps.ipynb

# Testing
python simple_test.py

# Evaluation (legacy)
python run_evals.py --quick
python eval_history.py --detail latest

# Data validation (one-liner)
python -c "import json; [json.loads(l) for l in open('data/ccp_training_with_reasoning.jsonl')]"
```
