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

Run evaluations to track model performance:

```bash
# Quick eval (5 examples)
python run_evals.py --quick

# Full eval (30 examples)
python run_evals.py --model finetuned

# View history
python eval_history.py
python eval_history.py --compare 2  # Compare last 2 runs
```

Key metrics:
- **JSON Valid Rate**: Target 95%+
- **Field Accuracy**: Target 70%+
- **Intent Type Accuracy**: Target 80%+

Results saved to `eval_results/` and summarized in `EVAL_RESULTS.md`.

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

## Key Commands

```bash
# Training
jupyter notebook fine_tune_llm_mac_mps.ipynb

# Testing
python simple_test.py

# Evaluation
python run_evals.py --quick
python eval_history.py --detail latest

# Data validation
python -c "import json; [json.loads(l) for l in open('data/ccp_training_with_reasoning.jsonl')]"
```
