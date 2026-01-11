# Phoenix Context Collapse Parser (CCP)

A fine-tuned small language model that transforms ambiguous GTM (Go-To-Market) prompts into structured, executable intent (GTM Intent IR).

## Quick Start

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Download base model (if not already done)
huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2

# 3. Prepare training data
# Create ./data/ccp_training.jsonl (see format below)

# 4. Run the notebook
jupyter notebook fine_tune_llm_mac_mps.ipynb
```

## Project Structure

```
local-llm/
├── fine_tune_llm_mac_mps.ipynb  # Main training notebook
├── PRD.md                       # Product requirements
├── gtm_domain_knowledge.md      # Domain knowledge for training
├── models/
│   └── phi-2/                   # Base model (downloaded)
├── data/
│   └── ccp_training.jsonl       # Training data (you create)
└── ccp-adapter/                 # Output (after training)
```

## Model Setup

### Download Base Model

The project uses Microsoft's Phi-2 (~2.7B parameters) as the base model. Download it from Hugging Face:

```bash
# Ensure you're in the project directory with venv activated
source .venv/bin/activate

# Download phi-2 (~5.5GB)
huggingface-cli download microsoft/phi-2 --local-dir ./models/phi-2
```

**Alternative models** (edit `BASE_MODEL_PATH` in notebook):
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — Smaller, faster (~2GB)
- `mistralai/Mistral-7B-Instruct-v0.2` — Larger, more capable (~14GB)

### Hugging Face Authentication (if needed)

Some models require authentication:

```bash
huggingface-cli login
# Enter your HF token from https://huggingface.co/settings/tokens
```

## Training Data Format

Create `./data/ccp_training.jsonl` with GTM prompt → Intent IR pairs:

```json
{"gtm_prompt": "Show me my best accounts", "intent_ir": {"intent_type": "account_discovery", "motion": "expansion", "role_assumption": "sales_rep", "account_scope": "existing", "icp_selector": "default", "time_horizon": "this_quarter", "confidence_scores": {"intent_type": 0.85, "motion": 0.7}, "assumptions_applied": ["Assumed sales rep role"]}}
{"gtm_prompt": "Which deals are at risk?", "intent_ir": {"intent_type": "churn_risk_assessment", "motion": "churn_prevention", "role_assumption": "sales_manager", "account_scope": "existing", "time_horizon": "this_quarter", "confidence_scores": {"intent_type": 0.95}}}
```

See `gtm_domain_knowledge.md` for the full IR schema and example mappings.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11
- ~16GB RAM (24GB recommended)
- ~10GB disk space for model + adapter

### Python Dependencies

Already installed in `.venv/`:
- torch (with MPS support)
- transformers
- peft
- bitsandbytes
- datasets

To reinstall:
```bash
pip install torch transformers peft bitsandbytes datasets accelerate
```

## Usage After Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + adapter (tokenizer from base model)
base_model = AutoModelForCausalLM.from_pretrained("./models/phi-2")
model = PeftModel.from_pretrained(base_model, "./ccp-adapter")
tokenizer = AutoTokenizer.from_pretrained("./models/phi-2")

# Parse GTM intent
prompt = "Show me my best accounts"
input_text = f"### GTM Prompt:\n{prompt}\n\n### Intent IR:\n"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200, temperature=0.1)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Evaluation

### Quick Evaluation

Compare base model vs fine-tuned model performance:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run quick evaluation (5 examples) on both models
python ccp_cli.py eval --model both --quick

# View comparison
python ccp_cli.py history --compare 2
```

### Full Evaluation

Run comprehensive evaluation on the full test set (30 examples):

```bash
# Evaluate both models
python ccp_cli.py eval --model both

# View side-by-side comparison
python ccp_cli.py history --model all

# View detailed results for latest run
python ccp_cli.py history --detail latest
```

### Evaluation Commands

```bash
# Evaluate only the base model
python ccp_cli.py eval --model base --quick

# Evaluate only the fine-tuned model
python ccp_cli.py eval --model finetuned

# Filter by category
python ccp_cli.py eval --category adversarial
python ccp_cli.py eval --category slang

# Export results to CSV
python ccp_cli.py history --export results.csv
```

### Semantic Evaluation (Recommended)

The standard eval tests exact field matching, which measures syntax correctness but not GTM understanding. **Semantic evaluation** tests business value:

```bash
# Quick semantic evaluation
python ccp_cli.py eval-semantic --quick

# Compare semantic vs exact field matching
python ccp_cli.py eval-semantic --quick --compare-with-exact

# Full semantic evaluation
python ccp_cli.py eval-semantic --model finetuned

# Test both models
python ccp_cli.py eval-semantic --model both
```

**Semantic evaluation tests:**
- **Context understanding**: Does "best accounts" mean existing (not prospects)?
- **Tool selection accuracy**: Would the IR select the right tool categories?
- **Slang interpretation**: Does "ghosting us" mean churn risk?
- **Business logic**: Does "QBR prep" imply reviewing existing accounts?

**Key differences:**
- Exact eval: `motion: "expansion"` must match exactly (41% accuracy)
- Semantic eval: `motion: "expansion"` ≈ `motion: "upsell"` (higher accuracy, tests understanding)

### Key Metrics

- **JSON Valid Rate**: Percentage of outputs that are valid JSON (target: 95%+)
- **Field Accuracy**: Average accuracy across all expected fields (target: 70%+)
- **Intent Type Accuracy**: Accuracy of intent classification (target: 80%+)

### Expected Performance

Based on evaluations, fine-tuning provides dramatic improvements:

| Metric | Base Model | Fine-Tuned | Improvement |
|--------|------------|------------|-------------|
| JSON Valid Rate | 0% | 100% | +100% |
| Field Accuracy | 0% | 41% | +41% |

**Fine-tuning is essential** - the base model cannot perform GTM intent parsing without domain-specific training.

## Testing

Run a quick inference test:

```bash
python simple_test.py
```
