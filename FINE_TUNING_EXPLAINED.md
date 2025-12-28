# How Fine-Tuning Works: A Step-by-Step Guide

This document explains how the Phoenix Context Collapse Parser (CCP) fine-tuning process works, breaking down each step of the `fine_tune_llm_mac_mps.ipynb` notebook.

---

## Table of Contents

1. [Overview: What Are We Building?](#1-overview-what-are-we-building)
2. [The Base Model](#2-the-base-model)
3. [QLoRA: Efficient Fine-Tuning](#3-qlora-efficient-fine-tuning)
4. [Step-by-Step Training Process](#4-step-by-step-training-process)
5. [Training Data Format](#5-training-data-format)
6. [The Training Loop](#6-the-training-loop)
7. [What the Model Learns](#7-what-the-model-learns)
8. [Inference After Training](#8-inference-after-training)
9. [v2: Chain-of-Thought Training](#9-v2-chain-of-thought-training)

---

## 1. Overview: What Are We Building?

We're taking a pre-trained language model (Phi-2, ~3B parameters) and teaching it a new specialized skill: **translating ambiguous GTM (Go-To-Market) prompts into structured JSON**.

**Before fine-tuning:**
```
Input:  "Show me my best accounts"
Output: (generic, unpredictable response)
```

**After fine-tuning (v2 with Chain-of-Thought):**
```
Input:  "Show me my best accounts"
Output:
<reasoning>
- 'my accounts' indicates personally assigned accounts, suggesting sales rep role
- 'best' is ambiguous: could mean highest ARR, best fit, or most engaged
- No time horizon mentioned, defaulting to current quarter
- Expansion motion inferred for existing customer accounts
</reasoning>

<intent_ir>
{
  "intent_type": "account_discovery",
  "motion": "expansion",
  "role_assumption": "sales_rep",
  "account_scope": "existing",
  ...
}
</intent_ir>
```

The model learns to:
- **Reason about the request first** (v2 Chain-of-Thought)
- Infer implied context (role, motion, time horizon)
- Output structured JSON with specific fields
- Provide confidence scores for its inferences

---

## 2. The Base Model

### What is a Base Model?

A **base model** (like Phi-2) is a large neural network pre-trained on massive amounts of text. It already understands:
- Grammar and language structure
- General knowledge
- How to follow instructions

We don't train from scratch — we leverage this existing knowledge.

### Loading the Model (Cell 5)

```python
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": "mps"},  # Use Apple Silicon GPU
    local_files_only=True,
)
```

**Key points:**
- `AutoModelForCausalLM`: Loads a model designed for text generation
- `device_map={"": "mps"}`: Uses Mac's Metal Performance Shaders (GPU acceleration)
- `quantization_config`: Applies 4-bit quantization to reduce memory

---

## 3. QLoRA: Efficient Fine-Tuning

### The Problem: Full Fine-Tuning is Expensive

A 3B parameter model has 3 billion numbers (weights) to potentially update. Training all of them requires:
- Massive GPU memory (50+ GB)
- Hours/days of training
- Risk of "catastrophic forgetting" (losing original capabilities)

### The Solution: LoRA (Low-Rank Adaptation)

LoRA freezes the original model and adds small "adapter" layers that learn the new task.

```
┌─────────────────────────────────────────────┐
│  Original Model (FROZEN - 3B parameters)   │
│  ┌───────────────────────────────────────┐  │
│  │  Attention Layer (q_proj, v_proj)     │  │
│  │         │                             │  │
│  │    ┌────┴────┐                        │  │
│  │    │  LoRA   │  ← Only these train    │  │
│  │    │ Adapter │    (~0.1% of params)   │  │
│  │    └────┬────┘                        │  │
│  │         │                             │  │
│  └─────────┴─────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### QLoRA: Adding Quantization

**Q**LoRA adds 4-bit quantization:
- Original weights stored in 4-bit format (¼ the memory)
- LoRA adapters train in higher precision
- Result: Fine-tune a 3B model on a laptop with 16GB RAM

### LoRA Configuration (Cell 6)

```python
lora_config = LoraConfig(
    r=16,                        # Rank of adapter matrices
    lora_alpha=32,               # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Which layers to adapt
    lora_dropout=0.05,           # Regularization
    bias="none",
    task_type="CAUSAL_LM"
)
```

**What these parameters mean:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `r=16` | Rank | Size of adapter matrices. Higher = more capacity, more memory |
| `lora_alpha=32` | Scale | How much the adapter affects output. Often set to 2×r |
| `target_modules` | Attention layers | Which model layers get adapters. q_proj and v_proj are attention query/value projections |
| `lora_dropout` | 0.05 | 5% dropout for regularization (prevents overfitting) |

### Quantization Configuration (Cell 5)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,           # 4-bit quantization
    bnb_4bit_compute_dtype=torch.float32,  # Compute in float32 (macOS requirement)
    bnb_4bit_use_double_quant=True,        # Quantize the quantization constants too
    bnb_4bit_quant_type="nf4"              # NormalFloat4 - optimized for neural networks
)
```

---

## 4. Step-by-Step Training Process

Here's what happens in each cell of the notebook:

### Step 1: Environment Check (Cells 1-2)
```python
print("MPS available:", torch.backends.mps.is_available())
```
Verifies Mac GPU (Metal) is available for acceleration.

### Step 2: Configuration (Cell 4)
```python
BASE_MODEL_PATH = "./models/phi-2"
DATASET_PATH = "./data/ccp_training.jsonl"
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
```
Sets paths and hyperparameters.

### Step 3: Load Tokenizer (Cell 8)
```python
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
```
The tokenizer converts text → numbers (tokens) and back. Example:
```
"Show me accounts" → [2438, 502, 13307] → (model processes) → [876, 323, ...] → "{...json...}"
```

### Step 4: Load Model with Quantization (Cell 10)
Loads the 3B parameter model in 4-bit format (~1.5GB instead of ~6GB).

### Step 5: Apply LoRA (Cell 12)
```python
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 2,621,440 || all params: 2,779,683,840 || trainable%: 0.09%
```
Only ~0.1% of parameters will be trained!

### Step 6: Load & Format Dataset (Cells 14, 16)
Loads training examples and formats them with the instruction template.

### Step 7: Tokenize (Cell 18)
Converts text examples to token IDs the model can process.

### Step 8: Training (Cell 26)
```python
trainer.train()
```
The actual learning happens here (1-3 hours).

### Step 9: Save Adapter (Cells 28, 32)
Saves only the LoRA weights (~10MB) — not the full model.

---

## 5. Training Data Format

### Input Format (JSONL)

Each training example is a JSON object:

```json
{
  "gtm_prompt": "Show me my best accounts",
  "intent_ir": {
    "intent_type": "account_discovery",
    "motion": "expansion",
    "role_assumption": "sales_rep",
    "account_scope": "existing",
    "confidence_scores": {...},
    "assumptions_applied": [...]
  }
}
```

### Instruction Template (Cell 16)

The data is formatted into an instruction-following template:

```
<s>[INST] You are the Phoenix Context Collapse Parser (CCP)...

User request: Show me my best accounts [/INST]
{
  "intent_type": "account_discovery",
  ...
}</s>
```

**Why this format?**
- `<s>` and `</s>`: Start/end tokens
- `[INST]...[/INST]`: Marks the instruction section
- Text after `[/INST]`: The expected output

The model learns: "When I see text between [INST] tags, I should generate text that looks like what comes after."

---

## 6. The Training Loop

### What Happens During Training?

```
┌──────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                         │
│                                                          │
│  For each epoch (3 total):                               │
│    For each batch of examples:                           │
│                                                          │
│      1. FORWARD PASS                                     │
│         Input tokens → Model → Predicted tokens          │
│                                                          │
│      2. LOSS CALCULATION                                 │
│         Compare predictions vs actual target             │
│         Loss = how wrong the model was                   │
│                                                          │
│      3. BACKWARD PASS                                    │
│         Calculate gradients (how to adjust weights)      │
│         Only for LoRA parameters (0.1% of model)         │
│                                                          │
│      4. WEIGHT UPDATE                                    │
│         Adjust LoRA weights to reduce loss               │
│         learning_rate controls step size                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Training Arguments (Cell 22)

```python
training_args = TrainingArguments(
    per_device_train_batch_size=1,    # Process 1 example at a time
    gradient_accumulation_steps=8,     # Accumulate gradients over 8 steps
    num_train_epochs=3,                # Pass through data 3 times
    learning_rate=2e-4,                # Step size for weight updates
)
```

**Key concepts:**

| Parameter | Value | Why? |
|-----------|-------|------|
| `batch_size=1` | Small batches | Limited GPU memory on laptops |
| `gradient_accumulation_steps=8` | Simulate larger batch | Accumulate 8 mini-batches before updating weights (effective batch = 8) |
| `num_train_epochs=3` | 3 passes | Enough to learn without overfitting |
| `learning_rate=2e-4` | 0.0002 | Standard for LoRA fine-tuning |

### Loss Function: Next Token Prediction

The model learns by predicting the next token. For this input:

```
[INST] ... User request: Show me accounts [/INST]
{"intent_type":
```

The model predicts what comes next (`"account_discovery"`). The loss measures how wrong it was.

---

## 7. What the Model Learns

### Before Fine-Tuning

The base model might respond to "Show me my best accounts" with:
- A conversational answer
- General advice
- Unpredictable output format

### After Fine-Tuning

The model learns specific patterns:

1. **Output Structure**: Always produce valid JSON with specific fields
2. **Field Values**: Use the defined enum values (e.g., `"outbound"`, `"expansion"`)
3. **Context Inference**:
   - "my accounts" → `account_scope: "existing"`, `role_assumption: "sales_rep"`
   - "at risk" → `intent_type: "churn_risk_assessment"`, `motion: "churn_prevention"`
4. **Confidence Scoring**: Provide calibrated confidence based on signal strength

### The Learning Process Visualized

```
Epoch 1: Model outputs random JSON structure
         Loss: HIGH
         ↓
Epoch 2: Model outputs correct fields, wrong values
         Loss: MEDIUM
         ↓
Epoch 3: Model outputs correct structure and values
         Loss: LOW
```

---

## 8. Inference After Training

### Loading the Fine-Tuned Model

```python
# Load base model (still needed)
model = AutoModelForCausalLM.from_pretrained("./models/phi-2", ...)

# Load LoRA adapter on top
model = PeftModel.from_pretrained(model, "./ccp-adapter")
```

### Running Inference (Cell 30)

```python
def parse_gtm_intent(user_prompt: str) -> dict:
    # Format with instruction template
    input_text = f"<s>[INST] {CCP_SYSTEM_PROMPT}\n\nUser request: {user_prompt} [/INST]\n"

    # Tokenize
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")

    # Generate
    output = model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.1,  # Low temperature for deterministic output
    )

    # Decode and parse JSON
    generated = tokenizer.decode(output[0])
    json_str = generated.split("[/INST]")[-1].strip()
    return json.loads(json_str)
```

**Generation parameters:**

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `temperature=0.1` | Very low | Deterministic output (less randomness) |
| `top_p=0.95` | High | Consider most probable tokens |
| `max_new_tokens=500` | Limit | Prevent runaway generation |

### Example Flow

```
User Input: "Which deals are at risk this quarter?"
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Tokenization                           │
│  "Which deals..." → [1234, 5678, ...]   │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Model Forward Pass                     │
│  (Base Model + LoRA Adapter)            │
│  Input tokens → Hidden states → Output  │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Token Generation (autoregressive)      │
│  Generate one token at a time:          │
│  "{" → "\"" → "i" → "n" → "t" → ...     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  Decode & Parse                         │
│  [tokens] → "{\"intent_type\":..."      │
│  → JSON object                          │
└─────────────────────────────────────────┘
                    │
                    ▼
Output: {
  "intent_type": "churn_risk_assessment",
  "motion": "churn_prevention",
  "time_horizon": "this_quarter",
  ...
}
```

---

## Summary

| Step | What Happens | Why |
|------|--------------|-----|
| 1. Load base model | Get pre-trained language understanding | Don't start from scratch |
| 2. Apply 4-bit quantization | Reduce memory usage | Run on laptops |
| 3. Add LoRA adapters | Create trainable parameters | Efficient fine-tuning |
| 4. Format training data | Structure input → output pairs | Teach the task format |
| 5. Train | Adjust LoRA weights | Learn GTM → JSON mapping |
| 6. Save adapter | Store learned weights | ~10MB vs ~6GB |
| 7. Inference | Generate structured output | Use the trained model |

The key insight: **We're not teaching the model language — it already knows that. We're teaching it a specific input/output format and the domain knowledge to fill in that format correctly.**

---

## 9. v2: Chain-of-Thought Training

### The Problem with Pure JSON Training

In v1, we trained the model to output JSON directly:

```
Input:  "Show me my best accounts"
Output: {"intent_type": "account_discovery", ...}
```

**The risk**: The model might learn to produce syntactically valid JSON without deeply understanding GTM semantics. It could be pattern-matching keywords to JSON fields rather than reasoning about implied context.

Signs this is happening:
- Model outputs valid JSON but wrong field values
- Confidence scores are arbitrary (always 0.7)
- `assumptions_applied` are generic/templated
- Model confidently produces JSON for nonsense prompts

### The Solution: Chain-of-Thought + JSON

v2 forces the model to **reason first, then structure**:

```
┌─────────────────────────────────────────────────────────────┐
│                    v1: Direct JSON                          │
│                                                             │
│  "Show me my best accounts"                                 │
│           │                                                 │
│           ▼                                                 │
│  {"intent_type": "account_discovery", ...}                  │
│                                                             │
│  Problem: Model may just pattern-match keywords             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                v2: Chain-of-Thought + JSON                  │
│                                                             │
│  "Show me my best accounts"                                 │
│           │                                                 │
│           ▼                                                 │
│  <reasoning>                                                │
│  - 'my accounts' signals existing book of business          │
│  - 'best' is ambiguous (ARR? fit? engagement?)              │
│  - Assumed sales rep from possessive language               │
│  </reasoning>                                               │
│           │                                                 │
│           ▼                                                 │
│  <intent_ir>                                                │
│  {"intent_type": "account_discovery", ...}                  │
│  </intent_ir>                                               │
│                                                             │
│  Benefit: Model must explain WHY before WHAT                │
└─────────────────────────────────────────────────────────────┘
```

### Training Data Format (v2)

Each training example now includes a reasoning chain:

```json
{
  "gtm_prompt": "Which deals are at risk this quarter?",
  "reasoning": "- 'at risk' explicitly signals churn/risk concern\n- 'deals' indicates active pipeline\n- 'this quarter' explicitly sets time horizon (confidence: 1.0)\n- Question about deal-level risk suggests manager perspective",
  "intent_ir": {
    "intent_type": "churn_risk_assessment",
    "motion": "churn_prevention",
    ...
  }
}
```

### Instruction Template (v2)

The formatted training example:

```
<s>[INST] You are the Phoenix Context Collapse Parser (CCP)...

IMPORTANT: You must REASON about the request before producing structured output.

User request: Which deals are at risk this quarter? [/INST]
<reasoning>
- 'at risk' explicitly signals churn/risk concern
- 'deals' indicates active pipeline, not just accounts
- 'this quarter' explicitly sets time horizon (confidence: 1.0)
- Question about deal-level risk suggests manager perspective
</reasoning>

<intent_ir>
{
  "intent_type": "churn_risk_assessment",
  "motion": "churn_prevention",
  ...
}
</intent_ir></s>
```

### Benefits of Chain-of-Thought

1. **Interpretability**: You can inspect the reasoning to see if the model actually understands GTM semantics

2. **Better learning signal**: The reasoning chain forces the model to engage with semantics before formatting

3. **Debugging**: When the JSON is wrong, the reasoning shows *why*

4. **Validation**: If reasoning is shallow or wrong, you know the model is guessing

### Inference Changes (v2)

The inference function now extracts both reasoning and IR:

```python
def parse_gtm_intent(user_prompt: str) -> dict:
    # ... generate response ...

    # Extract reasoning section
    reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract intent_ir section
    ir_match = re.search(r'<intent_ir>\s*(.*?)\s*</intent_ir>', response, re.DOTALL)
    json_str = ir_match.group(1).strip() if ir_match else response

    return {
        "reasoning": reasoning,      # For interpretability
        "intent_ir": json.loads(json_str),  # For downstream use
        "_valid": True
    }
```

### Adversarial Testing

v2 includes adversarial test cases to validate semantic learning:

| Prompt | Expected Behavior | Why |
|--------|-------------------|-----|
| "accounts" | Very low confidence, clarification needed | Too minimal to infer context |
| "Show me the zorbax metrics" | Flag unknown term, high uncertainty | Made-up jargon |
| "What's the weather?" | Wrong domain flag | Non-GTM request |
| "Show me accounts" vs "Show me my accounts" | Different role confidence | "my" provides signal |

If the model confidently produces JSON for nonsense prompts, it's learning syntax over semantics.

### Generating Reasoning Chains

Use the provided script to add reasoning chains to existing training data:

```bash
# Preview reasoning for first 5 examples
python scripts/generate_reasoning_chains.py --preview 5

# Generate full dataset with reasoning
python scripts/generate_reasoning_chains.py \
  --input data/ccp_training.jsonl \
  --output data/ccp_training_with_reasoning.jsonl
```

---

## Summary (v2)

| Step | v1 | v2 |
|------|----|----|
| Training data | `{prompt, intent_ir}` | `{prompt, reasoning, intent_ir}` |
| Output format | JSON only | `<reasoning>` + `<intent_ir>` |
| Max sequence length | 1024 | 1536 (longer for reasoning) |
| What model learns | JSON formatting + some semantics | Reasoning process + semantics + formatting |
| Interpretability | Low (black box) | High (can inspect reasoning) |
| Adversarial robustness | Unknown | Tested with edge cases |

**The key insight for v2**: By forcing the model to explain its reasoning, we ensure it learns GTM semantics rather than just JSON formatting patterns.
