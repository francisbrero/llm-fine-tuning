#!/usr/bin/env python3
"""
Test CCP v2 trained model with standard and adversarial prompts.
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== CONFIG =====
BASE_MODEL_PATH = "./models/phi-2"
ADAPTER_PATH = "./ccp-adapter"

# System prompt (same as training)
CCP_SYSTEM_PROMPT = """You are the Phoenix Context Collapse Parser (CCP). Your job is to transform ambiguous GTM (Go-To-Market) prompts into structured GTM Intent IR.

IMPORTANT: You must REASON about the request before producing structured output. This ensures you understand the GTM semantics, not just the JSON format.

Given a user's GTM request:
1. First, analyze the prompt in a <reasoning> section:
   - Identify explicit signals (keywords, jargon, metrics mentioned)
   - Infer implicit context (role, motion, time horizon)
   - Note any ambiguities that affect confidence
   - Explain WHY you chose each field value

2. Then, output the structured intent in an <intent_ir> section as JSON with:
   - intent_type: The primary intent category
   - motion: The GTM motion (outbound, expansion, renewal, etc.)
   - role_assumption: Inferred user role
   - account_scope: Which accounts (net_new, existing, all)
   - icp_selector: Which ICP to apply (default, or specific product/segment)
   - icp_resolution_required: true if ICP needs downstream resolution
   - geography_scope: Geographic filter if mentioned (null if global)
   - time_horizon: Time scope for the request
   - output_format: How results should be presented
   - confidence_scores: 0.0-1.0 confidence for each inferred field
   - assumptions_applied: List of assumptions made
   - clarification_needed: true if request is too ambiguous

Format your response EXACTLY as:
<reasoning>
[Your analysis here]
</reasoning>

<intent_ir>
[Valid JSON here]
</intent_ir>"""


def load_model():
    """Load the fine-tuned CCP v2 model."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float32,
        device_map={"": "mps"},
        local_files_only=True,
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()

    print("Model loaded!\n")
    return model, tokenizer


def parse_gtm_intent(model, tokenizer, user_prompt: str, max_new_tokens: int = 800) -> dict:
    """Run CCP v2 inference."""
    input_text = f"<s>[INST] {CCP_SYSTEM_PROMPT}\n\nUser request: {user_prompt} [/INST]\n"
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract content after [/INST]
    if "[/INST]" in generated:
        response = generated.split("[/INST]")[-1].strip()
    else:
        response = generated

    response = response.replace("</s>", "").strip()

    # Extract reasoning section
    reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract intent_ir section
    ir_match = re.search(r'<intent_ir>\s*(.*?)\s*</intent_ir>', response, re.DOTALL)
    json_str = ir_match.group(1).strip() if ir_match else response

    # Parse JSON
    try:
        intent_ir = json.loads(json_str)
        return {
            "reasoning": reasoning,
            "intent_ir": intent_ir,
            "_valid": True,
            "_raw": response
        }
    except json.JSONDecodeError as e:
        return {
            "reasoning": reasoning,
            "intent_ir": {},
            "_valid": False,
            "_error": str(e),
            "_raw": response
        }


def main():
    model, tokenizer = load_model()

    # Standard GTM test prompts
    standard_prompts = [
        "Show me my best accounts",
        "Which deals are at risk this quarter?",
        "Find me companies like Acme Corp",
        "I need to hit my number, what should I focus on?",
        "Give me expansion opportunities in EMEA",
    ]

    # Adversarial test prompts
    adversarial_prompts = [
        "accounts",  # Too minimal
        "Show me the zorbax metrics",  # Made-up jargon
        "What's the weather like?",  # Non-GTM
        "Make me a sandwich",  # Completely unrelated
    ]

    print("=" * 70)
    print("CCP v2 INFERENCE TEST - Chain-of-Thought")
    print("=" * 70)

    print("\n### STANDARD GTM PROMPTS ###\n")
    for prompt in standard_prompts:
        print(f"PROMPT: {prompt}")
        print("-" * 50)

        result = parse_gtm_intent(model, tokenizer, prompt)

        print("REASONING:")
        print(result.get("reasoning", "(no reasoning)") or "(empty)")
        print()

        if result.get("_valid"):
            print("INTENT IR:")
            ir = result["intent_ir"]
            # Print key fields
            print(f"  intent_type: {ir.get('intent_type', 'N/A')}")
            print(f"  motion: {ir.get('motion', 'N/A')}")
            print(f"  role_assumption: {ir.get('role_assumption', 'N/A')}")
            print(f"  account_scope: {ir.get('account_scope', 'N/A')}")
            print(f"  time_horizon: {ir.get('time_horizon', 'N/A')}")
            print(f"  clarification_needed: {ir.get('clarification_needed', 'N/A')}")
            if ir.get("confidence_scores"):
                print(f"  confidence_scores: {ir.get('confidence_scores')}")
        else:
            print(f"[INVALID JSON] Error: {result.get('_error')}")
            print(f"Raw output: {result.get('_raw', '')[:300]}...")

        print("=" * 70)

    print("\n### ADVERSARIAL PROMPTS (Should Show Uncertainty) ###\n")
    for prompt in adversarial_prompts:
        print(f"PROMPT: {prompt}")
        print("-" * 50)

        result = parse_gtm_intent(model, tokenizer, prompt)

        print("REASONING:")
        print(result.get("reasoning", "(no reasoning)") or "(empty)")
        print()

        if result.get("_valid"):
            ir = result["intent_ir"]
            clarification = ir.get("clarification_needed", False)
            confidence_scores = ir.get("confidence_scores", {})
            low_confidence = any(score is not None and score < 0.5 for score in confidence_scores.values()) if confidence_scores else False

            if clarification or low_confidence:
                print("[GOOD] Model showed appropriate uncertainty")
            else:
                print("[WARNING] Model may be over-confident on ambiguous input")

            print("INTENT IR:")
            print(f"  intent_type: {ir.get('intent_type', 'N/A')}")
            print(f"  clarification_needed: {ir.get('clarification_needed', 'N/A')}")
            if confidence_scores:
                print(f"  confidence_scores: {confidence_scores}")
        else:
            print(f"[INVALID JSON] {result.get('_error')}")

        print("=" * 70)


if __name__ == "__main__":
    main()
