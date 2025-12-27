#!/usr/bin/env python3
"""
CCP Inference Test — Compare base phi-2 vs fine-tuned CCP

Tests both models sequentially to avoid memory issues.
"""

import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Test prompts covering various GTM scenarios
TEST_PROMPTS = [
    "Show me my best accounts",
    "Which deals are at risk this quarter?",
    "Find me fintech companies hiring SDRs in NYC",
    "I need to hit my number",
    "Accounts with declining NPS scores",
    "Companies using Salesforce looking to switch",
    "What's my pipeline coverage?",
    "Expansion opportunities in EMEA",
    "Who should I call right now?",
    "Customers who haven't logged in this month",
]

def generate_response(model, tokenizer, prompt, max_new_tokens=300):
    """Generate model response for a GTM prompt"""
    input_text = f"### GTM Prompt:\n{prompt}\n\n### Intent IR:\n"
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

    if "### Intent IR:" in generated:
        response = generated.split("### Intent IR:")[-1].strip()
    else:
        response = generated

    if "### GTM Prompt:" in response:
        response = response.split("### GTM Prompt:")[0].strip()

    return response

def validate_json(response):
    """Check if response is valid JSON and has expected fields"""
    try:
        # Try to extract JSON if there's extra text
        response = response.strip()
        if response.startswith("{"):
            # Find the end of JSON
            brace_count = 0
            end_idx = 0
            for i, char in enumerate(response):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            response = response[:end_idx]

        data = json.loads(response)
        required_fields = ["intent_type", "motion", "role_assumption", "account_scope"]
        has_required = all(f in data for f in required_fields)
        return True, has_required, data
    except json.JSONDecodeError:
        return False, False, None

def test_base_model():
    """Test base phi-2 model"""
    print("\n" + "=" * 70)
    print("TESTING BASE PHI-2 MODEL (without fine-tuning)")
    print("=" * 70)

    print("\nLoading base phi-2 model...")
    model = AutoModelForCausalLM.from_pretrained(
        "./models/phi-2",
        torch_dtype=torch.float32,
        device_map={"": "mps"},
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("./models/phi-2", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] \"{prompt}\"")
        response = generate_response(model, tokenizer, prompt)
        valid, has_fields, data = validate_json(response)
        results.append({
            "prompt": prompt,
            "response": response[:500],
            "valid_json": valid,
            "has_required_fields": has_fields,
            "data": data
        })
        status = "✓" if valid and has_fields else "✗"
        print(f"    {status} valid_json={valid}, has_fields={has_fields}")

    # Cleanup
    del model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()

    return results

def test_finetuned_model():
    """Test fine-tuned CCP model"""
    print("\n" + "=" * 70)
    print("TESTING FINE-TUNED CCP MODEL")
    print("=" * 70)

    print("\nLoading fine-tuned CCP model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "./models/phi-2",
        torch_dtype=torch.float32,
        device_map={"": "mps"},
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, "./ccp-adapter")
    tokenizer = AutoTokenizer.from_pretrained("./ccp-adapter", local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}] \"{prompt}\"")
        response = generate_response(model, tokenizer, prompt)
        valid, has_fields, data = validate_json(response)
        results.append({
            "prompt": prompt,
            "response": response[:500],
            "valid_json": valid,
            "has_required_fields": has_fields,
            "data": data
        })
        status = "✓" if valid and has_fields else "✗"
        print(f"    {status} valid_json={valid}, has_fields={has_fields}")

    # Cleanup
    del model
    del base_model
    del tokenizer
    gc.collect()
    torch.mps.empty_cache()

    return results

def print_comparison(base_results, ft_results):
    """Print side-by-side comparison"""
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON")
    print("=" * 70)

    for base, ft in zip(base_results, ft_results):
        print(f"\n{'─'*70}")
        print(f"PROMPT: {base['prompt']}")
        print("─" * 70)

        print("\n▸ BASE PHI-2:")
        if base['valid_json'] and base['data']:
            print(json.dumps(base['data'], indent=2)[:400])
        else:
            print(base['response'][:300])
        print(f"  [valid_json={base['valid_json']}, has_fields={base['has_required_fields']}]")

        print("\n▸ FINE-TUNED CCP:")
        if ft['valid_json'] and ft['data']:
            print(json.dumps(ft['data'], indent=2)[:400])
        else:
            print(ft['response'][:300])
        print(f"  [valid_json={ft['valid_json']}, has_fields={ft['has_required_fields']}]")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    base_valid = sum(1 for r in base_results if r['valid_json'])
    base_fields = sum(1 for r in base_results if r['has_required_fields'])
    ft_valid = sum(1 for r in ft_results if r['valid_json'])
    ft_fields = sum(1 for r in ft_results if r['has_required_fields'])
    total = len(TEST_PROMPTS)

    print(f"\n{'Metric':<35} {'Base phi-2':<15} {'Fine-tuned CCP':<15} {'Improvement':<15}")
    print("─" * 80)
    print(f"{'Valid JSON responses':<35} {base_valid}/{total:<14} {ft_valid}/{total:<14} {'+' if ft_valid > base_valid else ''}{ft_valid - base_valid}")
    print(f"{'Has all required IR fields':<35} {base_fields}/{total:<14} {ft_fields}/{total:<14} {'+' if ft_fields > base_fields else ''}{ft_fields - base_fields}")
    print(f"{'Success rate':<35} {base_fields/total*100:.0f}%{'':<13} {ft_fields/total*100:.0f}%{'':<13} {'+' if ft_fields > base_fields else ''}{(ft_fields - base_fields)/total*100:.0f}%")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

def main():
    print("=" * 70)
    print("CCP INFERENCE TEST — BEFORE vs AFTER FINE-TUNING")
    print("=" * 70)

    # Test base model first
    base_results = test_base_model()

    # Test fine-tuned model
    ft_results = test_finetuned_model()

    # Print comparison
    print_comparison(base_results, ft_results)

if __name__ == "__main__":
    main()
