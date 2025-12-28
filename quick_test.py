#!/usr/bin/env python3
"""Quick CCP test - just the fine-tuned model with fewer tokens"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TEST_PROMPTS = [
    "Show me my best accounts",
    "Which deals are at risk?",
    "Companies hiring SDRs",
    "I need to hit my number",
    "Expansion opportunities in EMEA",
]

print("Loading fine-tuned CCP model...")
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

print("Model loaded!\n")
print("=" * 60)

for prompt in TEST_PROMPTS:
    print(f"\nPROMPT: {prompt}")
    print("-" * 40)

    input_text = f"### GTM Prompt:\n{prompt}\n\n### Intent IR:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.1,
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

    # Try to parse as JSON
    try:
        if response.startswith("{"):
            brace_count = 0
            end_idx = 0
            for i, char in enumerate(response):
                if char == "{": brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            json_str = response[:end_idx]
            data = json.loads(json_str)
            print(json.dumps(data, indent=2))
            print("\n✓ Valid JSON with fields:", list(data.keys())[:5], "...")
        else:
            print(response[:300])
    except:
        print(response[:300])
        print("\n✗ Not valid JSON")

print("\n" + "=" * 60)
print("TEST COMPLETE")
