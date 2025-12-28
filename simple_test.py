#!/usr/bin/env python3
"""Simple A/B test: Base phi-2 vs Fine-tuned CCP (3 prompts)"""

import json
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROMPTS = [
    "Show me my top accounts",
    "Help me forecast my quarter for my pipeline review?",
    "I'm running out of inbound leads, which accounts should I target to hit my number?",
]

def generate(model, tokenizer, prompt):
    input_text = f"### GTM Prompt:\n{prompt}\n\n### Intent IR:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150, temperature=0.1,
                            do_sample=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Intent IR:" in text:
        text = text.split("### Intent IR:")[-1].strip()
    if "### GTM" in text:
        text = text.split("### GTM")[0].strip()
    return text[:500]

# ===== TEST BASE MODEL =====
print("=" * 60)
print("LOADING BASE PHI-2 (not fine-tuned)")
print("=" * 60)
base_model = AutoModelForCausalLM.from_pretrained(
    "./models/phi-2", torch_dtype=torch.float32, device_map={"": "mps"},
    local_files_only=True, trust_remote_code=True)
base_tok = AutoTokenizer.from_pretrained("./models/phi-2", local_files_only=True)
if base_tok.pad_token is None: base_tok.pad_token = base_tok.eos_token

base_results = []
for p in PROMPTS:
    print(f"\nGenerating for: {p}...")
    base_results.append(generate(base_model, base_tok, p))

del base_model, base_tok
gc.collect()
torch.mps.empty_cache()

# ===== TEST FINE-TUNED MODEL =====
print("\n" + "=" * 60)
print("LOADING FINE-TUNED CCP")
print("=" * 60)
ft_base = AutoModelForCausalLM.from_pretrained(
    "./models/phi-2", torch_dtype=torch.float32, device_map={"": "mps"},
    local_files_only=True, trust_remote_code=True)
ft_model = PeftModel.from_pretrained(ft_base, "./ccp-adapter")
ft_tok = AutoTokenizer.from_pretrained("./ccp-adapter", local_files_only=True)
if ft_tok.pad_token is None: ft_tok.pad_token = ft_tok.eos_token

ft_results = []
for p in PROMPTS:
    print(f"\nGenerating for: {p}...")
    ft_results.append(generate(ft_model, ft_tok, p))

# ===== PRINT COMPARISON =====
print("\n" + "=" * 60)
print("RESULTS COMPARISON: BASE vs FINE-TUNED")
print("=" * 60)

for i, prompt in enumerate(PROMPTS):
    print(f"\n{'─'*60}")
    print(f"PROMPT: {prompt}")
    print("─" * 60)

    print("\n▶ BASE PHI-2 (before fine-tuning):")
    print(base_results[i][:400])

    print("\n▶ FINE-TUNED CCP (after fine-tuning):")
    print(ft_results[i][:400])

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
