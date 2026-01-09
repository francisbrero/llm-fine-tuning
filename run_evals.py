#!/usr/bin/env python3
"""
CCP Evaluation Framework

Runs evaluation suite against base and fine-tuned models, tracking results over time.

Usage:
    python run_evals.py                    # Run full eval suite
    python run_evals.py --model finetuned  # Only test fine-tuned model
    python run_evals.py --model base       # Only test base model
    python run_evals.py --category slang   # Only test slang category
    python run_evals.py --quick            # Run subset (5 samples) for quick check
"""

import argparse
import json
import os
import re
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ===== CONFIGURATION =====
BASE_MODEL_PATH = "./models/phi-2"
ADAPTER_PATH = "./ccp-adapter"
EVAL_DATASET_PATH = "./data/ccp_eval.jsonl"
RESULTS_DIR = "./eval_results"

# Fields to evaluate for accuracy
EVAL_FIELDS = [
    "intent_type",
    "motion",
    "role_assumption",
    "account_scope",
    "time_horizon",
    "geography_scope",
    "output_format",
    "clarification_needed",
]

# CCP v2 System Prompt
CCP_SYSTEM_PROMPT = """You are the Phoenix Context Collapse Parser (CCP). Your job is to transform ambiguous GTM (Go-To-Market) prompts into structured GTM Intent IR.

IMPORTANT: You must REASON about the request before producing structured output.

Given a user's GTM request:
1. First, analyze the prompt in a <reasoning> section
2. Then, output the structured intent in an <intent_ir> section as JSON

Format your response EXACTLY as:
<reasoning>
[Your analysis here]
</reasoning>

<intent_ir>
[Valid JSON here]
</intent_ir>"""


def load_eval_dataset(path: str, category: Optional[str] = None) -> list[dict]:
    """Load evaluation dataset, optionally filtering by category."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                if category is None or example.get("category") == category:
                    examples.append(example)
    return examples


def generate_base(model, tokenizer, prompt: str) -> str:
    """Generate with base model (simple prompt format)."""
    input_text = f"### GTM Prompt:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "### Response:" in text:
        text = text.split("### Response:")[-1].strip()
    if "### GTM" in text:
        text = text.split("### GTM")[0].strip()
    return text


def generate_finetuned(model, tokenizer, prompt: str) -> str:
    """Generate with fine-tuned CCP model (Chain-of-Thought format)."""
    input_text = f"<s>[INST] {CCP_SYSTEM_PROMPT}\n\nUser request: {prompt} [/INST]\n"
    inputs = tokenizer(input_text, return_tensors="pt").to("mps")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=700,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if "[/INST]" in text:
        text = text.split("[/INST]")[-1].strip()
    return text.replace("</s>", "").strip()


def parse_ccp_response(response: str) -> tuple[str, dict, bool]:
    """
    Parse CCP response into (reasoning, intent_ir, is_valid_json).
    """
    # Extract reasoning
    reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

    # Extract intent_ir JSON
    ir_match = re.search(r'<intent_ir>\s*(.*?)\s*</intent_ir>', response, re.DOTALL)
    if ir_match:
        try:
            intent_ir = json.loads(ir_match.group(1).strip())
            return reasoning, intent_ir, True
        except json.JSONDecodeError:
            pass

    # Fallback: try to find any JSON in the response
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            intent_ir = json.loads(json_match.group(0))
            return reasoning, intent_ir, True
        except json.JSONDecodeError:
            pass

    return reasoning, {}, False


def try_parse_base_response(response: str) -> dict:
    """Try to extract any structured info from base model response."""
    # Base model doesn't output structured JSON, so we return empty
    # This gives a baseline of what the model can do without fine-tuning
    return {}


def score_response(predicted: dict, expected: dict, is_valid_json: bool) -> dict:
    """
    Score a single response against expected output.

    Returns dict with:
    - json_valid: bool
    - field_scores: dict of field -> 1.0/0.0
    - field_accuracy: float (average across expected fields)
    - reasoning_present: bool
    - clarification_correct: bool (if applicable)
    """
    scores = {
        "json_valid": is_valid_json,
        "field_scores": {},
        "field_accuracy": 0.0,
        "reasoning_present": False,
        "clarification_correct": None,
    }

    if not is_valid_json:
        return scores

    # Score each expected field
    matched = 0
    total = 0

    for field in EVAL_FIELDS:
        if field in expected:
            total += 1
            pred_value = predicted.get(field)
            exp_value = expected[field]

            # Handle null/None comparison
            if pred_value is None and exp_value is None:
                match = True
            elif pred_value is None or exp_value is None:
                match = False
            else:
                match = str(pred_value).lower() == str(exp_value).lower()

            scores["field_scores"][field] = 1.0 if match else 0.0
            if match:
                matched += 1

    scores["field_accuracy"] = matched / total if total > 0 else 0.0

    # Special handling for clarification_needed
    if "clarification_needed" in expected:
        pred_clarification = predicted.get("clarification_needed", False)
        exp_clarification = expected["clarification_needed"]
        scores["clarification_correct"] = pred_clarification == exp_clarification

    return scores


def run_evaluation(
    model,
    tokenizer,
    eval_data: list[dict],
    model_type: str,  # "base" or "finetuned"
    verbose: bool = False
) -> dict:
    """
    Run evaluation on a model.

    Returns aggregate metrics and per-example results.
    """
    results = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "total_examples": len(eval_data),
        "examples": [],
        "aggregate": {
            "json_valid_rate": 0.0,
            "field_accuracy": 0.0,
            "by_category": {},
            "by_difficulty": {},
            "by_field": {},
        }
    }

    generate_fn = generate_finetuned if model_type == "finetuned" else generate_base

    json_valid_count = 0
    total_field_accuracy = 0.0
    category_scores = {}
    difficulty_scores = {}
    field_totals = {f: {"correct": 0, "total": 0} for f in EVAL_FIELDS}

    for i, example in enumerate(eval_data):
        prompt = example["gtm_prompt"]
        expected = example["expected"]
        category = example.get("category", "unknown")
        difficulty = example.get("difficulty", "unknown")

        if verbose:
            print(f"[{i+1}/{len(eval_data)}] {prompt[:50]}...")

        # Generate response
        start_time = time.time()
        raw_response = generate_fn(model, tokenizer, prompt)
        latency_ms = (time.time() - start_time) * 1000

        # Parse response
        if model_type == "finetuned":
            reasoning, intent_ir, is_valid = parse_ccp_response(raw_response)
        else:
            reasoning = ""
            intent_ir = try_parse_base_response(raw_response)
            is_valid = bool(intent_ir)

        # Score response
        scores = score_response(intent_ir, expected, is_valid)
        scores["reasoning_present"] = len(reasoning) > 20

        # Aggregate
        if scores["json_valid"]:
            json_valid_count += 1
        total_field_accuracy += scores["field_accuracy"]

        # By category
        if category not in category_scores:
            category_scores[category] = {"total": 0, "accuracy_sum": 0.0, "json_valid": 0}
        category_scores[category]["total"] += 1
        category_scores[category]["accuracy_sum"] += scores["field_accuracy"]
        if scores["json_valid"]:
            category_scores[category]["json_valid"] += 1

        # By difficulty
        if difficulty not in difficulty_scores:
            difficulty_scores[difficulty] = {"total": 0, "accuracy_sum": 0.0}
        difficulty_scores[difficulty]["total"] += 1
        difficulty_scores[difficulty]["accuracy_sum"] += scores["field_accuracy"]

        # By field
        for field, score in scores["field_scores"].items():
            field_totals[field]["total"] += 1
            field_totals[field]["correct"] += score

        # Store example result
        results["examples"].append({
            "prompt": prompt,
            "expected": expected,
            "predicted": intent_ir,
            "reasoning": reasoning[:500] if reasoning else "",
            "scores": scores,
            "latency_ms": latency_ms,
            "category": category,
            "difficulty": difficulty,
        })

    # Compute aggregates
    n = len(eval_data)
    results["aggregate"]["json_valid_rate"] = json_valid_count / n if n > 0 else 0.0
    results["aggregate"]["field_accuracy"] = total_field_accuracy / n if n > 0 else 0.0

    for cat, data in category_scores.items():
        results["aggregate"]["by_category"][cat] = {
            "accuracy": data["accuracy_sum"] / data["total"] if data["total"] > 0 else 0.0,
            "json_valid_rate": data["json_valid"] / data["total"] if data["total"] > 0 else 0.0,
            "count": data["total"],
        }

    for diff, data in difficulty_scores.items():
        results["aggregate"]["by_difficulty"][diff] = {
            "accuracy": data["accuracy_sum"] / data["total"] if data["total"] > 0 else 0.0,
            "count": data["total"],
        }

    for field, data in field_totals.items():
        if data["total"] > 0:
            results["aggregate"]["by_field"][field] = {
                "accuracy": data["correct"] / data["total"],
                "count": data["total"],
            }

    return results


def print_results(results: dict, compare_to: Optional[dict] = None):
    """Print evaluation results with optional comparison."""
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {results['model_type'].upper()}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Examples: {results['total_examples']}")
    print("=" * 70)

    agg = results["aggregate"]

    # Overall metrics
    print(f"\n{'METRIC':<30} {'SCORE':>10}", end="")
    if compare_to:
        print(f" {'BASELINE':>10} {'DELTA':>10}")
    else:
        print()

    print("-" * 70)

    def print_metric(name: str, value: float, baseline_value: Optional[float] = None):
        print(f"{name:<30} {value:>10.1%}", end="")
        if baseline_value is not None:
            delta = value - baseline_value
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            print(f" {baseline_value:>10.1%} {delta_str:>10}")
        else:
            print()

    baseline_agg = compare_to["aggregate"] if compare_to else None

    print_metric(
        "JSON Valid Rate",
        agg["json_valid_rate"],
        baseline_agg["json_valid_rate"] if baseline_agg else None
    )
    print_metric(
        "Field Accuracy (avg)",
        agg["field_accuracy"],
        baseline_agg["field_accuracy"] if baseline_agg else None
    )

    # By category
    print(f"\n{'BY CATEGORY':<30} {'ACCURACY':>10} {'JSON OK':>10} {'COUNT':>8}")
    print("-" * 70)
    for cat, data in sorted(agg["by_category"].items()):
        print(f"{cat:<30} {data['accuracy']:>10.1%} {data['json_valid_rate']:>10.1%} {data['count']:>8}")

    # By difficulty
    print(f"\n{'BY DIFFICULTY':<30} {'ACCURACY':>10} {'COUNT':>8}")
    print("-" * 70)
    for diff, data in sorted(agg["by_difficulty"].items()):
        print(f"{diff:<30} {data['accuracy']:>10.1%} {data['count']:>8}")

    # By field
    print(f"\n{'BY FIELD':<30} {'ACCURACY':>10} {'COUNT':>8}")
    print("-" * 70)
    for field, data in sorted(agg["by_field"].items(), key=lambda x: -x[1]["accuracy"]):
        print(f"{field:<30} {data['accuracy']:>10.1%} {data['count']:>8}")

    # Show failures
    failures = [ex for ex in results["examples"] if ex["scores"]["field_accuracy"] < 0.5]
    if failures:
        print(f"\n{'LOW ACCURACY EXAMPLES (<50%)':<70}")
        print("-" * 70)
        for ex in failures[:5]:
            print(f"  Prompt: {ex['prompt'][:60]}...")
            print(f"  Expected: {ex['expected']}")
            print(f"  Got: {ex['predicted']}")
            print(f"  Accuracy: {ex['scores']['field_accuracy']:.1%}")
            print()


def save_results(results: dict, output_dir: str):
    """Save results to JSON file with timestamp."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = results["model_type"]
    filename = f"eval_{model_type}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {filepath}")

    # Also update latest symlink/copy
    latest_path = os.path.join(output_dir, f"eval_{model_type}_latest.json")
    with open(latest_path, "w") as f:
        json.dump(results, f, indent=2)

    return filepath


def load_latest_results(output_dir: str, model_type: str) -> Optional[dict]:
    """Load most recent results for comparison."""
    latest_path = os.path.join(output_dir, f"eval_{model_type}_latest.json")
    if os.path.exists(latest_path):
        with open(latest_path, "r") as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="CCP Evaluation Framework")
    parser.add_argument("--model", choices=["base", "finetuned", "both"], default="both",
                       help="Which model(s) to evaluate")
    parser.add_argument("--category", type=str, default=None,
                       help="Filter to specific category (standard, slang, adversarial, ambiguous)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: only run 5 examples")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show progress during evaluation")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save results to disk")
    args = parser.parse_args()

    # Load eval dataset
    print(f"Loading eval dataset from {EVAL_DATASET_PATH}...")
    eval_data = load_eval_dataset(EVAL_DATASET_PATH, args.category)

    if args.quick:
        eval_data = eval_data[:5]
        print(f"Quick mode: using {len(eval_data)} examples")
    else:
        print(f"Loaded {len(eval_data)} examples")

    if args.category:
        print(f"Filtered to category: {args.category}")

    results = {}

    # ===== BASE MODEL =====
    if args.model in ["base", "both"]:
        print("\n" + "=" * 70)
        print("LOADING BASE MODEL (phi-2, not fine-tuned)")
        print("=" * 70)

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            device_map={"": "mps"},
            local_files_only=True,
            trust_remote_code=True
        )
        base_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token

        print("Running base model evaluation...")
        results["base"] = run_evaluation(
            base_model, base_tokenizer, eval_data, "base", verbose=args.verbose
        )

        # Clean up
        del base_model, base_tokenizer
        gc.collect()
        torch.mps.empty_cache()

    # ===== FINE-TUNED MODEL =====
    if args.model in ["finetuned", "both"]:
        print("\n" + "=" * 70)
        print("LOADING FINE-TUNED CCP MODEL")
        print("=" * 70)

        ft_base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
            device_map={"": "mps"},
            local_files_only=True,
            trust_remote_code=True
        )
        ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_PATH)
        ft_model.eval()
        ft_tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH, local_files_only=True)
        if ft_tokenizer.pad_token is None:
            ft_tokenizer.pad_token = ft_tokenizer.eos_token

        print("Running fine-tuned model evaluation...")
        results["finetuned"] = run_evaluation(
            ft_model, ft_tokenizer, eval_data, "finetuned", verbose=args.verbose
        )

        del ft_model, ft_base, ft_tokenizer
        gc.collect()
        torch.mps.empty_cache()

    # ===== PRINT & SAVE RESULTS =====
    for model_type, result in results.items():
        # Load previous results for comparison
        compare_to = None
        if not args.no_save:
            compare_to = load_latest_results(RESULTS_DIR, model_type)

        print_results(result, compare_to)

        if not args.no_save:
            save_results(result, RESULTS_DIR)

    # ===== COMPARISON SUMMARY =====
    if len(results) == 2:
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)

        base_acc = results["base"]["aggregate"]["field_accuracy"]
        ft_acc = results["finetuned"]["aggregate"]["field_accuracy"]
        base_json = results["base"]["aggregate"]["json_valid_rate"]
        ft_json = results["finetuned"]["aggregate"]["json_valid_rate"]

        print(f"\n{'METRIC':<30} {'BASE':>12} {'FINE-TUNED':>12} {'IMPROVEMENT':>12}")
        print("-" * 70)
        print(f"{'JSON Valid Rate':<30} {base_json:>12.1%} {ft_json:>12.1%} {ft_json - base_json:>+12.1%}")
        print(f"{'Field Accuracy':<30} {base_acc:>12.1%} {ft_acc:>12.1%} {ft_acc - base_acc:>+12.1%}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
