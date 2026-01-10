"""
CCP eval command.

Runs evaluation on the fine-tuned CCP model.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import click
import torch

from ccp.config import (
    CCP_SYSTEM_PROMPT,
    EVAL_FIELDS,
)
from ccp.model import load_finetuned_model, load_base_model, load_tokenizer, clear_memory
from ccp.data import load_eval_dataset
from ccp.utils.parsing import parse_ccp_response, format_inference_prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 800,
    temperature: float = 0.1,
) -> str:
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from response
    if prompt in response:
        response = response[len(prompt):]

    return response


def score_response(
    predicted: Optional[Dict],
    expected: Dict,
    is_valid_json: bool
) -> Dict[str, Any]:
    """Score a predicted response against expected."""
    scores = {
        "json_valid": is_valid_json,
        "field_scores": {},
        "field_accuracy": 0.0,
        "reasoning_present": False,
        "clarification_correct": False,
    }

    if not is_valid_json or predicted is None:
        return scores

    # Score each field
    matches = 0
    total = 0

    for field in EVAL_FIELDS:
        if field in expected:
            total += 1
            pred_val = predicted.get(field)
            exp_val = expected.get(field)

            # Handle None/null comparison
            if pred_val is None and exp_val is None:
                match = True
            elif pred_val is None or exp_val is None:
                match = False
            else:
                match = str(pred_val).lower() == str(exp_val).lower()

            scores["field_scores"][field] = match
            if match:
                matches += 1

    scores["field_accuracy"] = matches / total if total > 0 else 0.0

    # Check clarification_needed specifically
    if "clarification_needed" in expected:
        scores["clarification_correct"] = (
            predicted.get("clarification_needed") == expected.get("clarification_needed")
        )

    return scores


@click.command("eval")
@click.option(
    "--model", "model_type",
    default="finetuned",
    type=click.Choice(["finetuned", "base", "both"]),
    help="Which model(s) to evaluate"
)
@click.option(
    "--model-path",
    default="./models/phi-2",
    help="Path to base model"
)
@click.option(
    "--adapter-path",
    default="./ccp-adapter",
    help="Path to LoRA adapter"
)
@click.option(
    "--eval-data",
    default="./data/ccp_eval.jsonl",
    help="Path to evaluation data"
)
@click.option(
    "--category",
    default=None,
    type=click.Choice(["standard", "slang", "adversarial", "ambiguous"]),
    help="Filter by category"
)
@click.option(
    "--quick",
    is_flag=True,
    help="Quick eval with 5 examples"
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit number of examples"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed output"
)
@click.option(
    "--no-save",
    is_flag=True,
    help="Don't save results to file"
)
@click.option(
    "--results-dir",
    default="./eval_results",
    help="Directory to save results"
)
@click.option(
    "--compare",
    is_flag=True,
    help="Compare with previous run"
)
def eval_cmd(
    model_type: str,
    model_path: str,
    adapter_path: str,
    eval_data: str,
    category: Optional[str],
    quick: bool,
    limit: Optional[int],
    verbose: bool,
    no_save: bool,
    results_dir: str,
    compare: bool,
):
    """Evaluate CCP model performance.

    Runs the model on evaluation examples and scores the results
    for JSON validity, field accuracy, and per-field metrics.

    Examples:

        # Quick evaluation (5 examples)
        ccp eval --quick

        # Full evaluation of fine-tuned model
        ccp eval --model finetuned

        # Evaluate specific category
        ccp eval --category adversarial

        # Compare both models
        ccp eval --model both
    """
    click.echo("=" * 60)
    click.echo("CCP Evaluation")
    click.echo("=" * 60)

    # Determine limit
    if quick:
        limit = 5
    elif limit is None:
        limit = None  # No limit

    # Load eval data
    if not Path(eval_data).exists():
        click.echo(f"Error: Eval data not found: {eval_data}", err=True)
        raise SystemExit(1)

    examples = load_eval_dataset(eval_data, category=category, limit=limit)
    click.echo(f"\nLoaded {len(examples)} evaluation examples")
    if category:
        click.echo(f"  Category filter: {category}")

    results = {}

    # Evaluate models
    models_to_eval = []
    if model_type in ["finetuned", "both"]:
        models_to_eval.append("finetuned")
    if model_type in ["base", "both"]:
        models_to_eval.append("base")

    for mt in models_to_eval:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Evaluating: {mt.upper()} model")
        click.echo("=" * 60)

        # Load model
        if mt == "finetuned":
            if not Path(adapter_path).exists():
                click.echo(f"Error: Adapter not found: {adapter_path}", err=True)
                continue
            model, tokenizer = load_finetuned_model(model_path, adapter_path)
        else:
            tokenizer = load_tokenizer(model_path)
            model = load_base_model(model_path)
            model.eval()

        # Run evaluation
        eval_results = run_evaluation(model, tokenizer, examples, mt, verbose)
        results[mt] = eval_results

        # Print results
        print_results(eval_results, mt)

        # Clear memory
        del model
        clear_memory()

    # Save results
    if not no_save:
        save_results(results, results_dir, model_type)

    # Compare with previous
    if compare and not no_save:
        compare_with_previous(results, results_dir, model_type)


def run_evaluation(
    model,
    tokenizer,
    examples: List[Dict],
    model_type: str,
    verbose: bool
) -> Dict[str, Any]:
    """Run evaluation on examples."""
    results = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(examples),
        "overall": {
            "json_valid": 0,
            "field_accuracy": 0.0,
            "total_latency_ms": 0,
        },
        "by_category": {},
        "by_difficulty": {},
        "by_field": {field: {"correct": 0, "total": 0} for field in EVAL_FIELDS},
        "examples": [],
    }

    for idx, example in enumerate(examples):
        if verbose:
            click.echo(f"\n[{idx+1}/{len(examples)}] {example['gtm_prompt'][:50]}...")

        # Format prompt
        if model_type == "finetuned":
            prompt = format_inference_prompt(example["gtm_prompt"], CCP_SYSTEM_PROMPT)
        else:
            prompt = f"### GTM Prompt:\n{example['gtm_prompt']}\n\n### Response:\n"

        # Generate
        start_time = time.time()
        response = generate_response(model, tokenizer, prompt)
        latency_ms = (time.time() - start_time) * 1000

        # Parse response
        reasoning, intent_ir, is_valid = parse_ccp_response(response)

        # Score
        expected = example.get("expected", example.get("intent_ir", {}))
        scores = score_response(intent_ir, expected, is_valid)
        scores["latency_ms"] = latency_ms
        scores["reasoning_present"] = reasoning is not None and len(reasoning or "") > 10

        # Update overall stats
        if is_valid:
            results["overall"]["json_valid"] += 1
        results["overall"]["field_accuracy"] += scores["field_accuracy"]
        results["overall"]["total_latency_ms"] += latency_ms

        # Update by category
        cat = example.get("category", "unknown")
        if cat not in results["by_category"]:
            results["by_category"][cat] = {"json_valid": 0, "field_accuracy": 0, "count": 0}
        results["by_category"][cat]["count"] += 1
        if is_valid:
            results["by_category"][cat]["json_valid"] += 1
        results["by_category"][cat]["field_accuracy"] += scores["field_accuracy"]

        # Update by difficulty
        diff = example.get("difficulty", "unknown")
        if diff not in results["by_difficulty"]:
            results["by_difficulty"][diff] = {"json_valid": 0, "field_accuracy": 0, "count": 0}
        results["by_difficulty"][diff]["count"] += 1
        if is_valid:
            results["by_difficulty"][diff]["json_valid"] += 1
        results["by_difficulty"][diff]["field_accuracy"] += scores["field_accuracy"]

        # Update by field
        for field, correct in scores["field_scores"].items():
            results["by_field"][field]["total"] += 1
            if correct:
                results["by_field"][field]["correct"] += 1

        # Store example result
        results["examples"].append({
            "prompt": example["gtm_prompt"],
            "category": cat,
            "difficulty": diff,
            "is_valid": is_valid,
            "field_accuracy": scores["field_accuracy"],
            "latency_ms": latency_ms,
        })

        if verbose:
            status = "VALID" if is_valid else "INVALID"
            click.echo(f"  JSON: {status}, Accuracy: {scores['field_accuracy']:.1%}, Latency: {latency_ms:.0f}ms")

    # Compute averages
    n = len(examples)
    if n > 0:
        results["overall"]["json_valid_rate"] = results["overall"]["json_valid"] / n
        results["overall"]["field_accuracy"] = results["overall"]["field_accuracy"] / n
        results["overall"]["avg_latency_ms"] = results["overall"]["total_latency_ms"] / n

    for cat in results["by_category"]:
        c = results["by_category"][cat]["count"]
        if c > 0:
            results["by_category"][cat]["json_valid_rate"] = results["by_category"][cat]["json_valid"] / c
            results["by_category"][cat]["field_accuracy"] = results["by_category"][cat]["field_accuracy"] / c

    for diff in results["by_difficulty"]:
        d = results["by_difficulty"][diff]["count"]
        if d > 0:
            results["by_difficulty"][diff]["json_valid_rate"] = results["by_difficulty"][diff]["json_valid"] / d
            results["by_difficulty"][diff]["field_accuracy"] = results["by_difficulty"][diff]["field_accuracy"] / d

    return results


def print_results(results: Dict, model_type: str):
    """Print evaluation results."""
    click.echo(f"\n{'Results':=^60}")

    overall = results["overall"]
    click.echo(f"\n  JSON Valid Rate:    {overall.get('json_valid_rate', 0):.1%}")
    click.echo(f"  Field Accuracy:     {overall.get('field_accuracy', 0):.1%}")
    click.echo(f"  Avg Latency:        {overall.get('avg_latency_ms', 0):.0f}ms")

    # By category
    if results["by_category"]:
        click.echo(f"\n  {'By Category':-^50}")
        for cat, stats in sorted(results["by_category"].items()):
            click.echo(
                f"    {cat:15} JSON: {stats.get('json_valid_rate', 0):.1%}, "
                f"Accuracy: {stats.get('field_accuracy', 0):.1%} (n={stats['count']})"
            )

    # By field
    click.echo(f"\n  {'By Field':-^50}")
    for field, stats in results["by_field"].items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            click.echo(f"    {field:25} {acc:.1%} ({stats['correct']}/{stats['total']})")


def save_results(results: Dict, results_dir: str, model_type: str):
    """Save results to file."""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for mt, data in results.items():
        filename = f"eval_{mt}_{timestamp}.json"
        filepath = results_path / filename

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        click.echo(f"\nResults saved to: {filepath}")

        # Update latest symlink
        latest_link = results_path / f"latest_{mt}.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(filename)


def compare_with_previous(results: Dict, results_dir: str, model_type: str):
    """Compare with previous evaluation results."""
    results_path = Path(results_dir)

    for mt in results:
        latest_link = results_path / f"latest_{mt}.json"
        if not latest_link.exists():
            continue

        # Find previous (second latest)
        all_files = sorted(
            results_path.glob(f"eval_{mt}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if len(all_files) < 2:
            continue

        previous_file = all_files[1]
        with open(previous_file) as f:
            previous = json.load(f)

        current = results[mt]

        click.echo(f"\n{'Comparison with Previous':=^60}")
        click.echo(f"  Previous: {previous_file.name}")

        prev_json = previous["overall"].get("json_valid_rate", 0)
        curr_json = current["overall"].get("json_valid_rate", 0)
        diff_json = curr_json - prev_json

        prev_acc = previous["overall"].get("field_accuracy", 0)
        curr_acc = current["overall"].get("field_accuracy", 0)
        diff_acc = curr_acc - prev_acc

        click.echo(f"\n  JSON Valid Rate:  {curr_json:.1%} ({diff_json:+.1%})")
        click.echo(f"  Field Accuracy:   {curr_acc:.1%} ({diff_acc:+.1%})")
