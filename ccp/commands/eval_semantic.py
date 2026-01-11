"""
CCP semantic evaluation command.

Evaluates GTM context understanding and business value,
not just syntactic field matching.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import click
import torch

from ccp.config import CCP_SYSTEM_PROMPT
from ccp.model import load_finetuned_model, load_base_model, load_tokenizer, clear_memory
from ccp.data import load_eval_dataset
from ccp.utils.parsing import parse_ccp_response, format_inference_prompt
from ccp.eval.semantic_scoring import (
    score_semantic_understanding,
    compare_semantic_vs_exact,
)


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


@click.command("eval-semantic")
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
    "--compare-with-exact",
    is_flag=True,
    help="Compare semantic scoring with exact matching"
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
def eval_semantic_cmd(
    model_type: str,
    model_path: str,
    adapter_path: str,
    eval_data: str,
    category: Optional[str],
    quick: bool,
    limit: Optional[int],
    verbose: bool,
    compare_with_exact: bool,
    no_save: bool,
    results_dir: str,
):
    """Evaluate CCP with semantic understanding metrics.

    Tests GTM context understanding and business value,
    not just syntactic correctness.

    Examples:

        # Quick semantic evaluation
        ccp eval-semantic --quick

        # Compare semantic vs exact matching
        ccp eval-semantic --compare-with-exact

        # Evaluate both models
        ccp eval-semantic --model both
    """
    click.echo("=" * 60)
    click.echo("CCP Semantic Evaluation")
    click.echo("=" * 60)
    click.echo("\nTests: GTM context understanding & business value")

    # Determine limit
    if quick:
        limit = 5
    elif limit is None:
        limit = None

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
        eval_results = run_semantic_evaluation(
            model, tokenizer, examples, mt, verbose, compare_with_exact
        )
        results[mt] = eval_results

        # Print results
        print_semantic_results(eval_results, mt, compare_with_exact)

        # Clear memory
        del model
        clear_memory()

    # Save results
    if not no_save:
        save_semantic_results(results, results_dir, model_type)


def run_semantic_evaluation(
    model,
    tokenizer,
    examples: List[Dict],
    model_type: str,
    verbose: bool,
    compare_with_exact: bool,
) -> Dict[str, Any]:
    """Run semantic evaluation on examples."""
    results = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "num_examples": len(examples),
        "eval_type": "semantic",
        "overall": {
            "json_valid": 0,
            "critical_field_accuracy": 0.0,
            "tool_selection_accuracy": 0.0,
            "overall_semantic_accuracy": 0.0,
            "total_latency_ms": 0,
        },
        "by_category": {},
        "by_difficulty": {},
        "examples": [],
    }

    if compare_with_exact:
        results["comparison_with_exact"] = {
            "exact_field_accuracy": 0.0,
            "semantic_field_accuracy": 0.0,
            "improvement": 0.0,
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

        # Semantic scoring
        expected = example.get("expected", example.get("intent_ir", {}))
        semantic_score = score_semantic_understanding(intent_ir, expected, is_valid)

        # Update overall stats
        if is_valid:
            results["overall"]["json_valid"] += 1
        results["overall"]["critical_field_accuracy"] += semantic_score["critical_field_accuracy"]
        results["overall"]["tool_selection_accuracy"] += (1.0 if semantic_score["tool_selection_valid"] else 0.0)
        results["overall"]["overall_semantic_accuracy"] += semantic_score["overall_semantic_accuracy"]
        results["overall"]["total_latency_ms"] += latency_ms

        # Update by category
        cat = example.get("category", "unknown")
        if cat not in results["by_category"]:
            results["by_category"][cat] = {
                "json_valid": 0,
                "critical_field_accuracy": 0.0,
                "tool_selection_accuracy": 0.0,
                "count": 0
            }
        results["by_category"][cat]["count"] += 1
        if is_valid:
            results["by_category"][cat]["json_valid"] += 1
        results["by_category"][cat]["critical_field_accuracy"] += semantic_score["critical_field_accuracy"]
        results["by_category"][cat]["tool_selection_accuracy"] += (1.0 if semantic_score["tool_selection_valid"] else 0.0)

        # Update by difficulty
        diff = example.get("difficulty", "unknown")
        if diff not in results["by_difficulty"]:
            results["by_difficulty"][diff] = {
                "json_valid": 0,
                "critical_field_accuracy": 0.0,
                "tool_selection_accuracy": 0.0,
                "count": 0
            }
        results["by_difficulty"][diff]["count"] += 1
        if is_valid:
            results["by_difficulty"][diff]["json_valid"] += 1
        results["by_difficulty"][diff]["critical_field_accuracy"] += semantic_score["critical_field_accuracy"]
        results["by_difficulty"][diff]["tool_selection_accuracy"] += (1.0 if semantic_score["tool_selection_valid"] else 0.0)

        # Compare with exact matching if requested
        if compare_with_exact:
            comparison = compare_semantic_vs_exact(intent_ir, expected, is_valid)
            results["comparison_with_exact"]["exact_field_accuracy"] += comparison.get("exact_field_accuracy", 0.0)
            results["comparison_with_exact"]["semantic_field_accuracy"] += comparison.get("semantic_field_accuracy", 0.0)

        # Store example result
        example_result = {
            "prompt": example["gtm_prompt"],
            "category": cat,
            "difficulty": diff,
            "is_valid": is_valid,
            "critical_field_accuracy": semantic_score["critical_field_accuracy"],
            "tool_selection_valid": semantic_score["tool_selection_valid"],
            "overall_semantic_accuracy": semantic_score["overall_semantic_accuracy"],
            "latency_ms": latency_ms,
        }

        if compare_with_exact:
            example_result["exact_vs_semantic"] = comparison.get("comparison", {})

        results["examples"].append(example_result)

        if verbose:
            status = "VALID" if is_valid else "INVALID"
            tool_status = "✓" if semantic_score["tool_selection_valid"] else "✗"
            click.echo(f"  JSON: {status}, Critical: {semantic_score['critical_field_accuracy']:.1%}, "
                      f"Tools: {tool_status}, Overall: {semantic_score['overall_semantic_accuracy']:.1%}")

    # Compute averages
    n = len(examples)
    if n > 0:
        results["overall"]["json_valid_rate"] = results["overall"]["json_valid"] / n
        results["overall"]["critical_field_accuracy"] = results["overall"]["critical_field_accuracy"] / n
        results["overall"]["tool_selection_accuracy"] = results["overall"]["tool_selection_accuracy"] / n
        results["overall"]["overall_semantic_accuracy"] = results["overall"]["overall_semantic_accuracy"] / n
        results["overall"]["avg_latency_ms"] = results["overall"]["total_latency_ms"] / n

    for cat in results["by_category"]:
        c = results["by_category"][cat]["count"]
        if c > 0:
            results["by_category"][cat]["json_valid_rate"] = results["by_category"][cat]["json_valid"] / c
            results["by_category"][cat]["critical_field_accuracy"] = results["by_category"][cat]["critical_field_accuracy"] / c
            results["by_category"][cat]["tool_selection_accuracy"] = results["by_category"][cat]["tool_selection_accuracy"] / c

    for diff in results["by_difficulty"]:
        d = results["by_difficulty"][diff]["count"]
        if d > 0:
            results["by_difficulty"][diff]["json_valid_rate"] = results["by_difficulty"][diff]["json_valid"] / d
            results["by_difficulty"][diff]["critical_field_accuracy"] = results["by_difficulty"][diff]["critical_field_accuracy"] / d
            results["by_difficulty"][diff]["tool_selection_accuracy"] = results["by_difficulty"][diff]["tool_selection_accuracy"] / d

    if compare_with_exact and n > 0:
        results["comparison_with_exact"]["exact_field_accuracy"] /= n
        results["comparison_with_exact"]["semantic_field_accuracy"] /= n
        results["comparison_with_exact"]["improvement"] = (
            results["comparison_with_exact"]["semantic_field_accuracy"] -
            results["comparison_with_exact"]["exact_field_accuracy"]
        )

    return results


def print_semantic_results(results: Dict, model_type: str, compare_with_exact: bool):
    """Print semantic evaluation results."""
    click.echo(f"\n{'Semantic Evaluation Results':=^60}")

    overall = results["overall"]
    click.echo(f"\n  JSON Valid Rate:              {overall.get('json_valid_rate', 0):.1%}")
    click.echo(f"  Critical Field Accuracy:      {overall.get('critical_field_accuracy', 0):.1%}")
    click.echo(f"  Tool Selection Accuracy:      {overall.get('tool_selection_accuracy', 0):.1%}")
    click.echo(f"  Overall Semantic Score:       {overall.get('overall_semantic_accuracy', 0):.1%}")
    click.echo(f"  Avg Latency:                  {overall.get('avg_latency_ms', 0):.0f}ms")

    # By category
    if results["by_category"]:
        click.echo(f"\n  {'By Category':-^50}")
        for cat, stats in sorted(results["by_category"].items()):
            click.echo(
                f"    {cat:15} Critical: {stats.get('critical_field_accuracy', 0):.1%}, "
                f"Tools: {stats.get('tool_selection_accuracy', 0):.1%} (n={stats['count']})"
            )

    # Comparison with exact matching
    if compare_with_exact and "comparison_with_exact" in results:
        comp = results["comparison_with_exact"]
        click.echo(f"\n  {'Exact vs Semantic Comparison':-^50}")
        click.echo(f"    Exact Matching Accuracy:    {comp['exact_field_accuracy']:.1%}")
        click.echo(f"    Semantic Matching Accuracy: {comp['semantic_field_accuracy']:.1%}")
        click.echo(f"    Improvement:                {comp['improvement']:+.1%}")


def save_semantic_results(results: Dict, results_dir: str, model_type: str):
    """Save semantic results to file."""
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for mt, data in results.items():
        filename = f"eval_semantic_{mt}_{timestamp}.json"
        filepath = results_path / filename

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        click.echo(f"\nResults saved to: {filepath}")

        # Update latest symlink
        latest_link = results_path / f"latest_semantic_{mt}.json"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(filename)
