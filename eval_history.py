#!/usr/bin/env python3
"""
CCP Evaluation History Viewer

View and compare evaluation results over time.

Usage:
    python eval_history.py                    # Show summary of all runs
    python eval_history.py --compare 2        # Compare last 2 runs
    python eval_history.py --detail latest    # Show detailed latest results
    python eval_history.py --export csv       # Export history to CSV
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

RESULTS_DIR = "./eval_results"


def load_all_results(results_dir: str) -> list[dict]:
    """Load all evaluation results, sorted by timestamp."""
    results = []

    if not os.path.exists(results_dir):
        return results

    for filename in os.listdir(results_dir):
        if filename.startswith("eval_") and filename.endswith(".json") and "latest" not in filename:
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                    data["_filename"] = filename
                    results.append(data)
            except (json.JSONDecodeError, IOError):
                continue

    # Sort by timestamp
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return results


def print_history_summary(results: list[dict]):
    """Print summary table of all evaluation runs."""
    if not results:
        print("No evaluation results found.")
        return

    print("\n" + "=" * 90)
    print("CCP EVALUATION HISTORY")
    print("=" * 90)

    # Group by model type
    by_model = {}
    for r in results:
        model_type = r.get("model_type", "unknown")
        if model_type not in by_model:
            by_model[model_type] = []
        by_model[model_type].append(r)

    for model_type in ["finetuned", "base"]:
        if model_type not in by_model:
            continue

        model_results = by_model[model_type]
        print(f"\n{model_type.upper()} MODEL RUNS ({len(model_results)} total)")
        print("-" * 90)
        print(f"{'DATE':<20} {'EXAMPLES':>10} {'JSON OK':>10} {'ACCURACY':>10} {'TREND':>10}")
        print("-" * 90)

        prev_acc = None
        for r in model_results[:10]:  # Show last 10
            timestamp = r.get("timestamp", "unknown")
            try:
                dt = datetime.fromisoformat(timestamp)
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                date_str = timestamp[:16]

            n_examples = r.get("total_examples", 0)
            json_rate = r.get("aggregate", {}).get("json_valid_rate", 0)
            accuracy = r.get("aggregate", {}).get("field_accuracy", 0)

            # Trend indicator
            if prev_acc is not None:
                if accuracy > prev_acc + 0.01:
                    trend = "    ^"
                elif accuracy < prev_acc - 0.01:
                    trend = "    v"
                else:
                    trend = "    -"
            else:
                trend = ""

            prev_acc = accuracy

            print(f"{date_str:<20} {n_examples:>10} {json_rate:>10.1%} {accuracy:>10.1%} {trend:>10}")


def compare_runs(results: list[dict], n: int = 2, model_type: str = "finetuned"):
    """Compare the last N runs for a model type."""
    model_results = [r for r in results if r.get("model_type") == model_type]

    if len(model_results) < n:
        print(f"Only {len(model_results)} {model_type} runs available.")
        return

    runs_to_compare = model_results[:n]

    print("\n" + "=" * 90)
    print(f"COMPARING LAST {n} {model_type.upper()} RUNS")
    print("=" * 90)

    # Header
    header = f"{'METRIC':<30}"
    for i, r in enumerate(runs_to_compare):
        timestamp = r.get("timestamp", "unknown")
        try:
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime("%m/%d %H:%M")
        except:
            date_str = f"Run {i+1}"
        header += f" {date_str:>12}"

    if n == 2:
        header += f" {'DELTA':>12}"

    print(header)
    print("-" * 90)

    # Metrics to compare
    metrics = [
        ("JSON Valid Rate", lambda r: r.get("aggregate", {}).get("json_valid_rate", 0)),
        ("Field Accuracy", lambda r: r.get("aggregate", {}).get("field_accuracy", 0)),
    ]

    # Add category breakdowns
    categories = set()
    for r in runs_to_compare:
        categories.update(r.get("aggregate", {}).get("by_category", {}).keys())

    for cat in sorted(categories):
        metrics.append((
            f"  {cat}",
            lambda r, c=cat: r.get("aggregate", {}).get("by_category", {}).get(c, {}).get("accuracy", 0)
        ))

    for metric_name, metric_fn in metrics:
        row = f"{metric_name:<30}"
        values = [metric_fn(r) for r in runs_to_compare]

        for v in values:
            row += f" {v:>12.1%}"

        if n == 2:
            delta = values[0] - values[1]
            delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"
            row += f" {delta_str:>12}"

        print(row)


def show_detailed_results(results: list[dict], run_id: str = "latest", model_type: str = "finetuned"):
    """Show detailed results for a specific run."""
    model_results = [r for r in results if r.get("model_type") == model_type]

    if not model_results:
        print(f"No {model_type} results found.")
        return

    if run_id == "latest":
        result = model_results[0]
    else:
        # Try to find by filename or index
        try:
            idx = int(run_id)
            result = model_results[idx]
        except:
            result = next((r for r in model_results if run_id in r.get("_filename", "")), None)
            if not result:
                print(f"Run '{run_id}' not found.")
                return

    print("\n" + "=" * 90)
    print(f"DETAILED RESULTS: {result.get('_filename', 'unknown')}")
    print(f"Model: {result.get('model_type', 'unknown')}")
    print(f"Timestamp: {result.get('timestamp', 'unknown')}")
    print(f"Examples: {result.get('total_examples', 0)}")
    print("=" * 90)

    agg = result.get("aggregate", {})

    print(f"\nOVERALL METRICS")
    print(f"  JSON Valid Rate: {agg.get('json_valid_rate', 0):.1%}")
    print(f"  Field Accuracy:  {agg.get('field_accuracy', 0):.1%}")

    print(f"\nBY CATEGORY")
    for cat, data in sorted(agg.get("by_category", {}).items()):
        print(f"  {cat:<20} {data.get('accuracy', 0):.1%} ({data.get('count', 0)} examples)")

    print(f"\nBY DIFFICULTY")
    for diff, data in sorted(agg.get("by_difficulty", {}).items()):
        print(f"  {diff:<20} {data.get('accuracy', 0):.1%} ({data.get('count', 0)} examples)")

    print(f"\nBY FIELD")
    for field, data in sorted(agg.get("by_field", {}).items(), key=lambda x: -x[1].get("accuracy", 0)):
        print(f"  {field:<25} {data.get('accuracy', 0):.1%}")

    # Show worst examples
    examples = result.get("examples", [])
    failures = sorted(examples, key=lambda x: x.get("scores", {}).get("field_accuracy", 0))[:5]

    if failures:
        print(f"\nWORST PERFORMING EXAMPLES")
        print("-" * 90)
        for ex in failures:
            acc = ex.get("scores", {}).get("field_accuracy", 0)
            print(f"\n  Prompt: {ex.get('prompt', '')[:70]}...")
            print(f"  Category: {ex.get('category', 'unknown')} | Difficulty: {ex.get('difficulty', 'unknown')}")
            print(f"  Accuracy: {acc:.1%}")
            print(f"  Expected: {ex.get('expected', {})}")
            print(f"  Predicted: {ex.get('predicted', {})}")


def export_to_csv(results: list[dict], output_path: str = "eval_history.csv"):
    """Export history to CSV for external analysis."""
    import csv

    rows = []
    for r in results:
        agg = r.get("aggregate", {})
        row = {
            "timestamp": r.get("timestamp", ""),
            "model_type": r.get("model_type", ""),
            "total_examples": r.get("total_examples", 0),
            "json_valid_rate": agg.get("json_valid_rate", 0),
            "field_accuracy": agg.get("field_accuracy", 0),
        }

        # Add category accuracies
        for cat, data in agg.get("by_category", {}).items():
            row[f"cat_{cat}_accuracy"] = data.get("accuracy", 0)

        # Add difficulty accuracies
        for diff, data in agg.get("by_difficulty", {}).items():
            row[f"diff_{diff}_accuracy"] = data.get("accuracy", 0)

        rows.append(row)

    if not rows:
        print("No results to export.")
        return

    # Get all unique keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported {len(rows)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CCP Evaluation History Viewer")
    parser.add_argument("--compare", type=int, default=None,
                       help="Compare last N runs")
    parser.add_argument("--detail", type=str, default=None,
                       help="Show detailed results for a run (use 'latest' or filename)")
    parser.add_argument("--model", choices=["base", "finetuned"], default="finetuned",
                       help="Model type to view (default: finetuned)")
    parser.add_argument("--export", choices=["csv"], default=None,
                       help="Export history to CSV")
    args = parser.parse_args()

    results = load_all_results(RESULTS_DIR)

    if args.export == "csv":
        export_to_csv(results)
    elif args.compare:
        compare_runs(results, args.compare, args.model)
    elif args.detail:
        show_detailed_results(results, args.detail, args.model)
    else:
        print_history_summary(results)


if __name__ == "__main__":
    main()
