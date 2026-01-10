"""
CCP history command.

View and analyze evaluation history.
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import click


def load_all_results(results_dir: str) -> List[Dict]:
    """Load all evaluation results, sorted by timestamp descending."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    results = []
    for filepath in results_path.glob("eval_*.json"):
        # Skip symlinks
        if filepath.is_symlink():
            continue

        try:
            with open(filepath) as f:
                data = json.load(f)
                data["_filepath"] = str(filepath)
                data["_filename"] = filepath.name
                results.append(data)
        except (json.JSONDecodeError, IOError):
            continue

    # Sort by timestamp descending
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return results


@click.command("history")
@click.option(
    "--results-dir",
    default="./eval_results",
    help="Directory containing evaluation results"
)
@click.option(
    "--model", "model_type",
    default="finetuned",
    type=click.Choice(["finetuned", "base", "all"]),
    help="Filter by model type"
)
@click.option(
    "--compare",
    default=None,
    type=int,
    help="Compare last N runs"
)
@click.option(
    "--detail",
    default=None,
    help="Show detailed results (use 'latest' or index number)"
)
@click.option(
    "--export",
    default=None,
    type=click.Path(),
    help="Export history to CSV file"
)
def history(
    results_dir: str,
    model_type: str,
    compare: Optional[int],
    detail: Optional[str],
    export: Optional[str],
):
    """View and analyze evaluation history.

    Shows a summary of past evaluation runs with metrics and trends.

    Examples:

        # View history summary
        ccp history

        # Compare last 2 runs
        ccp history --compare 2

        # Show detailed results for latest run
        ccp history --detail latest

        # Export to CSV
        ccp history --export results.csv
    """
    results = load_all_results(results_dir)

    if not results:
        click.echo(f"No evaluation results found in {results_dir}")
        return

    # Filter by model type
    if model_type != "all":
        results = [r for r in results if r.get("model_type") == model_type]

    if not results:
        click.echo(f"No results found for model type: {model_type}")
        return

    if export:
        export_to_csv(results, export)
        return

    if detail:
        show_detail(results, detail)
        return

    if compare:
        compare_runs(results, compare)
        return

    # Default: show summary
    show_summary(results, model_type)


def show_summary(results: List[Dict], model_type: str):
    """Show summary of evaluation history."""
    click.echo("=" * 70)
    click.echo(f"CCP Evaluation History ({model_type})")
    click.echo("=" * 70)

    click.echo(f"\n{'#':<4} {'Date':<20} {'Examples':<10} {'JSON OK':<10} {'Accuracy':<10} {'Trend':<8}")
    click.echo("-" * 70)

    prev_accuracy = None
    for idx, result in enumerate(results[:20]):  # Show last 20
        timestamp = result.get("timestamp", "Unknown")
        try:
            dt = datetime.fromisoformat(timestamp)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            date_str = timestamp[:16] if timestamp else "Unknown"

        n_examples = result.get("num_examples", 0)
        overall = result.get("overall", {})
        json_rate = overall.get("json_valid_rate", 0)
        accuracy = overall.get("field_accuracy", 0)

        # Trend indicator
        if prev_accuracy is not None:
            diff = accuracy - prev_accuracy
            if diff > 0.01:
                trend = "UP"
            elif diff < -0.01:
                trend = "DOWN"
            else:
                trend = "-"
        else:
            trend = "-"
        prev_accuracy = accuracy

        click.echo(
            f"{idx:<4} {date_str:<20} {n_examples:<10} {json_rate:<10.1%} {accuracy:<10.1%} {trend:<8}"
        )

    click.echo("\n" + "=" * 70)
    click.echo(f"Total runs: {len(results)}")


def compare_runs(results: List[Dict], n: int):
    """Compare last N runs."""
    if len(results) < n:
        click.echo(f"Only {len(results)} runs available, comparing all")
        n = len(results)

    runs_to_compare = results[:n]

    click.echo("=" * 70)
    click.echo(f"Comparing Last {n} Runs")
    click.echo("=" * 70)

    # Header
    header = f"{'Metric':<25}"
    for idx, run in enumerate(runs_to_compare):
        timestamp = run.get("timestamp", "Unknown")[:10]
        header += f" {timestamp:<12}"
    click.echo(header)
    click.echo("-" * 70)

    # Metrics
    metrics = [
        ("Examples", "num_examples"),
        ("JSON Valid Rate", ("overall", "json_valid_rate"), True),
        ("Field Accuracy", ("overall", "field_accuracy"), True),
        ("Avg Latency (ms)", ("overall", "avg_latency_ms"), False),
    ]

    for metric_name, path, is_pct in [m if len(m) == 3 else (*m, False) for m in metrics]:
        row = f"{metric_name:<25}"
        for run in runs_to_compare:
            if isinstance(path, tuple):
                value = run.get(path[0], {}).get(path[1], 0)
            else:
                value = run.get(path, 0)

            if is_pct:
                row += f" {value:<12.1%}"
            elif isinstance(value, float):
                row += f" {value:<12.1f}"
            else:
                row += f" {value:<12}"
        click.echo(row)

    # Delta if comparing 2
    if n == 2:
        click.echo("\n" + "-" * 70)
        click.echo("Delta (newer - older):")

        curr = runs_to_compare[0]
        prev = runs_to_compare[1]

        curr_json = curr.get("overall", {}).get("json_valid_rate", 0)
        prev_json = prev.get("overall", {}).get("json_valid_rate", 0)
        curr_acc = curr.get("overall", {}).get("field_accuracy", 0)
        prev_acc = prev.get("overall", {}).get("field_accuracy", 0)

        click.echo(f"  JSON Valid Rate: {curr_json - prev_json:+.1%}")
        click.echo(f"  Field Accuracy:  {curr_acc - prev_acc:+.1%}")


def show_detail(results: List[Dict], detail: str):
    """Show detailed results for a specific run."""
    if detail == "latest":
        idx = 0
    else:
        try:
            idx = int(detail)
        except ValueError:
            click.echo(f"Invalid detail argument: {detail}")
            return

    if idx >= len(results):
        click.echo(f"Run index {idx} not found (only {len(results)} runs available)")
        return

    result = results[idx]

    click.echo("=" * 70)
    click.echo("Detailed Evaluation Results")
    click.echo("=" * 70)

    click.echo(f"\nFile: {result.get('_filename', 'Unknown')}")
    click.echo(f"Timestamp: {result.get('timestamp', 'Unknown')}")
    click.echo(f"Model Type: {result.get('model_type', 'Unknown')}")
    click.echo(f"Examples: {result.get('num_examples', 0)}")

    # Overall metrics
    overall = result.get("overall", {})
    click.echo(f"\n{'Overall Metrics':-^60}")
    click.echo(f"  JSON Valid Rate:    {overall.get('json_valid_rate', 0):.1%}")
    click.echo(f"  Field Accuracy:     {overall.get('field_accuracy', 0):.1%}")
    click.echo(f"  Avg Latency:        {overall.get('avg_latency_ms', 0):.0f}ms")

    # By category
    by_category = result.get("by_category", {})
    if by_category:
        click.echo(f"\n{'By Category':-^60}")
        for cat, stats in sorted(by_category.items()):
            click.echo(
                f"  {cat:15} JSON: {stats.get('json_valid_rate', 0):.1%}, "
                f"Acc: {stats.get('field_accuracy', 0):.1%} (n={stats.get('count', 0)})"
            )

    # By difficulty
    by_difficulty = result.get("by_difficulty", {})
    if by_difficulty:
        click.echo(f"\n{'By Difficulty':-^60}")
        for diff, stats in sorted(by_difficulty.items()):
            click.echo(
                f"  {diff:15} JSON: {stats.get('json_valid_rate', 0):.1%}, "
                f"Acc: {stats.get('field_accuracy', 0):.1%} (n={stats.get('count', 0)})"
            )

    # By field
    by_field = result.get("by_field", {})
    if by_field:
        click.echo(f"\n{'By Field':-^60}")
        for field, stats in by_field.items():
            total = stats.get("total", 0)
            correct = stats.get("correct", 0)
            if total > 0:
                acc = correct / total
                click.echo(f"  {field:25} {acc:.1%} ({correct}/{total})")


def export_to_csv(results: List[Dict], output_path: str):
    """Export results to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            "timestamp",
            "model_type",
            "num_examples",
            "json_valid_rate",
            "field_accuracy",
            "avg_latency_ms",
            "filename"
        ])

        # Data
        for result in results:
            overall = result.get("overall", {})
            writer.writerow([
                result.get("timestamp", ""),
                result.get("model_type", ""),
                result.get("num_examples", 0),
                overall.get("json_valid_rate", 0),
                overall.get("field_accuracy", 0),
                overall.get("avg_latency_ms", 0),
                result.get("_filename", ""),
            ])

    click.echo(f"Exported {len(results)} results to {output_path}")
