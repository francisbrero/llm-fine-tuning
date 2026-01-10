"""
CCP validate-data command.

Validates training data schema and content before training.
"""

import click
from pathlib import Path

from ccp.data import validate_training_data, load_jsonl
from ccp.config import (
    REQUIRED_IR_FIELDS,
    INTENT_TYPES,
    MOTIONS,
    ROLES,
)


@click.command("validate-data")
@click.option(
    "--file", "-f",
    default="./data/ccp_training_with_reasoning.jsonl",
    help="Path to training data JSONL file"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed error information"
)
@click.option(
    "--strict",
    is_flag=True,
    help="Fail on any warnings (not just errors)"
)
def validate_data(file: str, verbose: bool, strict: bool):
    """Validate training data schema and content.

    Checks that all training examples have:
    - Valid JSON structure
    - Required fields (gtm_prompt, intent_ir)
    - Valid enum values for intent_type, motion, role_assumption, etc.
    - Proper confidence_scores format
    """
    click.echo("=" * 60)
    click.echo("CCP Training Data Validation")
    click.echo("=" * 60)

    file_path = Path(file)
    if not file_path.exists():
        click.echo(f"\nError: File not found: {file}", err=True)
        raise SystemExit(1)

    click.echo(f"\nValidating: {file_path}")

    is_valid, stats = validate_training_data(str(file_path), verbose=verbose)

    # Print results
    click.echo(f"\n{'Results':=^60}")
    click.echo(f"  Total examples:   {stats['total_examples']}")
    click.echo(f"  Valid examples:   {stats['valid_examples']}")
    click.echo(f"  Invalid examples: {stats['invalid_examples']}")
    click.echo(f"  Has reasoning:    {stats['has_reasoning']}")

    # Field coverage
    click.echo(f"\n{'Field Coverage':=^60}")
    total = stats['total_examples']
    for field in REQUIRED_IR_FIELDS:
        count = stats['field_coverage'].get(field, 0)
        pct = (count / total * 100) if total > 0 else 0
        status = "OK" if pct == 100 else "WARN" if pct > 90 else "FAIL"
        click.echo(f"  {field:25} {count:5}/{total} ({pct:5.1f}%) [{status}]")

    # Intent type distribution
    if stats['intent_type_distribution']:
        click.echo(f"\n{'Intent Type Distribution':=^60}")
        for intent_type, count in sorted(
            stats['intent_type_distribution'].items(),
            key=lambda x: -x[1]
        ):
            pct = (count / total * 100) if total > 0 else 0
            click.echo(f"  {intent_type:25} {count:5} ({pct:5.1f}%)")

    # Motion distribution
    if stats['motion_distribution']:
        click.echo(f"\n{'Motion Distribution':=^60}")
        for motion, count in sorted(
            stats['motion_distribution'].items(),
            key=lambda x: -x[1]
        ):
            pct = (count / total * 100) if total > 0 else 0
            click.echo(f"  {motion:25} {count:5} ({pct:5.1f}%)")

    # Errors
    if stats['errors'] and verbose:
        click.echo(f"\n{'Errors':=^60}")
        for error in stats['errors'][:20]:  # Limit to first 20
            if isinstance(error, dict):
                click.echo(f"\n  Example {error['index']}: {error['prompt']}...")
                for e in error['errors']:
                    click.echo(f"    - {e}")
            else:
                click.echo(f"  {error}")

        if len(stats['errors']) > 20:
            click.echo(f"\n  ... and {len(stats['errors']) - 20} more errors")

    # Final status
    click.echo("\n" + "=" * 60)
    if is_valid:
        click.echo("PASSED: All training data is valid")
        click.echo("=" * 60)
    else:
        click.echo("FAILED: Training data has validation errors", err=True)
        click.echo("=" * 60)
        raise SystemExit(1)

    # Warnings (non-fatal unless --strict)
    warnings = []
    if stats['has_reasoning'] < stats['total_examples']:
        missing = stats['total_examples'] - stats['has_reasoning']
        warnings.append(f"{missing} examples missing reasoning field")

    for field in REQUIRED_IR_FIELDS:
        count = stats['field_coverage'].get(field, 0)
        if count < stats['total_examples']:
            missing = stats['total_examples'] - count
            warnings.append(f"{missing} examples missing {field}")

    if warnings:
        click.echo("\nWarnings:")
        for w in warnings:
            click.echo(f"  - {w}")

        if strict:
            click.echo("\n--strict mode: treating warnings as errors")
            raise SystemExit(1)
