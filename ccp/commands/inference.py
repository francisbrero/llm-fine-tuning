"""
CCP inference command.

Run single inference on a GTM prompt.
"""

import json
import time

import click
import torch

from ccp.config import CCP_SYSTEM_PROMPT
from ccp.model import load_finetuned_model, clear_memory
from ccp.utils.parsing import parse_ccp_response, format_inference_prompt, validate_intent_ir


@click.command("inference")
@click.argument("prompt")
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
    "--max-tokens",
    default=800,
    type=int,
    help="Maximum tokens to generate"
)
@click.option(
    "--temperature",
    default=0.1,
    type=float,
    help="Sampling temperature"
)
@click.option(
    "--raw",
    is_flag=True,
    help="Show raw model output instead of parsed"
)
@click.option(
    "--json-only",
    is_flag=True,
    help="Output only JSON (for piping)"
)
def inference(
    prompt: str,
    model_path: str,
    adapter_path: str,
    max_tokens: int,
    temperature: float,
    raw: bool,
    json_only: bool,
):
    """Run inference on a single GTM prompt.

    Takes a GTM prompt and returns the parsed intent IR.

    Examples:

        # Basic inference
        ccp inference "Show me my best accounts"

        # Output raw model response
        ccp inference "Show me my pipeline" --raw

        # JSON output only (for piping)
        ccp inference "Accounts at risk" --json-only | jq .
    """
    if not json_only:
        click.echo("Loading model...")

    # Load model
    model, tokenizer = load_finetuned_model(model_path, adapter_path)

    # Format prompt
    formatted_prompt = format_inference_prompt(prompt, CCP_SYSTEM_PROMPT)

    # Generate
    if not json_only:
        click.echo("Generating response...")

    start_time = time.time()

    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    latency_ms = (time.time() - start_time) * 1000

    # Remove prompt from response
    if formatted_prompt in response:
        response = response[len(formatted_prompt):]

    if raw:
        click.echo("\n" + "=" * 60)
        click.echo("Raw Model Output")
        click.echo("=" * 60)
        click.echo(response)
        click.echo("\n" + "=" * 60)
        click.echo(f"Latency: {latency_ms:.0f}ms")
        return

    # Parse response
    reasoning, intent_ir, is_valid = parse_ccp_response(response)

    if json_only:
        if is_valid and intent_ir:
            click.echo(json.dumps(intent_ir, indent=2))
        else:
            click.echo(json.dumps({"error": "Failed to parse response", "raw": response}))
        return

    # Pretty output
    click.echo("\n" + "=" * 60)
    click.echo("CCP Inference Result")
    click.echo("=" * 60)

    click.echo(f"\nPrompt: {prompt}")
    click.echo(f"Latency: {latency_ms:.0f}ms")

    if reasoning:
        click.echo(f"\n{'Reasoning':-^60}")
        click.echo(reasoning)

    if is_valid and intent_ir:
        click.echo(f"\n{'Intent IR':-^60}")
        click.echo(json.dumps(intent_ir, indent=2))

        # Validate
        valid, errors = validate_intent_ir(intent_ir)
        if valid:
            click.echo(f"\n{'Status':-^60}")
            click.echo("VALID: Intent IR passes schema validation")
        else:
            click.echo(f"\n{'Validation Errors':-^60}")
            for error in errors:
                click.echo(f"  - {error}")
    else:
        click.echo(f"\n{'Error':-^60}")
        click.echo("Failed to parse valid JSON from response")
        click.echo(f"\nRaw output:\n{response[:500]}...")

    # Cleanup
    del model
    clear_memory()
