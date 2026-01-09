"""
CCP train command.

Fine-tunes the CCP model with Chain-of-Thought training.
"""

import json
import os
from pathlib import Path

import click
import torch
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from ccp.config import (
    CCPConfig,
    ModelConfig,
    LoraConfig,
    TrainingConfig,
    DataConfig,
    CCP_SYSTEM_PROMPT,
)
from ccp.model import (
    load_tokenizer,
    load_base_model,
    apply_lora,
    clear_memory,
)
from ccp.data import load_training_dataset


@click.command("train")
@click.option(
    "--model-path",
    default="./models/phi-2",
    help="Path to base model"
)
@click.option(
    "--data-path",
    default="./data/ccp_training_with_reasoning.jsonl",
    help="Path to training data JSONL"
)
@click.option(
    "--output-dir",
    default="./ccp-adapter",
    help="Output directory for adapter"
)
@click.option(
    "--epochs",
    default=3,
    type=int,
    help="Number of training epochs"
)
@click.option(
    "--learning-rate", "--lr",
    default=2e-4,
    type=float,
    help="Learning rate"
)
@click.option(
    "--batch-size",
    default=1,
    type=int,
    help="Per-device batch size"
)
@click.option(
    "--gradient-accumulation", "--grad-accum",
    default=16,
    type=int,
    help="Gradient accumulation steps"
)
@click.option(
    "--max-seq-length",
    default=1024,
    type=int,
    help="Maximum sequence length"
)
@click.option(
    "--save-steps",
    default=100,
    type=int,
    help="Save checkpoint every N steps"
)
@click.option(
    "--logging-steps",
    default=10,
    type=int,
    help="Log every N steps"
)
@click.option(
    "--resume-from",
    default=None,
    help="Resume from checkpoint (path or 'latest')"
)
@click.option(
    "--device",
    default="auto",
    type=click.Choice(["auto", "mps", "cuda", "cpu"]),
    help="Device to train on"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate setup without training"
)
def train(
    model_path: str,
    data_path: str,
    output_dir: str,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    gradient_accumulation: int,
    max_seq_length: int,
    save_steps: int,
    logging_steps: int,
    resume_from: str,
    device: str,
    dry_run: bool,
):
    """Fine-tune CCP model with Chain-of-Thought training.

    Trains a LoRA adapter on the base model using the provided training data.
    Supports checkpoint resume for recovery from OOM or interruptions.

    Examples:

        # Basic training
        ccp train

        # Custom parameters
        ccp train --epochs 5 --learning-rate 1e-4

        # Resume from checkpoint
        ccp train --resume-from ccp-adapter/checkpoint-400

        # Resume from latest checkpoint
        ccp train --resume-from latest

        # Dry run to validate setup
        ccp train --dry-run
    """
    click.echo("=" * 60)
    click.echo("CCP Training - Chain-of-Thought + JSON")
    click.echo("=" * 60)

    # Resolve device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    click.echo(f"\nDevice: {device}")
    click.echo(f"PyTorch version: {torch.__version__}")

    # Resolve checkpoint path
    checkpoint_path = None
    if resume_from:
        if resume_from == "latest":
            # Find latest checkpoint in output_dir
            output_path = Path(output_dir)
            if output_path.exists():
                checkpoints = sorted(
                    output_path.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 0
                )
                if checkpoints:
                    checkpoint_path = str(checkpoints[-1])
                    click.echo(f"Resuming from latest checkpoint: {checkpoint_path}")
                else:
                    click.echo("Warning: No checkpoints found, starting fresh")
        else:
            checkpoint_path = resume_from
            if not Path(checkpoint_path).exists():
                click.echo(f"Error: Checkpoint not found: {checkpoint_path}", err=True)
                raise SystemExit(1)
            click.echo(f"Resuming from checkpoint: {checkpoint_path}")

    # Step 1: Load tokenizer
    click.echo("\n[1/6] Loading tokenizer...")
    tokenizer = load_tokenizer(model_path)
    click.echo(f"  Tokenizer: {tokenizer.__class__.__name__}")

    # Step 2: Load model
    click.echo("\n[2/6] Loading base model...")
    model = load_base_model(model_path, device=device, for_training=True)
    click.echo(f"  Model: {model.__class__.__name__}")

    # Step 3: Apply LoRA
    click.echo("\n[3/6] Applying LoRA adapters...")
    lora_config = LoraConfig()
    model = apply_lora(model, lora_config)

    # Step 4: Load dataset
    click.echo("\n[4/6] Loading training dataset...")
    if not Path(data_path).exists():
        click.echo(f"Error: Training data not found: {data_path}", err=True)
        raise SystemExit(1)

    tokenized_dataset = load_training_dataset(
        data_path,
        tokenizer,
        max_seq_length=max_seq_length,
    )
    click.echo(f"  Loaded {len(tokenized_dataset)} examples")

    # Step 5: Setup training
    click.echo("\n[5/6] Setting up training...")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        report_to="none",
        optim="adamw_torch",
        bf16=False,
        fp16=False,
        # For checkpoint resume
        save_safetensors=True,
        load_best_model_at_end=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Print config summary
    click.echo(f"\n  Epochs:           {epochs}")
    click.echo(f"  Batch size:       {batch_size} (effective: {batch_size * gradient_accumulation})")
    click.echo(f"  Learning rate:    {learning_rate}")
    click.echo(f"  Max seq length:   {max_seq_length}")
    click.echo(f"  Save steps:       {save_steps}")
    click.echo(f"  Output dir:       {output_dir}")

    if dry_run:
        click.echo("\n" + "=" * 60)
        click.echo("DRY RUN: Setup validated, skipping training")
        click.echo("=" * 60)
        return

    # Step 6: Train
    click.echo("\n[6/6] Starting training...")

    # Clear memory before training
    if device == "mps":
        torch.mps.empty_cache()
        click.echo("  MPS cache cleared")

    try:
        if checkpoint_path:
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            trainer.train()
    except KeyboardInterrupt:
        click.echo("\n\nTraining interrupted! Saving checkpoint...")
        trainer.save_model()
        click.echo(f"Checkpoint saved to {output_dir}")
        raise SystemExit(130)

    # Save final model
    click.echo("\n" + "=" * 60)
    click.echo("Training complete! Saving adapter...")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save metadata
    metadata = {
        "schema_version": "2.0.0",
        "base_model": model_path,
        "training_approach": "chain_of_thought",
        "output_format": {
            "reasoning": "<reasoning>...</reasoning>",
            "intent_ir": "<intent_ir>{JSON}</intent_ir>"
        },
        "training_config": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation": gradient_accumulation,
            "max_seq_length": max_seq_length,
        }
    }

    with open(os.path.join(output_dir, "ccp_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"\nCCP adapter saved to {output_dir}")
    click.echo("=" * 60)
