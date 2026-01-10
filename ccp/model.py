"""
Model loading utilities for CCP.

Handles loading base models, tokenizers, and LoRA adapters.
"""

import torch
from pathlib import Path
from typing import Tuple, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig as PeftLoraConfig, get_peft_model

from ccp.config import ModelConfig, LoraConfig


def load_tokenizer(model_path: str) -> AutoTokenizer:
    """
    Load tokenizer from model path.

    Args:
        model_path: Path to the model directory

    Returns:
        Loaded tokenizer with pad token set
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_base_model(
    model_path: str,
    device: str = "auto",
    for_training: bool = False
) -> AutoModelForCausalLM:
    """
    Load base model for training or inference.

    Args:
        model_path: Path to the model directory
        device: Device to load model on (auto, mps, cuda, cpu)
        for_training: If True, enables gradient checkpointing

    Returns:
        Loaded model
    """
    # Resolve device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"  Loading model on device: {device}")

    # Device map for loading
    device_map = {"": device}

    # Load model
    # Note: MPS doesn't support bitsandbytes quantization, use float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device_map,
        local_files_only=True,
    )

    if for_training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    return model


def apply_lora(
    model: AutoModelForCausalLM,
    lora_config: LoraConfig
) -> AutoModelForCausalLM:
    """
    Apply LoRA adapters to a model for training.

    Args:
        model: Base model to apply LoRA to
        lora_config: LoRA configuration

    Returns:
        Model with LoRA adapters applied
    """
    peft_config = PeftLoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.alpha,
        target_modules=lora_config.target_modules,
        lora_dropout=lora_config.dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def load_finetuned_model(
    base_model_path: str,
    adapter_path: str,
    device: str = "auto"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a fine-tuned model with LoRA adapter for inference.

    Args:
        base_model_path: Path to base model
        adapter_path: Path to LoRA adapter
        device: Device to load on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Resolve device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"  Loading fine-tuned model on device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer(base_model_path)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map={"": device},
        local_files_only=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def clear_memory(device: str = "auto"):
    """Clear GPU/MPS memory cache."""
    import gc
    gc.collect()

    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
