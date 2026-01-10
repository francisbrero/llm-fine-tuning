"""
Data loading and validation utilities for CCP.

Handles loading, validating, and processing training and evaluation data.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset, Dataset

from ccp.config import (
    CCP_SYSTEM_PROMPT,
    REQUIRED_IR_FIELDS,
    INTENT_TYPES,
    MOTIONS,
    ROLES,
    ACCOUNT_SCOPES,
    TIME_HORIZONS,
    OUTPUT_FORMATS,
)
from ccp.utils.parsing import validate_intent_ir


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
    return data


def validate_training_data(file_path: str, verbose: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate training data schema and content.

    Args:
        file_path: Path to training data JSONL
        verbose: Print detailed output

    Returns:
        Tuple of (is_valid, stats_dict)
    """
    stats = {
        "total_examples": 0,
        "valid_examples": 0,
        "invalid_examples": 0,
        "errors": [],
        "field_coverage": {},
        "intent_type_distribution": {},
        "motion_distribution": {},
        "has_reasoning": 0,
    }

    try:
        data = load_jsonl(file_path)
    except Exception as e:
        stats["errors"].append(f"Failed to load file: {e}")
        return False, stats

    stats["total_examples"] = len(data)

    for idx, example in enumerate(data):
        example_errors = []

        # Check required top-level fields
        if "gtm_prompt" not in example:
            example_errors.append("Missing 'gtm_prompt' field")

        if "intent_ir" not in example:
            example_errors.append("Missing 'intent_ir' field")
        else:
            intent_ir = example["intent_ir"]

            # Handle string vs dict
            if isinstance(intent_ir, str):
                try:
                    intent_ir = json.loads(intent_ir)
                except json.JSONDecodeError:
                    example_errors.append("intent_ir is invalid JSON string")
                    intent_ir = None

            if intent_ir:
                is_valid, ir_errors = validate_intent_ir(intent_ir)
                example_errors.extend(ir_errors)

                # Track distributions
                if "intent_type" in intent_ir:
                    it = intent_ir["intent_type"]
                    stats["intent_type_distribution"][it] = \
                        stats["intent_type_distribution"].get(it, 0) + 1

                if "motion" in intent_ir:
                    m = intent_ir["motion"]
                    stats["motion_distribution"][m] = \
                        stats["motion_distribution"].get(m, 0) + 1

                # Track field coverage
                for field in REQUIRED_IR_FIELDS:
                    if field in intent_ir:
                        stats["field_coverage"][field] = \
                            stats["field_coverage"].get(field, 0) + 1

        # Check reasoning (optional but tracked)
        if "reasoning" in example and example["reasoning"]:
            stats["has_reasoning"] += 1

        if example_errors:
            stats["invalid_examples"] += 1
            if verbose:
                stats["errors"].append({
                    "index": idx,
                    "prompt": example.get("gtm_prompt", "N/A")[:50],
                    "errors": example_errors
                })
        else:
            stats["valid_examples"] += 1

    is_valid = stats["invalid_examples"] == 0
    return is_valid, stats


def load_training_dataset(
    file_path: str,
    tokenizer,
    max_seq_length: int = 1024,
    system_prompt: str = CCP_SYSTEM_PROMPT
) -> Dataset:
    """
    Load and prepare training dataset.

    Args:
        file_path: Path to training JSONL
        tokenizer: Tokenizer to use
        max_seq_length: Maximum sequence length
        system_prompt: System prompt for formatting

    Returns:
        Tokenized HuggingFace Dataset
    """
    # Load raw dataset
    dataset = load_dataset(
        "json",
        data_files=file_path,
        split="train"
    )

    # Format examples with CoT template
    def format_example(example):
        prompt = example["gtm_prompt"].strip()
        reasoning = example.get("reasoning", "").strip()
        if not reasoning:
            reasoning = "- Analyzing prompt for GTM signals\n- Inferring context from keywords"

        ir_json = example["intent_ir"]
        if isinstance(ir_json, str):
            ir_output = ir_json
        else:
            ir_output = json.dumps(ir_json, indent=2)

        text = f"""<s>[INST] {system_prompt}

User request: {prompt} [/INST]
<reasoning>
{reasoning}
</reasoning>

<intent_ir>
{ir_output}
</intent_ir></s>"""

        return {"text": text}

    dataset = dataset.map(format_example)

    # Tokenize
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def load_eval_dataset(
    file_path: str,
    category: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset with optional filtering.

    Args:
        file_path: Path to eval JSONL
        category: Filter by category (standard, slang, adversarial, ambiguous)
        limit: Limit number of examples

    Returns:
        List of eval examples
    """
    data = load_jsonl(file_path)

    if category:
        data = [d for d in data if d.get("category") == category]

    if limit:
        data = data[:limit]

    return data
