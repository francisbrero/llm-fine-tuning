"""
Response parsing utilities for CCP.

Handles extraction and validation of reasoning and intent_ir from model outputs.
"""

import json
import re
from typing import Dict, Any, Tuple, Optional

from ccp.config import (
    REQUIRED_IR_FIELDS,
    INTENT_TYPES,
    MOTIONS,
    ROLES,
    ACCOUNT_SCOPES,
    TIME_HORIZONS,
    OUTPUT_FORMATS,
)


def parse_ccp_response(response: str) -> Tuple[Optional[str], Optional[Dict], bool]:
    """
    Parse a CCP model response to extract reasoning and intent_ir.

    Args:
        response: Raw model output string

    Returns:
        Tuple of (reasoning_text, intent_ir_dict, is_valid_json)
    """
    reasoning = None
    intent_ir = None
    is_valid = False

    # Extract reasoning section
    reasoning_match = re.search(
        r"<reasoning>(.*?)</reasoning>",
        response,
        re.DOTALL | re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Extract intent_ir section
    intent_match = re.search(
        r"<intent_ir>(.*?)</intent_ir>",
        response,
        re.DOTALL | re.IGNORECASE
    )

    if intent_match:
        json_str = intent_match.group(1).strip()
        try:
            intent_ir = json.loads(json_str)
            is_valid = True
        except json.JSONDecodeError:
            # Try to extract JSON from the string (might have extra content)
            json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
            if json_match:
                try:
                    intent_ir = json.loads(json_match.group(0))
                    is_valid = True
                except json.JSONDecodeError:
                    pass

    # Fallback: try to find raw JSON in response
    if intent_ir is None:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                intent_ir = json.loads(json_match.group(0))
                is_valid = True
            except json.JSONDecodeError:
                pass

    return reasoning, intent_ir, is_valid


def validate_intent_ir(intent_ir: Dict[str, Any]) -> Tuple[bool, list]:
    """
    Validate an intent_ir dictionary against the schema.

    Args:
        intent_ir: The parsed intent IR dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if not isinstance(intent_ir, dict):
        return False, ["intent_ir is not a dictionary"]

    # Check required fields
    for field in REQUIRED_IR_FIELDS:
        if field not in intent_ir:
            errors.append(f"Missing required field: {field}")

    # Validate enum fields
    if "intent_type" in intent_ir:
        if intent_ir["intent_type"] not in INTENT_TYPES:
            errors.append(f"Invalid intent_type: {intent_ir['intent_type']}")

    if "motion" in intent_ir:
        if intent_ir["motion"] not in MOTIONS:
            errors.append(f"Invalid motion: {intent_ir['motion']}")

    if "role_assumption" in intent_ir:
        if intent_ir["role_assumption"] not in ROLES:
            errors.append(f"Invalid role_assumption: {intent_ir['role_assumption']}")

    if "account_scope" in intent_ir:
        if intent_ir["account_scope"] not in ACCOUNT_SCOPES:
            errors.append(f"Invalid account_scope: {intent_ir['account_scope']}")

    if "time_horizon" in intent_ir:
        if intent_ir["time_horizon"] not in TIME_HORIZONS:
            errors.append(f"Invalid time_horizon: {intent_ir['time_horizon']}")

    if "output_format" in intent_ir:
        if intent_ir["output_format"] not in OUTPUT_FORMATS:
            errors.append(f"Invalid output_format: {intent_ir['output_format']}")

    # Validate confidence_scores
    if "confidence_scores" in intent_ir:
        scores = intent_ir["confidence_scores"]
        if not isinstance(scores, dict):
            errors.append("confidence_scores must be a dictionary")
        else:
            for key, value in scores.items():
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    errors.append(f"Invalid confidence score for {key}: {value}")

    # Validate assumptions_applied
    if "assumptions_applied" in intent_ir:
        if not isinstance(intent_ir["assumptions_applied"], list):
            errors.append("assumptions_applied must be a list")

    # Validate clarification_needed
    if "clarification_needed" in intent_ir:
        if not isinstance(intent_ir["clarification_needed"], bool):
            errors.append("clarification_needed must be a boolean")

    return len(errors) == 0, errors


def format_training_example(
    gtm_prompt: str,
    reasoning: str,
    intent_ir: Dict[str, Any],
    system_prompt: str
) -> str:
    """
    Format a training example with Chain-of-Thought structure.

    Args:
        gtm_prompt: The user's GTM prompt
        reasoning: The reasoning explanation
        intent_ir: The intent IR dictionary
        system_prompt: The system prompt to use

    Returns:
        Formatted training string
    """
    if isinstance(intent_ir, str):
        ir_output = intent_ir
    else:
        ir_output = json.dumps(intent_ir, indent=2)

    return f"""<s>[INST] {system_prompt}

User request: {gtm_prompt} [/INST]
<reasoning>
{reasoning}
</reasoning>

<intent_ir>
{ir_output}
</intent_ir></s>"""


def format_inference_prompt(gtm_prompt: str, system_prompt: str) -> str:
    """
    Format a prompt for inference.

    Args:
        gtm_prompt: The user's GTM prompt
        system_prompt: The system prompt to use

    Returns:
        Formatted inference prompt
    """
    return f"<s>[INST] {system_prompt}\n\nUser request: {gtm_prompt} [/INST]\n"
