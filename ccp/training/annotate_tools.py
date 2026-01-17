"""
Annotate existing training data with Phoenix MCP tool selections.

This script adds phoenix_tools field to existing training examples based on
their intent_type, motion, and account_scope.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple


# Phoenix MCP tool mappings based on GTM intent patterns
INTENT_TOOL_MAPPINGS = {
    "account_discovery": {
        "default": {
            "primary": ["company_search", "company_firmographic", "company_intent"],
            "secondary": ["company_technographic"],
            "avoid": ["get_product_reviews", "list_vendors"],
        },
        "existing": {
            "primary": ["company_fai", "company_firmographic", "company_intent"],
            "secondary": ["company_technographic"],
            "avoid": ["company_search", "get_product_reviews"],
        },
    },
    "lead_prioritization": {
        "default": {
            "primary": ["company_intent", "company_firmographic"],
            "secondary": ["company_technographic"],
            "avoid": ["company_search", "company_fai", "company_spend", "get_product_reviews"],
        },
    },
    "churn_risk_assessment": {
        "default": {
            "primary": ["company_intent", "company_technographic"],
            "secondary": ["company_spend", "company_fai"],
            "avoid": ["company_search", "get_product_reviews", "list_vendors"],
        },
    },
    "expansion_identification": {
        "default": {
            "primary": ["company_fai", "company_technographic", "company_firmographic"],
            "secondary": ["company_intent"],
            "avoid": ["company_search", "get_product_reviews", "company_spend"],
        },
    },
    "pipeline_analysis": {
        "default": {
            "primary": ["company_firmographic", "company_intent"],
            "secondary": ["company_technographic"],
            "avoid": ["get_product_reviews", "list_vendors"],
        },
        "existing": {
            "primary": ["company_firmographic", "company_intent", "company_fai"],
            "secondary": ["company_technographic"],
            "avoid": ["company_search", "get_product_reviews"],
        },
    },
    "forecast_review": {
        "default": {
            "primary": ["company_firmographic", "company_intent", "company_search"],
            "secondary": ["company_technographic"],
            "avoid": ["get_product_reviews", "list_vendors"],
        },
        "existing": {
            "primary": ["company_firmographic", "company_intent", "company_fai"],
            "secondary": ["company_technographic"],
            "avoid": ["company_search", "get_product_reviews"],
        },
    },
    "competitive_analysis": {
        "default": {
            "primary": ["company_technographic", "get_product_information"],
            "secondary": ["company_intent", "get_product_reviews"],
            "avoid": ["company_fai", "company_spend"],
        },
    },
    "engagement_summary": {
        "default": {
            "primary": ["company_intent", "company_technographic", "company_fai"],
            "secondary": ["company_firmographic"],
            "avoid": ["company_search", "get_product_reviews", "company_spend"],
        },
    },
    "qualification_analysis": {
        "default": {
            "primary": ["company_firmographic", "company_intent"],
            "secondary": ["company_technographic"],
            "avoid": ["company_search", "company_fai", "get_product_reviews"],
        },
    },
    "performance_tracking": {
        "default": {
            "primary": ["company_firmographic"],
            "secondary": ["company_intent"],
            "avoid": ["get_product_reviews", "company_spend", "list_vendors"],
        },
    },
    "meeting_prep": {
        "default": {
            "primary": ["company_firmographic", "company_technographic", "company_intent"],
            "secondary": ["get_product_information", "web_search"],
            "avoid": ["company_search", "get_product_reviews"],
        },
    },
    "relationship_building": {
        "default": {
            "primary": ["company_firmographic", "company_fai"],
            "secondary": ["company_intent"],
            "avoid": ["get_product_reviews", "company_spend"],
        },
    },
}


def get_phoenix_tools(intent_type: str, motion: str, account_scope: str) -> Dict[str, List[str]]:
    """
    Get Phoenix MCP tools for a given intent, motion, and scope.

    Args:
        intent_type: The GTM intent type
        motion: The GTM motion (outbound, inbound, expansion, renewal, churn_prevention)
        account_scope: The account scope (net_new, existing, churned, all)

    Returns:
        Dict with primary, secondary, and avoid tool lists
    """
    # Get base mapping for intent type
    intent_mapping = INTENT_TOOL_MAPPINGS.get(intent_type, {})

    # Determine which variant to use based on account scope
    if account_scope == "existing" and "existing" in intent_mapping:
        tools = intent_mapping["existing"]
    else:
        tools = intent_mapping.get("default", {
            "primary": ["company_firmographic"],
            "secondary": ["company_intent"],
            "avoid": ["get_product_reviews"],
        })

    # Make a copy to avoid modifying the original
    result = {
        "primary": list(tools.get("primary", [])),
        "secondary": list(tools.get("secondary", [])),
        "avoid": list(tools.get("avoid", [])),
    }

    # Motion-specific adjustments
    if motion == "outbound" and account_scope == "net_new":
        # Outbound prospecting needs company_search
        if "company_search" not in result["primary"] and intent_type not in ["lead_prioritization"]:
            result["primary"].insert(0, "company_search")
        # Remove company_fai (only for existing customers)
        result["avoid"].append("company_fai")
        if "company_fai" in result["primary"]:
            result["primary"].remove("company_fai")
        if "company_fai" in result["secondary"]:
            result["secondary"].remove("company_fai")

    if motion == "expansion" and account_scope == "existing":
        # Expansion needs company_fai
        if "company_fai" not in result["primary"] and "company_fai" not in result["secondary"]:
            result["primary"].append("company_fai")
        # Avoid company_search for existing accounts
        if "company_search" not in result["avoid"]:
            result["avoid"].append("company_search")
        if "company_search" in result["primary"]:
            result["primary"].remove("company_search")

    if motion == "churn_prevention" or motion == "renewal":
        # Churn prevention needs engagement tools
        if "company_intent" not in result["primary"]:
            result["primary"].insert(0, "company_intent")
        if "company_technographic" not in result["primary"]:
            result["primary"].append("company_technographic")

    # Account scope adjustments
    if account_scope == "net_new":
        # Net new should avoid customer-only tools
        for tool in ["company_fai", "company_spend", "company_cloud_spend"]:
            if tool not in result["avoid"]:
                result["avoid"].append(tool)
            if tool in result["primary"]:
                result["primary"].remove(tool)
            if tool in result["secondary"]:
                result["secondary"].remove(tool)

    if account_scope == "existing":
        # Existing should avoid search (usually)
        if intent_type not in ["performance_tracking", "forecast_review"]:
            if "company_search" not in result["avoid"]:
                result["avoid"].append("company_search")

    # Deduplicate and ensure avoid doesn't contain primary/secondary tools
    result["avoid"] = [t for t in result["avoid"] if t not in result["primary"] and t not in result["secondary"]]
    result["avoid"] = list(set(result["avoid"]))

    return result


def enhance_reasoning_with_tools(reasoning: str, phoenix_tools: Dict[str, List[str]]) -> str:
    """
    Enhance existing reasoning to mention tool selection.

    Args:
        reasoning: Original reasoning text
        phoenix_tools: Tool selections (primary, secondary, avoid)

    Returns:
        Enhanced reasoning with tool selection rationale
    """
    # Add tool selection reasoning
    primary_tools = ", ".join(phoenix_tools["primary"])
    avoid_tools = ", ".join(phoenix_tools["avoid"][:3])  # Limit to 3 for brevity

    tool_reasoning = f"\n- Should use {primary_tools} for this intent"
    if phoenix_tools["avoid"]:
        tool_reasoning += f"\n- Should NOT use {avoid_tools} (not applicable to this context)"

    return reasoning + tool_reasoning


def annotate_training_data(
    input_file: str,
    output_file: str,
    enhance_reasoning: bool = True,
    verbose: bool = False,
) -> Tuple[int, int, List[str]]:
    """
    Annotate training data with Phoenix MCP tool selections.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output JSONL file
        enhance_reasoning: Whether to add tool reasoning to existing reasoning
        verbose: Print progress

    Returns:
        Tuple of (processed_count, error_count, error_messages)
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    processed = 0
    errors = 0
    error_messages = []

    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                example = json.loads(line)

                # Check if already has phoenix_tools
                if "phoenix_tools" in example:
                    if verbose:
                        print(f"Line {line_num}: Already has phoenix_tools, skipping")
                    outfile.write(json.dumps(example) + "\n")
                    processed += 1
                    continue

                # Extract intent IR fields
                intent_ir = example.get("intent_ir", {})
                intent_type = intent_ir.get("intent_type", "unknown")
                motion = intent_ir.get("motion", "outbound")
                account_scope = intent_ir.get("account_scope", "net_new")

                # Get Phoenix tools
                phoenix_tools = get_phoenix_tools(intent_type, motion, account_scope)

                # Add phoenix_tools to example
                example["phoenix_tools"] = phoenix_tools

                # Enhance reasoning if requested
                if enhance_reasoning and "reasoning" in example:
                    example["reasoning"] = enhance_reasoning_with_tools(
                        example["reasoning"],
                        phoenix_tools
                    )

                # Write annotated example
                outfile.write(json.dumps(example) + "\n")
                processed += 1

                if verbose and processed % 100 == 0:
                    print(f"Processed {processed} examples...")

            except json.JSONDecodeError as e:
                errors += 1
                error_msg = f"Line {line_num}: JSON decode error: {e}"
                error_messages.append(error_msg)
                if verbose:
                    print(error_msg)
            except Exception as e:
                errors += 1
                error_msg = f"Line {line_num}: Error: {e}"
                error_messages.append(error_msg)
                if verbose:
                    print(error_msg)

    return processed, errors, error_messages


def validate_annotations(file_path: str, sample_size: int = 10) -> None:
    """
    Validate annotated training data by showing samples.

    Args:
        file_path: Path to annotated JSONL file
        sample_size: Number of examples to sample
    """
    import random

    with open(file_path, 'r') as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"Validation Sample from {file_path}")
    print(f"Total examples: {len(examples)}")
    print(f"{'='*60}\n")

    # Count examples with phoenix_tools
    with_tools = sum(1 for ex in examples if "phoenix_tools" in ex)
    print(f"Examples with phoenix_tools: {with_tools}/{len(examples)} ({with_tools/len(examples)*100:.1f}%)\n")

    # Sample random examples
    sample = random.sample(examples, min(sample_size, len(examples)))

    for i, ex in enumerate(sample, 1):
        print(f"\n--- Example {i} ---")
        print(f"Prompt: {ex.get('gtm_prompt', 'N/A')[:80]}...")

        intent_ir = ex.get("intent_ir", {})
        print(f"Intent: {intent_ir.get('intent_type', 'N/A')}")
        print(f"Motion: {intent_ir.get('motion', 'N/A')}")
        print(f"Scope: {intent_ir.get('account_scope', 'N/A')}")

        if "phoenix_tools" in ex:
            pt = ex["phoenix_tools"]
            print(f"\nPhoenix Tools:")
            print(f"  Primary: {', '.join(pt.get('primary', []))}")
            if pt.get('secondary'):
                print(f"  Secondary: {', '.join(pt.get('secondary', []))}")
            print(f"  Avoid: {', '.join(pt.get('avoid', []))}")
        else:
            print("\n⚠️  Missing phoenix_tools field!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate training data with Phoenix MCP tools")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument("--enhance-reasoning", action="store_true", default=True,
                        help="Add tool selection reasoning (default: True)")
    parser.add_argument("--no-enhance-reasoning", dest="enhance_reasoning", action="store_false",
                        help="Don't enhance reasoning")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress")
    parser.add_argument("--validate", action="store_true",
                        help="Validate output after annotation")
    parser.add_argument("--sample-size", type=int, default=10,
                        help="Sample size for validation (default: 10)")

    args = parser.parse_args()

    print(f"Annotating {args.input} -> {args.output}")
    print(f"Enhance reasoning: {args.enhance_reasoning}")
    print()

    processed, errors, error_messages = annotate_training_data(
        args.input,
        args.output,
        enhance_reasoning=args.enhance_reasoning,
        verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print(f"Annotation Complete")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    if error_messages:
        print(f"\nError details:")
        for msg in error_messages[:10]:  # Show first 10 errors
            print(f"  {msg}")
        if len(error_messages) > 10:
            print(f"  ... and {len(error_messages) - 10} more")

    print(f"\nOutput written to: {args.output}")

    if args.validate:
        validate_annotations(args.output, args.sample_size)
