#!/usr/bin/env python3
"""
Generate reasoning chains for CCP training data.

This script transforms the existing training data format to include
explicit reasoning chains that explain WHY each inference was made,
implementing the Chain-of-Thought + JSON approach.

Input format (existing):
{
    "gtm_prompt": "Show me my best accounts",
    "intent_ir": {...}
}

Output format (new):
{
    "gtm_prompt": "Show me my best accounts",
    "reasoning": "- 'my accounts' signals...\n- 'best' is ambiguous...",
    "intent_ir": {...}
}
"""

import json
import argparse
from pathlib import Path


# Reasoning templates based on GTM domain knowledge
REASONING_PATTERNS = {
    # Intent type reasoning
    "account_discovery": "This is an account discovery request - user wants to find/view accounts matching criteria",
    "pipeline_analysis": "This is a pipeline analysis request - user wants to review deal pipeline status",
    "expansion_identification": "This is an expansion identification request - user wants upsell/cross-sell opportunities",
    "churn_risk_assessment": "This is a churn risk assessment - user wants to identify at-risk accounts",
    "lead_prioritization": "This is a lead prioritization request - user wants to rank/order leads by importance",
    "territory_planning": "This is a territory planning request - user wants to analyze territory coverage",
    "forecast_review": "This is a forecast review request - user wants to check quota/target progress",
    "competitive_analysis": "This is a competitive analysis request - user wants to understand competitive landscape",
    "engagement_summary": "This is an engagement summary request - user wants activity history on accounts",

    # Motion reasoning
    "outbound": "Outbound motion inferred - prospecting/net-new focus",
    "inbound": "Inbound motion inferred - responding to existing demand/leads",
    "expansion": "Expansion motion inferred - growing existing customer relationships",
    "renewal": "Renewal motion inferred - retaining existing ARR",
    "churn_prevention": "Churn prevention motion inferred - saving at-risk accounts",

    # Role reasoning
    "sales_rep": "Sales rep role assumed from individual contributor language",
    "sales_manager": "Sales manager role assumed from team/aggregate perspective",
    "sdr": "SDR role assumed from prospecting/lead context",
    "revops": "RevOps role assumed from operational/metrics focus",
    "marketing": "Marketing role assumed from demand gen/content context",
    "cs": "Customer Success role assumed from health/renewal context",
    "exec": "Executive role assumed from board/strategic context",

    # Account scope reasoning
    "net_new": "Net-new scope - targeting accounts not yet customers",
    "existing": "Existing scope - targeting current customers or pipeline",
    "churned": "Churned scope - targeting former customers",
    "all": "All accounts scope - no specific customer status filter",

    # Time horizon reasoning
    "immediate": "Immediate time horizon - urgent/today action needed",
    "this_week": "This week time horizon - near-term focus",
    "this_month": "This month time horizon - monthly planning cycle",
    "this_quarter": "This quarter time horizon - quarterly business cycle",
    "this_year": "This year time horizon - annual planning view",
}

# Signal keywords to reasoning mapping
KEYWORD_SIGNALS = {
    # Possessive/personal signals
    "my accounts": "- 'my accounts' indicates personally assigned accounts, suggesting sales rep role",
    "my book": "- 'my book' refers to book of business - existing customer portfolio",
    "my pipe": "- 'my pipe' is shorthand for personal pipeline - active deals",
    "my territory": "- 'my territory' indicates assigned geographic/account territory",
    "my number": "- 'my number' refers to quota/target - quota-carrying role",

    # Risk signals
    "at risk": "- 'at risk' explicitly signals churn concern - needs immediate attention",
    "red accounts": "- 'red accounts' means unhealthy/at-risk customers (health score terminology)",
    "churned": "- 'churned' explicitly indicates former customers",
    "risk": "- risk mentioned suggests concern about losing accounts/deals",

    # Expansion signals
    "upsell": "- 'upsell' explicitly indicates expansion motion within existing accounts",
    "cross-sell": "- 'cross-sell' indicates selling additional products to existing customers",
    "whitespace": "- 'whitespace' means untapped opportunity in existing accounts",
    "expand": "- 'expand' or 'expansion' signals growth within customer base",
    "land and expand": "- 'land and expand' is classic expansion strategy terminology",

    # Prospecting signals
    "prospects": "- 'prospects' indicates net-new/outbound targeting",
    "new logos": "- 'new logos' means acquiring new customers (not existing)",
    "greenfield": "- 'greenfield' means untouched/virgin territory - net-new accounts",
    "net new": "- 'net new' explicitly indicates non-customer accounts",

    # Inbound signals
    "mql": "- 'MQL' (Marketing Qualified Lead) indicates inbound motion",
    "lead": "- 'lead' typically indicates inbound demand",
    "demo request": "- 'demo request' is an inbound intent signal",
    "hand-raiser": "- 'hand-raiser' indicates inbound prospect showing interest",

    # Renewal signals
    "renewal": "- 'renewal' explicitly indicates retention motion",
    "nrr": "- 'NRR' (Net Revenue Retention) is a renewal/retention metric",
    "contract": "- contract mention suggests renewal/retention context",

    # Role signals
    "pipeline": "- 'pipeline' analysis typically suggests manager or rep perspective",
    "forecast": "- 'forecast' analysis typically suggests manager/revops/exec view",
    "health score": "- 'health score' is a CSM metric - suggests customer success role",
    "board": "- 'board' mention indicates executive-level context",

    # Time signals
    "this quarter": "- 'this quarter' explicitly sets quarterly time horizon",
    "this week": "- 'this week' explicitly sets weekly time horizon",
    "this month": "- 'this month' explicitly sets monthly time horizon",
    "today": "- 'today' indicates immediate time horizon",
    "q1": "- quarter reference (Q1/Q2/Q3/Q4) sets explicit time horizon",
    "q2": "- quarter reference (Q1/Q2/Q3/Q4) sets explicit time horizon",
    "q3": "- quarter reference (Q1/Q2/Q3/Q4) sets explicit time horizon",
    "q4": "- quarter reference (Q1/Q2/Q3/Q4) sets explicit time horizon",

    # Geography signals
    "emea": "- EMEA geography explicitly mentioned",
    "apac": "- APAC geography explicitly mentioned",
    "latam": "- LATAM geography explicitly mentioned",
    "na ": "- North America geography explicitly mentioned",
    "north america": "- North America geography explicitly mentioned",

    # Format signals
    "chart": "- 'chart' indicates visualization output format",
    "export": "- 'export' indicates export/download output format",
    "csv": "- 'CSV' indicates export output format",
    "summarize": "- 'summarize' indicates summary output format",
    "list": "- 'list' or 'show me' indicates list output format",
}


def detect_signals(prompt: str) -> list[str]:
    """Detect GTM signals present in the prompt."""
    prompt_lower = prompt.lower()
    signals = []

    for keyword, reasoning in KEYWORD_SIGNALS.items():
        if keyword.lower() in prompt_lower:
            signals.append(reasoning)

    return signals


def generate_reasoning_chain(example: dict) -> str:
    """Generate a reasoning chain for a training example."""
    prompt = example["gtm_prompt"]
    ir = example["intent_ir"]

    reasoning_parts = []

    # Start with signal detection from the prompt
    signals = detect_signals(prompt)
    if signals:
        reasoning_parts.extend(signals)

    # Add intent type reasoning
    intent_type = ir.get("intent_type")
    if intent_type and intent_type in REASONING_PATTERNS:
        reasoning_parts.append(f"- {REASONING_PATTERNS[intent_type]}")

    # Add motion reasoning with confidence context
    motion = ir.get("motion")
    confidence_scores = ir.get("confidence_scores", {})
    motion_conf = confidence_scores.get("motion", 0.5)

    if motion and motion in REASONING_PATTERNS:
        if motion_conf >= 0.9:
            reasoning_parts.append(f"- {REASONING_PATTERNS[motion]} (explicit in prompt)")
        elif motion_conf >= 0.7:
            reasoning_parts.append(f"- {REASONING_PATTERNS[motion]} (strong signal)")
        else:
            reasoning_parts.append(f"- {REASONING_PATTERNS[motion]} (inferred, moderate confidence)")

    # Add role reasoning
    role = ir.get("role_assumption")
    role_conf = confidence_scores.get("role_assumption", 0.5)

    if role and role in REASONING_PATTERNS:
        if role_conf >= 0.8:
            reasoning_parts.append(f"- {REASONING_PATTERNS[role]}")
        else:
            reasoning_parts.append(f"- {REASONING_PATTERNS[role]} (default assumption)")

    # Add account scope reasoning
    scope = ir.get("account_scope")
    if scope and scope in REASONING_PATTERNS:
        reasoning_parts.append(f"- {REASONING_PATTERNS[scope]}")

    # Add time horizon reasoning
    time_horizon = ir.get("time_horizon")
    time_conf = confidence_scores.get("time_horizon", 0.5)

    if time_horizon and time_horizon in REASONING_PATTERNS:
        if time_conf >= 0.9:
            reasoning_parts.append(f"- {REASONING_PATTERNS[time_horizon]} (explicit)")
        else:
            reasoning_parts.append(f"- {REASONING_PATTERNS[time_horizon]} (default for this intent)")

    # Add ICP reasoning if relevant
    icp_resolution = ir.get("icp_resolution_required", False)
    if icp_resolution:
        reasoning_parts.append("- ICP resolution required - need to determine specific targeting criteria")

    # Add clarification reasoning if needed
    clarification = ir.get("clarification_needed", False)
    if clarification:
        reasoning_parts.append("- Request is ambiguous - clarification may be needed before execution")

    # Add assumptions from the original data
    assumptions = ir.get("assumptions_applied", [])
    for assumption in assumptions:
        # Avoid duplicating signals we already captured
        assumption_lower = assumption.lower()
        already_captured = any(assumption_lower in part.lower() for part in reasoning_parts)
        if not already_captured:
            reasoning_parts.append(f"- {assumption}")

    # Deduplicate while preserving order
    seen = set()
    unique_parts = []
    for part in reasoning_parts:
        normalized = part.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_parts.append(part)

    return "\n".join(unique_parts)


def transform_training_data(input_path: Path, output_path: Path) -> dict:
    """Transform training data to include reasoning chains."""
    stats = {
        "total": 0,
        "processed": 0,
        "errors": 0
    }

    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line_num, line in enumerate(infile, 1):
            stats["total"] += 1

            try:
                example = json.loads(line.strip())

                # Generate reasoning chain
                reasoning = generate_reasoning_chain(example)

                # Create new example with reasoning
                new_example = {
                    "gtm_prompt": example["gtm_prompt"],
                    "reasoning": reasoning,
                    "intent_ir": example["intent_ir"]
                }

                outfile.write(json.dumps(new_example) + "\n")
                stats["processed"] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
                stats["errors"] += 1
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                stats["errors"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate reasoning chains for CCP training data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/ccp_training.jsonl"),
        help="Input JSONL file with training data"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ccp_training_with_reasoning.jsonl"),
        help="Output JSONL file with reasoning chains added"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Preview N examples without writing output"
    )

    args = parser.parse_args()

    if args.preview > 0:
        # Preview mode
        print(f"Previewing {args.preview} examples from {args.input}\n")
        print("=" * 60)

        with open(args.input, "r") as f:
            for i, line in enumerate(f):
                if i >= args.preview:
                    break

                example = json.loads(line.strip())
                reasoning = generate_reasoning_chain(example)

                print(f"\nPrompt: {example['gtm_prompt']}")
                print("-" * 40)
                print("Reasoning:")
                print(reasoning)
                print("-" * 40)
                print("Intent IR:")
                print(json.dumps(example["intent_ir"], indent=2))
                print("=" * 60)
    else:
        # Transform mode
        print(f"Transforming {args.input} -> {args.output}")

        stats = transform_training_data(args.input, args.output)

        print(f"\nResults:")
        print(f"  Total lines: {stats['total']}")
        print(f"  Processed: {stats['processed']}")
        print(f"  Errors: {stats['errors']}")
        print(f"\nOutput written to {args.output}")


if __name__ == "__main__":
    main()
