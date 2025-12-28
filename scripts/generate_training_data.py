#!/usr/bin/env python3
"""
Training Data Generator

Generates GTM prompt → Intent IR training examples from scraped vocabulary.
Output: data/vocabulary_training.jsonl
"""

import json
import random
from typing import Dict, List

# Load vocabulary
with open("data/vocabulary.json") as f:
    vocab = json.load(f)

# Category → Intent IR mapping
CATEGORY_INTENT_MAP = {
    "quota_forecast": {
        "intent_types": ["forecast_review", "quota_analysis", "performance_tracking"],
        "motions": ["outbound", "inbound", "expansion"],
        "roles": ["sales_rep", "sales_manager"],
    },
    "pipeline_deals": {
        "intent_types": ["pipeline_analysis", "deal_review", "opportunity_scoring"],
        "motions": ["outbound", "expansion", "renewal"],
        "roles": ["sales_rep", "sales_manager", "ae"],
    },
    "lead_qualification": {
        "intent_types": ["lead_scoring", "account_discovery", "qualification_analysis"],
        "motions": ["inbound", "outbound"],
        "roles": ["sdr", "bdr", "sales_rep"],
    },
    "gtm_motion": {
        "intent_types": ["account_discovery", "expansion_opportunity", "churn_risk_assessment"],
        "motions": ["outbound", "inbound", "expansion", "renewal", "churn_prevention"],
        "roles": ["sales_rep", "csm", "sales_manager"],
    },
    "metrics_kpi": {
        "intent_types": ["performance_tracking", "forecast_review", "metrics_analysis"],
        "motions": ["expansion", "renewal"],
        "roles": ["sales_manager", "revops", "cro"],
    },
    "roles": {
        "intent_types": ["team_analysis", "performance_tracking", "resource_planning"],
        "motions": ["outbound", "inbound"],
        "roles": ["sales_manager", "vp_sales"],
    },
    "methodology": {
        "intent_types": ["qualification_analysis", "deal_review"],
        "motions": ["outbound", "expansion"],
        "roles": ["sales_rep", "ae"],
    },
    "account_health": {
        "intent_types": ["churn_risk_assessment", "health_scoring", "expansion_opportunity"],
        "motions": ["renewal", "expansion", "churn_prevention"],
        "roles": ["csm", "am", "sales_rep"],
    },
    "forecast": {
        "intent_types": ["forecast_review", "quota_analysis", "pipeline_analysis"],
        "motions": ["outbound", "expansion"],
        "roles": ["sales_rep", "sales_manager"],
    },
    "compensation": {
        "intent_types": ["performance_tracking", "quota_analysis"],
        "motions": ["outbound", "expansion"],
        "roles": ["sales_rep", "sales_manager"],
    },
    "meetings": {
        "intent_types": ["relationship_building", "deal_review", "performance_tracking"],
        "motions": ["expansion", "renewal"],
        "roles": ["ae", "csm", "sales_rep"],
    },
    "activity": {
        "intent_types": ["prospecting_analysis", "activity_tracking", "pipeline_generation"],
        "motions": ["outbound", "inbound"],
        "roles": ["sdr", "bdr", "sales_rep"],
    },
    "general": {
        "intent_types": ["account_discovery", "pipeline_analysis", "performance_tracking"],
        "motions": ["outbound", "inbound", "expansion"],
        "roles": ["sales_rep", "sales_manager"],
    },
}

# Prompt templates for each category
PROMPT_TEMPLATES = {
    "quota_forecast": [
        "How am I tracking against {term}?",
        "What's my {term} status?",
        "Show me {term} for this quarter",
        "Help me with {term}",
        "I need to {term}",
        "Where do I stand on {term}?",
        "Am I on track to {term}?",
        "{term} - what should I focus on?",
    ],
    "pipeline_deals": [
        "Show me my {term}",
        "What's happening with {term}?",
        "Help me understand {term}",
        "Review my {term}",
        "Which {term} need attention?",
        "I have a {term} coming up",
        "Let's look at {term}",
        "{term} - give me insights",
    ],
    "lead_qualification": [
        "Is this lead {term}?",
        "Help me qualify using {term}",
        "What's the {term} status?",
        "Show me {term} leads",
        "Check {term} for this prospect",
        "Does this account meet {term}?",
        "{term} - what should I look for?",
    ],
    "gtm_motion": [
        "I need to run {term}",
        "Help me with {term} strategy",
        "What's our {term} approach?",
        "Show me {term} opportunities",
        "{term} - where should I focus?",
        "I'm doing {term}, what accounts?",
        "Find me {term} targets",
    ],
    "metrics_kpi": [
        "What's our {term}?",
        "Show me {term} trends",
        "How is {term} looking?",
        "Calculate {term} for my territory",
        "{term} - how are we doing?",
        "Track {term} this quarter",
        "I need to improve {term}",
    ],
    "roles": [
        "As a {term}, what should I prioritize?",
        "Show me what a {term} should focus on",
        "Help me as a {term}",
        "I'm a {term}, show me my accounts",
        "{term} view of my pipeline",
    ],
    "methodology": [
        "Help me apply {term}",
        "Run {term} on this deal",
        "What's the {term} for this opportunity?",
        "Use {term} to analyze this",
        "{term} - check this deal",
    ],
    "account_health": [
        "Show me {term}",
        "Which accounts are {term}?",
        "Find {term} in my portfolio",
        "Check for {term}",
        "{term} - what's the status?",
    ],
    "forecast": [
        "What's my {term}?",
        "Show me {term} for the quarter",
        "{term} - where do I stand?",
        "Calculate my {term}",
        "Help me with {term}",
    ],
    "compensation": [
        "What's my {term}?",
        "Show me {term} potential",
        "Calculate {term} if I close this",
        "{term} - what can I earn?",
    ],
    "meetings": [
        "Prepare for my {term}",
        "I have a {term} coming up",
        "Help me with {term}",
        "What should I cover in {term}?",
    ],
    "activity": [
        "Track my {term}",
        "How many {term} today?",
        "Show me {term} metrics",
        "{term} - what's my rate?",
        "Improve my {term}",
    ],
    "general": [
        "Tell me about {term}",
        "What is {term}?",
        "Help me understand {term}",
        "Show me {term}",
        "{term} - explain this",
    ],
}

# Colloquial prompt variations (more natural phrasing)
COLLOQUIAL_PROMPTS = {
    "my number": [
        "I need to hit my number this quarter",
        "How close am I to my number?",
        "Am I going to make my number?",
        "What do I need to close to hit my number?",
        "I'm behind on my number, what should I do?",
        "Show me how I'm tracking against my number",
    ],
    "hit my number": [
        "Help me hit my number",
        "What accounts will help me hit my number?",
        "I need to hit my number by end of quarter",
        "Can I still hit my number?",
    ],
    "pipeline review": [
        "I have pipeline review with my manager",
        "Help me prep for pipeline review",
        "What should I bring to pipeline review?",
        "Pipeline review is tomorrow, what's at risk?",
    ],
    "my book": [
        "Show me my book of business",
        "What's happening in my book?",
        "Prioritize accounts in my book",
        "My book needs attention, where should I start?",
    ],
    "pipe": [
        "I need to build more pipe",
        "My pipe is thin, help me",
        "Show me my pipe for Q4",
        "How's my pipe looking?",
    ],
    "green account": [
        "Which accounts are green?",
        "Show me green accounts for expansion",
        "Focus on green accounts",
    ],
    "red account": [
        "Which accounts are red?",
        "Show me red accounts that need attention",
        "Alert me on red accounts",
    ],
    "new logo": [
        "I need new logos this quarter",
        "Find me new logo opportunities",
        "How many new logos have I closed?",
    ],
    "deal slip": [
        "Which deals might slip?",
        "Show me deals at risk of slipping",
        "Prevent deal slip this quarter",
    ],
    "commit": [
        "What's my commit for the quarter?",
        "Is this deal commit or best case?",
        "Show me my commit deals",
    ],
    "upside": [
        "What's my upside this quarter?",
        "Show me upside opportunities",
        "Can I pull in any upside?",
    ],
    "disco call": [
        "I have a disco call tomorrow",
        "Help me prep for disco",
        "What should I ask in the disco call?",
    ],
    "QBR": [
        "Prepare for the QBR",
        "I have a QBR next week",
        "What should I cover in the QBR?",
    ],
    "OTE": [
        "What's my OTE?",
        "Am I on track for OTE?",
        "Show me path to OTE",
    ],
    "spiff": [
        "What spiffs are available?",
        "How can I earn the spiff?",
        "Show me spiff opportunities",
    ],
    "land and expand": [
        "I want to land and expand in this account",
        "Show me land and expand opportunities",
        "What's our land and expand strategy?",
    ],
    "whitespace": [
        "Find whitespace in my accounts",
        "Show me whitespace opportunities",
        "Where's the whitespace in my territory?",
    ],
    "greenfield": [
        "Find me greenfield accounts",
        "Show me greenfield opportunities in EMEA",
        "I need greenfield prospects",
    ],
}


def generate_intent_ir(term: Dict, category: str) -> Dict:
    """Generate Intent IR for a term"""
    mapping = CATEGORY_INTENT_MAP.get(category, CATEGORY_INTENT_MAP["general"])

    intent_ir = {
        "intent_type": random.choice(mapping["intent_types"]),
        "motion": random.choice(mapping["motions"]),
        "role_assumption": random.choice(mapping["roles"]),
        "account_scope": random.choice(["existing", "new", "all"]),
        "time_horizon": random.choice(["this_quarter", "this_month", "this_year", "next_quarter"]),
        "confidence_scores": {
            "intent_type": round(random.uniform(0.7, 0.95), 2),
            "motion": round(random.uniform(0.6, 0.9), 2),
        },
        "assumptions_applied": [],
    }

    # Add contextual assumptions
    term_lower = term["term"].lower()

    if "quota" in term_lower or "number" in term_lower:
        intent_ir["intent_type"] = "forecast_review"
        intent_ir["assumptions_applied"].append(f"Interpreted '{term['term']}' as sales quota/target")

    if "pipeline" in term_lower or "pipe" in term_lower:
        intent_ir["intent_type"] = "pipeline_analysis"
        intent_ir["assumptions_applied"].append(f"'{term['term']}' refers to active sales opportunities")

    if "review" in term_lower:
        intent_ir["assumptions_applied"].append("Meeting context - likely needs preparation insights")

    if "churn" in term_lower or "risk" in term_lower or "red" in term_lower:
        intent_ir["intent_type"] = "churn_risk_assessment"
        intent_ir["motion"] = "churn_prevention"

    if "expansion" in term_lower or "upsell" in term_lower or "whitespace" in term_lower:
        intent_ir["intent_type"] = "expansion_opportunity"
        intent_ir["motion"] = "expansion"
        intent_ir["account_scope"] = "existing"

    if "outbound" in term_lower or "cold" in term_lower or "prospect" in term_lower:
        intent_ir["motion"] = "outbound"

    if "inbound" in term_lower:
        intent_ir["motion"] = "inbound"

    if "new logo" in term_lower or "greenfield" in term_lower or "net-new" in term_lower:
        intent_ir["motion"] = "outbound"
        intent_ir["account_scope"] = "new"

    if "sdr" in term_lower or "bdr" in term_lower:
        intent_ir["role_assumption"] = "sdr"

    if "ae" in term_lower or "close" in term_lower:
        intent_ir["role_assumption"] = "ae"

    if "csm" in term_lower or "customer success" in term_lower:
        intent_ir["role_assumption"] = "csm"

    if "manager" in term_lower:
        intent_ir["role_assumption"] = "sales_manager"

    if not intent_ir["assumptions_applied"]:
        intent_ir["assumptions_applied"].append(f"Interpreted as {category} context")

    return intent_ir


def generate_prompt(term: Dict, category: str) -> str:
    """Generate a natural prompt using the term"""
    templates = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["general"])
    template = random.choice(templates)
    return template.format(term=term["term"])


def generate_colloquial_examples() -> List[Dict]:
    """Generate examples from colloquial prompts with targeted Intent IR"""
    examples = []

    for term, prompts in COLLOQUIAL_PROMPTS.items():
        for prompt in prompts:
            intent_ir = {
                "intent_type": "account_discovery",
                "motion": "outbound",
                "role_assumption": "sales_rep",
                "account_scope": "existing",
                "time_horizon": "this_quarter",
                "confidence_scores": {
                    "intent_type": round(random.uniform(0.8, 0.95), 2),
                    "motion": round(random.uniform(0.7, 0.9), 2),
                },
                "assumptions_applied": [],
            }

            # Specific mappings for colloquialisms
            if "number" in term:
                intent_ir["intent_type"] = "forecast_review"
                intent_ir["assumptions_applied"].append("'my number' refers to quarterly sales quota")

            if "pipeline review" in term:
                intent_ir["intent_type"] = "pipeline_analysis"
                intent_ir["assumptions_applied"].append("Pipeline review is recurring meeting with sales manager to review deals")

            if "book" in term:
                intent_ir["intent_type"] = "account_discovery"
                intent_ir["assumptions_applied"].append("'my book' refers to assigned account portfolio")

            if "pipe" in term:
                intent_ir["intent_type"] = "pipeline_analysis"
                intent_ir["assumptions_applied"].append("'pipe' is shorthand for sales pipeline")

            if "green" in term:
                intent_ir["intent_type"] = "expansion_opportunity"
                intent_ir["motion"] = "expansion"
                intent_ir["assumptions_applied"].append("Green accounts are healthy with positive engagement")

            if "red" in term:
                intent_ir["intent_type"] = "churn_risk_assessment"
                intent_ir["motion"] = "churn_prevention"
                intent_ir["assumptions_applied"].append("Red accounts are at-risk showing churn signals")

            if "logo" in term:
                intent_ir["intent_type"] = "account_discovery"
                intent_ir["motion"] = "outbound"
                intent_ir["account_scope"] = "new"
                intent_ir["assumptions_applied"].append("'new logo' means net-new customer acquisition")

            if "slip" in term:
                intent_ir["intent_type"] = "pipeline_analysis"
                intent_ir["assumptions_applied"].append("Deal slip means close date moving to later period")

            if "commit" in term:
                intent_ir["intent_type"] = "forecast_review"
                intent_ir["assumptions_applied"].append("Commit deals are high-confidence this-quarter closes")

            if "upside" in term:
                intent_ir["intent_type"] = "forecast_review"
                intent_ir["assumptions_applied"].append("Upside deals are stretch opportunities beyond commit")

            if "disco" in term:
                intent_ir["intent_type"] = "meeting_prep"
                intent_ir["motion"] = "outbound"
                intent_ir["assumptions_applied"].append("'disco call' is discovery call to understand prospect needs")

            if "QBR" in term:
                intent_ir["intent_type"] = "meeting_prep"
                intent_ir["motion"] = "expansion"
                intent_ir["account_scope"] = "existing"
                intent_ir["assumptions_applied"].append("QBR is Quarterly Business Review with customer")

            if "OTE" in term:
                intent_ir["intent_type"] = "performance_tracking"
                intent_ir["assumptions_applied"].append("OTE is On-Target Earnings (base + commission at quota)")

            if "spiff" in term:
                intent_ir["intent_type"] = "performance_tracking"
                intent_ir["assumptions_applied"].append("Spiff is short-term bonus for specific sales behaviors")

            if "land and expand" in term:
                intent_ir["intent_type"] = "expansion_opportunity"
                intent_ir["motion"] = "expansion"
                intent_ir["assumptions_applied"].append("Land and expand means starting small then growing within account")

            if "whitespace" in term:
                intent_ir["intent_type"] = "expansion_opportunity"
                intent_ir["motion"] = "expansion"
                intent_ir["account_scope"] = "existing"
                intent_ir["assumptions_applied"].append("Whitespace is untapped opportunity within existing customer")

            if "greenfield" in term:
                intent_ir["intent_type"] = "account_discovery"
                intent_ir["motion"] = "outbound"
                intent_ir["account_scope"] = "new"
                intent_ir["assumptions_applied"].append("Greenfield means net-new prospect with no existing relationship")

            examples.append({
                "gtm_prompt": prompt,
                "intent_ir": intent_ir,
            })

    return examples


def generate_vocabulary_examples() -> List[Dict]:
    """Generate training examples from vocabulary terms"""
    examples = []

    for category, terms in vocab["by_category"].items():
        for term in terms:
            # Skip terms that are too generic
            if len(term["term"]) < 3:
                continue

            prompt = generate_prompt(term, category)
            intent_ir = generate_intent_ir(term, category)

            examples.append({
                "gtm_prompt": prompt,
                "intent_ir": intent_ir,
            })

    return examples


def main():
    print("=" * 60)
    print("TRAINING DATA GENERATOR")
    print("=" * 60)

    # Generate from vocabulary
    print("\nGenerating from vocabulary...")
    vocab_examples = generate_vocabulary_examples()
    print(f"  Generated {len(vocab_examples)} vocabulary examples")

    # Generate from colloquialisms
    print("\nGenerating from colloquialisms...")
    colloquial_examples = generate_colloquial_examples()
    print(f"  Generated {len(colloquial_examples)} colloquial examples")

    # Combine
    all_examples = vocab_examples + colloquial_examples

    # Shuffle
    random.shuffle(all_examples)

    # Save to JSONL
    output_path = "data/vocabulary_training.jsonl"
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n" + "=" * 60)
    print(f"RESULTS")
    print("=" * 60)
    print(f"Total examples: {len(all_examples)}")
    print(f"Saved to: {output_path}")

    # Sample output
    print(f"\nSample examples:")
    for i, ex in enumerate(all_examples[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {ex['gtm_prompt']}")
        print(f"Intent: {ex['intent_ir']['intent_type']}")
        print(f"Motion: {ex['intent_ir']['motion']}")
        if ex['intent_ir']['assumptions_applied']:
            print(f"Assumption: {ex['intent_ir']['assumptions_applied'][0]}")


if __name__ == "__main__":
    main()
