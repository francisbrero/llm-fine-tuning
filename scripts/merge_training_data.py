#!/usr/bin/env python3
"""
Merge Training Data

Combines existing training data with newly generated vocabulary examples.
Deduplicates and validates the merged dataset.
"""

import json
from typing import Dict, List, Set

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    examples = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def get_prompt_key(prompt: str) -> str:
    """Normalize prompt for deduplication"""
    return prompt.lower().strip()


def validate_example(ex: Dict) -> bool:
    """Validate example structure"""
    if "gtm_prompt" not in ex or "intent_ir" not in ex:
        return False

    ir = ex["intent_ir"]
    required_fields = ["intent_type"]
    for field in required_fields:
        if field not in ir:
            return False

    # Ensure confidence_scores are numbers, not strings
    if "confidence_scores" in ir:
        for key, val in ir["confidence_scores"].items():
            if isinstance(val, str):
                try:
                    ir["confidence_scores"][key] = float(val)
                except:
                    ir["confidence_scores"][key] = 0.8

    return True


def main():
    print("=" * 60)
    print("MERGE TRAINING DATA")
    print("=" * 60)

    # Load existing training data
    print("\nLoading existing training data...")
    existing = load_jsonl("data/ccp_training.jsonl")
    print(f"  Loaded {len(existing)} existing examples")

    # Load new vocabulary training data
    print("\nLoading vocabulary training data...")
    vocabulary = load_jsonl("data/vocabulary_training.jsonl")
    print(f"  Loaded {len(vocabulary)} vocabulary examples")

    # Deduplicate by prompt
    print("\nDeduplicating...")
    seen_prompts: Set[str] = set()
    merged = []

    # Prioritize existing examples
    for ex in existing:
        if validate_example(ex):
            key = get_prompt_key(ex["gtm_prompt"])
            if key not in seen_prompts:
                seen_prompts.add(key)
                merged.append(ex)

    existing_count = len(merged)

    # Add vocabulary examples
    for ex in vocabulary:
        if validate_example(ex):
            key = get_prompt_key(ex["gtm_prompt"])
            if key not in seen_prompts:
                seen_prompts.add(key)
                merged.append(ex)

    new_count = len(merged) - existing_count

    # Save merged data
    output_path = "data/ccp_training_merged.jsonl"
    with open(output_path, "w") as f:
        for ex in merged:
            f.write(json.dumps(ex) + "\n")

    print(f"\n" + "=" * 60)
    print(f"RESULTS")
    print("=" * 60)
    print(f"Existing examples kept: {existing_count}")
    print(f"New vocabulary examples added: {new_count}")
    print(f"Total merged examples: {len(merged)}")
    print(f"Duplicates removed: {len(existing) + len(vocabulary) - len(merged)}")
    print(f"\nSaved to: {output_path}")

    # Category distribution
    print(f"\nIntent type distribution:")
    intent_counts = {}
    for ex in merged:
        intent = ex["intent_ir"].get("intent_type", "unknown")
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {intent}: {count}")


if __name__ == "__main__":
    main()
