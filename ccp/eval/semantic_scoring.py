"""
Semantic evaluation scoring for CCP.

Tests GTM context understanding and business value,
not just syntactic field matching.
"""

from typing import Dict, List, Any, Optional, Set
import re


# Semantic equivalence mappings
INTENT_TYPE_EQUIVALENTS = {
    "churn_risk_assessment": ["churn_risk_analysis", "at_risk_identification", "churn_risk_forecast"],
    "account_discovery": ["account_search", "account_ranking", "top_accounts", "account_identification"],
    "pipeline_analysis": ["deal_analysis", "pipeline_review", "deal_stage_review"],
    "forecast_review": ["forecast_analysis", "forecast_check", "quota_review"],
    "lead_prioritization": ["lead_scoring", "lead_ranking", "lead_qualification"],
    "expansion_identification": ["upsell_identification", "expansion_opportunity"],
}

MOTION_EQUIVALENTS = {
    "churn_prevention": ["retention", "save"],
    "expansion": ["upsell", "cross_sell", "growth"],
    "outbound": ["prospecting", "new_business"],
    "inbound": ["response", "inquiry_handling"],
    "renewal": ["renewal_management"],
}

ROLE_EQUIVALENTS = {
    "sales_rep": ["ae", "account_executive", "sales"],
    "sales_manager": ["sales_lead", "vp_sales"],
    "cs": ["csm", "customer_success", "account_manager", "am"],
    "sdr": ["bdr", "sales_dev"],
    "revops": ["operations", "sales_ops"],
}

ACCOUNT_SCOPE_EQUIVALENTS = {
    "existing": ["current", "active"],
    "net_new": ["new", "prospecting", "prospect"],
    "churned": ["lost", "former"],
    "all": ["any", "both"],
}

# Tool category mappings (Intent IR → Expected Tool Categories)
INTENT_TO_TOOL_CATEGORIES = {
    "churn_risk_assessment": {"correct": ["risk_scoring", "engagement_analysis", "health_metrics", "churn_prediction"], "wrong": ["prospecting", "lead_gen", "outbound_tools"]},
    "account_discovery": {"correct": ["account_search", "icp_filter", "firmographic_data", "account_scoring"], "wrong": ["lead_routing", "churn_tools"]},
    "pipeline_analysis": {"correct": ["deal_stage", "forecast", "pipeline_health", "win_rate"], "wrong": ["prospecting", "churn_tools"]},
    "forecast_review": {"correct": ["forecast", "quota_tracking", "pipeline_health"], "wrong": ["prospecting", "lead_gen"]},
    "lead_prioritization": {"correct": ["lead_scoring", "intent_signals", "engagement_tracking"], "wrong": ["churn_tools", "account_expansion"]},
}

MOTION_TO_TOOL_CATEGORIES = {
    "churn_prevention": {"correct": ["retention_tools", "engagement_tools", "health_monitoring", "risk_alerts"], "wrong": ["prospecting", "lead_gen", "cold_outreach"]},
    "expansion": {"correct": ["upsell_signals", "expansion_opportunities", "account_scoring", "whitespace_analysis"], "wrong": ["prospecting", "lead_gen"]},
    "outbound": {"correct": ["prospecting", "lead_gen", "cold_outreach", "target_account_list"], "wrong": ["churn_tools", "retention_tools"]},
    "inbound": {"correct": ["lead_routing", "response_management", "qualification", "intent_capture"], "wrong": ["prospecting", "cold_outreach"]},
}

ACCOUNT_SCOPE_TO_TOOLS = {
    "existing": {"correct": ["account_data", "engagement_history", "usage_data"], "wrong": ["lead_gen", "prospecting_lists"]},
    "net_new": {"correct": ["prospecting_lists", "firmographic_data", "lead_sources"], "wrong": ["usage_data", "renewal_tracking"]},
}


def normalize_value(value: Any) -> str:
    """Normalize a field value for comparison."""
    if value is None:
        return ""
    return str(value).lower().strip().replace("-", "_").replace(" ", "_")


def is_semantically_equivalent(predicted: Any, expected: Any, equivalents: Dict[str, List[str]]) -> bool:
    """Check if predicted value is semantically equivalent to expected."""
    pred_norm = normalize_value(predicted)
    exp_norm = normalize_value(expected)

    # Exact match
    if pred_norm == exp_norm:
        return True

    # Check if predicted is in the equivalents list for expected
    if exp_norm in equivalents:
        acceptable = [normalize_value(v) for v in equivalents[exp_norm]]
        if pred_norm in acceptable:
            return True

    # Check reverse (sometimes we expect a variant)
    for canonical, variants in equivalents.items():
        variants_norm = [normalize_value(v) for v in variants]
        if exp_norm in variants_norm and pred_norm == normalize_value(canonical):
            return True

    return False


def score_intent_type(predicted_ir: Dict, expected_ir: Dict) -> Dict[str, Any]:
    """Score intent_type understanding."""
    predicted = predicted_ir.get("intent_type")
    expected = expected_ir.get("intent_type")

    exact_match = normalize_value(predicted) == normalize_value(expected)
    semantic_match = is_semantically_equivalent(predicted, expected, INTENT_TYPE_EQUIVALENTS)

    return {
        "field": "intent_type",
        "exact_match": exact_match,
        "semantic_match": semantic_match,
        "predicted": predicted,
        "expected": expected,
    }


def score_motion(predicted_ir: Dict, expected_ir: Dict) -> Dict[str, Any]:
    """Score motion understanding."""
    predicted = predicted_ir.get("motion")
    expected = expected_ir.get("motion")

    exact_match = normalize_value(predicted) == normalize_value(expected)
    semantic_match = is_semantically_equivalent(predicted, expected, MOTION_EQUIVALENTS)

    return {
        "field": "motion",
        "exact_match": exact_match,
        "semantic_match": semantic_match,
        "predicted": predicted,
        "expected": expected,
    }


def score_role_assumption(predicted_ir: Dict, expected_ir: Dict) -> Dict[str, Any]:
    """Score role_assumption understanding."""
    predicted = predicted_ir.get("role_assumption")
    expected = expected_ir.get("role_assumption")

    exact_match = normalize_value(predicted) == normalize_value(expected)
    semantic_match = is_semantically_equivalent(predicted, expected, ROLE_EQUIVALENTS)

    return {
        "field": "role_assumption",
        "exact_match": exact_match,
        "semantic_match": semantic_match,
        "predicted": predicted,
        "expected": expected,
    }


def score_account_scope(predicted_ir: Dict, expected_ir: Dict) -> Dict[str, Any]:
    """Score account_scope understanding."""
    predicted = predicted_ir.get("account_scope")
    expected = expected_ir.get("account_scope")

    exact_match = normalize_value(predicted) == normalize_value(expected)
    semantic_match = is_semantically_equivalent(predicted, expected, ACCOUNT_SCOPE_EQUIVALENTS)

    return {
        "field": "account_scope",
        "exact_match": exact_match,
        "semantic_match": semantic_match,
        "predicted": predicted,
        "expected": expected,
    }


def infer_tool_categories(intent_ir: Dict) -> Set[str]:
    """Infer which tool categories would be selected from an Intent IR."""
    tools = set()

    intent_type = normalize_value(intent_ir.get("intent_type", ""))
    motion = normalize_value(intent_ir.get("motion", ""))
    account_scope = normalize_value(intent_ir.get("account_scope", ""))

    # Add tools from intent_type
    if intent_type in INTENT_TO_TOOL_CATEGORIES:
        tools.update(INTENT_TO_TOOL_CATEGORIES[intent_type]["correct"])

    # Add tools from motion
    if motion in MOTION_TO_TOOL_CATEGORIES:
        tools.update(MOTION_TO_TOOL_CATEGORIES[motion]["correct"])

    # Add tools from account_scope
    if account_scope in ACCOUNT_SCOPE_TO_TOOLS:
        tools.update(ACCOUNT_SCOPE_TO_TOOLS[account_scope]["correct"])

    return tools


def score_tool_selection(predicted_ir: Dict, expected_ir: Dict) -> Dict[str, Any]:
    """
    Score whether the predicted IR would lead to correct tool selection.

    This is the KEY business value metric - does the IR result in the right actions?
    """
    predicted_tools = infer_tool_categories(predicted_ir)
    expected_tools = infer_tool_categories(expected_ir)

    # Get wrong tools that should NOT be selected
    wrong_tools = set()
    intent_type = normalize_value(expected_ir.get("intent_type", ""))
    motion = normalize_value(expected_ir.get("motion", ""))

    if intent_type in INTENT_TO_TOOL_CATEGORIES:
        wrong_tools.update(INTENT_TO_TOOL_CATEGORIES[intent_type]["wrong"])
    if motion in MOTION_TO_TOOL_CATEGORIES:
        wrong_tools.update(MOTION_TO_TOOL_CATEGORIES[motion]["wrong"])

    # Check overlap
    correct_overlap = len(predicted_tools & expected_tools)
    wrong_overlap = len(predicted_tools & wrong_tools)

    # Tool selection is valid if:
    # 1. At least some correct tools are selected
    # 2. No catastrophically wrong tools are selected
    has_correct_tools = correct_overlap > 0
    has_wrong_tools = wrong_overlap > 0

    tool_selection_valid = has_correct_tools and not has_wrong_tools

    return {
        "tool_selection_valid": tool_selection_valid,
        "correct_tools_count": correct_overlap,
        "wrong_tools_count": wrong_overlap,
        "predicted_tools": list(predicted_tools),
        "expected_tools": list(expected_tools),
        "wrong_tools_selected": list(predicted_tools & wrong_tools),
    }


def score_critical_fields(predicted_ir: Dict, expected_ir: Dict, critical_fields: List[str]) -> Dict[str, Any]:
    """
    Score critical fields that determine tool selection.

    These fields MUST be correct for the system to work.
    """
    scores = {}
    correct_count = 0

    for field in critical_fields:
        if field == "intent_type":
            result = score_intent_type(predicted_ir, expected_ir)
        elif field == "motion":
            result = score_motion(predicted_ir, expected_ir)
        elif field == "role_assumption":
            result = score_role_assumption(predicted_ir, expected_ir)
        elif field == "account_scope":
            result = score_account_scope(predicted_ir, expected_ir)
        else:
            # Generic exact match for other fields
            predicted = predicted_ir.get(field)
            expected = expected_ir.get(field)
            result = {
                "field": field,
                "exact_match": normalize_value(predicted) == normalize_value(expected),
                "semantic_match": normalize_value(predicted) == normalize_value(expected),
                "predicted": predicted,
                "expected": expected,
            }

        scores[field] = result
        if result["semantic_match"]:
            correct_count += 1

    return {
        "field_scores": scores,
        "critical_accuracy": correct_count / len(critical_fields) if critical_fields else 0.0,
        "correct_count": correct_count,
        "total_count": len(critical_fields),
    }


def score_semantic_understanding(
    predicted_ir: Optional[Dict],
    expected_ir: Dict,
    is_valid_json: bool,
    critical_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Main semantic scoring function.

    Returns a score that measures GTM understanding and business value,
    not just syntactic correctness.
    """
    if critical_fields is None:
        critical_fields = ["intent_type", "motion", "account_scope"]

    score = {
        "json_valid": is_valid_json,
        "semantic_scores": {},
        "critical_field_accuracy": 0.0,
        "tool_selection_valid": False,
        "overall_semantic_accuracy": 0.0,
    }

    if not is_valid_json or predicted_ir is None:
        return score

    # Score critical fields
    critical_result = score_critical_fields(predicted_ir, expected_ir, critical_fields)
    score["critical_field_accuracy"] = critical_result["critical_accuracy"]
    score["semantic_scores"] = critical_result["field_scores"]

    # Score tool selection (most important!)
    tool_result = score_tool_selection(predicted_ir, expected_ir)
    score["tool_selection_valid"] = tool_result["tool_selection_valid"]
    score["tool_selection_details"] = tool_result

    # Overall semantic accuracy (weighted)
    # Tool selection is weighted 2x because it's what matters for downstream execution
    overall = (
        score["critical_field_accuracy"] * 1.0 +
        (1.0 if score["tool_selection_valid"] else 0.0) * 2.0
    ) / 3.0

    score["overall_semantic_accuracy"] = overall

    return score


def compare_semantic_vs_exact(
    predicted_ir: Optional[Dict],
    expected_ir: Dict,
    is_valid_json: bool
) -> Dict[str, Any]:
    """
    Compare semantic scoring vs exact matching to show the difference.
    """
    result = {
        "exact_match_scores": {},
        "semantic_scores": {},
        "comparison": {}
    }

    if not is_valid_json or predicted_ir is None:
        return result

    # Exact match scoring (current approach)
    exact_matches = 0
    exact_total = 0
    for field in ["intent_type", "motion", "role_assumption", "account_scope"]:
        if field in expected_ir:
            exact_total += 1
            predicted = normalize_value(predicted_ir.get(field))
            expected = normalize_value(expected_ir.get(field))
            match = predicted == expected
            result["exact_match_scores"][field] = match
            if match:
                exact_matches += 1

    result["exact_field_accuracy"] = exact_matches / exact_total if exact_total > 0 else 0.0

    # Semantic scoring (proposed approach)
    semantic_score = score_semantic_understanding(predicted_ir, expected_ir, is_valid_json)
    result["semantic_field_accuracy"] = semantic_score["critical_field_accuracy"]
    result["tool_selection_valid"] = semantic_score["tool_selection_valid"]
    result["overall_semantic_accuracy"] = semantic_score["overall_semantic_accuracy"]

    # Comparison
    result["comparison"] = {
        "exact_accuracy": result["exact_field_accuracy"],
        "semantic_accuracy": result["semantic_field_accuracy"],
        "improvement": result["semantic_field_accuracy"] - result["exact_field_accuracy"],
        "tool_selection_correct": semantic_score["tool_selection_valid"],
    }

    return result
