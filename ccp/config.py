"""
CCP Configuration Module

Centralized configuration for all CCP operations.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class ModelConfig:
    """Model and adapter configuration."""
    base_model_path: str = "./models/phi-2"
    adapter_path: str = "./ccp-adapter"
    device: str = "auto"  # auto, mps, cuda, cpu

    def resolve_device(self) -> str:
        """Resolve 'auto' to actual device."""
        if self.device != "auto":
            return self.device

        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"


@dataclass
class LoraConfig:
    """LoRA adapter configuration."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    num_epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 1024
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    warmup_steps: int = 0
    weight_decay: float = 0.0

    # Checkpoint resume
    resume_from: Optional[str] = None


@dataclass
class DataConfig:
    """Data paths and settings."""
    training_data_path: str = "./data/ccp_training_with_reasoning.jsonl"
    eval_data_path: str = "./data/ccp_eval.jsonl"
    results_dir: str = "./eval_results"


@dataclass
class CCPConfig:
    """Main CCP configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Schema version
    ir_schema_version: str = "2.0.0"


# System prompt for CCP v2 (Chain-of-Thought)
CCP_SYSTEM_PROMPT = """You are the Phoenix Context Collapse Parser (CCP). Your job is to transform ambiguous GTM (Go-To-Market) prompts into structured GTM Intent IR.

IMPORTANT: You must REASON about the request before producing structured output. This ensures you understand the GTM semantics, not just the JSON format.

Given a user's GTM request:
1. First, analyze the prompt in a <reasoning> section:
   - Identify explicit signals (keywords, jargon, metrics mentioned)
   - Infer implicit context (role, motion, time horizon)
   - Note any ambiguities that affect confidence
   - Explain WHY you chose each field value

2. Then, output the structured intent in an <intent_ir> section as JSON with:
   - intent_type: The primary intent category
   - motion: The GTM motion (outbound, expansion, renewal, etc.)
   - role_assumption: Inferred user role
   - account_scope: Which accounts (net_new, existing, all)
   - icp_selector: Which ICP to apply (default, or specific product/segment)
   - icp_resolution_required: true if ICP needs downstream resolution
   - geography_scope: Geographic filter if mentioned (null if global)
   - time_horizon: Time scope for the request
   - output_format: How results should be presented
   - confidence_scores: 0.0-1.0 confidence for each inferred field
   - assumptions_applied: List of assumptions made
   - clarification_needed: true if request is too ambiguous

Format your response EXACTLY as:
<reasoning>
[Your analysis here]
</reasoning>

<intent_ir>
[Valid JSON here]
</intent_ir>"""


# GTM Intent IR Schema definitions
INTENT_TYPES = [
    "account_discovery",
    "pipeline_analysis",
    "expansion_identification",
    "churn_risk_assessment",
    "lead_prioritization",
    "territory_planning",
    "forecast_review",
    "competitive_analysis",
    "engagement_summary",
    # Extended types from training data
    "performance_tracking",
    "qualification_analysis",
    "lead_scoring",
    "deal_review",
    "quota_analysis",
    "opportunity_scoring",
    "metrics_analysis",
    "expansion_opportunity",
    "resource_planning",
    "team_analysis",
    "meeting_prep",
    "relationship_building",
    "prospecting_analysis",
    "pipeline_generation",
    "health_scoring",
    "activity_tracking",
]

MOTIONS = [
    "outbound",
    "inbound",
    "expansion",
    "renewal",
    "churn_prevention",
]

ROLES = [
    "sales_rep",
    "sales_manager",
    "revops",
    "marketing",
    "cs",
    "exec",
    "sdr",   # Sales Development Representative
    "bdr",   # Business Development Representative
    "ae",    # Account Executive
    "am",    # Account Manager
    "csm",   # Customer Success Manager
    "cro",   # Chief Revenue Officer
    "vp_sales",
    "director",
]

ACCOUNT_SCOPES = [
    "net_new",
    "existing",
    "churned",
    "all",
    "new",  # Alias for net_new in some training data
]

TIME_HORIZONS = [
    "immediate",
    "this_week",
    "this_month",
    "this_quarter",
    "this_year",
    "next_quarter",
    "next_month",
    "custom",
]

OUTPUT_FORMATS = [
    "list",
    "summary",
    "detailed",
    "export",
    "visualization",
]

# Required fields for validation (core fields that must be present)
REQUIRED_IR_FIELDS = [
    "intent_type",
    "motion",
    "role_assumption",
    "account_scope",
    "time_horizon",
    "confidence_scores",
    "assumptions_applied",
]

# Optional fields (nice to have but not required for training)
OPTIONAL_IR_FIELDS = [
    "output_format",
    "clarification_needed",
    "icp_selector",
    "icp_resolution_required",
    "geography_scope",
]

# Fields used for evaluation scoring
EVAL_FIELDS = [
    "intent_type",
    "motion",
    "role_assumption",
    "account_scope",
    "time_horizon",
    "geography_scope",
    "output_format",
    "clarification_needed",
]
