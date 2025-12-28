# GTM Domain Knowledge for Context Collapse Parser

This document defines the domain-specific knowledge that CCP must internalize to accurately collapse implied GTM context. Use this to guide training data generation and prompt engineering.

---

## 1. GTM Motions

Enterprise B2B sales operates through distinct **motions** — each with different intents, metrics, and workflows.

| Motion | Description | Typical User | Key Signals |
|--------|-------------|--------------|-------------|
| **Outbound** | Prospecting net-new accounts | SDR, AE | "prospects", "new logos", "pipeline generation", "cold" |
| **Inbound** | Responding to inbound demand | SDR, AE | "leads", "MQLs", "demo requests", "hand-raisers" |
| **Expansion** | Growing existing customers | AE, CSM | "upsell", "cross-sell", "whitespace", "land and expand" |
| **Renewal** | Retaining existing ARR | CSM, AM | "renewal", "churn risk", "retention", "NRR" |
| **Churn Prevention** | Saving at-risk accounts | CSM, AM | "at risk", "health score", "red accounts", "save" |

### Implied Context Rules

- If user mentions "prospects" or "net new" → motion = outbound
- If user mentions "existing customers" or "book of business" → motion = expansion OR renewal
- If user says "my accounts" without qualification:
  - Sales rep → likely existing pipeline (expansion)
  - CSM → likely existing customers (renewal/churn prevention)
  - SDR → likely assigned prospects (outbound)

---

## 2. User Roles & Assumptions

CCP must infer the user's role when not stated, as it dramatically affects interpretation.

| Role | Alias Terms | Default Motion | Default Time Horizon |
|------|-------------|----------------|---------------------|
| **sales_rep** (AE) | "rep", "AE", "account exec" | expansion | this_quarter |
| **sales_manager** | "manager", "director of sales" | pipeline_analysis | this_quarter |
| **sdr** | "SDR", "BDR", "prospector" | outbound | this_week |
| **revops** | "ops", "revenue ops", "sales ops" | forecast_review | this_quarter |
| **marketing** | "marketing", "demand gen" | lead_prioritization | this_month |
| **cs** (CSM) | "CSM", "customer success" | renewal | this_quarter |
| **exec** | "VP", "CRO", "CEO" | forecast_review | this_quarter/this_year |

### Role Inference Signals

- "My number" / "my quota" → sales_rep or sales_manager
- "Pipeline" / "forecast" → sales_manager or revops
- "Leads" / "MQLs" → marketing or SDR
- "Health score" / "renewal" → cs
- "Board" / "investors" → exec

---

## 3. ICP (Ideal Customer Profile) Context

CCP does NOT store ICP definitions. Instead, it determines:
1. Whether ICP filtering is implied
2. Which ICP selector to emit (for downstream resolution)

### ICP Selector Patterns

| Selector | When to Use |
|----------|-------------|
| `default` | No specific ICP mentioned, use org's primary ICP |
| `product_{name}` | User mentions specific product line |
| `segment_{name}` | User mentions segment (SMB, mid-market, enterprise) |
| `geo_{region}` | ICP varies by geography |
| `all` | User explicitly wants to see everything, including non-ICP |

### Signals for ICP Resolution Required

Set `icp_resolution_required: true` when:
- Multiple products exist and user didn't specify
- User asks about "fit" or "qualified" accounts
- User mentions "ICP" explicitly but not which one

---

## 4. Account Scope

| Scope | Description | Signals |
|-------|-------------|---------|
| **net_new** | Accounts not in CRM / never engaged | "prospects", "greenfield", "new logos" |
| **existing** | Current customers or active pipeline | "customers", "my accounts", "book of business" |
| **churned** | Former customers | "churned", "lost", "win-back" |
| **all** | All accounts regardless of status | "all accounts", "entire database" |

### Default Scope by Motion

- outbound → net_new
- inbound → net_new (unless "customer" mentioned)
- expansion → existing
- renewal → existing
- churn_prevention → existing

---

## 5. Time Horizon

| Horizon | Interpretation | Signals |
|---------|---------------|---------|
| **immediate** | Right now, today | "now", "today", "urgent", "asap" |
| **this_week** | Next 7 days | "this week", "by Friday" |
| **this_month** | Current calendar month | "this month", "by end of month" |
| **this_quarter** | Current fiscal quarter | "this quarter", "Q1/Q2/Q3/Q4", "by quarter end" |
| **this_year** | Current fiscal year | "this year", "annual", "FY" |
| **custom** | Specific date range mentioned | "next 90 days", "Jan-Mar" |

### Default Time Horizon by Intent

- pipeline_analysis → this_quarter
- forecast_review → this_quarter
- account_discovery → this_month
- churn_risk_assessment → immediate or this_month
- lead_prioritization → this_week

---

## 6. Intent Types

| Intent Type | Description | Example Prompts |
|-------------|-------------|-----------------|
| **account_discovery** | Find accounts matching criteria | "Show me accounts like X", "Find prospects in fintech" |
| **pipeline_analysis** | Analyze deal pipeline | "What's in my pipe?", "Show me my deals" |
| **expansion_identification** | Find upsell/cross-sell opportunities | "Where can I expand?", "Whitespace analysis" |
| **churn_risk_assessment** | Identify at-risk customers | "Which accounts are at risk?", "Red accounts" |
| **lead_prioritization** | Rank/prioritize leads | "Which leads should I call first?", "Best MQLs" |
| **territory_planning** | Analyze territory coverage | "Show me my territory", "Account distribution" |
| **forecast_review** | Review sales forecast | "Where's the forecast?", "Will we hit the number?" |
| **competitive_analysis** | Analyze competitive deals | "Deals against Competitor X", "Win/loss" |
| **engagement_summary** | Summarize account activity | "What's happening with Account X?", "Activity summary" |

---

## 7. Common GTM Shorthand

CCP must understand enterprise GTM jargon and abbreviations:

| Term | Meaning | Context |
|------|---------|---------|
| "my number" | quota/target | sales_rep or sales_manager |
| "pipe" / "pipeline" | active opportunities | sales |
| "ARR" | Annual Recurring Revenue | expansion, renewal |
| "NRR" | Net Revenue Retention | renewal, churn |
| "MQL" | Marketing Qualified Lead | inbound |
| "SQL" | Sales Qualified Lead | inbound |
| "ACV" | Annual Contract Value | pipeline |
| "land and expand" | small initial deal → grow | expansion |
| "whitespace" | untapped opportunity in existing account | expansion |
| "green/red accounts" | healthy/at-risk accounts | churn |
| "book of business" | assigned accounts | existing scope |
| "hit the number" | achieve quota | sales_rep/manager |
| "logo" | customer (as a count) | "new logos" = new customers |

---

## 8. Geography Scope

When geography is mentioned, emit `geography_scope`. Otherwise, leave null (global/all).

| Pattern | geography_scope |
|---------|-----------------|
| "in EMEA" | "EMEA" |
| "North America" / "NA" | "NA" |
| "APAC" / "Asia" | "APAC" |
| "LATAM" | "LATAM" |
| "US" / "United States" | "US" |
| Specific country | country name |
| No mention | null |

---

## 9. Output Format

Infer desired output format from how user asks:

| Format | Signals |
|--------|---------|
| **list** | "show me", "give me", "list" |
| **summary** | "summarize", "overview", "brief" |
| **detailed** | "detailed", "deep dive", "analysis" |
| **export** | "export", "download", "CSV", "spreadsheet" |
| **visualization** | "chart", "graph", "visualize" |

Default: `list` for discovery, `summary` for analysis.

---

## 10. Confidence Scoring Guidelines

CCP must output confidence scores (0.0-1.0) for each inferred field:

| Confidence | When to Apply |
|------------|---------------|
| **0.9-1.0** | Explicit mention in prompt |
| **0.7-0.9** | Strong implicit signal |
| **0.5-0.7** | Reasonable inference from context |
| **0.3-0.5** | Weak signal, default assumption |
| **< 0.3** | Consider setting `clarification_needed: true` |

### Fields That Often Need Clarification

- `icp_selector` when multiple ICPs exist
- `account_scope` when ambiguous ("accounts" without context)
- `time_horizon` when urgency is unclear
- `role_assumption` when no role signals present

---

## 11. Example Prompt → IR Mappings

### Example 1: "Show me my best accounts"

```json
{
  "intent_type": "account_discovery",
  "motion": "expansion",
  "role_assumption": "sales_rep",
  "account_scope": "existing",
  "icp_selector": "default",
  "icp_resolution_required": false,
  "geography_scope": null,
  "time_horizon": "this_quarter",
  "output_format": "list",
  "confidence_scores": {
    "intent_type": 0.85,
    "motion": 0.7,
    "role_assumption": 0.6,
    "account_scope": 0.8,
    "time_horizon": 0.5
  },
  "assumptions_applied": [
    "Assumed 'best' means high-fit existing accounts",
    "Assumed sales rep role from 'my accounts'",
    "Assumed expansion motion for existing accounts"
  ],
  "clarification_needed": false
}
```

### Example 2: "Which deals are at risk this quarter?"

```json
{
  "intent_type": "churn_risk_assessment",
  "motion": "churn_prevention",
  "role_assumption": "sales_manager",
  "account_scope": "existing",
  "icp_selector": "default",
  "icp_resolution_required": false,
  "geography_scope": null,
  "time_horizon": "this_quarter",
  "output_format": "list",
  "confidence_scores": {
    "intent_type": 0.95,
    "motion": 0.9,
    "role_assumption": 0.7,
    "account_scope": 0.95,
    "time_horizon": 1.0
  },
  "assumptions_applied": [
    "Assumed manager role from deal-level risk question",
    "Time horizon explicit in prompt"
  ],
  "clarification_needed": false
}
```

### Example 3: "I need to hit my number"

```json
{
  "intent_type": "pipeline_analysis",
  "motion": "expansion",
  "role_assumption": "sales_rep",
  "account_scope": "existing",
  "icp_selector": "default",
  "icp_resolution_required": false,
  "geography_scope": null,
  "time_horizon": "this_quarter",
  "output_format": "summary",
  "confidence_scores": {
    "intent_type": 0.7,
    "motion": 0.6,
    "role_assumption": 0.85,
    "account_scope": 0.6,
    "time_horizon": 0.7
  },
  "assumptions_applied": [
    "'Hit my number' implies quota attainment goal",
    "Assumed pipeline analysis to identify gap-closing opportunities",
    "Assumed quarterly quota cycle"
  ],
  "clarification_needed": true
}
```

---

## 12. Training Data Generation Guidelines

When generating training data for CCP:

1. **Vary prompt styles**: formal, casual, shorthand, incomplete sentences
2. **Cover all roles**: Ensure each role has sufficient examples
3. **Include ambiguous cases**: Where clarification_needed should be true
4. **Mix explicit and implicit**: Some prompts state intent clearly, others require inference
5. **Include negative examples**: Non-GTM prompts should trigger high uncertainty
6. **Geographic diversity**: Include regional variations in terminology
7. **Multi-signal prompts**: "My EMEA enterprise deals at risk" has multiple signals

### Training Data Format

```json
{
  "gtm_prompt": "Show me my best accounts",
  "intent_ir": {
    "intent_type": "account_discovery",
    "motion": "expansion",
    ...
  }
}
```
