---
description: GTM (Go-To-Market) domain knowledge for CCP intent parsing
globs:
  - "data/**/*.jsonl"
  - "**/gtm*.md"
alwaysApply: false
---

# GTM Domain Knowledge

## Overview
CCP parses GTM (Go-To-Market) prompts into structured intent. This skill covers domain terminology and expected mappings.

## GTM Intent Types

| Intent Type | Description | Example Prompts |
|-------------|-------------|-----------------|
| `forecast_review` | Review pipeline/quota progress | "How are we tracking?", "QBR prep" |
| `pipeline_analysis` | Analyze deal pipeline | "Show me my pipeline", "What's in stage 3?" |
| `lead_prioritization` | Rank/score leads | "Who should I call?", "Best leads" |
| `churn_risk_assessment` | Identify at-risk accounts | "Who's ghosting us?", "Untouched accounts" |
| `account_discovery` | Find new accounts | "Companies like Acme", "ICP matches" |
| `deal_coaching` | Help close deals | "How do I close this?", "Objection handling" |

## GTM Motions

| Motion | Description | Indicators |
|--------|-------------|------------|
| `outbound` | Proactive prospecting | "net new", "cold outreach", "prospect" |
| `inbound` | Lead response | "webinar", "demo request", "MQL" |
| `expansion` | Grow existing accounts | "upsell", "cross-sell", "expand" |
| `renewal` | Retain customers | "renewal", "contract", "anniversary" |
| `churn_prevention` | Stop churn | "at-risk", "ghosting", "silent" |

## Roles

| Role | Description | Typical Requests |
|------|-------------|-----------------|
| `sales_rep` | Individual contributor | Deal-specific, account-specific |
| `sales_manager` | Team lead | Team metrics, forecast roll-up |
| `sdr` | Sales Development Rep | Lead volume, conversion |
| `cs` | Customer Success | Health scores, renewals |
| `revops` | Revenue Operations | Cross-team metrics |
| `marketing` | Marketing | Campaign performance, MQLs |
| `exec` | Executive | High-level summaries |

## B2B Slang Mappings

| Slang | Meaning | Maps To |
|-------|---------|---------|
| "ghosting us" | Accounts not responding | `churn_risk_assessment` |
| "going dark" | Lost contact | `churn_risk_assessment` |
| "QBR" | Quarterly Business Review | `forecast_review` |
| "SQLs" | Sales Qualified Leads | `lead_prioritization` |
| "MQLs" | Marketing Qualified Leads | `lead_prioritization` |
| "ARR" | Annual Recurring Revenue | `forecast_review` |
| "NRR" | Net Revenue Retention | `churn_risk_assessment` |
| "pipe" | Pipeline | `pipeline_analysis` |

## Account Scopes

| Scope | Description |
|-------|-------------|
| `net_new` | Never been a customer |
| `existing` | Current customers |
| `churned` | Former customers |
| `all` | No filter |

## Time Horizons

| Horizon | Description |
|---------|-------------|
| `this_week` | Current 7 days |
| `this_month` | Current month |
| `this_quarter` | Current quarter |
| `next_quarter` | Following quarter |
| `this_year` | Current year |

## Key Files
- `gtm_domain_knowledge.md` - Detailed domain reference
- `data/ccp_training_with_reasoning.jsonl` - Training examples
- `data/ccp_eval.jsonl` - Evaluation test cases

## Resources
- PRD.md - CCP architecture and goals
