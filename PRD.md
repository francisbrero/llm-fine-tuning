# PRD — Phoenix Context Collapse Parser (CCP)

## 1. Overview

The Phoenix Context Collapse Parser (CCP) is a domain-specific intent parsing layer designed to translate human GTM language into structured, executable intent for AI agents. CCP focuses on collapsing implied, latent enterprise GTM context that is typically unstated by users but assumed by humans. This collapsed context is passed to a downstream general-purpose LLM responsible for planning and tool execution via Phoenix MCP tools.

CCP is explicitly **not** a system prompt and **not** a full agent. It is a probabilistic, dynamic front-end that transforms ambiguous natural language into a compact, observable intermediate representation (IR) optimized for GTM workflows.

---

## 2. Problem Statement

Enterprise GTM users communicate in shorthand, assuming shared context around roles, motions, ICPs, geographies, timing, and workflows. General-purpose LLMs embedded in platforms such as Copilot Studio, Agentforce, or watsonX struggle to reliably infer this implied context, leading to:

* Incorrect tool selection
* Over-clarification or unnecessary follow-up questions
* Confident but incorrect execution
* High variance in agent behavior

Static system prompts and deterministic rules fail to address this problem because implied context is situational, user-dependent, and probabilistic.

---

## 3. Goals and Non-Goals

### Goals

* Accurately infer and collapse implied GTM context from short, ambiguous user prompts
* Produce a structured, testable, and versioned GTM Intent IR
* Reduce ambiguity before planning and execution
* Improve downstream tool selection accuracy and agent reliability
* Enable observability, governance, and auditability of inferred assumptions

### Non-Goals

* Full task planning or multi-step reasoning
* Tool execution
* Long-horizon conversational memory
* Replacing or competing with general-purpose LLMs

---

## 4. Key Concept: Context Collapse

Context collapse refers to the process of layering latent enterprise GTM context on top of a user request in order to transform its meaning, not merely rephrase it. CCP infers unstated assumptions across multiple GTM dimensions and emits them explicitly as structured intent.

CCP differs from prompt engineering in that it:

* Operates per request
* Is probabilistic rather than deterministic
* Produces observable, machine-readable outputs
* Is evaluated independently of downstream execution quality

---

## 5. Architecture Overview

User Prompt
→ Context Collapse Parser (SLM)
→ GTM Intent IR
→ General-Purpose LLM (Planner / Executor)
→ Phoenix MCP Tools

CCP is implemented as a **small language model (SLM)** optimized for enterprise GTM semantics and Phoenix MCP tool literacy. It may be deployed via Phoenix-managed cloud infrastructure or customer-managed environments.

---

## 6. GTM Intent IR

The GTM Intent IR is a structured representation of user intent enriched with inferred context. It is versioned and backward compatible.

### Critical Constraint: ICP Handling

ICP definitions **do not live in the model**. CCP does not encode customer-specific ICP logic. Instead, it:

* Assumes GTM questions are implicitly scoped to one or more active ICPs
* Infers *which* ICP(s) may be relevant when multiple exist
* Emits references or selectors, not concrete ICP definitions

Concrete ICP resolution is delegated to Phoenix MCP tools downstream.

### Illustrative IR Fields (Non-Exhaustive)

* intent_type (e.g., account_discovery, pipeline_analysis, expansion_identification)
* motion (outbound, expansion, renewal)
* role_assumption (sales_rep, sales_manager, revops, exec)
* geography_scope
* icp_selector (e.g., default, product_A, product_B)
* icp_resolution_required (boolean)
* account_scope (net_new, existing, all)
* time_horizon
* output_format
* confidence_scores (per inferred field)
* assumptions_applied (human-readable)

---

## 7. Inputs to Context Collapse

CCP may leverage multiple context sources, with configurable weighting:

* User prompt text
* Organizational configuration (ICP selectors, territories, GTM motions)
* User metadata (role, team)
* Limited conversation context
* Environmental signals (time, quarter)

---

## 8. Outputs

For each request, CCP produces:

* A structured GTM Intent IR
* Confidence scores for each inferred field
* An explicit list of assumptions applied
* Optional flags indicating when clarification is required

All outputs are observable and auditable independently of downstream execution.

---

## 9. Training and Model Strategy

### Model Type

* Small instruction-tuned language model (~3B parameters)
* Optimized for fast inference and low operational cost

### Fine-Tuning Approach

* PEFT / QLoRA-based fine-tuning
* Domain-specific GTM datasets
* Instruction → structured-output supervision

### Dataset Characteristics

* Short, ambiguous GTM prompts
* Explicit target IR outputs
* Coverage across roles, motions, ICP selectors, and geographies

---

## 10. Functional Requirements

* Deterministic IR schema validation
* Versioned IR schema and model artifacts
* Configurable confidence thresholds for clarification
* Ability to retrain and iterate rapidly

---

## 11. Non-Functional Requirements

* Low-latency inference (sub-100ms local target)
* Fully offline-capable after initial setup
* Runs on commodity hardware (MacBook, CPU-based VPCs)
* Strong observability and logging guarantees

---

## 12. Success Metrics

* Reduction in downstream tool call failures
* Improvement in first-pass execution accuracy
* Reduction in unnecessary clarification questions
* Consistency of intent interpretation across semantically similar prompts

---

## 13. Risks and Mitigations

* Overfitting to narrow GTM patterns → diversify prompts and organizational contexts
* Ambiguous inference under uncertainty → confidence thresholds and clarification flags
* Schema drift → strict versioning and backward compatibility guarantees

---

## 14. Open Questions

* Should CCP emit a single best IR or multiple ranked hypotheses under ambiguity?
* What is the minimal viable IR schema for v1?
* How tightly should CCP be coupled to Phoenix MCP tool schemas?
* What confidence thresholds trigger clarification vs silent assumptions?

---

## 15. Out of Scope (v1)

* Multi-turn conversational grounding
* Autonomous execution
* Cross-domain (non-GTM) intent parsing
