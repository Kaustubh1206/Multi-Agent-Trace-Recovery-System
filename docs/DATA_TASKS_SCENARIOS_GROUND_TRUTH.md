# MA-TRACE — Retail Data, Tasks, Scenarios, Ground Truth & Future Fault Injection

**Project:** MA-TRACE (Multi-Agent Tracking, Recovery, Attribution, and Consistent Evaluation)  
**Current scope:** Phase 1 — build a functioning, policy-compliant **retail multi-agent system (MAS)** with LangGraph.  
**Benchmark decision:** Pin **Sierra Research τ²-bench `v1.0.1`**, English/text Retail task assets, rather than mixing benchmark releases.  
**Status:** This is an *implementation specification*, not a claim that any scenarios, oracles, faults, or repairs have already been implemented.

---

## 1. What data are we using, and why?

τ²-bench provides a customer-service environment. In Retail, an agent interacts with a customer (normally a simulated user), follows a written policy, invokes retail database tools, and is evaluated on the outcome. This is **not a supervised ML table of labeled agent failures**. The task definitions, policy, initial retail database, simulation behavior, and evaluation logic collectively form a reproducible task environment.

We will write **our own LangGraph MAS**, and use τ²-bench Retail as the **task and environment source**, not claim that τ²-bench or AgentTether already provides our multi-agent architecture.

| Asset | Source and meaning | Phase 1 use |
|---|---|---|
| `tasks.json` | Benchmark customer tasks, simulated-user instructions, evaluation criteria | Load and select authentic retail tasks |
| `split_tasks.json` | IDs divided into train/test/base | Keep development and held-out testing separate |
| `db.json` | Retail users, orders, products, payment methods | Source of initial database state; reset between runs |
| `policy.md` | Retail operational rules and prohibitions | Shared policy reference; hard guards for writes |
| `src/tau2/domains/retail/tools.py` | Reference retail tool behavior | Reuse via official environment/adapter instead of reimplementing inaccurately |
| `src/tau2/domains/retail/environment.py` | Environment/task construction | Integration point for official tasks and tools |
| User simulator & native evaluator | Multi-turn customer behavior; checks against benchmark expectations | Use for benchmark-aligned runs as integration matures |
| AgentTether anonymous source | Recovery research implementation, **not** the authoritative retail dataset | Reference for subsequent tracing/diagnosis/recovery phase |

**Exact upstream paths (pinned to `v1.0.1`):**

- [Retail tasks](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/data/tau2/domains/retail/tasks.json)
- [Task splits](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/data/tau2/domains/retail/split_tasks.json)
- [Retail policy](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/data/tau2/domains/retail/policy.md)
- [Retail database](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/data/tau2/domains/retail/db.json)
- [Retail tools](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/src/tau2/domains/retail/tools.py)
- [Retail environment](https://github.com/sierra-research/tau2-bench/blob/v1.0.1/src/tau2/domains/retail/environment.py)
- [AgentTether paper](https://arxiv.org/html/2607.06273v1)
- [AgentTether anonymous code lead](https://anonymous.4open.science/r/AgentTether-9416/README.md)

**Version warning:** The upstream `main` branch has evolved; its newer benchmark configurations and language variants are not interchangeable with `v1.0.1`. Do not silently upgrade or combine its `tasks.json`, `db.json`, policy, tools, and evaluator with another release. Record the *resolved commit SHA* in the manifest before experiments. Newer τ²-bench benchmark results are not automatically comparable with earlier ones.

**AgentTether distinction:** The *paper* reports 114 Retail benchmark tasks and 26 *initially failed tasks* in its specific experiment. The 26 are **not** 26 supplied injected-fault labels. The anonymous code mirror could not be independently inspected in the source review for this guide; after cloning it locally, identify the real scripts and run artifacts before making file-level or experiment-compatibility claims. Its paper also includes Airline and Banking; our current scope is Retail only.

---

## 2. Dataset size and split selection

In the pinned official `v1.0.1` Retail split file:

| Split | Task count | Purpose |
|---|---:|---|
| `train` | **74** | Build and debug the MAS; select pilot examples; prototype future faults |
| `test` | **40** | Hold out for evaluation; do not tune prompts/guards against these cases |
| `base` | **114** | Union of all official Retail tasks; not an independent third split |

The IDs in this benchmark are **strings** (e.g., `"0"`, `"47"`) rather than our own invented `TASK-001` IDs. Preserve them as `source_task_id` and additionally assign a local `scenario_id` only for our own tests. Do not assume IDs are meaningful category labels.

**Suggested rollout:**

1. **Smoke tests:** 3–5 verified `train` tasks, starting with one read-only task and one explicit-confirmation mutation task.
2. **Functional pilot:** 10–15 diverse `train` tasks, including multi-turn corrections and multi-item requests.
3. **Integration:** run the MAS against the real benchmark user simulator and evaluator, one task at a time.
4. **Held-out test:** once prompts/architecture are frozen, evaluate selected or all 40 `test` tasks. Save trial counts, evaluator model, database snapshot and outputs.

Counts above describe the official split; pilot counts are **our own targets**, not published results.

---

## 3. What does one official task record contain?

The `v1.0.1` file includes fields shaped like the following. Field names below are **source-supported**; the values in this illustration are abbreviated.

```json
{
  "id": "0",
  "description": {
    "purpose": null,
    "relevant_policies": null,
    "notes": null
  },
  "user_scenario": {
    "persona": null,
    "instructions": {
      "task_instructions": "...",
      "domain": "retail",
      "reason_for_call": "...",
      "known_info": "...",
      "unknown_info": "..."
    }
  },
  "initial_state": null,
  "evaluation_criteria": {
    "actions": [
      {"action_id": "0_0", "name": "find_user_id_by_name_zip", "arguments": {"first_name": "...", "last_name": "...", "zip": "..."}, "info": null}
    ],
    "communicate_info": [],
    "nl_assertions": null,
    "reward_basis": ["DB", "NL_ASSERTION"]
  }
}
```

### Field meanings and information-access boundaries

| Field | Meaning | Who may see it? |
|---|---|---|
| `id` | Official task identifier | Runner/logger/evaluator; may be passed as run metadata |
| `user_scenario.instructions.reason_for_call` | The simulated customer's private objective | **User simulator only**; never inject verbatim into MAS system prompts |
| `known_info`, `unknown_info`, `task_instructions` | Simulator's persona/knowledge/behavior | **User simulator only** |
| `initial_state` | Optional per-task environment override; `null` for sampled cases | Runner/environment, never a shortcut to hidden answers |
| `evaluation_criteria.actions` | Reference actions/arguments to evaluate | **Evaluator only**; never exposed as the MAS plan |
| `communicate_info` | Required information to communicate when applicable | **Evaluator only** |
| `nl_assertions` | Natural-language expectations when present | **Evaluator only** |
| `reward_basis` | Type(s) of benchmark reward | Runner/evaluator, not agent decision hints |

**Important:** A task's `evaluation_criteria.actions` represents benchmark expectations, **not a guaranteed unique valid sequence of internal LangGraph agent steps**. Our agents may delegate differently while producing an equivalent, policy-compliant result. Preserve official scoring separately from our supplemental assertions.

**No answer leakage:** Do *not* load `evaluation_criteria` or hidden simulator instructions into the Supervisor, specialist agents, RAG retrieval context, or tool-return messages. For an interactive demo, a human customer can supply the facts step by step. For official evaluation, use the benchmark's user simulator.

---

## 4. Retail policy: constraints the MAS must enforce

The `v1.0.1` policy establishes these requirements; the implementation must read the full authoritative `policy.md` rather than substituting this summary:

| Rule | Enforcement intention |
|---|---|
| Authenticate before providing account/order/profile information | Locate user via email **or** name + ZIP, even if user supplied a user ID |
| Serve one customer per conversation | Reject other-user operations |
| Mutating operations require explicit confirmation | State action details and obtain a clear **yes** before any write |
| One tool call at a time | Do not parallelize retail-domain tool calls; do not tool-call and reply to user in same benchmark step |
| Pending vs delivered eligibility | Cancellation/modification generally concern pending orders; return/exchange concern delivered orders |
| Pending cancellation reason | Only accepted reasons: `no longer needed` or `ordered by mistake` |
| Variant modification/exchange | New available variant of the **same product**, with **different options**; product ID and item ID differ |
| Item modification/exchange calls | Collect all items and perform the permitted once-per-order operation rather than partial duplicate writes |
| Payment methods | Credit card, PayPal, gift card; enforce payment and balance rules for the relevant operation |
| Unsupported actions | Deny or transfer under the policy's transfer conditions, not invent permissions |
| Transfer protocol | Use the transfer tool, then the specified transfer message |

Source nuance: Some order mutations yield additional statuses such as `return requested`, `exchange requested`, or `pending (items modifed)`; do not hard-code a simplified enum that rejects a legitimate post-action status. Read the tool data models and actual transition logic.

### Tool ownership plan (our architecture, not a τ² benchmark requirement)

- **Supervisor Agent:** captures intent, missing details, tracks workflow progress; routes rather than directly changing retail DB.
- **Customer/Identity Agent:** customer lookup/authentication using authorized tools.
- **Order Agent:** retrieves order details and item/product options; builds factual context.
- **Policy Agent:** interprets the authoritative retail rules for the requested action; returns rationale and constraints.
- **Payment Agent:** validates/selects eligible payment methods and computes price differences; **never** directly performs refund writes on its own.
- **Decision/Response Agent:** proposes a user-facing action summary, obtains confirmation when required, and explains outcome.
- **Action Executor (non-agent tool boundary):** performs one sanctioned mutation **only after** all required checks succeed.

Not every request needs every agent. Read-only requests skip Payment/Executor; payment-related checks occur only if relevant. Agent outputs should be structured, small, and explicit; hard policy gates are deterministic Python, not an LLM's confident statement.

---

## 5. Task taxonomy: scenarios we want to support

The categories below are our **derived taxonomy**, not official task-classification labels. Populate category memberships by reading and validating task text, not by assigning arbitrary task IDs.

| Scenario family | Customer objective | Typical agents | Success condition |
|---|---|---|---|
| S01 — Profile or order inquiry | Obtain correct account/order/product details | Identity → Order → Response | Correct authorized information, **no DB mutation** |
| S02 — Cancel pending order | Cancel an eligible pending order | Identity → Order → Policy → Confirmation → Executor | Correct order canceled once, correct reason, no unrelated change |
| S03 — Modify pending address | Update shipping address | Identity → Order → Policy → Confirmation → Executor | Authorized address changed for correct eligible order |
| S04 — Modify pending payment | Change payment method | Identity → Order → Policy → Payment → Confirmation → Executor | Allowed payment change, balance rules satisfied |
| S05 — Modify pending variants | Change one or more item options | Identity → Order → Policy → Payment → Confirmation → Executor | Valid same-product variants, one allowed write, expected total |
| S06 — Return delivered items | Start return/refund for eligible items | Identity → Order → Policy → Payment → Confirmation → Executor | Correct items returned, allowed refund method, no duplication |
| S07 — Exchange delivered items | Swap eligible item variants | Identity → Order → Policy → Payment → Confirmation → Executor | Correct new variants and price difference, one allowed write |
| S08 — Multi-request conversation | Combine eligible actions, possibly across orders | Supervisor + relevant specialists | All feasible authorized subgoals addressed, dependencies respected |
| S09 — Ambiguous or corrected details | Wrong order ID, unclear item, or revised choice | Supervisor → Identity/Order → clarification | Corrected fact used **after** customer resolves ambiguity |
| S10 — Unsupported/handoff request | Customer requests something policy cannot perform | Identity/Policy → Decision/Transfer | Correct refusal/transfer; no unsupported DB write |

For each family, build **at least one clean runnable example** before adding synthetic faults.

---

## 6. Examples of real, source-backed training tasks

These are task **ideas from published retail benchmark entries**, not a preexisting MA-TRACE labeled failure set. Confirm exact contents when importing from the pinned `v1.0.1` file. All example IDs below appear in that release's `train` split.

| Official task ID | What it illustrates | Important difficulty for our MAS |
|---|---|---|
| `"0"` | Customer asks to exchange a keyboard and thermostat in order `#W2378156`; preferences constrain replacement variants | Multiple changes, product vs item IDs, confirmation, one exchange call |
| `"1"` | Same general shopping context but a **different fallback preference** from task `0` | Do not generalize one answer across superficially similar tasks |
| `"47"` | Customer initially gives an incorrect order ID, then corrects it if asked | Clarification, evidence refresh, no premature wrong-order action |
| `"50"` | Customer wants to undo an existing cancellation and insists on exceptional delivery | Policy boundary and transfer-to-human rather than invented undo tool |
| `"69"` | A pending-order cancellation case involving `#W2417020` | Authentication, cancellation reason, explicit approval, DB outcome |
| `"84"` | Customer initially asks to return one tablet, then changes preference on confirmation | User's *latest confirmed intention* must supersede early assumptions |

**Important:** A task can contain conditional branches or customer changes of mind. Do not turn its first user utterance into a static complete task; preserve the multi-turn interaction.

### Fully illustrative demo flow — cancellation

The following is our **proposed interaction flow**, not a transcription of task `69`'s exact reference trajectory:

1. Customer requests cancellation; Supervisor identifies intent and any missing information.
2. Identity Agent authenticates by the benchmark-approved lookup process.
3. Order Agent fetches the actual order and confirms its owner/status.
4. Policy Agent establishes whether cancellation is permitted and what reason is allowed.
5. Decision Agent summarizes **which order**, **what reason**, and consequences; asks for a clear yes/no.
6. If **yes**, Action Executor calls the official cancellation tool **once**; if no or ambiguous, it does not mutate.
7. Read resulting tool output/current state, then give a factual final reply.
8. Independent test code compares the resulting database state with the task oracle.

### Illustrative future fault case — wrong-order handoff

- Clean: Order Agent obtains order `A`; downstream work targets order `A`.
- Injected: handoff replaces `order_id=A` with another valid-looking `order_id=B`.
- Risk: Decision Agent proposes an action on `B` instead of `A`.
- Future MA-TRACE goal: localize the corrupted handoff, prevent a write to `B`, repair downstream context, and independently prove `A` changed appropriately and `B` did not.
- This is a **synthetic scenario we intend to construct**, not a published τ²-bench failure label.

---

## 7. Ground truth: what we create NOW versus LATER

### 7.1 Build **task ground truth NOW** (Phase 1)

Ground truth must be separate from the MAS runtime and invisible to all business agents.

For **each selected official task**, create a test/evaluation record containing:

| Ground-truth field | Derivation/meaning | Confidence |
|---|---|---|
| `source_task_id`, `benchmark_ref`, `split` | IDs and pinned benchmark version | Directly verified |
| `initial_database_hash` and fixture reference | Snapshot of benchmark-provided DB before execution | Computed by harness |
| `policy_version` | Pinned policy document/hash | Computed by harness |
| `official_evaluation_criteria` | Original official criteria kept verbatim | Source-provided, **not automatically flawless** |
| `expected_effect` | Read-only, valid change, denial, or handoff | Derived and manually checked |
| `required_preconditions` | Auth, ownership, status, eligibility, explicit consent, etc. | Policy-derived |
| `allowed_actions`/`forbidden_actions` | Extra safety assertions appropriate to task | Our checked oracle annotations |
| `final_state_assertions` | Exact allowed changes and forbidden effects | Deterministic assertions where expressible |
| `communication_assertions` | Necessary response/transfer facts | Native benchmark judge + optional checked rules |
| `label_review_status` | `draft` → `validated`, include reviewer/date | Our annotation provenance |

The official benchmark's grader remains the primary point of comparison. Supplement it with strict process checks (unauthorized write, duplicate mutation) that it may not fully cover. Do **not** assume that its action list is perfect: benchmark task corrections have been published, so visibly mark disputed annotations and use verified fixtures.

### 7.2 Do **not** create these as if known in Phase 1

- `root_cause_agent_id` / `root_cause_step_id` for runs **not yet executed**.
- A claimed causal ordering based only on time adjacency.
- A fixed recoverability label for every possible fault.
- A fabricated list of AgentTether's 26 failed task IDs.
- Traces, impact graphs, success rates, or repair metrics that do not exist yet.

### 7.3 Create these once controlled failures exist (later phases)

1. **Injection label:** exact tampered boundary, agent, step, field, injected value, seed, and whether injected corruption actually took effect.
2. **Observed failure label:** was the task or state actually wrong? Was this a harmless/no-effect perturbation?
3. **Adjudicated decisive-cause label:** controlled comparison or replay shows which event(s) caused the failure; can be ambiguous or multi-step.
4. **Recovery oracle:** what a correct recovery is allowed to redo; which effects cannot be duplicated; final task/state assertions.
5. **Outcome label:** valid recovery, invalid accepted recovery, not recoverable, safe escalation, or unnecessary intervention.

**Golden rule:** Injection location is known experimentally, but it is **not necessarily the decisive root-cause location**. A robust research design separates those labels.

### 7.4 Suggested test-only oracle record (illustrative schema)

```json
{
  "scenario_id": "retail-clean-0001",
  "source_task_id": "69",
  "benchmark": {"repo": "sierra-research/tau2-bench", "tag": "v1.0.1", "commit": "TO_RESOLVE"},
  "split": "train",
  "scenario_kind": "clean",
  "initial_database_hash": "TO_CALCULATE",
  "policy_hash": "TO_CALCULATE",
  "expected_effect": "cancel_correct_pending_order_only_after_confirmation",
  "oracle": {
    "must_authenticate_first": true,
    "must_confirm_before_write": true,
    "forbid_other_user_mutation": true,
    "forbid_duplicate_write": true,
    "final_state_assertions": ["validated_against_real_task_and_db"],
    "official_grader": "tau2_retail_v1.0.1"
  },
  "review_status": "draft_unvalidated"
}
```

This JSON is a **template**, not ready-to-run verified benchmark ground truth. Do not pass it to the agents. Replace placeholders and validate its task-specific fields before marking it `validated`.

---

## 8. Data integration and runtime flow (Phase 1)

```text
Pinned τ²-bench Retail assets
   ├── tasks + split  ──────────────> Task runner / scenario selector
   ├── policy ──────────────────────> Policy lookup / hard validation
   ├── users/orders/products DB ────> Official retail environment + tools
   └── simulated user instructions ─> User simulator ONLY
                                          |
                                          v
                               Customer utterance
                                          |
                                          v
                              Our LangGraph Retail MAS
              Supervisor → Identity → Order / Policy / Payment
                              → Confirm → Execute → Respond
                                          |
                               One allowed tool call at a time
                                          |
                                          v
                              Simulated database state
                                          |
                                          v
            Separate benchmark evaluator + test-only oracle assertions
                                          |
                            Task outcome / failure record
```

### Required reproducibility controls

- Reset from a fresh official database copy **for each task/trial**; never reuse a mutated global fixture.
- Pin benchmark tag **and full resolved Git commit**, domain, split, policy, tool behavior, simulator/evaluator versions, LLM model/config, seed (where supported), and code revision.
- Maintain clear ownership: the runtime state stores *observed* values; the oracle stores *expected* values.
- Do not reuse source task's `reason_for_call` as if it were a customer-visible message.
- Limit tool concurrency to comply with Retail policy.
- Store IDs as strings: order IDs like `#W...` and item IDs are distinct from product IDs.
- Simulated tools only. No real order, payment, refund, or human-contact side effects.
- Keep initial/development results separate from held-out evaluation.

### Proposed local files (not a claim these exist upstream)

```text
data/
  README.md                          # points to this document and pinned assets
  manifests/
    retail_benchmark_manifest.json   # exact versions, source, selected IDs
    pilot_train_ids.json             # selected train IDs only
  fixtures/
    README.md                        # where frozen DB copies are stored
  ground_truth/
    clean_tasks.jsonl                # our validated task-level assertions, test-only
    README.md                        # review/label definitions, anti-leakage rules
  faults/                             # FUTURE PHASE; do not implement now
    fault_scenarios.jsonl
```

No need to copy the full source benchmark's database into Git by default. Prefer a reproducible retrieval command, pinned checkout, and hash. Check upstream license and citation obligations before redistributing benchmark data.

---

## 9. Future fault-injection design — **DOCUMENT ONLY in Phase 1**

Below are **our proposed experimental fault families**, not claims about specific faults provided by AgentTether. These will be implemented only *after* a stable, baseline retail MAS and task oracle exist.

| Fault ID | Proposed injection boundary | Example corruption | Detection/oracle expectation | Possible downstream impact |
|---|---|---|---|---|
| F01 Wrong order reference | Order Agent → Supervisor/Decision handoff | Swap correct `order_id` for wrong valid order ID | Task-target and ownership checks; wrong order unchanged | Policy/payment/action targeting wrong order |
| F02 Wrong user identity | Identity result → Order Agent | Replace `user_id` with unrelated customer | Must reject; cannot read/mutate another customer's records | Privacy or unauthorized updates |
| F03 Stale order status | Tool result → Order Agent | Show `pending` for already processed/delivered order | Pre-write actual-state version/eligibility check | Invalid cancellation/modification |
| F04 Stale policy information | Policy lookup/result | Provide outdated eligibility rule | Compare pinned policy, deterministic guard | Premature/forbidden action |
| F05 Invalid product variant | Order/Product → Decision | Substitute variant from another product or unavailable option | Same-product/available/different-option assertion | Wrong item exchanged/modified |
| F06 Invalid payment selection | Payment → Decision | Unapproved refund destination or insufficient gift-card balance | Payment policy + owner/balance guard | Invalid refund/payment accounting |
| F07 Missing consent | Confirmation → Executor | Mark `confirmed=true` without customer yes | Conversation confirmation evidence check | Unauthorized DB write |
| F08 Duplicate write | Executor retry boundary | Repeat cancellation/return/exchange call | Operation ledger and DB diff; at most one logical commit | Double refund / inconsistent state |
| F09 Faulty handoff | Specialist → Supervisor | Drop or overwrite requested item IDs | Contract validation + task oracle | Omitted/incorrect part of multi-item request |
| F10 Tool error/time-out | Official tool adapter | Controlled exception or missing response | Log real exception; safe no-write retry/escalation | Repeated call, incomplete task |
| F11 Conflicting multi-turn preference | User update → Decision | Preserve earlier choice instead of latest confirmed choice | Final choice and consent against user messages | Wrong return/exchange target |
| F12 Missing required step | Workflow routing | Bypass identity/policy check | Deterministic preconditions reject | Process violation despite plausible answer |

### Future fault record format

```json
{
  "fault_scenario_id": "fault-F01-0001",
  "source_task_id": "69",
  "base_clean_scenario_id": "retail-clean-0001",
  "fault_type": "wrong_order_reference",
  "injection_spec": {
    "target_agent": "order_agent",
    "boundary": "handoff_to_decision",
    "field": "order_id",
    "original_value": "FROM_FROZEN_RUN",
    "injected_value": "VALIDATED_DISTINCT_ORDER_ID",
    "trigger": "first_matching_handoff",
    "seed": 17
  },
  "injection_applied": null,
  "observed_failure": null,
  "adjudicated_root_cause": null,
  "recoverability": "not_yet_determined",
  "review_status": "planned"
}
```

**Do not generate thousands of fake failure rows now.** Start with validated pilot tasks, then perhaps 2–3 independently verified instances per selected fault type. Only scale after demonstrating that fault injection actually changes an outcome or exercises a meaningful safety boundary.

### Future experiment protocol

1. Reset the clean official DB and run a **no-fault control** on the selected task.
2. Replay under an identical frozen initial DB/policy, with one registered fault injected at a known boundary.
3. Record the injection **attempt**, whether it **actually fired**, and the observed output/state difference.
4. Compare against task-level oracle. Mark as failed, safely contained, no-effect, or inconclusive; do not force every injection into the failure bucket.
5. Independently label decisive causal event(s), using controlled counterfactual replay where feasible.
6. Only after those labels are validated should MA-TRACE diagnosis/recovery methods be compared.
7. Repeat with controlled seeds/models; record success/failure denominators explicitly.

### Metrics enabled by this documentation (implemented later)

Attribution (agent+step), recovery success, containment/false holds, state safety, efficiency/cost, tool correctness, and end-to-end task completion. In Phase 1, measure **basic task completion, policy compliance, incorrect writes, and test coverage** only; tracing-based attribution and recovery statistics do not exist yet.

---

## 10. What is intentionally outside Phase 1?

No AgentTether CTG/HGT model, graph anomaly detector, diagnostic analyst, trace dependency DAG, replay planner, repair memory, fault execution middleware, DeepEval scoring, mobility domain, or real payment integration. We **will** use typed state, stable task/run IDs, the official policy and tools, separate executor boundary, and test-only oracles so adding tracing later will not require rewriting the retail MAS.

A simple checkpointer is reasonable to support multi-turn confirmation now, but **checkpointing is not tracing or rollback of external effects**. Do not promise recovery semantics in Phase 1.

---

## 11. Phase 1 acceptance checklist

- [ ] `v1.0.1` task, split, DB, policy, and tool versions pinned together; commit and hashes recorded.
- [ ] Select a small **train-only** pilot list; task IDs source-verified.
- [ ] Retail MAS has clearly separate Supervisor, Identity, Order, Policy, Payment, Decision roles and a guarded executor.
- [ ] Authentication required before protected reads or writes.
- [ ] Mutating action details displayed and explicit customer confirmation recorded **before** any write.
- [ ] One-at-a-time retail tool calls; no wrong-customer access.
- [ ] Read-only paths terminate without mutating DB.
- [ ] Policy-ineligible paths deny/transfer appropriately.
- [ ] Multiple variants handled in one allowed mutation when necessary.
- [ ] Fresh per-task database fixtures and deterministic integration tests for key guards.
- [ ] Native benchmark evaluator and our **separate** clean-task oracles can assess pilot runs.
- [ ] Test-only ground truth cannot be imported by MAS modules or exposed to agent prompts.
- [ ] Phase 1 README explains run commands and limitations.
- [ ] No claims of diagnosed, recovered, or injected failures before those features exist.

## Sources and verification notes

- Sierra Research τ²-bench `v1.0.1` tagged source: links above, specifically `split_tasks.json`, first entries of `tasks.json`, `policy.md`, `tools.py`, and `environment.py` inspected for this guide.
- Zhao et al. (2026), *AgentTether*, arXiv:2607.06273v1, Section IV (Retail task counts and failure subset), used only for research framing.
- [τ²-bench-Verified](https://github.com/amazon-agi/tau2-bench-verified): useful caution that published benchmark task/ground-truth annotations may require auditing; not automatically interchangeable with our pinned release.
- The anonymous AgentTether source mirror was **not successfully inspected file by file**. We should inspect it locally when available and amend this documentation if its actual benchmark setup differs.
