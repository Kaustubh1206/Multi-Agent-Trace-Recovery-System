# Phase 1 implementation notes

Scope: functional retail MAS only. No tracing framework, failure attribution, injector,
anomaly detector, recovery, replay, or repair memory was implemented. The attached proposal
is reference context; its later-phase architecture did not expand this implementation's scope.

## Audit and provenance

- Target repository: `Kaustubh1206/Multi-Agent-Trace-Recovery-System`, initial commit
  `285ce7c` (`Setup and Enviornment`). The separately attached `MATRACE` checkout was untouched.
- Initial target files: `main.py`, `pyproject.toml`, `uv.lock`, `requirements.txt`, empty
  `README.md`, `.python-version`. No business implementation existed to overwrite.
- Style reviewed: `states.py`, `nodes.py`, `edges.py`, `objects.py`, `prompts.py`, `models.py`
  and `src/creating_analysts.py` in the user's research-assistant repository. The implementation
  retains small functions, typed state and structured outputs, with current graph APIs.
- The uploaded data guide is preserved under its original name. The proposal's text was
  extracted from the supplied DOCX to `MA_TRACE_Final_Retail_MAS_Proposal.txt`; figures are
  not reproduced in the text extraction.
- AgentTether's anonymous README was inaccessible through the web tool. No source code,
  repair architecture, or compatibility claims were inferred from it.
- τ² `v1.0.1` annotated tag resolves to full commit
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`. The manifest contains source URLs and SHA-256
  hashes for tasks, splits, DB, policy, and the three retail source interfaces.
- Reviewed official `RetailTools`, `RetailDB`, environment construction, task split loader,
  initial-state model and ENV evaluator. Tests compare installed retail source bytes with
  the pinned manifest. No custom replacement retail database or rewritten tool suite exists.
- `pyproject.toml` and `uv.lock` are the dependency authority. The old broad provider/ML/UI
  list was replaced with Phase 1 requirements; `requirements.txt` delegates to the project.
  τ² transitively installs LiteLLM and some ML utilities, but no model weights are downloaded.

## Runtime boundaries

One graph, adapter and memory checkpointer belong to **one conversation**. Create all three
again for a new task/trial. The CLI does this on startup; multiple requests from the same
authenticated customer may continue in that session. Do not reuse a compiled graph for
multiple customers/threads. Persistence across process restarts is not implemented.

Agents receive an allowlisted context: actual dialogue, scoped read results, current purpose
and validation errors. They receive no task record, task ID, expected operations or oracle.
The CLI loads a train task only to validate its ID and initial fixture. The task's original
user instructions and evaluation criteria stay in the original pinned assets and test harness.
All six pilots have null initial state. Non-null overrides/history explicitly fail closed.

Identity, Order and Decision use structured agent steps; Policy and Payment have independent
structured finding schemas and prompts. They may share a configured model, while keeping
independent responsibilities and controlled handoffs. A model is never the authority for a
Boolean authorization or confirmation rule.

Protected reads check the adapter's authenticated identity, never an agent-supplied flag.
Order ownership is checked before the official result is returned. One `RLock` serializes
retail calls. Tool steps are separate from customer-response steps.

A proposal runs signature/type/policy validation and official preconditions on a disposable
copy. This preflight is not an executed customer action. The exact canonical arguments and
financial consequences are shown to the customer. An exact affirmative reply binds consent
to an action fingerprint including relevant user/order/product state. No/ambiguous/changed
replies write nothing. Changed choices restart planning and require a new summary/approval.

The executor independently rechecks identity, ownership, eligibility, parameters, freshness,
exact approval and duplicate commit. It executes the unchanged official API on another copy
and commits that copy only on success. Failed execution consumes approval. This is an
in-memory transaction boundary over the simulated DB, not a durable external-payment ledger.

## Verified source nuances and limits

- Task `69` begins as a laptop return inquiry, with cancellation as fallback. The real laptop
  order `#W2417020` is pending. The tested dialogue asks for a valid cancellation reason and
  obtains separate exact consent before cancellation.
- Tasks `0` and `1` have different keyboard fallback preferences. The offline test doubles
  choose different item lists; these are not universal answers embedded in runtime code.
- Task `47` includes deliberately invalid early order references. The reference evaluator
  logs their failures and continues. Our adapter returns a generic unavailable-order error;
  the graph asks for the customer's corrections. The test also checks the remaining-items
  calculation. This is not an annotation inconsistency.
- In task `84`, the two tablets are in different delivered orders. Both are looked up. The
  changed choice and refund method invalidate the first proposal and trigger fresh consent.
- Task `50` has no DB mutation in its reference. ENV reward alone cannot establish transfer
  correctness. A supplementary test verifies the real transfer API call and exact message.
- The official pending-item modification implementation uses the last `variant` variable
  when assigning every changed item's price/options. Multi-item pending modifications with
  different replacement price/options are therefore blocked and should be handed to a human.
  Overlapping pending replacement IDs are also blocked because upstream resolves them
  sequentially. Upstream was not patched. Multi-item **exchange** remains runnable.
- Policy says further cancellation/modification is unavailable after a pending item change.
  Some official address/payment helpers accept any status containing `pending`; our guard
  follows the stricter policy and requires exactly `pending`.
- Python is currently constrained to 3.12: the pinned package imports `audioop` transitively,
  which is removed in Python 3.13. The observed deprecation warning is upstream.

## Evaluation status

Offline tests invoke the actual LangGraph graph with scripted model outputs, real fresh DBs
and official retail tools. Evaluator-only helpers capture actual committed operations for
the native `EnvironmentEvaluator` with strict output checking. They also compare the actual
entire DB against a separately constructed reference DB. This ENV-only integration is not
the official user simulator or complete benchmark scoring path.

Remaining: wire a half-duplex adapter to the official user simulator/orchestrator, run live
specialist LLM dialogues, integrate full communication/NL assertion evaluation and review the
draft labels independently. No API credentials were configured in this environment, so no
paid LLM calls or full simulated-user trials were run. No test-split task was executed.
Multi-request tasks are handled sequentially through customer turns; autonomous completion
of an entire queue of subgoals has not been demonstrated with a live model.

The original data guide documents future fault families. Those families and causal labels
remain future work; none were generated in this phase.
