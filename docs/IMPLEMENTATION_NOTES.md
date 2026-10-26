# Custom five-agent implementation

The original seven-agent experiment is preserved on `phase-1-retail-mas`.
This branch keeps the same package layout, dependencies, pinned data and native evaluator.

Supervisor, Identity, Order/Product, Policy and Decision remain independently prompted
LLM roles. The normal path is sequential. Deterministic routing handles authentication
prerequisites and handoffs; tool calls remain serial. Decision pauses for confirmation,
then invokes the unchanged guarded adapter and deterministic DB verification directly.
There are no Action/Payment or Verification LLM agents.

Cost controls include a full-policy prompt only for Policy, recent dialogue plus the
maintained purpose, deduplicated observed reads, bounded model requests/output, and
per-role provider usage counters. These are architectural reductions; live quality and
dollar savings remain unmeasured.

Later actions for multi-order requests are queued, not approved. Each is prepared
against current DB state and separately confirmed. Changed replies clear the queue;
failed verification stops it. Successful writes clear stale tool facts. The native
runner records verification even when it immediately pauses for a subsequent action.

The experiment manifest fixes 17 development and 57 training-baseline tasks; the 40
official test tasks remain held out. Baseline feedback used to change the system is
development evidence. Final test execution needs a separately frozen protocol.

No recovery, fault injection, causal labels or replay framework was added.
