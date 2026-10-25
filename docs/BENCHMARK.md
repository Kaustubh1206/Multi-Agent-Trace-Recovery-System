# Automatic Retail conversations and evaluator reports

The separate `retail_benchmark` package connects the official τ²-bench v1.0.1 text
`UserSimulator` to the existing seven-agent LangGraph MAS and invokes native
`evaluate_simulation(ALL)`. The source is commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992`. DB, policy, tasks, split and simulator
guidelines are hash-pinned. The conversation loop is our serial bridge; the upstream
`Orchestrator` is not used. No business-agent nodes were added.

## Start with one task

Keep your existing OpenAI API configuration in `.env`:

```bash
git switch phase-1-retail-mas
git pull --ff-only
uv sync --locked
uv run python scripts/fetch_retail_data.py
uv run python -m retail_benchmark --tasks 69 --trials 1 --verbose
```

No terminal customer input is needed. The official simulator receives the private user
scenario, generates customer replies, and answers clarification/confirmation pauses. Its
STOP/TRANSFER/OUT-OF-SCOPE tokens end the dialogue. Only generated customer utterances
enter the MAS; original simulator instructions and reference criteria stay outside it.

The MAS uses `RETAIL_MODEL`. The simulator defaults to the pinned upstream model
`gpt-4.1-2025-04-14`, with temperature 0, one retry and a 60-second request timeout.
Override it with `--user-model` or `RETAIL_USER_MODEL`. Optional
`RETAIL_USER_REASONING_EFFORT` sets reasoning effort and omits simulator temperature.
The native NL judge retains upstream `gpt-4.1-2025-04-14` and temperature 0.
Model IDs in CLI/env are native OpenAI IDs; the simulator adapter adds the LiteLLM
provider prefix. Simulator, MAS and judge use the OpenAI key and incur API usage.

Run five training tasks, then repeat trials if useful:

```bash
uv run python -m retail_benchmark --tasks 69 0 1 47 50 --output work/pilot-five
uv run python -m retail_benchmark --tasks 69 --trials 3 --seed 300 --max-turns 20
```

Trials run serially with fresh DB, identity, approval, graph/checkpointer and thread ID.
The default limit is 30 simulator replies including the stop reply. Existing graph step
limits also apply. The simulator seed increases by one per trial; API sampling and MAS
outputs are not guaranteed deterministic. Reports preserve the model settings and seed.
The 40-task test split stays held out: `--tasks 5` is rejected before generation.
Non-null initial-state/history overrides remain unsupported and fail explicitly.

## Saved files

Files default to the Git-ignored `work/benchmark/` directory. Each trial has a unique stem
`task-<id>-trial-<n>-<run_id>`:

| File | Contents |
|---|---|
| `*.simulation.json` | Native `SimulationRun`: dialogue/tool messages, termination, raw reward |
| `*.report.json` | Provenance, models, hashes, native ENV/ALL results, checks and errors |
| `*.final_db.json` | Actual final simulated DB snapshot |
| `summary.json` | Most recent batch; older trial files remain available |

The initial DB is reproducible from the pinned asset/hash, loaded afresh per trial.
Tool calls run serially and their messages contain no simultaneous customer response.
Preflight copies are excluded. Successful identity lookups record the official scalar
output for replay; the MAS keeps its normal authenticated-identity adapter result.
Rejected reads, including guard refusals and lookup errors, are reported separately and
omitted from native replay. Strict replay checks successful tool outputs, and a separate
assertion compares the actual DB with that replayed DB.

`official_result` and `native_reward` preserve the upstream score.
`supplementary_checks` separately checks authentication, owned writes, exact consent,
duplicate writes, unrelated record changes, verification and required human handoff.
`validated_success` requires completed native grading, reward 1 and all those checks.
This is our stricter trial flag, not a published benchmark metric or a reviewed label.

The runner saves an ENV result first so it survives a later judge failure. Model/turn
failures are `incomplete_run`; grading failures have `evaluation_status=error`. Missing
NL judge assertions are flagged while preserving any raw native result. These cases
cannot be `validated_success`. The CLI exits 1 for incomplete/error evaluations.
A fully scored reward-0 trial remains a valid evaluator result and exits normally.
Independent trials continue after reported model/conversation failures.

## Natural-language confirmation

Confirmation accepts a conservative set of explicit phrases such as "yes, please proceed".
If a reply names an operation, it must match the prepared tool type.
Completeness acknowledgements such as
"that covers everything I want to exchange" and "that includes all the items I want
to return" are supported only for a matching item operation, alongside an explicit
approval. They do not select different items. Trailing "thank you" is supported.
An altered item/order/payment, a condition, a question or an unmatched phrase is not
approval and follows clarification/replanning. Approval remains bound to the exact
prepared arguments, financial consequences and DB snapshot; fingerprint, eligibility
and duplicate-write guards are unchanged.

The previous exact-word parser rejected the natural approvals supplied for live tasks
69 and 0 and discarded their proposals. Regression tests use those exact final replies
with scripted specialists and real tools. The task 0 rerun also exposed a repeated
confirmation loop with a different item-list acknowledgement. Its exact first approval
now has an automatic regression requiring one proposal, one committed exchange and
native DB agreement. The runner still stops before passing any STOP-bearing simulator
message to the MAS. Passing these checks does not establish new live model success.

The historical generic ValueError reports do not retain the exact throwing line. New
reports preserve safe protocol errors and a project-frame `run_error_location` without
paths, locals or raw provider bodies.

An additional task 1 failure asked the customer to confirm the exchange in prose without
preparing the official call. Common action-approval questions from Order/Decision are now
intercepted before reaching the customer. The same role gets one corrective retry: Order
hands off ready facts; Decision proposes the complete call. Repeated noncompliance fails
closed with a safe protocol error. Supervisor approval questions instead defer to the
specialists. Ordinary missing-fact questions continue to pause for the customer.

This is a conservative text check for common approval-request wording, not a semantic
classifier for every possible sentence. The actual security boundary remains preparation,
exact approval, authentication, ownership and fresh eligibility checks in the executor.
Existing informal customer approvals never become consent for an unprepared action.

## Clarification, capabilities and return totals

Clarification replies retain the current request's observed tool facts and purpose.
They still clear consent, discard the old proposal, reassess policy and prepare a new
exact action. A new top-level customer request clears those observations; successful
writes are followed by fresh reads for later requests. The internal continuation flag
is not accepted as a public customer input.

Every role now sees actual supported team write names, alongside its own scoped tools.
Prompts distinguish missing facts, missing consent and genuinely unsupported operations.
Decision proposes the tool before the application asks for approval; Order gathers facts
and Policy assesses eligibility rather than denying actions absent from its own scope.

Return preflight computes the selected items' purchased-price sum and the amount paid
for remaining items directly from the current order. The verified final response includes
both totals. No task IDs, expected evaluator amounts or reference tool arguments enter
these calculations. A three-day refund guarantee is not introduced.

The task 47 rerun already communicated the correct amounts, but the customer tied
approval to a three-day refund expectation. Those replies remain unapproved. The
clarification handoff now retains that unresolved question/condition, so specialists
can address it. Every prepared return summary and verified response explicitly says
a refund by a requested deadline cannot be guaranteed. A new unqualified approval is
still required before writing.

The pinned return policy promises a return-request status and email instructions;
it specifies no return-refund deadline. The 5–7 business-day language elsewhere applies
to cancellation/payment-change refunds and is not reused as a return promise. No
deadline or reference answer is added to the official tools or evaluator.

## Inspect an existing failed batch

Use the saved artifacts before paying for another model run:

```bash
uv run python -m retail_benchmark --inspect work/pilot-five
```

This imports no models, makes no API calls, and changes no trial artifacts. If
`summary.json` exists it selects that batch's run IDs, excluding stale reports from
earlier runs in the directory. Otherwise it summarizes all available report files.
It prints termination/errors, native DB/component results, failed supplementary
checks, failed NL/communication assertions and the last six dialogue messages.
Only message role/content is included; provider raw response metadata is omitted.

`validated_success` combines native reward and supplementary checks. Its being false
does not by itself mean a safety check failed. Incomplete runs must be diagnosed from
`termination_reason` and `run_error` before interpreting their native reward 0 as
a completed task-quality measurement.

## Interpretation and evidence

Many Retail tasks list NL_ASSERTION but provide no assertions. Native NL grading then
returns a vacuous success without a judge call. Task 47 has real assertions and uses the
paid judge. Communication grading is substring matching. Native action diagnostics may
miss reference reads even when an alternative valid path produces the expected DB.
ACTION is not in these pilot tasks' reward basis. Rejected lookup attempts also remain
outside native action diagnostics, in the rejection list and customer dialogue.

Task 50 can receive native reward 1 for unchanged DB without a human transfer. A separate
required-handoff check uses the official reference transfer action, actual transfer tool
and customer announcement to prevent marking that no-op as `validated_success`.
Reference actions, expected arguments and judge outcomes never become MAS inputs.

Offline checks run the official simulator class and native graders with intercepted model
responses. They cover train cases 0/1 (different exchange preferences), 69 (confirmation
and independent trials), 50 (actual/missing handoff), and 47 (communication, NL coverage
and judge failure). Task 47 deliberately uses incomplete read-only trajectories in these
grading checks. Existing task 84 graph/tool tests remain separate.

A subprocess CLI test exercises the real OpenAI SDK and LiteLLM against a local HTTP
model stub, including saved reports. All 84 tests passed in the last available workspace
run. The published copy is checked by the offline GitHub Actions workflow.

**No live benchmark trial or pass rate is established in this workspace.** Run a new
real API pilot on your machine and inspect the result before expanding it.
Labels remain draft. No causal attribution, fault injection, recovery or repair was added.

## LangSmith and module layout

Existing LangSmith tracing settings trace MAS calls in automatic trials. The thread ID
matches the report's `run_id`; metadata includes `execution_mode=simulated_user`, task,
trial and seed. Simulator instructions and grader references stay outside that graph.
Reports remain local; LangSmith dataset/experiment uploads are not implemented.

| Module | Responsibility |
|---|---|
| `simulator.py` | Official simulator subclass selecting pinned guidelines |
| `recording.py` | Successful tool results and consent evidence for grading |
| `runner.py` | Bounded dialogue, interrupt resumes, fresh trials and report construction |
| `evaluation.py` | Native grading on pinned fixtures and supplementary assertions |
| `__main__.py` | Task/trial CLI and JSON artifact saving |
