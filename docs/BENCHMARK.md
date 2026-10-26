# Retail simulator and evaluation on the custom branch

This branch uses the same hash-pinned τ²-bench v1.0.1 Retail fixtures, serial tool adapter,
official UserSimulator and native ENV/ALL evaluators as the earlier experiment.
The benchmark commit is `fc0055dc4e0a316c3f83133267fbd6faaa770992`.
No reference action, hidden user instruction or grading criterion is sent to an MAS agent.
The serial bridge remains custom; the upstream Orchestrator is not used.

## Run and inspect

```bash
uv run python scripts/fetch_retail_data.py
uv run python -m retail_benchmark --tasks 69 --trials 1 --verbose
uv run python -m retail_benchmark --cohort development --dry-run
uv run python -m retail_benchmark --cohort baseline --dry-run
uv run python -m retail_benchmark --cohort baseline --trials 1 --seed 300 --output work/custom-baseline
uv run python -m retail_benchmark --inspect work/custom-baseline
```

The 17/57 training cohorts are defined in `data/manifests/retail_experiment_manifest.json`.
All 40 test tasks remain excluded. Dry runs validate IDs/initial states and make no
model calls. Start with a small explicit task list before paying for a whole cohort.
Capture the Git revision and model/budget settings when comparing runs.

The MAS uses `RETAIL_MODEL`; the official simulator defaults to
`gpt-4.1-2025-04-14` through the existing LiteLLM adapter. Override with
`--user-model` or `RETAIL_USER_MODEL`. The native NL judge retains its pinned model.
These roles have independent API usage. The simulator seed is recorded; it does not
guarantee deterministic MAS outputs. Trials use fresh DBs, adapters and checkpoints.

## Outputs and usage

Each trial writes `*.simulation.json`, `*.report.json`, and `*.final_db.json`.
The latest batch is listed in `summary.json`. Unique run IDs preserve earlier trials.

Reports preserve native results separately from supplementary checks. They also include
`architecture=custom-retail-five`, cohort, `mas_call_budget` and `mas_usage`.
Usage counts logical MAS requests and provider-reported tokens, including schema/JSON
fallbacks. `responses_with_usage` exposes missing metadata; a zero token subtotal with
missing usage is not evidence of free requests. SDK transport retries may add HTTP calls.

MAS totals exclude the official simulator and NL judge. No estimated dollar amounts
are invented. Use LangSmith/provider billing alongside these counts for actual spend.
The CLI prints MAS usage and the read-only inspector includes the same saved fields.

Native reward is unchanged. `validated_success` additionally requires completed grading
and all supplementary safety/state checks. Incomplete runs and grading errors remain
separate statuses; the CLI exits nonzero for them. Fully evaluated reward-0 trials are
valid measurements and do not by themselves cause a nonzero exit.

The native ENV result is saved before ALL grading so a later judge error does not erase
the DB result. Graders are not weakened for missing writes, human-transfer shortcuts or
NL failures. Reference reads/action diagnostics may differ from a valid alternative path;
native reward bases remain untouched. Some tasks have no NL assertions; others invoke
the paid judge. Communication grading uses upstream substring matching.

## Actions and confirmation

All five graph roles are LLM agents. There are no Payment/Verification LLM roles.
Decision's approved-action handler executes the official write and deterministically
compares the resulting DB with the prepared expected state. Verification is against the
customer's prepared action, not the benchmark oracle.

For multiple requests, later tool calls are queued without approval. Each one gets fresh
preflight and its own exact confirmation. Changed or ambiguous replies discard queued
calls. Verification failures halt the queue. Native replay records actual tools only;
disposable preflight calls are excluded. Successful writes are verified even when the
same graph invocation pauses to confirm the next operation.

Natural confirmation remains conservative and tied to the prepared action/fingerprint.
Common item-list acknowledgements are recognized; changed parameters and conditional
refund expectations are clarified. Common unprepared approval questions from specialists
get one corrective retry. This wording check is not a complete semantic classifier.
Actual write authorization remains independent of model judgments.

Returns disclose that a requested refund deadline cannot be guaranteed. The pinned
return policy specifies return-request status and email instructions, not a refund
deadline. Cancellation/payment-change timing is not reused as a return promise.

## Limits

The full policy is supplied only to Policy; other role prompts define their scopes.
The latest eight dialogue messages plus the maintained request purpose form model
conversation context. Tool observations are deduplicated without truncating their values.
The native simulator/evaluator still receive the complete recorded trajectory.

All reported automated checks use offline responses unless explicitly identified as
API-backed runs. No live pass rate or dollar saving has been established for this branch.
Ground-truth labels remain draft. Non-null initial-state/history overrides, durable
sessions and multiworker DB synchronization are not supported. The documented upstream
pending multi-item modification defect remains guarded rather than patched silently.

## Follow-up clarification and variant selection

Fallback-choice and missing-information questions are distinct from final authorization.
A denied action may lead Decision to ask about an alternative; such questions do not
authorize a write and must not be rejected merely for mentioning confirmation. The
canonical prepared-action approval remains mandatory before the executor runs.

Exchange summaries now show each original/replacement pair's observed options and
replacement price, alongside IDs and the price difference. These values come from the
current official DB, are copied rather than shared with it, and are bound into the action
fingerprint. This makes an unintended option visible to the customer.

The supplied task 0 run selected a white-backlit keyboard. The pinned reference expects
the no-backlight fallback. The full supplied option-selection exchange confirms that
the MAS offered both options and the user simulator explicitly chose white. That choice
deviated from the simulator's private fallback instruction. The MAS should honor the
observed customer choice, while native reward 0 remains the recorded benchmark outcome.
Record this observed simulator deviation alongside the raw score; do not silently discard
the run, relabel it successful or force the hidden expected variant into the agents.
No expected variant IDs are inserted into MAS prompts/routing; evaluator criteria are unchanged.

The conservative approval grammar additionally handles "as described" and past-tense
item-list acknowledgements. Conditions, changed IDs/payment details and conflicting
operation names remain unapproved.
