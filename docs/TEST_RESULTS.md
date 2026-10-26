# Custom five-agent verification

This branch replaces the seven LLM roles with five: Supervisor, Identity, Order/Product,
Policy and Decision. Native τ²-bench source remains pinned to
`fc0055dc4e0a316c3f83133267fbd6faaa770992`. Python is 3.12 and dependencies are
unchanged in `pyproject.toml`/`uv.lock`.

The first implementation checkpoint `4313dc409ff141b8b99d37e6cb330f5bfedaf83c` passed
[GitHub Actions run 40290440326](https://github.com/Kaustubh1206/Multi-Agent-Trace-Recovery-System/actions/runs/40290440326):
**125 passed, 1 warning in 86.64s**; Ruff passed.

The queue/budget/cohort checkpoint `8a2eb0f8a6339a8792086e8d74ad1f854b60ecda` passed
[GitHub Actions run 40290986977](https://github.com/Kaustubh1206/Multi-Agent-Trace-Recovery-System/actions/runs/40290986977):
**131 passed, 1 warning in 102.68s**; Ruff passed.

Final code revision `588297b4a75c4e4e3ad0149316c98e57e52778d7` passed
[GitHub Actions run 40291989439](https://github.com/Kaustubh1206/Multi-Agent-Trace-Recovery-System/actions/runs/40291989439):
**133 passed, 1 warning in 94.79s**; Ruff passed. Final checks additionally cover changing
a queued request before approval, native structured-output usage through the real SDK,
and usage exposure in CLI batch summaries and read-only inspection.

## Abbreviated offline execution example

This describes the cancellation path exercised by the real-SDK local HTTP test and
real pinned retail tools. Customer/model responses are offline fixtures:

```text
Customer: Emma Smith, ZIP 10192, no email. Cancel #W2417020: no longer needed.
LLM roles: Supervisor → Identity → Order/Product → Policy → Decision
MAS: Proposed cancel_pending_order ... refund $2674.40 to original gift card.
     Proceed? Reply yes or no.
Committed writes before approval: 0
Customer: yes
Python: execute exact confirmed call → verify resulting DB
MAS: cancel_pending_order completed for #W2417020. Status: cancelled.
Committed writes: 1
MAS logical model requests: 6; additional model requests after approval: 0
```

The JSON summary is abbreviated above. Actual prompts show the complete prepared tool
arguments and financial effects. The native DB result is checked independently of the
role outputs. No live transcript or successful paid run is represented by this example.

## Evidence scope

All automated tests use scripted/intercepted model responses, real pinned fixtures/tools,
and native graders where applicable. They do not establish API-backed pass rates, dollar
savings, or correctness on all 57 baseline tasks. The warning is upstream's deprecated
audioop import.

Retained checks cover authenticated read-only behavior, identity lookup procedures,
ownership, exact consent, stale/tampered arguments, variants/payment restrictions,
duplicate writes, separate task DBs, all supported write APIs, draft-label isolation,
JSON fallback, local LangSmith tracing, Studio checkpoint lifecycles and simulator reports.

Train tasks 0, 1, 47, 50, 69 and 84 retain source-backed graph/tool regressions. The earlier
natural approvals and incorrect informal confirmation questions are exercised against the
new graph. Policy and Identity remain actual independently prompted LLM roles in tests.

New task 113 tests verify two queued cancellations with separate confirmations. The first
write is verified even while the graph pauses for the next approval. Saying no to the
second preserves only the first commit. Actual DB/native ENV checks remain real; the
paid NL judge is intercepted. These are offline outcomes, not live task 113 results.

The real SDK/subprocess CLI test now makes 18 local HTTP requests across a successful
and declined cancellation trial, down from 20 in the previous test fixture. Each new MAS
trial makes six logical model requests; the successful old path made eight. The two
removed requests were Payment and Verification LLM calls. This fixture count does not
predict total spend in real conversations.

Usage checks validate provider-reported input/output/cache counts per role, schema/JSON
fallback accounting and request-budget termination before extra model requests. Full
policy text is sent only to Policy. Cohort tests verify 17+57 disjoint train IDs,
held-out test isolation, null initial states and CLI dry-run behavior without API keys.

## Run locally

```bash
uv sync --locked
uv run python scripts/fetch_retail_data.py
uv run ruff check src tests scripts main.py
uv run pytest tests -q
uv run python -m retail_benchmark --cohort baseline --dry-run
```

LangSmith tests record locally and block uploads; cloud UI verification uses the user's
keys. Simulator and NL-judge tokens/costs remain outside MAS usage counters. SDK retries
may exceed logical request counts. No dollar estimates or recovery results are claimed.

The original experiment's records remain on
[phase-1-retail-mas](https://github.com/Kaustubh1206/Multi-Agent-Trace-Recovery-System/tree/phase-1-retail-mas).
Historical `offline_demo.txt` and `offline_pilot_results.json` belong to that earlier
checkpoint and are not refreshed run artifacts for this branch. Ground-truth labels
remain draft. Non-null initial-state/history overrides, durable/multiworker sessions and
the documented pending multi-item upstream defect remain limitations.

## Follow-up smoke-run regressions

User-supplied task 69 run `c68c6adc-230d-4491-90e2-8cdc24081c1f` stopped with
`decision repeated an unprepared approval request`. It used 12 MAS requests with
25,590 reported input tokens and 703 output tokens. The exact rejected question is not
present in the terminal excerpt. The approval-text filter could incorrectly intercept
missing-information/fallback-choice questions; it now allows these and permits Decision
to clarify an alternative after a Policy denial. This does not authorize a write.
Tests cover an ineligible pending-order return followed by reason clarification, a fresh
prepared cancellation, and exact approval, using task 69's real fixture/native grading.

Task 0 run `101b8e4f-6de7-4052-9922-f786b2278423` completed one exchange but received
native reward 0. The supplied full dialogue shows that the MAS offered white-backlit
and no-backlight full-size clicky keyboards, and the simulator explicitly chose white:
`6342039236`. Its private scenario instead specifies the no-backlight fallback:
`7706410293`. The catalog confirms these options and prices. This is an observed
simulator preference deviation, not evidence that the MAS ignored an explicit public
no-backlight choice. The reference criterion remains unchanged and the raw score stays 0.

The unnecessary repeated confirmation was a separate parser issue. The exact reply
"Yes, please proceed with the exchange as described. This covers everything I wanted
to change. Thank you!" now approves the prepared operation without replanning. Changed
IDs/payment, conditions and conflicting operation labels remain unapproved.

Exchange preflight now includes copied original/replacement options and replacement
prices in its financial-effect summary. This makes the chosen variant visible, binds it
to the fingerprint and prevents summary mutation from changing the DB before execution.
Generic role guidance preserves unspecified options and requires asking about unavailable
combinations. It contains no source-task IDs or expected replacement IDs.

Code revision `658625b0cc1f08ec00092a8e3f0248db572b90f2` passed
[GitHub Actions run 40297632461](https://github.com/Kaustubh1206/Multi-Agent-Trace-Recovery-System/actions/runs/40297632461):
**144 passed, 1 warning in 100.67s**; Ruff passed. The warning is upstream audioop.
Negative grading tests preserve native reward 0 for a confirmed white-backlit exchange;
the no-backlight counterpart retains native reward 1 in the offline fixture. Both use
real official tools/graders with scripted dialogue and specialist outputs.

No new live trial, benchmark pass rate or automatic simulator-fidelity classifier is
established here. The task 69 fix still needs an API-backed rerun. Existing draft oracle
labels, official criteria and held-out split are unchanged.
