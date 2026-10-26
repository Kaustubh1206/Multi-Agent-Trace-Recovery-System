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
