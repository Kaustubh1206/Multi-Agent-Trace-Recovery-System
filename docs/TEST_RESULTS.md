# Observed Phase 1 verification

Run date: October 25, 2026. Python 3.12.14; LangGraph 1.2.17; τ² 1.0.1 pinned
to `fc0055dc4e0a316c3f83133267fbd6faaa770992`. Exact dependencies are in `uv.lock`.

```bash
uv sync --locked
uv run python scripts/fetch_retail_data.py
uv run pytest tests -q --pilot-report work/offline_pilot_results.json
uv run ruff check src tests scripts main.py
uv run ruff format --check src tests scripts main.py
uv run python -m tests.demo
```

Last full workspace pytest output before the execution environment became unavailable:

```text
84 passed, 1 warning in 40.76s
```

Published integration commit `e66f453129566f063ea0a044b6b7cace89988e3e` was independently
verified by [GitHub Actions run 40230528922](https://github.com/Kaustubh1206/Multi-Agent-Trace-Recovery-System/actions/runs/40230528922)
on Python 3.12.15. Ruff passed; pytest reported **84 passed, 1 warning in 48.63s**.
The workflow uses intercepted/local model responses and no OpenAI or LangSmith secrets.

The warning is the pinned upstream's deprecated `audioop` import. Ruff lint and format
checks passed. `git diff --check` passed. The entry-point help command and cached-asset
verification command ran successfully. Checkpoint resumption was tested with strict
MessagePack type checking and an explicit allowlist for the project's output schemas.

The API compatibility tests use the real LangChain/OpenAI SDK with intercepted HTTP
responses. They reproduce a Groq `json_validate_failed` HTTP 400, check bounded JSON
fallback, verify JSON-text mode omits native schema parameters, and ensure unrelated bad
requests are not retried as format errors. Diagnostic tests check secret/URL redaction
and exclusion of failed generations. No live Groq request was made in these tests.

LangSmith visualization checks capture the graph locally across a confirmation pause/resume,
verify all seven agent names and shared task/thread metadata, and inspect scoped tool inputs.
Local SDK traces record blocked writes and one successful write/verification; tests ensure
tracing cannot bypass approval or capture the adapter/full fixture. Upload methods are
blocked in these checks, and the normal suite forces tracing off. The CLI starts and exits
with tracing disabled and no LangSmith key. Cloud upload/viewing remains to be verified with
the user's account. See `LANGSMITH.md` for configuration and the recorded-data boundaries.

Studio lifecycle tests check schema rendering without credentials, one adapter per thread,
reuse across confirmation resumes, JSON checkpoint round trips, and rejection of old
checkpoints after the retail session is lost. A local `langgraph dev --no-reload` HTTP smoke
check returned HTTP 200 for health, graph and schema endpoints. Two real server threads
used 14 intercepted OpenAI-compatible model responses: `yes` committed and verified a
cancellation, while `no` on a fresh thread made no change. This used the pinned official
DB/tools, no paid model calls and no cloud upload. The Studio browser UI was not automated.

## Official simulator bridge checks

The 15 added integration cases exercise the official text simulator class and native ALL
grading with intercepted model responses. Train tasks 0/1 use different exchange preferences;
69 covers yes/no/ambiguous approval and independent trials; 50 covers actual and missing
handoff; 47 covers communication/NL grading, missing assertions and judge failure.
Private simulator instructions and expected criteria remain outside all MAS role contexts.

A subprocess test runs the real CLI, OpenAI SDK and LiteLLM against a local HTTP model stub:
20 model requests, two independent trials, native results, final DB snapshots and a batch
summary. No paid model call, live benchmark score or cloud trace upload is established.
The published copy is also checked by the repository's offline GitHub Actions workflow.

## Source-backed offline pilots actually run

| Train ID | Graph/tool behavior checked | Committed writes | Native ENV DB match |
|---|---|---:|---|
| `69` | Return request → pending-order discovery → reason clarification → confirmed cancellation | 1 | true |
| `0` | Two-item exchange with no-backlight keyboard fallback | 1 | true |
| `1` | Different fallback: exchange thermostat only | 1 | true |
| `47` | Wrong references → customer corrections → return → remaining-items inquiry | 1 | true |
| `84` | Cheaper-tablet proposal → changed item/payment → fresh summary/consent | 1 | true |
| `50` | Unsupported reversal → actual human-transfer API and required message | 0 | true |

These are **scripted model integration tests using real DB fixtures and official tools**.
The native `EnvironmentEvaluator` was run with strict tool-output checks; each raw result
is preserved in `offline_pilot_results.json`. Its ENV reward was 1.0 for these scripted
trajectories. No communication/NL assertion reward or full benchmark score was computed.
For task 50, unchanged DB alone is insufficient, so a separate assertion verifies transfer.

The graph-node test checks exactly the seven proposal roles. Confirmation resumption does
not repeat planning or the proposal. Runtime Verification compares the committed DB with
the confirmed operation; a failed deterministic check cannot be approved by its LLM.

Additional tests cover authenticated read-only inquiry without mutation; blocked protected
activity without auth; wrong-customer references; exact approval and stale/tampered actions;
no/ambiguous approval; item/product confusion, variant availability, quantities and payment
rules; all seven retail write APIs; the pinned pending-item defect; unchanged unrelated
records; duplicate commit rejection; independent sequential tasks; agent step budget;
structured-output fallback failures; and oracle isolation across all seven role contexts.

## Reproducible example

The full generated output is in `offline_demo.txt`. This exercises task 69's cancellation
subpath with a manual-style request, scripted specialist outputs and official tools:

```text
Customer: I'm Emma Smith, ZIP 10192; I cannot remember email.
          Cancel #W2417020 because it is no longer needed.
Specialists: Supervisor → Identity → Order → Policy → Decision → Action/Payment → Verification
MAS: Proposed cancel_pending_order for #W2417020, reason no longer needed.
     Refund: $2674.40 to gift_card_8541487 immediately. Proceed? Reply yes or no.
Writes before confirmation: 0
Customer: yes
MAS: cancel_pending_order completed for #W2417020. Status: cancelled.
Writes after confirmation: 1
Gift-card balance: 2736.4
```

The excerpt abbreviates the canonical JSON summary printed by the application; it is not
a live-model or official-user-simulator conversation. That summary always contains the
actual exact tool arguments and validated financial consequences.

## Remaining integration work

- Real API-backed specialist dialogues have not been run: no API credentials were configured.
- The official text simulator and native full evaluator are integrated through a custom
  serial bridge and tested with intercepted model responses. Live evaluation remains
  pending. No held-out test task or claimed benchmark pass rate exists.
- Ground-truth labels remain draft pending independent review.
- Non-null task initial state/history, durable sessions and robust autonomous multi-goal
  completion are outside the demonstrated runner behavior.
- Affected pending multi-item modifications fail closed due to the documented upstream defect.

No tracing, causal attribution, fault injection or recovery results are claimed.
