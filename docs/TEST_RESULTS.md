# Observed Phase 1 verification

Run date: October 24, 2026. Python 3.12.14; LangGraph 1.2.17; τ² 1.0.1 pinned
to `fc0055dc4e0a316c3f83133267fbd6faaa770992`. Exact dependencies are in `uv.lock`.

```bash
uv sync --locked
uv run python scripts/fetch_retail_data.py
LANGGRAPH_STRICT_MSGPACK=true uv run pytest tests -q --pilot-report docs/offline_pilot_results.json
uv run ruff check src tests scripts main.py
uv run ruff format --check src tests scripts main.py
uv run python -m tests.demo
```

Observed final pytest output:

```text
.................................................                        [100%]
49 passed, 1 warning in 16.12s
```

The warning is the pinned upstream's deprecated `audioop` import. Ruff lint and format
checks passed. `git diff --check` passed. The entry-point help command and cached-asset
verification command ran successfully. Checkpoint resumption was tested with strict
MessagePack type checking and an explicit allowlist for the project's output schemas.

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

Additional tests cover authenticated read-only inquiry without mutation; blocked protected
activity without auth; wrong-customer references; exact approval and stale/tampered actions;
no/ambiguous approval; item/product confusion, variant availability, quantities and payment
rules; all seven retail write APIs; the pinned pending-item defect; unchanged unrelated
records; duplicate commit rejection; independent sequential tasks; agent step budget;
structured-output fallback failures; and oracle isolation across all six role contexts.

## Reproducible example

The full generated output is in `offline_demo.txt`. This exercises task 69's cancellation
subpath with a manual-style request, scripted specialist outputs and official tools:

```text
Customer: I'm Emma Smith, ZIP 10192; I cannot remember email.
          Cancel #W2417020 because it is no longer needed.
Specialists: Supervisor → Identity → Order → Decision → Policy → Payment
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
- The official user simulator/orchestrator bridge and full communication/NL evaluator remain
  unintegrated. No held-out test task or claimed benchmark pass rate exists.
- Ground-truth labels remain draft pending independent review.
- Non-null task initial state/history, durable sessions and robust autonomous multi-goal
  completion are outside the demonstrated runner behavior.
- Affected pending multi-item modifications fail closed due to the documented upstream defect.

No tracing, causal attribution, fault injection or recovery results are claimed.
