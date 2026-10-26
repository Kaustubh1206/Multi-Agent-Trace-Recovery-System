# LangSmith and Studio for the custom Retail MAS

Use the existing `langgraph.json` entry point. The graph has five LLM roles:
Supervisor, Identity, Order/Product, Policy, Decision. Execution and DB verification
are ordinary guarded Python operations inside Decision.

## Configuration

Keep credentials in your local `.env`:

```dotenv
OPENAI_API_KEY=your-openai-key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=matrace-retail-custom
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

Use a separate project for the new branch so its runs are easy to compare with the
earlier seven-agent experiment. Neither key is committed.

```bash
uv sync --locked
uv run python scripts/fetch_retail_data.py
uv run langgraph dev
```

Open Studio using the URL printed by the server. Create a thread and supply the
customer message/query, then resume customer interrupts with actual replies.
`source_task_id` is metadata; it does not inject the hidden task scenario.
For manual terminal mode use `uv run python main.py --task 69`.

For automatic conversations:

```bash
uv run python -m retail_benchmark --tasks 69 0 1 --output work/custom-smoke --verbose
```

## What to inspect

- Graph spans identify the five roles. OpenAI child spans expose provider usage when
  available. A confirmation resume performs no new LLM request for execution/verification.
- Tool spans retain scoped reads, actual writes and deterministic verification. They
  never serialize the adapter/full DB or preflight copies as tool inputs.
- Each thread/run ID links the MAS trace to its local benchmark report.
- `mas_usage` in reports gives logical request and token subtotals per role. SDK transport
  retries may exceed logical request counts. Missing usage is explicitly counted.
- Simulator and native NL-judge costs are separate from MAS usage. LangSmith project
  totals should not be assumed to include every upstream call; check provider billing.

The full benchmark trajectory and oracle remain in the benchmark harness, outside MAS
role prompts/state. No LangSmith dataset/experiment upload pipeline is implemented;
this integration provides tracing and Studio visualization for CLI benchmark runs.

## Session behavior and tests

Each Studio thread has one in-memory adapter/DB. Confirmation resumes reuse it; other
threads start fresh. Restart requires a new thread. Checkpoints are JSON-compatible,
but checkpoint rewinding does not rewind the retail DB. This is a single-process
development setup, not a durable multiworker deployment.

Offline tests capture graph/tool spans locally, block uploads, and verify guards, thread
isolation and pause/resume. No cloud upload or live model behavior is claimed by those
tests. Configure your own keys to verify the cloud UI.
