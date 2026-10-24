# View the Retail MAS experiment in LangSmith

The graph still contains the same seven proposal agents. Optional LangSmith Observability
shows their actual execution, model calls, scoped tool calls, outputs, errors and latency.
Model token/cost details depend on provider usage reporting and LangSmith's pricing catalog.
This is experiment visualization; MA-TRACE causal attribution/recovery is still future work.

## Run interactively in Studio

The repository includes `langgraph.json` and a graph factory in `src/retail_mas/studio.py`.
The command is **`langgraph dev`**, without `run` between the two words. From the repository
root, after configuring `.env` with your OpenAI and LangSmith keys:

```bash
git switch phase-1-retail-mas
git pull --ff-only
uv sync --locked
uv run python scripts/fetch_retail_data.py
uv run langgraph dev
```

The development CLI is in the `dev` dependency group, installed by `uv sync` by default.
Docker is not required. Keep the terminal running and open the Studio link printed there:
[local Studio](https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024).
The local API docs are at `http://127.0.0.1:2024/docs`.

Select the **`retail_mas`** graph and create a new thread. Use **Graph** mode and submit:

```json
{
  "customer_query": "I'm Emma Smith, ZIP 10192. I cannot remember my email. Cancel order #W2417020 because it is no longer needed.",
  "messages": [{
    "role": "human",
    "content": "I'm Emma Smith, ZIP 10192. I cannot remember my email. Cancel order #W2417020 because it is no longer needed."
  }],
  "new_request": true
}
```

The graph pauses at Decision when it needs a customer reply or confirmation. Use Studio's
interrupt **Resume** control with the string `yes` or `no` (the API equivalent is
`{"command": {"resume": "yes"}}`). Answer other clarification questions the same way.
Do not submit confirmation as a new request. For a later request on the same thread, submit
new `customer_query`/`messages` values and `new_request: true`. Inspect `response` in the
final state and expand the executed nodes and model/tool traces.

Studio runs manual customer conversations against the pinned DB. This example exercises
task 69's cancellation subpath; Studio does not run the official user simulator or compute
a benchmark score. The graph is unchanged: seven agent nodes, with guards inside tools.

Each Studio thread has its own fresh simulated DB, identity and approval state. The same
thread reuses its session across turns and confirmation resumes. Retail sessions remain
in memory: **after a server restart or hot reload, create a new thread**. Attempting to
resume an old session fails closed. Development checkpoints in `.langgraph_api/` are
ignored by Git. Rewinding/forking a thread's checkpoints does not rewind the retail DB;
start a fresh thread for an independent trial. This entry point is for a local development
server, not a multi-worker deployment.

Server startup, graph/schema endpoints and a scripted HTTP cancellation/confirmation run
are checked locally. No live OpenAI call or cloud trace upload is part of that smoke check.

The remaining sections describe tracing when using the terminal customer loop instead.

## 1. Configure your local account

Create a LangSmith API key in your workspace's settings. Keep the existing OpenAI settings
and add these lines to your local `.env`:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=matrace-retail-pilot
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
```

The LangSmith key is separate from `OPENAI_API_KEY`. Your `.env` is ignored by Git.
For an EU workspace, use `https://eu.api.smith.langchain.com` as the endpoint. Set
`LANGSMITH_WORKSPACE_ID` if your key requires choosing among multiple workspaces.
For other regions, use the API endpoint shown in your workspace settings.

## 2. Run one experiment conversation

```bash
git switch phase-1-retail-mas
git pull --ff-only
uv sync --locked
uv run python main.py --task 69
```

The CLI prints the project name and a unique conversation thread ID when tracing is enabled.
Supply the customer request and answer the confirmation prompts as usual. `/quit` closes
the session and flushes pending traces. Set `LANGSMITH_TRACING=false` to turn uploads off.

For the five-task pilot, run tasks `69`, `0`, `1`, `47`, `50` separately. Each new CLI process
gets a fresh DB and new thread ID. All tasks can use the same LangSmith project.

## 3. Open the trace view

Open [LangSmith](https://smith.langchain.com), select your workspace and the
`matrace-retail-pilot` tracing project. Run a conversation first so the project has traces.
Open the latest `retail-task-69` run and expand its child runs.

A typical confirmed change includes:

```text
retail-task-69
  supervisor
    ChatOpenAI
  identity
    ChatOpenAI
    retail.read
  order
    ChatOpenAI
    retail.read
  policy
    ChatOpenAI
  decision
    ChatOpenAI
    [customer confirmation pause]

retail-task-69  [resume; same thread ID]
  decision
  action_payment
    ChatOpenAI
    retail.execute
  verification
    retail.verify
      retail.read
    ChatOpenAI
```

Supervisor/order steps can repeat as the dialogue progresses. Read-only answers, refusal
and transfer take shorter paths. `retail.transfer` records an official simulated handoff.
`retail.execute` is the guarded write boundary; disposable preflight calls are not recorded
as executed customer actions.

Each `graph.invoke` and confirmation resume is a separate root trace. Use the **Threads**
view or filter by the printed `thread_id` to see the whole conversation together. Metadata
also includes `source_task_id`, `execution_mode=manual` and the pinned benchmark commit.
LangSmith traces are execution evidence, not benchmark scores.

## What is recorded

Enabling tracing uploads customer dialogue, shared runtime state, system prompts, model
outputs and observed simulated retail records. The tool wrappers exclude the adapter and
full fixture from their inputs. Agent/API keys are not supplied as trace arguments.
Private simulator instructions and official evaluation criteria remain outside the graph.
The official toolkit's internal copies are not instrumented as actual customer writes.

The normal pytest suite forces uploads off, even if a local `.env` enables tracing. Dedicated
checks use local trace capture and blocked upload methods. No live account upload was tested
in the implementation environment; complete the setup above to verify your own account.

If traces do not appear, check the project, endpoint region and API-key workspace access.
Shell-exported environment variables take precedence over `.env`. A model request failure
and a LangSmith upload failure are separate errors; inspect the terminal message and trace.

References: [LangGraph tracing](https://docs.langchain.com/langsmith/trace-with-langgraph),
[custom tool spans](https://docs.langchain.com/langsmith/annotate-code),
[conversation threads](https://docs.langchain.com/langsmith/threads).
