"""Real CLI + OpenAI SDK + LiteLLM + native grading, against a local HTTP model stub."""

import json
import os
import subprocess
import sys
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

from .support import after_write, cancellation_steps
from .test_benchmark import REQUEST


def test_cli_saves_independent_trial_reports_and_native_simulations(tmp_path):
    responses = deque([
        ("user", REQUEST), *cancellation_steps(False), ("user", "yes"), *after_write(),
        ("user", "###STOP###"),
        ("user", REQUEST), *cancellation_steps(False), ("user", "no"),
        ("user", "###STOP###"),
    ])
    received = []

    class ModelStub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            role, output = responses.popleft()
            is_user = request["model"] == "gpt-4.1-2025-04-14"
            assert is_user == (role == "user")
            received.append(role)
            content = output if is_user else json.dumps(output)
            body = json.dumps({
                "id": "offline-http-test", "object": "chat.completion", "created": 0,
                "model": request["model"], "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ModelStub)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    env = {
        **os.environ, "OPENAI_API_KEY": "offline-key", "OPENAI_BASE_URL": endpoint,
        "OPENAI_API_BASE": endpoint, "RETAIL_MODEL": "gpt-6-luna",
        "RETAIL_USER_MODEL": "gpt-4.1-2025-04-14", "RETAIL_USER_REASONING_EFFORT": "",
        "RETAIL_OUTPUT_METHOD": "json", "RETAIL_REASONING_EFFORT": "none",
        "LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "retail_benchmark", "--tasks", "69", "--trials", "2",
             "--output", str(tmp_path)],
            cwd=Path(__file__).resolve().parents[1], env=env,
            text=True, capture_output=True, timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not responses
    assert len(received) == 18
    reports = sorted(
        (json.loads(p.read_text()) for p in tmp_path.glob("*.report.json")),
        key=lambda r: r["trial"],
    )
    assert len(reports) == 2
    assert [r["mas_usage"]["calls"] for r in reports] == [6, 6]
    assert [r["mas_usage"]["input_tokens"] for r in reports] == [6, 6]
    assert [r["mas_usage"]["output_tokens"] for r in reports] == [6, 6]
    assert [r["native_reward"] for r in reports] == [1.0, 0.0]
    assert [r["committed_writes"] for r in reports] == [1, 0]
    assert [r["validated_success"] for r in reports] == [True, False]
    assert reports[0]["initial_database_hash"] == reports[1]["initial_database_hash"]
    assert reports[0]["run_id"] != reports[1]["run_id"]
    assert len(list(tmp_path.glob("*.simulation.json"))) == 2
    assert len(list(tmp_path.glob("*.final_db.json"))) == 2
    assert len(json.loads((tmp_path / "summary.json").read_text())["trials"]) == 2
