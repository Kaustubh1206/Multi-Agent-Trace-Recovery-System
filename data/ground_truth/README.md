# Test-only draft task labels

`clean_tasks.jsonl` preserves the pinned official evaluation criteria for train tasks
`0`, `1`, `47`, `50`, `69`, `84`, fixture/policy hashes and computed expected final DB hashes.
The original simulator instructions remain unchanged in the downloaded `tasks.json`.

Generate with `uv run python -m tests.build_labels`. The generator replays reference tools
on a fresh fixture, not an MAS run. All labels remain **draft** until independently reviewed.
Automated reference replay is insufficient to mark communication or task interpretation as
validated. No causal/fault/recovery labels are present.

Only `tests/` imports oracle logic. Runtime agents cannot read this file, evaluation criteria
or private simulator instructions through their supplied context/tools. Isolation tests
check every specialist's actual payload and runtime imports.

The benchmark data/code are Sierra Research's MIT-licensed assets. See
[upstream license](https://github.com/sierra-research/tau2-bench/blob/fc0055dc4e0a316c3f83133267fbd6faaa770992/LICENSE).
Downloaded source assets are cached locally rather than committed as full copies.
