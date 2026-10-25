"""Official text simulator with its guidelines read from our hash-verified assets."""

import os
from pathlib import Path

from tau2.config import DEFAULT_LLM_USER
from tau2.user.user_simulator import UserSimulator


class PinnedUserSimulator(UserSimulator):
    def __init__(self, instructions: str, guidelines_path: Path, model: str, seed: int):
        self.guidelines_path = guidelines_path
        # Explicit provider selection supports IDs unknown to LiteLLM's catalog.
        model = model.removeprefix("openai/")
        arguments = {"temperature": 0.0, "timeout": 60, "num_retries": 1}
        effort = os.getenv("RETAIL_USER_REASONING_EFFORT")
        if effort:
            arguments.pop("temperature")
            arguments["reasoning_effort"] = effort
        super().__init__(llm=f"openai/{model}", instructions=instructions, llm_args=arguments)
        self.set_seed(seed)

    @property
    def global_simulation_guidelines(self):
        return self.guidelines_path.read_text()


def create_simulator(task, dataset, model: str | None, seed: int):
    return PinnedUserSimulator(
        instructions=str(task.user_scenario),
        guidelines_path=dataset.root / "simulation_guidelines.md",
        model=model or os.getenv("RETAIL_USER_MODEL") or DEFAULT_LLM_USER,
        seed=seed,
    )
