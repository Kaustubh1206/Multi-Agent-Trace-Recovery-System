"""Offline cancellation subpath using scripted agent outputs and task 69's real DB."""

import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def main():
    from loguru import logger

    logger.disable("tau2")
    from retail_mas.data_loader import RetailDataset
    from retail_mas.tools import RetailAdapter

    from .support import cancellation_steps, resume, session

    dataset = RetailDataset()
    adapter = RetailAdapter(dataset.fresh_db(dataset.load_train_task("69")["initial_state"]))
    query = "I'm Emma Smith, ZIP 10192; I cannot remember email. Cancel #W2417020 because it is no longer needed."
    print("OFFLINE DEMO: scripted LLM outputs, official tools; no simulated-user benchmark score.")
    print("Customer:", query)
    graph, agents, config, result = session(adapter, dataset, cancellation_steps(), query, "69")
    print("MAS:", result["__interrupt__"][0].value["message"])
    print("Writes before confirmation:", adapter.write_count)
    print("Customer: yes")
    result = resume(graph, config, "yes")
    print("MAS:", result["response"])
    print("Specialists:", " -> ".join(role for role, _, _ in agents.contexts))
    print("Writes after confirmation:", adapter.write_count)
    print(
        "Gift-card balance:",
        adapter.snapshot().users["emma_smith_8564"].payment_methods["gift_card_8541487"].balance,
    )


if __name__ == "__main__":
    main()
