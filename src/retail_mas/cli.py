"""Manual customer mode. A fresh process/session gets its own benchmark DB."""

import argparse
import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Retail MAS — manual customer")
    parser.add_argument(
        "--task", help="Optional train task ID; metadata only, not a customer prompt"
    )
    args = parser.parse_args()
    load_dotenv()
    # tau2 imports LiteLLM as a transitive dependency; offline tests need no price-map fetch.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    from loguru import logger

    logger.disable("tau2")
    from .data_loader import RetailDataset
    from .graph import build_graph
    from .models import AgentModels
    from .tools import RetailAdapter

    dataset = RetailDataset()
    task = dataset.load_train_task(args.task) if args.task else None
    adapter = RetailAdapter(dataset.fresh_db(task["initial_state"] if task else None))
    agents = AgentModels()
    graph = build_graph(adapter, agents, dataset.policy)
    config = {"configurable": {"thread_id": str(uuid4())}, "recursion_limit": 512}
    print("Retail MAS (simulated retail data). Type /quit to stop. No benchmark score is computed.")
    try:
        while True:
            query = input("Customer: ").strip()
            if query == "/quit":
                break
            if not query:
                continue
            update = {
                "messages": [HumanMessage(content=query)],
                "customer_query": query,
                "new_request": True,
            }
            if args.task:
                update.update(source_task_id=args.task, scenario_id=f"manual-{args.task}")
            result = graph.invoke(update, config)
            while result.get("__interrupt__"):
                print("MAS:", result["__interrupt__"][0].value["message"])
                reply = input("Customer: ").strip()
                if reply == "/quit":
                    return
                result = graph.invoke(Command(resume=reply), config)
            print("MAS:", result["response"])
    except (EOFError, KeyboardInterrupt):
        print("\nSession closed.")
    except Exception as error:
        # Avoid printing provider payloads/URLs, which may contain sensitive configuration.
        print(f"Session stopped ({type(error).__name__}). No further actions will run.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
