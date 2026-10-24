"""One configurable API model; validated JSON fallback for compatible providers."""

import json
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError
from pydantic import ValidationError


def create_model():
    return ChatOpenAI(
        model=os.getenv("RETAIL_MODEL", "gpt-4.1-mini"),
        temperature=0,
        timeout=60,
        max_retries=1,
    )


class AgentModels:
    """Each role receives its own system prompt and a narrowly filtered context."""

    def __init__(self, model=None):
        self.model = model if model is not None else create_model()

    def invoke(self, role, schema, system: str, context: dict):
        messages = [SystemMessage(content=system), HumanMessage(content=json.dumps(context))]
        try:
            result = self.model.with_structured_output(schema, method="json_schema").invoke(
                messages
            )
            return schema.model_validate(result)
        except (NotImplementedError, ValueError, TypeError):
            # Do not swallow API authentication, network or rate-limit failures.
            pass
        except BadRequestError as error:
            if not any(
                word in str(error).lower()
                for word in ("response_format", "json_schema", "structured")
            ):
                raise
        instruction = "Return only a JSON object matching this schema: " + json.dumps(
            schema.model_json_schema()
        )
        for _ in range(2):
            output = self.model.invoke(
                [SystemMessage(content=system + "\n" + instruction), messages[1]]
            )
            try:
                content = output.content
                if not isinstance(content, str):
                    raise ValueError("Expected text JSON")
                if content.startswith("```json") and content.rstrip().endswith("```"):
                    content = content[7 : content.rfind("```")].strip()
                return schema.model_validate_json(content)
            except (ValidationError, ValueError):
                instruction += (
                    "\nThe last output was invalid. Return valid JSON, without commentary."
                )
        raise ValueError(f"Invalid structured output from {role}; no action taken")
