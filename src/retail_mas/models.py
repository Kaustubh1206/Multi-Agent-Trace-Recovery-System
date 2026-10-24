"""One configurable API model; validated JSON fallback for compatible providers."""

import json
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APIStatusError, BadRequestError
from pydantic import ValidationError


class ModelOutputError(ValueError):
    """A model response could not be validated after bounded retries."""


def create_model():
    return ChatOpenAI(
        model=os.getenv("RETAIL_MODEL", "gpt-6-luna"),
        reasoning_effort=os.getenv("RETAIL_REASONING_EFFORT") or None,
        timeout=60,
        max_retries=1,
        use_responses_api=False,
    )


def schema_error(error: BadRequestError) -> bool:
    """Retry format-specific rejections, never unrelated bad requests."""
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error", body)
    code = details.get("code", "") if isinstance(details, dict) else ""
    return code in {"json_validate_failed", "json_schema_failed"} or any(
        word in str(error).lower()
        for word in ("response_format", "json_schema", "structured output")
    )


def describe_model_error(error: Exception) -> str:
    """Show provider diagnostics without headers, request bodies or failed generations."""
    if isinstance(error, ModelOutputError):
        return str(error)
    if not isinstance(error, APIStatusError):
        return type(error).__name__
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error", body)
    if not isinstance(details, dict):
        details = {}
    message = str(details.get("message") or "Provider rejected the request.")
    code = str(details.get("code") or "unknown")
    parameter = str(details.get("param") or "unspecified")
    diagnostic = f"HTTP {error.status_code}; code={code}; parameter={parameter}: {message}"
    for name in ("OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        key = os.getenv(name)
        if key:
            diagnostic = diagnostic.replace(key, "[redacted]")
    diagnostic = re.sub(r"(?:gsk_|sk-)[A-Za-z0-9_-]+", "[redacted]", diagnostic)
    diagnostic = re.sub(r"https?://\S+", "[endpoint]", diagnostic)
    return " ".join(diagnostic.split())[:500]


class AgentModels:
    """Each role receives its own system prompt and a narrowly filtered context."""

    def __init__(self, model=None, output_method=None):
        self.model = model if model is not None else create_model()
        self.output_method = output_method or os.getenv("RETAIL_OUTPUT_METHOD", "json_schema")
        if self.output_method not in {"json", "json_schema"}:
            raise ValueError("RETAIL_OUTPUT_METHOD must be json or json_schema")

    def invoke(self, role, schema, system: str, context: dict):
        messages = [SystemMessage(content=system), HumanMessage(content=json.dumps(context))]
        if self.output_method == "json_schema":
            try:
                # A plain schema avoids SDK-enforced strict schemas, which cannot
                # represent our variable retail-tool argument dictionaries faithfully.
                result = self.model.with_structured_output(
                    schema.model_json_schema(), method="json_schema", strict=False
                ).invoke(messages)
                return schema.model_validate(result)
            except (NotImplementedError, ValueError, TypeError):
                pass
            except BadRequestError as error:
                if not schema_error(error):
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
        raise ModelOutputError(f"Invalid structured output from {role}; no action taken")
