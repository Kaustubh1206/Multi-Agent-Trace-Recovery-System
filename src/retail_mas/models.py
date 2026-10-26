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
        max_tokens=int(os.getenv("RETAIL_MAX_OUTPUT_TOKENS", "1024")),
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
    if isinstance(error, ValueError):
        message = str(error)
        protocol_error = re.fullmatch(
            r"Invalid (identity|order|decision) output kind: "
            r"(tool|ask|complete|respond|propose|transfer)", message
        )
        if protocol_error or message in {
            "A tool/proposal step requires a tool call",
            "RETAIL_OUTPUT_METHOD must be json or json_schema",
        }:
            return message
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

    def __init__(self, model=None, output_method=None, max_calls=None):
        self.model = model if model is not None else create_model()
        self.max_calls = int(max_calls if max_calls is not None else os.getenv("RETAIL_MAX_MODEL_CALLS", "32"))
        if self.max_calls < 1:
            raise ValueError("RETAIL_MAX_MODEL_CALLS must be positive")
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                      "cached_input_tokens": 0, "responses_with_usage": 0, "per_role": {}}
        self.output_method = output_method or os.getenv("RETAIL_OUTPUT_METHOD", "json_schema")
        if self.output_method not in {"json", "json_schema"}:
            raise ValueError("RETAIL_OUTPUT_METHOD must be json or json_schema")

    def call(self, role, function, messages):
        """Count logical model requests and provider-reported tokens, including fallbacks."""
        if self.usage["calls"] >= self.max_calls:
            raise ModelOutputError("MAS model-call budget reached; no further model calls")
        row = self.usage["per_role"].setdefault(role, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cached_input_tokens": 0, "responses_with_usage": 0,
        })
        self.usage["calls"] += 1
        row["calls"] += 1
        result = function.invoke(messages)
        raw = result.get("raw") if isinstance(result, dict) else result
        usage = getattr(raw, "usage_metadata", None) or {}
        if usage:
            values = {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cached_input_tokens": (usage.get("input_token_details") or {}).get("cache_read", 0),
                "responses_with_usage": 1,
            }
            for name, value in values.items():
                row[name] += value or 0
                self.usage[name] += value or 0
        return result

    def invoke(self, role, schema, system: str, context: dict):
        messages = [SystemMessage(content=system), HumanMessage(content=json.dumps(context))]
        if self.output_method == "json_schema":
            try:
                # A plain schema avoids SDK-enforced strict schemas, which cannot
                # represent our variable retail-tool argument dictionaries faithfully.
                structured = self.model.with_structured_output(
                    schema.model_json_schema(), method="json_schema", strict=False, include_raw=True
                )
                result = self.call(role, structured, messages)
                if isinstance(result, dict) and "raw" in result:
                    if result.get("parsing_error"):
                        raise ValueError("Invalid provider structured output")
                    result = result["parsed"]
                return schema.model_validate(result)
            except ModelOutputError:
                raise
            except (NotImplementedError, ValueError, TypeError):
                pass
            except BadRequestError as error:
                if not schema_error(error):
                    raise
        instruction = "Return only a JSON object matching this schema: " + json.dumps(
            schema.model_json_schema()
        )
        for _ in range(2):
            output = self.call(
                role, self.model, [SystemMessage(content=system + "\n" + instruction), messages[1]]
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
