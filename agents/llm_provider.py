#!/usr/bin/env python3
"""Swappable structured-output LLM providers for the audit agents (Part B).

Default provider is **Gemini** (Google Generative Language REST API) called
directly with ``requests`` — no SDK dependency. ``anthropic`` and ``openai``
REST providers and an offline ``mock`` provider are also available; select via
``config.yaml`` ``agents.provider``. Every provider returns output validated
against a Pydantic v2 schema, re-asking on invalid JSON up to
``agents.max_schema_retries`` times.

Only ``requests`` is required (present in the venv). Live Gemini calls need
``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) and network access.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# Pydantic v2 JSON-schema -> Gemini responseSchema (OpenAPI subset)
# ---------------------------------------------------------------------------
_JSON_TO_GEMINI_TYPE = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}


def pydantic_to_gemini_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model's JSON schema to Gemini's ``responseSchema``.

    Inlines ``$defs``/``$ref``, collapses ``Optional`` (``anyOf`` with null) to
    ``nullable``, maps enums and arrays, and preserves field order via
    ``propertyOrdering``. Drops keys Gemini rejects (``title``, ``default``).
    """
    root = model.model_json_schema()
    defs = root.get("$defs", {})

    def resolve(node: Dict[str, Any]) -> Dict[str, Any]:
        if "$ref" in node:
            ref_name = node["$ref"].split("/")[-1]
            return resolve(defs[ref_name])

        # Optional[X] / Union with null -> nullable
        if "anyOf" in node:
            variants = node["anyOf"]
            non_null = [v for v in variants if v.get("type") != "null"]
            has_null = any(v.get("type") == "null" for v in variants)
            base = resolve(non_null[0]) if non_null else {"type": "STRING"}
            if has_null:
                base["nullable"] = True
            if node.get("description"):
                base.setdefault("description", node["description"])
            return base

        out: Dict[str, Any] = {}
        if node.get("description"):
            out["description"] = node["description"]

        if "enum" in node:
            out["type"] = "STRING"
            out["enum"] = [str(v) for v in node["enum"]]
            return out

        json_type = node.get("type")
        if json_type == "object":
            out["type"] = "OBJECT"
            props = node.get("properties", {})
            out["properties"] = {k: resolve(v) for k, v in props.items()}
            if props:
                out["propertyOrdering"] = list(props.keys())
            required = node.get("required")
            if required:
                out["required"] = list(required)
            return out

        if json_type == "array":
            out["type"] = "ARRAY"
            items = node.get("items", {})
            out["items"] = resolve(items) if items else {"type": "STRING"}
            if "minItems" in node:
                out["minItems"] = node["minItems"]
            return out

        out["type"] = _JSON_TO_GEMINI_TYPE.get(json_type, "STRING")
        return out

    return resolve(root)


def schema_hint(model: Type[BaseModel]) -> str:
    """Compact JSON-schema string appended to prompts as a belt-and-braces guide."""
    return json.dumps(model.model_json_schema(), separators=(",", ":"))


def _extract_json(text: str) -> Any:
    """Parse JSON, tolerating ```json fences and surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if 0 <= start < end:
            return json.loads(text[start : end + 1])
        raise


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------
class StructuredLLMError(RuntimeError):
    pass


class ChatModel(ABC):
    """Common structured-output loop; providers implement ``_generate``."""

    provider_name = "base"

    def __init__(self, agents_cfg: Dict[str, Any]):
        self.cfg = agents_cfg
        self.model_name = agents_cfg.get("model", "")
        self.temperature = float(agents_cfg.get("temperature", 0.1))
        self.max_tokens = int(agents_cfg.get("max_tokens", 2048))
        self.max_schema_retries = int(agents_cfg.get("max_schema_retries", 2))
        self.timeout = float(agents_cfg.get("timeout_s", 60))
        self.last_raw: Optional[str] = None
        self.call_count = 0

    @abstractmethod
    def _generate(self, system: str, user: str, schema: Type[BaseModel], repair_hint: str) -> str:
        """Return the model's raw text response (expected to be JSON)."""

    def structured_output(self, system: str, user: str, schema: Type[BaseModel]) -> BaseModel:
        system_full = f"{system}\n\nReturn ONLY a JSON object matching this schema:\n{schema_hint(schema)}"
        last_err = ""
        for attempt in range(self.max_schema_retries + 1):
            hint = "" if attempt == 0 else (
                f"Your previous reply was rejected: {last_err}. "
                "Reply with ONLY valid JSON conforming to the schema — no prose, no code fences."
            )
            self.call_count += 1
            raw = self._generate(system_full, user, schema, hint)
            self.last_raw = raw
            try:
                return schema.model_validate(_extract_json(raw))
            except (json.JSONDecodeError, ValidationError, KeyError) as exc:
                last_err = str(exc)[:400]
        raise StructuredLLMError(
            f"{self.provider_name}:{self.model_name} did not return schema-valid JSON after "
            f"{self.max_schema_retries + 1} attempts. Last error: {last_err}. "
            f"Last raw (truncated): {(self.last_raw or '')[:600]}"
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


# ---------------------------------------------------------------------------
# Key resolution
# ---------------------------------------------------------------------------
def _resolve_key(agents_cfg: Dict[str, Any], extra_env: List[str]) -> str:
    candidates: List[str] = []
    primary = agents_cfg.get("api_key_env")
    if primary:
        candidates.append(primary)
    candidates.extend(extra_env)
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return value
    raise StructuredLLMError(
        "No API key found. Set one of these environment variables "
        f"(e.g. in .env): {', '.join(dict.fromkeys(candidates))}."
    )


# ---------------------------------------------------------------------------
# Gemini (default)
# ---------------------------------------------------------------------------
class GeminiProvider(ChatModel):
    provider_name = "gemini"

    def __init__(self, agents_cfg: Dict[str, Any]):
        super().__init__(agents_cfg)
        self.base_url = agents_cfg.get("base_url", "https://generativelanguage.googleapis.com").rstrip("/")
        self.model_name = agents_cfg.get("model", "gemini-2.5-flash")
        self.api_key = _resolve_key(agents_cfg, extra_env=["GEMINI_API_KEY", "GOOGLE_API_KEY"])

    def build_request(
        self, system: str, user: str, schema: Type[BaseModel], repair_hint: str = ""
    ) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
        """Construct (url, headers, body) without sending — used by tests too."""
        url = f"{self.base_url}/v1beta/models/{self.model_name}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        parts = [{"text": user}]
        if repair_hint:
            parts.append({"text": repair_hint})
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "responseMimeType": "application/json",
                "responseSchema": pydantic_to_gemini_schema(schema),
            },
        }
        return url, headers, body

    def _generate(self, system: str, user: str, schema: Type[BaseModel], repair_hint: str) -> str:
        import requests

        url, headers, body = self.build_request(system, user, schema, repair_hint)
        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise StructuredLLMError(f"Gemini HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise StructuredLLMError(f"Gemini returned no candidates: {json.dumps(data)[:500]}")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            reason = candidates[0].get("finishReason", "unknown")
            raise StructuredLLMError(f"Gemini empty response (finishReason={reason})")
        return text


# ---------------------------------------------------------------------------
# Anthropic (Messages API + forced tool)
# ---------------------------------------------------------------------------
class AnthropicProvider(ChatModel):
    provider_name = "anthropic"

    def __init__(self, agents_cfg: Dict[str, Any]):
        super().__init__(agents_cfg)
        self.base_url = agents_cfg.get("base_url", "https://api.anthropic.com").rstrip("/")
        self.model_name = agents_cfg.get("model", "claude-sonnet-5")
        self.api_key = _resolve_key(agents_cfg, extra_env=["ANTHROPIC_API_KEY"])
        self.version = agents_cfg.get("anthropic_version", "2023-06-01")

    def _generate(self, system: str, user: str, schema: Type[BaseModel], repair_hint: str) -> str:
        import requests

        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json",
        }
        content = user if not repair_hint else f"{user}\n\n{repair_hint}"
        body = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system,
            "tools": [
                {
                    "name": "emit_result",
                    "description": "Return the audit result.",
                    "input_schema": schema.model_json_schema(),
                }
            ],
            "tool_choice": {"type": "tool", "name": "emit_result"},
            "messages": [{"role": "user", "content": content}],
        }
        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise StructuredLLMError(f"Anthropic HTTP {resp.status_code}: {resp.text[:500]}")
        for block in resp.json().get("content", []):
            if block.get("type") == "tool_use":
                return json.dumps(block.get("input", {}))
        raise StructuredLLMError("Anthropic response contained no tool_use block")


# ---------------------------------------------------------------------------
# OpenAI (chat.completions, JSON mode)
# ---------------------------------------------------------------------------
class OpenAIProvider(ChatModel):
    provider_name = "openai"

    def __init__(self, agents_cfg: Dict[str, Any]):
        super().__init__(agents_cfg)
        self.base_url = agents_cfg.get("base_url", "https://api.openai.com").rstrip("/")
        self.model_name = agents_cfg.get("model", "gpt-4o-mini")
        self.api_key = _resolve_key(agents_cfg, extra_env=["OPENAI_API_KEY"])

    def _generate(self, system: str, user: str, schema: Type[BaseModel], repair_hint: str) -> str:
        import requests

        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if repair_hint:
            messages.append({"role": "user", "content": repair_hint})
        body = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": messages,
        }
        resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise StructuredLLMError(f"OpenAI HTTP {resp.status_code}: {resp.text[:500]}")
        return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Mock (offline / tests) — deterministic, no network
# ---------------------------------------------------------------------------
class MockProvider(ChatModel):
    """Returns canned/derived output keyed by schema class name.

    ``responder`` maps ``schema.__name__`` -> dict OR callable(system, user) ->
    dict. Enables full offline graph runs and unit tests without a key.
    """

    provider_name = "mock"

    def __init__(self, agents_cfg: Dict[str, Any], responder: Dict[str, Any]):
        super().__init__(agents_cfg)
        self.model_name = agents_cfg.get("model", "mock")
        self.responder = responder

    def _generate(self, system: str, user: str, schema: Type[BaseModel], repair_hint: str) -> str:
        handler = self.responder.get(schema.__name__)
        if handler is None:
            raise StructuredLLMError(f"MockProvider has no responder for {schema.__name__}")
        result = handler(system, user) if callable(handler) else handler
        return json.dumps(result)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_PROVIDERS = {
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_chat_model(
    config: Dict[str, Any],
    mock_responder: Optional[Dict[str, Any]] = None,
) -> ChatModel:
    """Build the configured provider. ``mock_responder`` (or provider==mock) -> MockProvider."""
    agents_cfg = dict(config.get("agents", {}))
    provider = str(agents_cfg.get("provider", "gemini")).lower()
    if mock_responder is not None or provider == "mock":
        return MockProvider(agents_cfg, mock_responder or {})
    if provider not in _PROVIDERS:
        raise StructuredLLMError(
            f"Unknown agents.provider {provider!r}. Options: {sorted(_PROVIDERS) + ['mock']}."
        )
    return _PROVIDERS[provider](agents_cfg)
