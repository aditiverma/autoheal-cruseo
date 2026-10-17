"""Crusoe Managed Inference client.

Thin wrapper over the OpenAI-compatible Crusoe endpoint. Its one real job is
`complete_json`: send a system+user prompt, get back an instance of a Pydantic
schema. All agent reasoning in AutoHeal flows through here, so the whole
recursive decision loop runs on Crusoe-hosted Nemotron inference.
"""
from __future__ import annotations

import json
import os
import re
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_BASE_URL = "https://api.inference.crusoecloud.com/v1"


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of a model response.

    Nemotron reasoning models often wrap JSON in prose or ```json fences, or
    emit <think>...</think> first. Be tolerant.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    # fall back to first balanced-looking {...}
    start = text.find("{")
    if start == -1:
        raise LLMError(f"No JSON object found in response: {text[:300]!r}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise LLMError(f"Unbalanced JSON in response: {text[:300]!r}")


class CrusoeLLM:
    def __init__(
        self,
        model: str | None = None,
        safety_model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.1,
    ):
        from openai import OpenAI  # local import so --offline works without key

        self.model = model or os.environ.get("CRUSOE_MODEL", "")
        self.safety_model = (
            safety_model or os.environ.get("CRUSOE_SAFETY_MODEL") or self.model
        )
        self.base_url = base_url or os.environ.get("CRUSOE_BASE_URL", DEFAULT_BASE_URL)
        self.temperature = temperature
        key = api_key or os.environ.get("CRUSOE_API_KEY")
        if not key:
            raise LLMError("CRUSOE_API_KEY is not set (see .env.example).")
        self.client = OpenAI(base_url=self.base_url, api_key=key)
        self.name = "crusoe"

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list().data)

    def _chat(self, model: str, system: str, user: str) -> str:
        import time

        from openai import APIConnectionError, RateLimitError

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        delay = 4.0
        last_err: Exception | None = None
        for attempt in range(5):  # bounded retry/backoff for transient throttling
            try:
                resp = self.client.chat.completions.create(
                    model=model, temperature=self.temperature, messages=messages,
                )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APIConnectionError) as e:
                last_err = e
                if attempt == 4:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise LLMError(f"Crusoe inference failed after retries: {last_err}")

    def complete_json(
        self, *, system: str, user: str, schema: Type[T], role: str = "reason"
    ) -> T:
        if not self.model:
            raise LLMError("CRUSOE_MODEL is not set; run `python main.py models` "
                           "and set it in .env.")
        model = self.safety_model if role == "safety" else self.model
        instruction = (
            "\n\nRespond with ONLY a single JSON object matching this schema. "
            "No prose, no markdown fences.\nJSON schema:\n"
            + json.dumps(schema.model_json_schema())
        )
        raw = self._chat(model, system, user + instruction)
        try:
            return schema.model_validate_json(_extract_json(raw))
        except Exception as first_err:  # one repair attempt
            repair = (
                f"Your previous reply could not be parsed into the schema "
                f"({first_err}). Previous reply:\n{raw}\n\n"
                "Return ONLY corrected JSON for the schema."
            )
            raw2 = self._chat(model, system, user + instruction + "\n\n" + repair)
            return schema.model_validate_json(_extract_json(raw2))
