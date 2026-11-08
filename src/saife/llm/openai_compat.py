"""Verifier client for any OpenAI-compatible endpoint.

Default target: ``gemini-3.1-flash-lite`` through an OpenAI-compatible gateway.
Gateways differ in what structured output they pass through, so three modes
are supported, strongest first:

- ``json_schema``: ``response_format={"type": "json_schema", ...}``
- ``json_object``: ``response_format={"type": "json_object"}`` + schema in the prompt
- ``text``: schema in the prompt, JSON extracted from the reply

Whatever the mode, output is validated with pydantic. One repair retry is allowed;
after that the call raises ``LLMOutputError`` (which detectors treat as UNVERIFIABLE).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from typing import Any, NoReturn, get_origin

import httpx2
import openai
from pydantic import BaseModel, ValidationError

from saife.config import STRUCTURED_MODES, VERIFIER, StructuredMode
from saife.errors import (
    LLMError,
    LLMOutputError,
    LLMRefusalError,
    LLMTruncatedError,
    LLMUnavailableError,
)
from saife.llm.base import Message, ParsedResult, StopReason, T, TextResult, Usage

log = logging.getLogger(__name__)

_FINISH_TO_STOP: dict[str, StopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
}

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_ARRAY_FENCE = re.compile(r"```(?:json)?\s*(\[.*\])\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Pull a JSON object out of free text: fenced block first, else outermost braces."""
    match = _FENCE.search(text)
    if match:
        return match.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text.strip()


def _wrap_bare_list(schema: type[BaseModel], text: str) -> str | None:
    """Some models answer ``{"claims": [...]}`` with just the bare ``[...]``. When the
    schema is a single list field, wrap the array under that field; pydantic still
    validates every item. Anything else returns None."""
    fields = schema.model_fields
    if len(fields) != 1:
        return None
    ((name, info),) = fields.items()
    if get_origin(info.annotation) is not list:
        return None
    stripped = text.strip()
    if fenced := _ARRAY_FENCE.fullmatch(stripped):
        stripped = fenced.group(1)
    if not stripped.startswith("["):
        return None
    try:
        data = json.loads(stripped)
    except ValueError:
        return None
    return json.dumps({name: data}) if isinstance(data, list) else None


def _schema_instruction(schema_json: dict[str, Any]) -> str:
    return (
        "Reply with a single JSON object and nothing else. It must validate against "
        "this JSON Schema:\n" + json.dumps(schema_json, indent=2)
    )


class OpenAICompatClient:
    provider = VERIFIER

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        structured_mode: StructuredMode = "json_schema",
        timeout: float = 30.0,
        max_retries: int = 2,
        http_client: httpx2.AsyncClient | None = None,
        provider: str = VERIFIER,
        extra_body: Mapping[str, Any] | None = None,
        structured_extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        """``provider`` names this endpoint in routing, errors and traces. ``extra_body``
        is merged into every request and ``structured_extra_body`` into requests that
        carry a ``response_format`` (e.g. OpenRouter's ``reasoning`` and ``provider``
        preferences)."""
        if structured_mode not in STRUCTURED_MODES:
            raise ValueError(f"unknown structured mode {structured_mode!r}")
        self.provider = provider
        self.structured_mode: StructuredMode = structured_mode
        self._extra_body = dict(extra_body or {})
        self._structured_extra_body = dict(structured_extra_body or {})
        self.last_attempts = 0  # model calls made by the most recent parse() (probe telemetry)
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = openai.AsyncOpenAI(**kwargs)

    # -- low level ---------------------------------------------------------

    def _map_error(self, exc: openai.APIError, model: str) -> NoReturn:
        if isinstance(exc, openai.BadRequestError):
            msg = f"bad request: {exc.message}"
            raise LLMError(msg, provider=self.provider, model=model) from exc
        raise LLMUnavailableError(
            f"{type(exc).__name__}: {getattr(exc, 'message', exc)}",
            provider=self.provider,
            model=model,
        ) from exc

    async def _create(
        self,
        *,
        model: str,
        messages: list[Message],
        max_tokens: int,
        temperature: float | None,
        response_format: dict[str, Any] | None,
    ) -> tuple[str, StopReason, Usage, str, str | None]:
        params: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            params["temperature"] = temperature
        if response_format is not None:
            params["response_format"] = response_format
        extra = {**self._extra_body, **(self._structured_extra_body if response_format else {})}
        if "deepseek" in model.lower():
            # DeepSeek (the generator) thinks first by default; calls here have a 30 s timeout.
            extra.setdefault("reasoning_effort", "none")
        if extra:
            params["extra_body"] = extra
        try:
            response: Any = await self._client.chat.completions.create(**params)
        except openai.APIError as exc:
            self._map_error(exc, model)

        if not response.choices:
            raise LLMOutputError("response had no choices", provider=self.provider, model=model)
        choice = response.choices[0]
        stop = _FINISH_TO_STOP.get(choice.finish_reason or "", "other")
        if stop == "refusal":
            raise LLMRefusalError("content filtered", provider=self.provider, model=model)
        if stop == "max_tokens":
            raise LLMTruncatedError("output hit max_tokens", provider=self.provider, model=model)
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise LLMRefusalError(f"model refused: {refusal}", provider=self.provider, model=model)

        u = response.usage
        usage = Usage(
            input_tokens=(getattr(u, "prompt_tokens", 0) or 0) if u else 0,
            output_tokens=(getattr(u, "completion_tokens", 0) or 0) if u else 0,
        )
        return (
            choice.message.content or "",
            stop,
            usage,
            response.model or model,
            getattr(response, "_request_id", None),
        )

    @staticmethod
    def _validate(schema: type[T], text: str) -> T:
        try:
            return schema.model_validate_json(text)
        except ValidationError:
            extracted = extract_json(text)
            for candidate in (text, extracted):
                wrapped = _wrap_bare_list(schema, candidate)
                if wrapped is not None:
                    return schema.model_validate_json(wrapped)
            if extracted == text:
                raise
            return schema.model_validate_json(extracted)

    # -- public ------------------------------------------------------------

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        max_tokens: int = 4096,
        temperature: float | None = None,
    ) -> TextResult:
        text, stop, usage, served, req = await self._create(
            model=model,
            messages=[{"role": "system", "content": system}, *messages],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=None,
        )
        return TextResult(
            text=text,
            stop_reason=stop,
            usage=usage,
            provider=self.provider,
            model=served,
            request_id=req,
        )

    async def parse(
        self,
        *,
        model: str,
        system: str,
        messages: list[Message],
        schema: type[T],
        max_tokens: int = 16000,
        temperature: float | None = None,
    ) -> ParsedResult[T]:
        schema_json = schema.model_json_schema()
        total = Usage()
        convo = list(messages)
        last_error = ""
        repairs_left = 1
        self.last_attempts = 0

        while True:
            mode = self.structured_mode
            sys_prompt = (
                system
                if mode == "json_schema"
                else (f"{system}\n\n{_schema_instruction(schema_json)}")
            )
            response_format: dict[str, Any] | None = None
            if mode == "json_schema":
                # strict=False: gateways vary in which schema features they accept for
                # strict mode. Pydantic validates the result either way.
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema_json,
                        "strict": False,
                    },
                }
            elif mode == "json_object":
                response_format = {"type": "json_object"}

            self.last_attempts += 1
            try:
                text, stop, usage, served, req = await self._create(
                    model=model,
                    messages=[{"role": "system", "content": sys_prompt}, *convo],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    response_format=response_format,
                )
            except LLMError as exc:
                # A 400 here usually means the gateway rejects this response_format:
                # degrade one level (sticky for this client) and retry. This doesn't
                # use up the repair retry.
                is_bad_request = type(exc) is LLMError
                if is_bad_request and mode != "text":
                    self.structured_mode = STRUCTURED_MODES[STRUCTURED_MODES.index(mode) + 1]
                    log.warning(
                        "verifier rejected %s; downgrading to %s", mode, self.structured_mode
                    )
                    continue
                raise

            total = Usage(
                input_tokens=total.input_tokens + usage.input_tokens,
                output_tokens=total.output_tokens + usage.output_tokens,
            )
            try:
                parsed = self._validate(schema, text)
            except ValidationError as exc:
                last_error = str(exc)[:800]
                if repairs_left == 0:
                    break
                repairs_left -= 1
                convo = [
                    *convo,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply did not validate against the required JSON "
                            f"schema:\n{last_error}\nReply with only the corrected JSON object."
                        ),
                    },
                ]
                continue
            return ParsedResult[schema](  # type: ignore[valid-type]
                parsed=parsed,
                raw_text=text,
                stop_reason=stop,
                usage=total,
                provider=self.provider,
                model=served,
                request_id=req,
            )

        raise LLMOutputError(
            f"output failed schema validation: {last_error}", provider=self.provider, model=model
        )
