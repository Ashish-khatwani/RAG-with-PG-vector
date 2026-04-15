from __future__ import annotations

import abc
import json
from collections.abc import AsyncIterator

import httpx

from app.core.exceptions import LLMServiceError, ValidationError


class LLMProvider(abc.ABC):
    @abc.abstractmethod
    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        raise NotImplementedError

    @abc.abstractmethod
    async def healthcheck(self) -> str:
        raise NotImplementedError


class OllamaProvider(LLMProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    async def generate(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": False},
            )
        if response.status_code >= 400:
            raise LLMServiceError(f"Ollama returned {response.status_code}: {response.text}")
        payload = response.json()
        return payload.get("response", "").strip()

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/generate",
                json={"model": self._model, "prompt": prompt, "stream": True},
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMServiceError(
                        f"Ollama returned {response.status_code}: {body.decode('utf-8', errors='ignore')}"
                    )
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    token = payload.get("response", "")
                    if token:
                        yield token

    async def healthcheck(self) -> str:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{self._base_url}/api/tags")
        if response.status_code >= 400:
            raise LLMServiceError(f"Ollama healthcheck failed with {response.status_code}")
        return "healthy"


class MistralProvider(LLMProvider):
    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int) -> None:
        if not api_key.strip():
            raise ValidationError("MISTRAL_API_KEY must be configured when llm_provider is mistral.")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers,
                json=payload,
            )
        if response.status_code >= 400:
            raise LLMServiceError(f"Mistral returned {response.status_code}: {response.text}")
        body = response.json()
        return self._extract_message_text(body).strip()

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        headers = {**self._headers, "Accept": "text/event-stream"}
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = await response.aread()
                    raise LLMServiceError(
                        f"Mistral returned {response.status_code}: {body.decode('utf-8', errors='ignore')}"
                    )
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    payload = json.loads(data)
                    token = self._extract_delta_text(payload)
                    if token:
                        yield token

    async def healthcheck(self) -> str:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{self._base_url}/v1/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        if response.status_code >= 400:
            raise LLMServiceError(f"Mistral healthcheck failed with {response.status_code}")
        return "healthy"

    @staticmethod
    def _extract_message_text(payload: dict[str, object]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        content = message.get("content", "")
        return MistralProvider._normalize_content(content)

    @staticmethod
    def _extract_delta_text(payload: dict[str, object]) -> str:
        choices = payload.get("choices", [])
        if not choices:
            return ""
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        return MistralProvider._normalize_content(content)

    @staticmethod
    def _normalize_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, str):
                    fragments.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
            return "".join(fragments)
        return ""
