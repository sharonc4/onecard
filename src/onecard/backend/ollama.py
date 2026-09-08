import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from onecard.errors import BackendError
from onecard.gpu.consumer import DEFAULT_FOOTPRINT_MB, Residency

BYTES_PER_MIB = 1024 * 1024
DEFAULT_TAG = "latest"


def canonical_ref(ref: str) -> str:
    """Ollama's own name for a model ref, so our key space and its key space agree.

    /api/ps always reports fully qualified names ("nomic-embed-text:latest")
    while a config ref is often written bare ("nomic-embed-text"). The chosen
    canonical form is the FULLY QUALIFIED one: a ref with no tag gets ":latest"
    appended. Every key this consumer accepts or reports is canonicalized, so
    the arbiter's confirmed-eviction check compares like with like instead of
    silently finding no match and declaring a still-resident model gone.
    """
    return ref if ":" in ref else f"{ref}:{DEFAULT_TAG}"


class OllamaConsumer:
    """GpuConsumer backed by an Ollama server.

    Residency policy is expressed entirely through `keep_alive`: -1 pins,
    0 evicts immediately, a duration sets a timeout. Ground truth about what
    is loaded always comes from /api/ps, never from our own bookkeeping.
    """

    name = "ollama"

    def __init__(self, base_url: str, client: httpx.AsyncClient) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client

    async def _post(self, path: str, payload: dict[str, Any]) -> httpx.Response:
        try:
            resp = await self._client.post(path, json=payload, timeout=300.0)
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama unreachable at {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"ollama returned {resp.status_code}: {resp.text[:200]}")
        return resp

    def key_for(self, ref: str) -> str:
        return canonical_ref(ref)

    async def residents(self) -> list[Residency]:
        try:
            resp = await self._client.get("/api/ps", timeout=30.0)
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama unreachable at {self.base_url}: {exc}") from exc
        if resp.status_code >= 400:
            raise BackendError(f"ollama returned {resp.status_code}: {resp.text[:200]}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise BackendError(
                f"ollama at {self.base_url} returned a non-JSON response from /api/ps: "
                f"{resp.text[:200]!r}"
            ) from exc
        return [
            Residency(
                consumer=self.name,
                key=canonical_ref(entry["name"]),
                footprint_mb=int(entry.get("size_vram", 0)) // BYTES_PER_MIB,
            )
            for entry in data.get("models", [])
        ]

    async def load(self, key: str, *, pinned: bool, footprint_hint_mb: int | None) -> Residency:
        ref = canonical_ref(key)
        await self._post("/api/generate", {"model": ref, "keep_alive": -1 if pinned else "10m"})
        for r in await self.residents():
            if r.key == ref:
                # A measured footprint always beats an estimate we were handed.
                return Residency(
                    consumer=self.name,
                    key=ref,
                    footprint_mb=r.footprint_mb or (footprint_hint_mb or DEFAULT_FOOTPRINT_MB),
                    pinned=pinned,
                )
        raise BackendError(f"ollama accepted a load of '{ref}' but does not report it as resident")

    async def release(self, key: str) -> None:
        await self._post("/api/generate", {"model": canonical_ref(key), "keep_alive": 0})

    async def release_all(self) -> None:
        for r in await self.residents():
            await self.release(r.key)

    async def chat(self, model_ref: str, prompt: str, params: dict[str, Any]) -> AsyncIterator[str]:
        payload = {
            "model": model_ref,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "options": params,
        }
        try:
            async with self._client.stream(
                "POST", "/api/chat", json=payload, timeout=600.0
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode()[:200]
                    raise BackendError(f"ollama returned {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise BackendError(
                            f"ollama at {self.base_url} streamed a non-JSON line from "
                            f"/api/chat: {line[:200]!r}"
                        ) from exc
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        yield content
        except httpx.HTTPError as exc:
            raise BackendError(f"ollama unreachable at {self.base_url}: {exc}") from exc
