import json

import httpx
import pytest

from onecard.backend.ollama import OllamaConsumer
from onecard.errors import BackendError

PS_TWO = {
    "models": [
        {"name": "llama3.1:8b", "size_vram": 5_000_000_000},
        {"name": "qwen2.5:1.5b", "size_vram": 900_000_000},
    ]
}


def consumer(handler) -> OllamaConsumer:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://ollama:11434"
    )
    return OllamaConsumer(base_url="http://ollama:11434", client=client)


async def test_residents_reports_vram_in_mib():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(200, json=PS_TWO)

    residents = await consumer(handler).residents()
    assert {r.key for r in residents} == {"llama3.1:8b", "qwen2.5:1.5b"}
    big = next(r for r in residents if r.key == "llama3.1:8b")
    assert big.footprint_mb == 4768  # 5e9 bytes // 1024^2
    assert big.consumer == "ollama"


async def test_load_sends_keep_alive_minus_one_when_pinned():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"done": True})
        return httpx.Response(200, json=PS_TWO)

    await consumer(handler).load("llama3.1:8b", pinned=True, footprint_hint_mb=None)
    assert seen["keep_alive"] == -1
    assert seen["model"] == "llama3.1:8b"


async def test_release_sends_keep_alive_zero():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            seen.update(json.loads(request.content))
        return httpx.Response(200, json={"done": True})

    await consumer(handler).release("llama3.1:8b")
    assert seen["keep_alive"] == 0


async def test_release_all_releases_every_resident_model():
    released: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json=PS_TWO)
        released.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={"done": True})

    await consumer(handler).release_all()
    assert sorted(released) == ["llama3.1:8b", "qwen2.5:1.5b"]


async def test_unreachable_backend_raises_backend_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(BackendError, match="unreachable"):
        await consumer(handler).residents()


async def test_http_error_raises_backend_error_with_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(BackendError, match="500"):
        await consumer(handler).residents()


async def test_load_that_does_not_become_resident_raises():
    """Never report success for a load the backend did not actually perform."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={"done": True})
        return httpx.Response(200, json={"models": []})

    with pytest.raises(BackendError, match="does not report it as resident"):
        await consumer(handler).load("llama3.1:8b", pinned=False, footprint_hint_mb=None)


async def test_chat_streams_content_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            '{"message":{"content":"Hel"},"done":false}\n'
            '{"message":{"content":"lo"},"done":false}\n'
            '{"message":{"content":""},"done":true}\n'
        )
        return httpx.Response(200, text=body)

    chunks = [c async for c in consumer(handler).chat("m", "hi", {})]
    assert "".join(chunks) == "Hello"


async def test_residents_raises_backend_error_on_non_json_body():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(200, text="<html><body>502 Bad Gateway</body></html>")

    with pytest.raises(BackendError, match="not valid JSON|non-JSON"):
        await consumer(handler).residents()


async def test_chat_raises_backend_error_on_malformed_stream_line():
    def handler(request: httpx.Request) -> httpx.Response:
        body = '{"message":{"content":"Hel"},"done":false}\n' "not json at all\n"
        return httpx.Response(200, text=body)

    with pytest.raises(BackendError, match="not valid JSON|non-JSON"):
        _ = [c async for c in consumer(handler).chat("m", "hi", {})]
