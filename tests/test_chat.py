import httpx
from fastapi.testclient import TestClient

from api.index import app


class FakeAsyncClient:
    def __init__(self, response: httpx.Response):
        self.response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def post(self, *args, **kwargs):
        return self.response


client = TestClient(app)


def test_chat_returns_llm_reply(monkeypatch):
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Hello from the LLM"}}]},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "api.index.httpx.AsyncClient",
        lambda **kwargs: FakeAsyncClient(response),
    )

    result = client.post("/api/chat", json={"message": "Hello"})

    assert result.status_code == 200
    assert result.json() == {"reply": "Hello from the LLM"}


def test_chat_sends_expected_openrouter_request(monkeypatch):
    captured = {}
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "Reply"}}]},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )

    class CapturingAsyncClient(FakeAsyncClient):
        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return self.response

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "api.index.httpx.AsyncClient",
        lambda **kwargs: CapturingAsyncClient(response),
    )

    result = client.post("/api/chat", json={"message": "What is walking skeleton?"})

    assert result.status_code == 200
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["kwargs"]["headers"] == {"Authorization": "Bearer test-key"}
    assert captured["kwargs"]["json"] == {
        "model": "openrouter/free",
        "messages": [
            {"role": "user", "content": "What is walking skeleton?"},
        ],
    }


def test_chat_requires_message(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    result = client.post("/api/chat", json={})

    assert result.status_code == 422


def test_chat_returns_error_when_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = client.post("/api/chat", json={"message": "Hello"})

    assert result.status_code == 500
    assert result.json() == {"detail": "OPENROUTER_API_KEY is not configured"}
