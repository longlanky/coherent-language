"""Tests for the provider-agnostic cleanup layer.

All HTTP is mocked: these tests must never touch a real provider.
"""

import json
import os

import pytest

import llm_cleanup
from llm_cleanup import LLMConfig, LLMConfigError, LLMUpstreamError


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("LLM_", "OLLAMA_", "DEEPSEEK_", "ANTHROPIC_", "OPENAI_")):
            monkeypatch.delenv(name, raising=False)
    llm_cleanup.clear_model_cache()
    yield
    llm_cleanup.clear_model_cache()


def _fake_request(captured, response):
    def fake_request(method, url, **kwargs):
        captured.update(method=method, url=url, **kwargs)
        return response
    return fake_request


# --- configuration ---------------------------------------------------------

def test_ollama_defaults_to_loopback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    cfg = llm_cleanup.resolve_config("ollama", "llama3")
    assert cfg.provider == "ollama"
    assert cfg.base_url == "http://127.0.0.1:11434"
    assert cfg.model == "llama3"


def test_remote_ollama_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://10.200.100.5:11434/")
    cfg = llm_cleanup.resolve_config("ollama", "gemma4:e2b")
    assert cfg.base_url == "http://10.200.100.5:11434"


def test_deepseek_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    cfg = llm_cleanup.resolve_config("deepseek", "deepseek-flash")
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.api_key == "secret"


def test_anthropic_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    cfg = llm_cleanup.resolve_config("anthropic", "claude-sonnet-4-5")
    assert cfg.base_url == "https://api.anthropic.com"


def test_openai_requires_base_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "openai")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(LLMConfigError, match="base URL"):
        llm_cleanup.resolve_config("openai", "some-model")

    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234")
    cfg = llm_cleanup.resolve_config("openai", "some-model")
    assert cfg.base_url == "http://localhost:1234"
    assert cfg.api_key is None


def test_generic_base_url_and_key_apply_to_default_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "deepseek")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "generic-secret")
    cfg = llm_cleanup.resolve_config("deepseek", "deepseek-flash")
    assert cfg.base_url == "https://proxy.example/v1"
    assert cfg.api_key == "generic-secret"


def test_llm_model_applies_to_the_selected_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama,deepseek")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("LLM_MODEL", "deepseek-flash")
    cfg = llm_cleanup.resolve_config("deepseek")
    assert cfg.model == "deepseek-flash"


def test_provider_must_be_enabled(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    with pytest.raises(LLMConfigError, match="not enabled"):
        llm_cleanup.resolve_config("deepseek", "deepseek-flash")


def test_remote_provider_requires_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "deepseek")
    with pytest.raises(LLMConfigError, match="DEEPSEEK_API_KEY"):
        llm_cleanup.resolve_config("deepseek", "deepseek-flash")


@pytest.mark.parametrize("model", ["bad model", "-leading", "", "x" * 300])
def test_model_id_is_validated(monkeypatch, model):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    monkeypatch.setattr(
        llm_cleanup, "list_models",
        lambda provider=None, refresh=False: {"models": []},
    )
    with pytest.raises(LLMConfigError, match="model"):
        llm_cleanup.resolve_config("ollama", model)


@pytest.mark.parametrize("url", [
    "ftp://example.com",
    "http://user:pass@example.com",
    "https://example.com/v1?query=1",
])
def test_base_url_is_validated(monkeypatch, url):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", url)
    with pytest.raises(LLMConfigError):
        llm_cleanup.resolve_config("ollama", "llama3")


def test_ollama_model_falls_back_to_queried_list(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    monkeypatch.setattr(
        llm_cleanup, "list_models",
        lambda provider=None, refresh=False: {
            "provider": "ollama", "base_url": "http://127.0.0.1:11434",
            "models": [{"id": "first-model", "label": "first-model"}],
            "cached": False,
        },
    )
    cfg = llm_cleanup.resolve_config("ollama")
    assert cfg.model == "first-model"


def test_available_providers_never_exposes_the_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "super-secret")
    body = llm_cleanup.available_providers()
    assert body["providers"][0]["api_key_configured"] is True
    assert "super-secret" not in json.dumps(body)


# --- model discovery -------------------------------------------------------

def test_list_ollama_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    captured = {}
    monkeypatch.setattr(
        llm_cleanup.requests, "request",
        _fake_request(captured, FakeResponse({
            "models": [{"name": "llama3"}, {"name": "gemma4:e2b"}],
        })),
    )
    out = llm_cleanup.list_models("ollama")
    assert captured["method"] == "GET"
    assert captured["url"] == "http://127.0.0.1:11434/api/tags"
    assert [m["id"] for m in out["models"]] == ["gemma4:e2b", "llama3"]
    assert out["cached"] is False


def test_list_deepseek_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    captured = {}
    monkeypatch.setattr(
        llm_cleanup.requests, "request",
        _fake_request(captured, FakeResponse({
            "data": [
                {"id": "deepseek-v4-pro", "name": "DeepSeek-V4-Pro"},
                {"id": "deepseek-flash", "name": "DeepSeek-V4.1-Flash"},
            ],
        })),
    )
    out = llm_cleanup.list_models("deepseek")
    assert captured["url"] == "https://api.deepseek.com/models"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert [m["id"] for m in out["models"]] == ["deepseek-flash", "deepseek-v4-pro"]


def test_list_anthropic_models_paginates(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    pages = [
        FakeResponse({
            "data": [{"id": "claude-sonnet-4-5", "display_name": "Claude Sonnet 4.5"}],
            "has_more": True,
            "last_id": "claude-sonnet-4-5",
        }),
        FakeResponse({
            "data": [{"id": "claude-haiku-4-5", "display_name": "Claude Haiku 4.5"}],
            "has_more": False,
        }),
    ]
    urls = []

    def fake_request(method, url, **kwargs):
        urls.append(url)
        assert kwargs["headers"]["x-api-key"] == "secret"
        assert kwargs["headers"]["anthropic-version"] == "2023-06-01"
        return pages.pop(0)

    monkeypatch.setattr(llm_cleanup.requests, "request", fake_request)
    out = llm_cleanup.list_models("anthropic")
    assert urls[0].startswith("https://api.anthropic.com/v1/models?limit=1000")
    assert "after_id=claude-sonnet-4-5" in urls[1]
    assert [m["id"] for m in out["models"]] == ["claude-haiku-4-5", "claude-sonnet-4-5"]


def test_custom_openai_base_appends_v1(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "openai")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:1234")
    captured = {}
    monkeypatch.setattr(
        llm_cleanup.requests, "request",
        _fake_request(captured, FakeResponse({"data": [{"id": "local-model"}]})),
    )
    llm_cleanup.list_models("openai")
    assert captured["url"] == "http://localhost:1234/v1/models"
    assert "Authorization" not in captured["headers"]


def test_model_list_is_cached_until_refresh(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDERS", "ollama")
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(url)
        return FakeResponse({"models": [{"name": "llama3"}]})

    monkeypatch.setattr(llm_cleanup.requests, "request", fake_request)
    llm_cleanup.list_models("ollama")
    llm_cleanup.list_models("ollama")
    assert len(calls) == 1
    llm_cleanup.list_models("ollama", refresh=True)
    assert len(calls) == 2


# --- chat request shapes ---------------------------------------------------

def test_ollama_chat_request_shape(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        llm_cleanup.requests, "request",
        _fake_request(captured, FakeResponse({
            "message": {"content": '{"cleaned_text": "ok"}'},
        })),
    )
    cfg = LLMConfig("ollama", "http://127.0.0.1:11434", "gemma4:e2b", None,
                    timeout=5, unload_ollama=False)
    content = llm_cleanup.chat(cfg, [{"role": "user", "content": "hi"}])
    assert captured["url"] == "http://127.0.0.1:11434/api/chat"
    assert captured["json"]["format"] == "json"
    assert captured["json"]["stream"] is False
    assert captured["json"]["options"]["num_predict"] == -1
    assert json.loads(content)["cleaned_text"] == "ok"


def test_deepseek_chat_request_shape(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        llm_cleanup.requests, "request",
        _fake_request(captured, FakeResponse({
            "choices": [{"message": {"content": '{"cleaned_text": "ok"}'},
                         "finish_reason": "stop"}],
        })),
    )
    cfg = LLMConfig("deepseek", "https://api.deepseek.com", "deepseek-flash",
                    "secret", timeout=5, unload_ollama=False)
    llm_cleanup.chat(cfg, [{"role": "user", "content": "hi"}])
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_openai_retries_without_response_format(monkeypatch):
    responses = [
        FakeResponse({"error": {"message": "response_format unsupported"}}, 400),
        FakeResponse({"choices": [{"message": {"content": "{}"},
                                   "finish_reason": "stop"}]}),
    ]
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(llm_cleanup.requests, "request", fake_request)
    cfg = LLMConfig("openai", "http://localhost:1234/v1", "local", None,
                    timeout=5, unload_ollama=False)
    llm_cleanup.chat(cfg, [{"role": "user", "content": "hi"}])
    assert len(calls) == 2
    assert "response_format" not in calls[1]["json"]


def test_anthropic_chat_request_shape(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        llm_cleanup.requests, "request",
        _fake_request(captured, FakeResponse({
            "content": [{"type": "text", "text": '{"cleaned_text": "ok"}'}],
            "stop_reason": "end_turn",
        })),
    )
    cfg = LLMConfig("anthropic", "https://api.anthropic.com", "claude-sonnet-4-5",
                    "secret", timeout=5, unload_ollama=False)
    content = llm_cleanup.chat(cfg, [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ])
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "secret"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert captured["json"]["system"] == "sys"
    assert captured["json"]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["json"]["max_tokens"] == 8192
    assert json.loads(content)["cleaned_text"] == "ok"


def test_anthropic_truncation_is_an_error(monkeypatch):
    monkeypatch.setattr(
        llm_cleanup.requests, "request",
        _fake_request({}, FakeResponse({
            "content": [{"type": "text", "text": "{}"}],
            "stop_reason": "max_tokens",
        })),
    )
    cfg = LLMConfig("anthropic", "https://api.anthropic.com", "claude-sonnet-4-5",
                    "secret", timeout=5, unload_ollama=False)
    with pytest.raises(LLMUpstreamError, match="truncated"):
        llm_cleanup.chat(cfg, [{"role": "user", "content": "hi"}])


def test_redirects_are_refused(monkeypatch):
    monkeypatch.setattr(
        llm_cleanup.requests, "request",
        _fake_request({}, FakeResponse({"error": "moved"}, 302)),
    )
    cfg = LLMConfig("deepseek", "https://api.deepseek.com", "deepseek-flash",
                    "secret", timeout=5, unload_ollama=False)
    with pytest.raises(LLMUpstreamError, match="redirect"):
        llm_cleanup.chat(cfg, [{"role": "user", "content": "hi"}])


# --- cleanup orchestration -------------------------------------------------

def _cfg(provider="deepseek"):
    return LLMConfig(provider, "https://example.invalid", "model", "secret",
                     timeout=5, unload_ollama=False)


def test_clean_transcript_success(monkeypatch):
    monkeypatch.setattr(
        llm_cleanup, "chat",
        lambda config, messages: json.dumps({
            "cleaned_text": "Clean text.",
            "changes": [{"type": "capitalization"}],
            "warnings": [],
        }),
    )
    result = llm_cleanup.clean_transcript("raw text", None, _cfg())
    assert result["cleaned_text"] == "Clean text."
    assert result["changes"] == [{"type": "capitalization"}]
    assert result["warnings"] == []
    assert result["llm"] == {"provider": "deepseek", "model": "model"}


def test_clean_transcript_translates_in_a_second_pass(monkeypatch):
    calls = []

    def fake_chat(config, messages):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps({"cleaned_text": "Cleaned.", "changes": [],
                               "warnings": []})
        return json.dumps({"translation": "Traduit."})

    monkeypatch.setattr(llm_cleanup, "chat", fake_chat)
    result = llm_cleanup.clean_transcript("raw", "French", _cfg())
    assert len(calls) == 2
    assert result["cleaned_text"] == "Traduit."
    assert result["translation"] == {"target": "French"}


def test_clean_transcript_retries_invalid_json(monkeypatch):
    responses = iter([
        "not json at all",
        json.dumps({"cleaned_text": "ok", "changes": [], "warnings": []}),
    ])
    monkeypatch.setattr(llm_cleanup, "chat",
                        lambda config, messages: next(responses))
    result = llm_cleanup.clean_transcript("raw", None, _cfg())
    assert result["cleaned_text"] == "ok"


def test_clean_transcript_returns_error(monkeypatch):
    def boom(config, messages):
        raise llm_cleanup.LLMUpstreamError("provider is down")

    monkeypatch.setattr(llm_cleanup, "chat", boom)
    result = llm_cleanup.clean_transcript("raw", None, _cfg())
    assert "provider is down" in result["error"]


def test_clean_transcript_unloads_ollama(monkeypatch):
    cfg = LLMConfig("ollama", "http://127.0.0.1:11434", "gemma4:e2b", None,
                    timeout=5, unload_ollama=True)
    called = []
    monkeypatch.setattr(
        llm_cleanup, "chat",
        lambda config, messages: json.dumps({
            "cleaned_text": "ok", "changes": [], "warnings": [],
        }),
    )
    monkeypatch.setattr(llm_cleanup, "_unload_ollama", lambda c: called.append(c))
    llm_cleanup.clean_transcript("raw", None, cfg)
    assert called == [cfg]


def test_parse_json_tolerates_fences_and_prose():
    parsed = llm_cleanup._parse_json('Here: ```json\n{"cleaned_text": "ok"}\n``` done')
    assert parsed["cleaned_text"] == "ok"
