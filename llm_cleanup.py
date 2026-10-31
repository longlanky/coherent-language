"""Provider-agnostic LLM cleanup for ASR transcripts.

Supported providers:
  - ollama     local (loopback by default) or a remote Ollama instance
  - deepseek   https://api.deepseek.com (OpenAI-compatible)
  - anthropic  https://api.anthropic.com (native Messages API)
  - openai     any OpenAI-compatible endpoint (OpenAI, vLLM, LM Studio, …)

Configuration is read from environment variables. API keys stay on the
server: request bodies may select a provider and model, but never carry a
key. See README.md for the full environment table.

The module is also usable directly:

    python llm_cleanup.py transcript.txt --provider deepseek --model deepseek-flash
    python llm_cleanup.py --list-models deepseek
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import requests

log = logging.getLogger("cohere-asr.llm")

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "llm-post-processing-system-prompt.txt"), encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read()
with open(os.path.join(_HERE, "llm-post-processing-user-prompt.txt"), encoding="utf-8") as _f:
    USER_TEMPLATE = _f.read()

TRANSLATION_SYSTEM = (
    "You are a professional translator. Translate the user's text "
    "into {target}. Translate faithfully and completely — do "
    "not summarize, condense, or omit anything, and do not add "
    "commentary. Return ONLY a JSON object of the form "
    '{{"translation": "..."}}.'
)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# Wall-clock budget for one LLM cleanup call. ASR_OLLAMA_TIMEOUT is the old
# name and remains supported for existing deployments.
DEFAULT_TIMEOUT = _env_float("LLM_TIMEOUT", _env_float("ASR_OLLAMA_TIMEOUT", 600))
MODELS_TIMEOUT = _env_float("LLM_MODELS_TIMEOUT", 15)
MODEL_CACHE_TTL = _env_float("LLM_MODELS_TTL", 60)
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
ANTHROPIC_DEFAULT_MAX_TOKENS = 8192

_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+~-]{0,199}$")


class LLMError(Exception):
    """Base class for cleanup configuration and upstream failures."""


class LLMConfigError(LLMError):
    """The requested provider/model/endpoint is not usable."""


class LLMUpstreamError(LLMError):
    """The provider could not be reached or returned an error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    base_url_envs: tuple[str, ...]
    api_key_envs: tuple[str, ...]
    default_base_url: str | None
    models_path: str
    chat_path: str
    style: str  # "ollama" | "openai" | "anthropic"
    requires_key: bool = False
    ensure_v1: bool = False


PROVIDERS: dict[str, ProviderSpec] = {
    "ollama": ProviderSpec(
        id="ollama",
        label="Ollama",
        base_url_envs=("OLLAMA_HOST",),
        api_key_envs=("OLLAMA_API_KEY",),
        default_base_url="http://127.0.0.1:11434",
        models_path="/api/tags",
        chat_path="/api/chat",
        style="ollama",
    ),
    "deepseek": ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        base_url_envs=("DEEPSEEK_BASE_URL",),
        api_key_envs=("DEEPSEEK_API_KEY",),
        default_base_url="https://api.deepseek.com",
        models_path="/models",
        chat_path="/chat/completions",
        style="openai",
        requires_key=True,
    ),
    "anthropic": ProviderSpec(
        id="anthropic",
        label="Anthropic",
        base_url_envs=("ANTHROPIC_BASE_URL",),
        api_key_envs=("ANTHROPIC_API_KEY",),
        default_base_url="https://api.anthropic.com",
        models_path="/v1/models",
        chat_path="/v1/messages",
        style="anthropic",
        requires_key=True,
    ),
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI-compatible",
        base_url_envs=("OPENAI_BASE_URL",),
        api_key_envs=("OPENAI_API_KEY",),
        default_base_url=None,
        models_path="/models",
        chat_path="/chat/completions",
        style="openai",
        ensure_v1=True,
    ),
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str | None
    timeout: float = DEFAULT_TIMEOUT
    max_tokens: int | None = None
    unload_ollama: bool = True


# provider id -> (timestamp, model list)
_model_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}


def clear_model_cache():
    """Drop cached model lists (used by tests and the refresh button)."""
    _model_cache.clear()


# --- configuration ---------------------------------------------------------

def _env_api_key(provider: str) -> str | None:
    """Provider-specific key only; used during provider auto-discovery."""
    for env in PROVIDERS[provider].api_key_envs:
        value = os.environ.get(env)
        if value:
            return value.strip()
    return None


def enabled_provider_ids() -> list[str]:
    """Providers exposed by this server.

    LLM_PROVIDERS wins when set. Otherwise Ollama is always available and
    remote providers are enabled automatically when their key (or, for the
    generic OpenAI-compatible provider, its base URL) is configured.
    """
    raw = os.environ.get("LLM_PROVIDERS", "").strip()
    if raw:
        ids: list[str] = []
        for part in raw.split(","):
            pid = part.strip().lower()
            if not pid:
                continue
            if pid not in PROVIDERS:
                raise LLMConfigError(
                    f"Unknown LLM provider '{pid}'. Available: {sorted(PROVIDERS)}"
                )
            if pid not in ids:
                ids.append(pid)
        if not ids:
            raise LLMConfigError("LLM_PROVIDERS is set but empty")
    else:
        ids = ["ollama"]
        for pid in ("deepseek", "anthropic", "openai"):
            if pid in ids:
                continue
            if _env_api_key(pid) or (pid == "openai" and os.environ.get("OPENAI_BASE_URL")):
                ids.append(pid)

    # A configured default provider is always exposed, even if the operator
    # forgot to add it to LLM_PROVIDERS.
    preferred = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if preferred:
        if preferred not in PROVIDERS:
            raise LLMConfigError(
                f"Unknown LLM_PROVIDER '{preferred}'. Available: {sorted(PROVIDERS)}"
            )
        if preferred not in ids:
            ids.append(preferred)
    return ids


def default_provider_id() -> str:
    preferred = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if preferred:
        if preferred not in PROVIDERS:
            raise LLMConfigError(
                f"Unknown LLM_PROVIDER '{preferred}'. Available: {sorted(PROVIDERS)}"
            )
        return preferred
    return enabled_provider_ids()[0]


def _resolve_provider_id(provider: str | None) -> str:
    pid = (provider or default_provider_id()).strip().lower()
    if pid not in PROVIDERS:
        raise LLMConfigError(
            f"Unknown LLM provider '{pid}'. Available: {sorted(PROVIDERS)}"
        )
    return pid


def _generic_applies_to(provider: str) -> bool:
    """LLM_BASE_URL / LLM_API_KEY target the default or only provider."""
    preferred = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if preferred:
        return provider == preferred
    enabled = enabled_provider_ids()
    return len(enabled) == 1 and enabled[0] == provider


def _normalize_base_url(url: str) -> str:
    url = url.strip()
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise LLMConfigError("LLM base URL must start with http:// or https://")
    if not parts.netloc:
        raise LLMConfigError("LLM base URL is missing a host")
    if parts.username or parts.password:
        raise LLMConfigError("LLM base URL must not contain credentials")
    if parts.query or parts.fragment:
        raise LLMConfigError("LLM base URL must not contain a query or fragment")
    return url.rstrip("/")


def provider_base_url(provider: str) -> str:
    spec = PROVIDERS[provider]
    for env in spec.base_url_envs:
        value = os.environ.get(env)
        if value:
            return _normalize_base_url(value)
    generic = os.environ.get("LLM_BASE_URL")
    if generic and _generic_applies_to(provider):
        return _normalize_base_url(generic)
    if spec.default_base_url:
        return spec.default_base_url
    raise LLMConfigError(
        f"{spec.label} base URL is not configured; set {spec.base_url_envs[0]}"
    )


def provider_api_key(provider: str) -> str | None:
    spec = PROVIDERS[provider]
    for env in spec.api_key_envs:
        value = os.environ.get(env)
        if value:
            return value.strip()
    generic = os.environ.get("LLM_API_KEY")
    if generic and _generic_applies_to(provider):
        return generic.strip()
    return None


def _require_configured(provider: str, api_key: str | None):
    spec = PROVIDERS[provider]
    if spec.requires_key and not api_key:
        key_env = spec.api_key_envs[0]
        raise LLMConfigError(
            f"{spec.label} is not configured; set {key_env} (or LLM_API_KEY)"
        )


def default_model_for(provider: str) -> str | None:
    specific = os.environ.get(f"LLM_MODEL_{provider.upper()}", "").strip()
    if specific:
        return specific
    generic = os.environ.get("LLM_MODEL", "").strip()
    if generic:
        return generic
    if provider == "ollama":
        return os.environ.get("OLLAMA_MODEL", "").strip() or None
    return None


def _configured_max_tokens() -> int | None:
    raw = os.environ.get("LLM_MAX_TOKENS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise LLMConfigError("LLM_MAX_TOKENS must be an integer") from exc
    return value if value > 0 else None


def _validate_model_id(model: str):
    if not _MODEL_RE.fullmatch(model):
        raise LLMConfigError(f"Invalid model id {model!r}")


def resolve_config(provider: str | None = None,
                   model: str | None = None) -> LLMConfig:
    """Resolve request/env settings into one concrete provider config.

    Request arguments win over environment defaults. API keys are read only
    from the environment.
    """
    pid = _resolve_provider_id(provider)
    if pid not in enabled_provider_ids():
        raise LLMConfigError(
            f"Provider '{pid}' is not enabled; add it to LLM_PROVIDERS"
        )
    base_url = provider_base_url(pid)
    api_key = provider_api_key(pid)
    _require_configured(pid, api_key)

    chosen = (model or default_model_for(pid) or "").strip()
    if not chosen and pid == "ollama":
        # Dynamic fallback: use the first model the endpoint advertises.
        listed = list_models(pid)["models"]
        if listed:
            chosen = listed[0]["id"]
            log.info("No Ollama model configured; using %s", chosen)
    if not chosen:
        raise LLMConfigError(
            f"No model selected for {pid}; pass llm_model or set LLM_MODEL "
            "(see /v1/llm/models)"
        )
    _validate_model_id(chosen)

    unload = os.environ.get("LLM_OLLAMA_UNLOAD", "1").strip().lower()
    return LLMConfig(
        provider=pid,
        base_url=base_url,
        model=chosen,
        api_key=api_key,
        timeout=DEFAULT_TIMEOUT,
        max_tokens=_configured_max_tokens(),
        unload_ollama=unload not in ("0", "false", "no", "off"),
    )


def _is_loopback(url: str | None) -> bool:
    if not url:
        return False
    host = (urlsplit(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def available_providers() -> dict:
    """Describe the providers exposed by this server (never the key itself)."""
    default = default_provider_id()
    providers = []
    for pid in enabled_provider_ids():
        spec = PROVIDERS[pid]
        try:
            base_url = _api_base(pid, provider_base_url(pid))
        except LLMConfigError:
            base_url = None
        api_key = provider_api_key(pid)
        providers.append({
            "id": pid,
            "label": spec.label,
            "base_url": base_url,
            "default_model": default_model_for(pid),
            "configured": bool(base_url) and (not spec.requires_key or bool(api_key)),
            "api_key_configured": bool(api_key),
            "requires_key": spec.requires_key,
            "local": _is_loopback(base_url),
        })
    return {"providers": providers, "default_provider": default}


# --- HTTP helpers ----------------------------------------------------------

def _api_base(provider: str, base_url: str) -> str:
    """OpenAI-compatible endpoints usually live under /v1."""
    if PROVIDERS[provider].ensure_v1:
        parts = urlsplit(base_url)
        if parts.path in ("", "/"):
            return base_url + "/v1"
    return base_url


def _join(base_url: str, path: str) -> str:
    base_url = base_url.rstrip("/")
    # Tolerate a configured base that already ends in /v1.
    if base_url.endswith("/v1") and path.startswith("/v1/"):
        return base_url + path[len("/v1"):]
    return base_url + path


def _headers(provider: str, api_key: str | None) -> dict[str, str]:
    spec = PROVIDERS[provider]
    headers = {"Content-Type": "application/json"}
    if api_key:
        if spec.style == "anthropic":
            headers["x-api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    if spec.style == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    return headers


def _error_message(resp) -> str:
    try:
        data = resp.json()
    except Exception:
        return (getattr(resp, "text", "") or "")[:300]
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        message = error.get("message") or error.get("type") or error
    elif isinstance(error, str):
        message = error
    elif isinstance(data, dict):
        message = data.get("message") or data.get("detail") or data
    else:
        message = data
    return str(message)[:300]


def _request(method: str, url: str, *, headers: dict, timeout: float,
             json_body: dict | None = None) -> dict | list:
    try:
        resp = requests.request(
            method, url, headers=headers, timeout=timeout,
            json=json_body, allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise LLMUpstreamError(f"{method} {url} failed: {exc}") from exc
    if 300 <= resp.status_code < 400:
        # Never forward an API key to a redirect target.
        raise LLMUpstreamError(
            f"provider redirected ({resp.status_code}); refusing to follow",
            resp.status_code,
        )
    if not resp.ok:
        raise LLMUpstreamError(
            f"provider returned HTTP {resp.status_code}: {_error_message(resp)}",
            resp.status_code,
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise LLMUpstreamError("provider returned invalid JSON") from exc


# --- model discovery -------------------------------------------------------

def _fetch_models(provider: str, base_url: str, api_key: str | None) -> list[dict]:
    spec = PROVIDERS[provider]
    base_url = _api_base(provider, base_url)
    headers = _headers(provider, api_key)
    models: list[dict] = []

    if spec.style == "ollama":
        data = _request("GET", _join(base_url, spec.models_path),
                        headers=headers, timeout=MODELS_TIMEOUT)
        if isinstance(data, dict):
            items = data.get("models") or data.get("data") or []
        else:
            items = data
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            model_id = item.get("name") or item.get("model") or item.get("id")
            if model_id:
                models.append({"id": model_id, "label": item.get("name") or model_id})

    elif spec.style == "anthropic":
        url = _join(base_url, spec.models_path) + "?limit=1000"
        for _ in range(20):
            data = _request("GET", url, headers=headers, timeout=MODELS_TIMEOUT)
            items = data.get("data") if isinstance(data, dict) else data
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if model_id:
                    models.append({
                        "id": model_id,
                        "label": item.get("display_name") or model_id,
                    })
            if not isinstance(data, dict) or not data.get("has_more"):
                break
            last_id = data.get("last_id")
            if not last_id:
                break
            url = _join(base_url, spec.models_path) + "?" + urlencode({
                "limit": 1000, "after_id": last_id,
            })

    else:  # openai-compatible (deepseek, openai, …)
        data = _request("GET", _join(base_url, spec.models_path),
                        headers=headers, timeout=MODELS_TIMEOUT)
        items = data.get("data") if isinstance(data, dict) else data
        if not isinstance(items, list) and isinstance(data, dict):
            items = data.get("models") or []
        for item in items if isinstance(items, list) else []:
            if isinstance(item, str):
                models.append({"id": item, "label": item})
                continue
            if not isinstance(item, dict):
                continue
            model_id = item.get("id") or item.get("name")
            if model_id:
                models.append({
                    "id": model_id,
                    "label": item.get("name") or item.get("display_name") or model_id,
                })

    # Stable order for the UI and deterministic tests.
    models.sort(key=lambda m: m["id"].lower())
    deduped = {}
    for model in models:
        deduped.setdefault(model["id"], model)
    return list(deduped.values())


def list_models(provider: str | None = None, *, refresh: bool = False) -> dict:
    """Query the provider's model list (cached briefly in-process)."""
    pid = _resolve_provider_id(provider)
    if pid not in enabled_provider_ids():
        raise LLMConfigError(
            f"Provider '{pid}' is not enabled; add it to LLM_PROVIDERS"
        )
    base_url = provider_base_url(pid)
    api_key = provider_api_key(pid)
    _require_configured(pid, api_key)

    cache_key = (pid, base_url)
    now = time.time()
    if not refresh and cache_key in _model_cache:
        cached_at, cached_models = _model_cache[cache_key]
        if now - cached_at < MODEL_CACHE_TTL:
            return {
                "provider": pid,
                "base_url": _api_base(pid, base_url),
                "models": [dict(m) for m in cached_models],
                "cached": True,
            }

    models = _fetch_models(pid, base_url, api_key)
    _model_cache[cache_key] = (now, models)
    return {
        "provider": pid,
        "base_url": _api_base(pid, base_url),
        "models": [dict(m) for m in models],
        "cached": False,
    }


# --- chat + cleanup --------------------------------------------------------

def chat(config: LLMConfig, messages: list[dict]) -> str:
    """Run one JSON-mode chat call and return the raw content string."""
    spec = PROVIDERS[config.provider]
    base_url = _api_base(config.provider, config.base_url)
    headers = _headers(config.provider, config.api_key)

    if spec.style == "ollama":
        options: dict = {"num_predict": -1}
        if config.max_tokens:
            options["num_predict"] = config.max_tokens
        payload = {
            "model": config.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": options,
        }
        data = _request("POST", _join(base_url, spec.chat_path),
                        headers=headers, timeout=config.timeout,
                        json_body=payload)
        content = (data.get("message") or {}).get("content") if isinstance(data, dict) else None
        if not content:
            raise LLMUpstreamError("provider returned an empty response")
        return content

    if spec.style == "anthropic":
        system_parts = [
            str(m.get("content", "")) for m in messages if m.get("role") == "system"
        ]
        convo = [
            {"role": m.get("role"), "content": m.get("content", "")}
            for m in messages if m.get("role") != "system"
        ]
        payload = {
            "model": config.model,
            "messages": convo,
            "max_tokens": config.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        data = _request("POST", _join(base_url, spec.chat_path),
                        headers=headers, timeout=config.timeout,
                        json_body=payload)
        if isinstance(data, dict) and data.get("stop_reason") == "max_tokens":
            raise LLMUpstreamError(
                "response truncated (stop_reason=max_tokens); increase LLM_MAX_TOKENS"
            )
        blocks = data.get("content") if isinstance(data, dict) else None
        content = "".join(
            block.get("text", "") for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ) if isinstance(blocks, list) else ""
        if not content:
            raise LLMUpstreamError("provider returned an empty response")
        return content

    # OpenAI-compatible (DeepSeek, OpenAI, vLLM, LM Studio, …)
    payload = {
        "model": config.model,
        "messages": messages,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    if config.max_tokens:
        payload["max_tokens"] = config.max_tokens
    try:
        data = _request("POST", _join(base_url, spec.chat_path),
                        headers=headers, timeout=config.timeout,
                        json_body=payload)
    except LLMUpstreamError as exc:
        # Some compatible servers do not implement response_format.
        if exc.status_code != 400:
            raise
        log.info("%s rejected response_format; retrying without it", config.provider)
        payload.pop("response_format", None)
        data = _request("POST", _join(base_url, spec.chat_path),
                        headers=headers, timeout=config.timeout,
                        json_body=payload)
    if not isinstance(data, dict):
        raise LLMUpstreamError("provider returned an unexpected response")
    choices = data.get("choices") or []
    if not choices:
        raise LLMUpstreamError("provider returned no choices")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    if choice.get("finish_reason") == "length":
        raise LLMUpstreamError(
            "response truncated (finish_reason=length); increase LLM_MAX_TOKENS"
        )
    message = choice.get("message") or {}
    content = message.get("content") if isinstance(message, dict) else None
    if content is None:
        content = choice.get("text")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        )
    if not content:
        raise LLMUpstreamError("provider returned an empty response")
    return content


def _parse_json(content: str) -> dict:
    """Parse a model response tolerantly.

    Handles stray markdown fences, raw control characters inside strings
    (strict=False), and leading/trailing prose around the JSON object.
    """
    cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        cleaned = cleaned[start:end + 1]
    data = json.loads(cleaned, strict=False)
    if not isinstance(data, dict):
        raise ValueError("unexpected payload (not a JSON object)")
    return data


def _chat_json(config: LLMConfig, messages: list[dict], attempts: int = 3) -> dict:
    """chat + parse with retries: models intermittently emit bad JSON."""
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        content = chat(config, messages)
        try:
            return _parse_json(content)
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            log.warning("LLM returned invalid JSON (attempt %d/%d): %s",
                        attempt, attempts, exc)
            # Never log the payload: it is transcript content (personal data).
            log.warning("invalid LLM payload: len=%d sha256=%s", len(content),
                        hashlib.sha256(content.encode()).hexdigest()[:16])
    assert last_exc is not None
    raise last_exc


def _unload_ollama(config: LLMConfig):
    """Ask Ollama to release the model (keep_alive=0)."""
    if config.provider != "ollama" or not config.unload_ollama:
        return
    try:
        _request(
            "POST", _join(config.base_url, "/api/generate"),
            headers=_headers(config.provider, config.api_key),
            timeout=30,
            json_body={"model": config.model, "keep_alive": 0},
        )
        log.info("Unloaded %s from ollama", config.model)
    except Exception:
        log.warning("Failed to unload %s from ollama", config.model, exc_info=True)


def clean_transcript(raw_text: str, translate_to: str | None = None,
                     config: LLMConfig | None = None) -> dict:
    """Post-process a transcript with the configured LLM.

    Returns {"cleaned_text", "changes", "warnings", "llm"} on success, or
    {"error": ...} on failure. Ollama models are unloaded afterwards when
    configured to do so.

    When translate_to is set (e.g. "French" or "Levantine Arabic") a second
    LLM pass translates the cleaned text — a single combined prompt proved
    unreliable. cleaned_text then holds the translation; changes/warnings come
    from the cleaning pass.
    """
    resolved: LLMConfig | None = None
    try:
        resolved = config or resolve_config()
        log.info("Running LLM cleanup provider=%s model=%s (%d chars, translate_to=%s)",
                 resolved.provider, resolved.model, len(raw_text), translate_to)
        data = _chat_json(resolved, [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.replace("{raw_transcript}", raw_text)},
        ])
        if "cleaned_text" not in data:
            raise ValueError("unexpected cleanup payload (missing cleaned_text)")
        data.setdefault("changes", [])
        data.setdefault("warnings", [])

        if translate_to:
            log.info("Translating cleaned text to %s", translate_to)
            tr_data = _chat_json(resolved, [
                {"role": "system", "content": TRANSLATION_SYSTEM.format(target=translate_to)},
                {"role": "user", "content": data["cleaned_text"]},
            ])
            translated = tr_data.get("translation") or tr_data.get("cleaned_text")
            if not translated:
                raise ValueError("unexpected translation payload (missing translation)")
            data["cleaned_text"] = translated
            data["translation"] = {"target": translate_to}

        data["llm"] = {"provider": resolved.provider, "model": resolved.model}
        log.info("Cleanup done: %d changes, %d warnings",
                 len(data["changes"]), len(data["warnings"]))
        return data
    except Exception as exc:
        log.exception("LLM cleanup failed")
        return {"error": str(exc)}
    finally:
        if resolved is not None:
            _unload_ollama(resolved)


# --- standalone CLI --------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Clean an ASR transcript with a configurable LLM provider.")
    parser.add_argument("input", nargs="?",
                        help="Text file to clean, or '-' to read stdin")
    parser.add_argument("--provider", help="ollama, deepseek, anthropic, or openai")
    parser.add_argument("--model", help="Model id from the provider's model list")
    parser.add_argument("--translate", metavar="LANG",
                        help="Also translate the cleaned text into LANG")
    parser.add_argument("--list-providers", action="store_true",
                        help="Show configured providers and exit")
    parser.add_argument("--list-models", nargs="?", const="", metavar="PROVIDER",
                        help="Query and show the provider's models, then exit")
    parser.add_argument("--json", action="store_true",
                        help="Print the full cleanup result as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.list_providers:
            print(json.dumps(available_providers(), indent=2, ensure_ascii=False))
            return 0
        if args.list_models is not None:
            print(json.dumps(list_models(args.list_models or args.provider or None),
                             indent=2, ensure_ascii=False))
            return 0
        if not args.input:
            parser.error("input is required unless --list-providers/--list-models is used")
        text = (sys.stdin.read() if args.input == "-"
                else Path(args.input).read_text(encoding="utf-8"))
        config = resolve_config(args.provider, args.model)
        result = clean_transcript(text, args.translate, config)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if "error" in result:
        print(f"error: {result['error']}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(result["cleaned_text"])
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
