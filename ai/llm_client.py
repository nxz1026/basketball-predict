"""
Generic LLM client for any OpenAI-compatible API.
No hardcoded provider — configure via env vars or params.
"""
import os
import json
import time
import requests
import sys

_last_call = 0.0

# Gemini fallback chain — only used when provider is "gemini"
GEMINI_FALLBACK = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
]


def generate(
    prompt: str,
    model: str = "",
    api_base: str = "",
    api_key: str = "",
    rate_limit: float = 0.0,
    provider: str = "openai",
) -> dict:
    """Call any OpenAI-compatible LLM API. Returns parsed JSON or {}.

    Model fallback chain is only used when provider='gemini'.
    For all other providers, single call — no fallback.
    """
    global _last_call

    # Resolve from env/params
    api_key = api_key or os.environ.get("LLM_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
    api_base = api_base or os.environ.get("LLM_API_BASE", "https://token.sensenova.cn/v1")
    model = model or os.environ.get("LLM_MODEL", "deepseek-v4-flash")

    if not api_key:
        return {}

    # Rate limiting (only if set > 0)
    if rate_limit > 0:
        elapsed = time.time() - _last_call
        if elapsed < rate_limit:
            time.sleep(rate_limit - elapsed)
        _last_call = time.time()

    if provider == "gemini":
        # Gemini REST API: different URL format, fallback chain
        return _call_gemini(prompt, model, api_key, rate_limit)
    else:
        # OpenAI-compatible: single model, single call
        return _call_openai(prompt, model, api_base, api_key)


def _call_openai(prompt: str, model: str, api_base: str, api_key: str) -> dict:
    """Call OpenAI-compatible chat completions endpoint."""
    url = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return {}
        return _parse_openai(resp)
    except requests.RequestException:
        return {}


def _call_gemini(prompt: str, model: str, api_key: str, rate_limit: float) -> dict:
    """Call Gemini REST API with model fallback on 429/404."""
    models = [model] + [m for m in GEMINI_FALLBACK if m != model] if model else GEMINI_FALLBACK
    for m in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.3,
                "maxOutputTokens": 2048,
            },
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 200:
                return _parse_gemini(resp)
            if resp.status_code in (429, 404):
                time.sleep(1)
                continue
            return {}
        except requests.RequestException:
            return {}
    return {}


def _parse_openai(resp) -> dict:
    """Extract JSON from OpenAI-compatible response."""
    try:
        text = resp.json()["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, KeyError, IndexError):
        return {}


def _parse_gemini(resp) -> dict:
    """Extract JSON from Gemini response."""
    try:
        text = (
            resp.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, KeyError):
        return {}