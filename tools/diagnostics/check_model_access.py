#!/usr/bin/env python3
"""Probe model access against an OpenAI/Anthropic-compatible base URL.

This script intentionally avoids Claude Code SDK and OpenAI/Anthropic client
libraries so it can distinguish raw HTTP/API access from SDK behavior.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


@dataclass
class ProbeResult:
    name: str
    ok: bool
    status: int | None
    body: Any
    error: str | None = None


def normalize_base_url(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and "](" in value and value.endswith(")"):
        value = value.split("](", 1)[1][:-1]
    return value.rstrip("/")


def redact(value: str) -> str:
    if len(value) <= 10:
        return "***"
    return f"{value[:5]}...{value[-4:]}"


def request_json(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any] | None = None,
    timeout: int,
) -> tuple[int, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers = {**headers, "Content-Type": "application/json"}

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, parse_body(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        return exc.code, parse_body(raw)


def parse_body(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw[:4000]


def has_error(body: Any) -> bool:
    if isinstance(body, dict):
        return bool(body.get("error"))
    return False


def probe_models(base_url: str, api_key: str, timeout: int) -> ProbeResult:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        status, body = request_json(
            method="GET",
            url=f"{base_url}/models",
            headers=headers,
            timeout=timeout,
        )
        return ProbeResult("GET /models", 200 <= status < 300 and not has_error(body), status, body)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("GET /models", False, None, None, f"{type(exc).__name__}: {exc}")


def probe_anthropic_messages(base_url: str, api_key: str, model: str, timeout: int) -> ProbeResult:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    }
    try:
        status, body = request_json(
            method="POST",
            url=f"{base_url}/messages",
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        return ProbeResult("POST /messages (Anthropic)", 200 <= status < 300 and not has_error(body), status, body)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("POST /messages (Anthropic)", False, None, None, f"{type(exc).__name__}: {exc}")


def probe_chat_completions(base_url: str, api_key: str, model: str, timeout: int) -> ProbeResult:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    }
    try:
        status, body = request_json(
            method="POST",
            url=f"{base_url}/chat/completions",
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        return ProbeResult("POST /chat/completions", 200 <= status < 300 and not has_error(body), status, body)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("POST /chat/completions", False, None, None, f"{type(exc).__name__}: {exc}")


def probe_responses(base_url: str, api_key: str, model: str, timeout: int) -> ProbeResult:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "input": "Reply with exactly: ok",
        "max_output_tokens": 16,
    }
    try:
        status, body = request_json(
            method="POST",
            url=f"{base_url}/responses",
            headers=headers,
            payload=payload,
            timeout=timeout,
        )
        return ProbeResult("POST /responses", 200 <= status < 300 and not has_error(body), status, body)
    except Exception as exc:  # noqa: BLE001
        return ProbeResult("POST /responses", False, None, None, f"{type(exc).__name__}: {exc}")


def summarize_body(body: Any) -> Any:
    if isinstance(body, dict):
        if "error" in body:
            return {"error": body["error"]}
        summary: dict[str, Any] = {}
        for key in ("id", "type", "model", "role", "stop_reason", "object"):
            if key in body:
                summary[key] = body[key]
        if "content" in body:
            summary["content"] = body["content"]
        if "choices" in body:
            summary["choices"] = body["choices"][:1]
        if "output" in body:
            summary["output"] = body["output"][:1]
        if "data" in body and isinstance(body["data"], list):
            summary["data_count"] = len(body["data"])
            summary["first_models"] = [
                item.get("id") or item.get("model")
                for item in body["data"][:10]
                if isinstance(item, dict)
            ]
        return summary or body
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("MODEL_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("MODEL_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="Print full JSON results.")
    args = parser.parse_args()

    if not args.base_url or not args.api_key:
        print("Set --base-url/--api-key or MODEL_BASE_URL/MODEL_API_KEY.", file=sys.stderr)
        return 2

    base_url = normalize_base_url(args.base_url)
    print(f"base_url: {base_url}")
    print(f"api_key: {redact(args.api_key)}")
    print(f"model: {args.model}")

    probes = [
        probe_models(base_url, args.api_key, args.timeout),
        probe_anthropic_messages(base_url, args.api_key, args.model, args.timeout),
        probe_chat_completions(base_url, args.api_key, args.model, args.timeout),
        probe_responses(base_url, args.api_key, args.model, args.timeout),
    ]

    for result in probes:
        print()
        print(f"== {result.name} ==")
        print(f"ok: {result.ok}")
        print(f"status: {result.status}")
        if result.error:
            print(f"error: {result.error}")
        else:
            body = result.body if args.json else summarize_body(result.body)
            print(json.dumps(body, ensure_ascii=False, indent=2))

    return 0 if any(result.ok for result in probes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
