#!/usr/bin/env python3
"""Probe OpenAI Responses streaming behavior for an endpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any


def normalize_base_url(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and "](" in value and value.endswith(")"):
        value = value.split("](", 1)[1][:-1]
    return value.rstrip("/")


def parse_sse_data(line: str) -> Any:
    data = line[len("data:") :].strip()
    if data == "[DONE]":
        return data
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("MODEL_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY") or os.environ.get("MODEL_API_KEY"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt-5"))
    parser.add_argument("--input", default="Create /tmp/stream_probe.txt with the text ok, then say done.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-events", type=int, default=200)
    args = parser.parse_args()

    if not args.base_url or not args.api_key:
        print("Set --base-url/--api-key or OPENAI_BASE_URL/OPENAI_API_KEY.", file=sys.stderr)
        return 2

    url = f"{normalize_base_url(args.base_url)}/responses"
    payload = {
        "model": args.model,
        "input": args.input,
        "stream": True,
        "max_output_tokens": 512,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )

    print(f"url: {url}")
    print(f"model: {args.model}")
    print(f"stream: true")

    seen_failed = False
    seen_completed = False
    events: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            print(f"status: {response.status}")
            for raw in response:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                    events.append(event_name)
                    print(f"event: {event_name}")
                    if event_name == "response.failed":
                        seen_failed = True
                    if event_name == "response.completed":
                        seen_completed = True
                    if len(events) >= args.max_events:
                        print("max-events reached")
                        break
                elif line.startswith("data:"):
                    data = parse_sse_data(line)
                    if isinstance(data, dict) and data.get("type") in {"response.failed", "error"}:
                        seen_failed = True
                        print(json.dumps(data, ensure_ascii=False)[:2000])
                    elif isinstance(data, dict) and data.get("type") == "response.completed":
                        seen_completed = True
                    elif data == "[DONE]":
                        print("data: [DONE]")
                        break
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"http_error: {exc.code}")
        print(body[:4000])
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}")
        return 1

    print(f"seen_failed: {seen_failed}")
    print(f"seen_completed: {seen_completed}")
    print(f"event_count: {len(events)}")
    return 1 if seen_failed or not seen_completed else 0


if __name__ == "__main__":
    raise SystemExit(main())
