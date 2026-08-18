#!/usr/bin/env python3
"""Lightweight Responses-to-Chat proxy for Codex CLI experiments.

This is a small fallback for environments where the official MoonBridge image
cannot be built. It exposes the OpenAI Responses paths Codex calls and forwards
text generation to an OpenAI-compatible `/chat/completions` upstream.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def normalize_base_url(value: str) -> str:
    value = value.strip()
    if value.startswith("[") and "](" in value and value.endswith(")"):
        value = value.split("](", 1)[1][:-1]
    return value.rstrip("/")


def redact(value: str) -> str:
    if len(value) <= 10:
        return "***"
    return f"{value[:5]}...{value[-4:]}"


def parse_json(raw: bytes) -> Any:
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def response_id() -> str:
    return f"resp_{int(time.time() * 1000)}"


def split_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        return "\n".join(parts)
    return "" if content is None else str(content)


def responses_input_to_messages(body: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    input_value = body.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            role = item.get("role") or "user"
            content = item.get("content")
            text = split_text_content(content)
            if text:
                chat_role = str(role)
                if chat_role == "developer":
                    chat_role = "system"
                elif chat_role not in {"system", "user", "assistant", "tool"}:
                    chat_role = "user"
                messages.append({"role": chat_role, "content": text})

    if not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def chat_payload_from_responses(body: dict[str, Any], default_model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": body.get("model") or default_model,
        "messages": responses_input_to_messages(body),
        "stream": bool(body.get("stream")),
    }
    max_tokens = body.get("max_output_tokens") or body.get("max_tokens")
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    temperature = body.get("temperature")
    if temperature is not None:
        payload["temperature"] = temperature
    top_p = body.get("top_p")
    if top_p is not None:
        payload["top_p"] = top_p
    return payload


def summarize_chat_text(chat_body: dict[str, Any]) -> str:
    choices = chat_body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else {}
    if not isinstance(message, dict):
        return ""
    return str(message.get("content") or message.get("reasoning_content") or "")


def responses_body_from_chat(chat_body: dict[str, Any], model: str, rid: str) -> dict[str, Any]:
    text = summarize_chat_text(chat_body)
    return {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"{rid}_msg",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                    }
                ],
            }
        ],
        "output_text": text,
        "usage": chat_body.get("usage"),
    }


def sse_event(event: str, data: Any) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8")


class ProxyHandler(BaseHTTPRequestHandler):
    server_version = "CodexResponsesChatProxy/0.1"

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        if path in ("/", "/health", "/v1", "/v1/"):
            self.send_json(200, {"ok": True, "upstream_base_url": self.server.upstream_base_url})
            return
        if path in ("/models", "/v1/models"):
            self.forward_models()
            return
        self.send_json(404, {"error": {"message": f"unsupported path: {path}"}})

    def do_HEAD(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        self.send_response(200 if path in ("/", "/health", "/v1", "/v1/") else 404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        if path not in ("/responses", "/v1/responses"):
            self.send_json(404, {"error": {"message": f"unsupported path: {path}"}})
            return
        raw = self.rfile.read(int(self.headers.get("content-length") or "0"))
        try:
            body = parse_json(raw)
        except json.JSONDecodeError as exc:
            self.send_json(400, {"error": {"message": f"invalid JSON: {exc}"}})
            return
        self.handle_responses(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        if self.server.verbose:
            super().log_message(fmt, *args)

    def send_json(self, status: int, payload: Any) -> None:
        raw = json_bytes(payload)
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def forward_models(self) -> None:
        status, headers, body = self.upstream_request(
            method="GET",
            path="/models",
            payload=None,
            stream=False,
        )
        self.send_response(status)
        self.send_header("content-type", headers.get("content-type", "application/json"))
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def handle_responses(self, body: dict[str, Any]) -> None:
        payload = chat_payload_from_responses(body, self.server.default_model)
        rid = response_id()
        if payload.get("stream"):
            self.handle_streaming_chat(payload, rid)
            return

        status, _headers, raw = self.upstream_request(
            method="POST",
            path="/chat/completions",
            payload=payload,
            stream=False,
        )
        if status < 200 or status >= 300:
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        chat_body = json.loads(raw.decode("utf-8"))
        self.send_json(200, responses_body_from_chat(chat_body, str(payload["model"]), rid))

    def handle_streaming_chat(self, payload: dict[str, Any], rid: str) -> None:
        upstream_url = f"{self.server.upstream_base_url}/chat/completions"
        req = urllib.request.Request(
            upstream_url,
            data=json_bytes(payload),
            headers=self.upstream_headers(content_type=True),
            method="POST",
        )
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()

        item_id = f"{rid}_msg"
        content_index = 0
        output_index = 0
        full_text: list[str] = []
        model = str(payload["model"])
        self.write_sse("response.created", {"type": "response.created", "response": {"id": rid, "status": "in_progress", "model": model}})
        self.write_sse("response.in_progress", {"type": "response.in_progress", "response": {"id": rid, "status": "in_progress", "model": model}})
        self.write_sse("response.output_item.added", {"type": "response.output_item.added", "output_index": output_index, "item": {"id": item_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}})
        self.write_sse("response.content_part.added", {"type": "response.content_part.added", "item_id": item_id, "output_index": output_index, "content_index": content_index, "part": {"type": "output_text", "text": "", "annotations": []}})

        try:
            with urllib.request.urlopen(req, timeout=self.server.timeout) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = self.extract_chat_delta(chunk)
                    if not delta:
                        continue
                    full_text.append(delta)
                    self.write_sse("response.output_text.delta", {"type": "response.output_text.delta", "item_id": item_id, "output_index": output_index, "content_index": content_index, "delta": delta})
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            self.write_sse("response.failed", {"type": "response.failed", "response": {"id": rid, "status": "failed", "error": {"message": raw or str(exc)}}})
            self.write_sse("done", "[DONE]")
            return

        text = "".join(full_text)
        self.write_sse("response.output_text.done", {"type": "response.output_text.done", "item_id": item_id, "output_index": output_index, "content_index": content_index, "text": text})
        self.write_sse("response.content_part.done", {"type": "response.content_part.done", "item_id": item_id, "output_index": output_index, "content_index": content_index, "part": {"type": "output_text", "text": text, "annotations": []}})
        self.write_sse("response.output_item.done", {"type": "response.output_item.done", "output_index": output_index, "item": {"id": item_id, "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}})
        self.write_sse("response.completed", {"type": "response.completed", "response": {"id": rid, "status": "completed", "model": model, "output": [{"id": item_id, "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}], "output_text": text}})
        self.write_sse("done", "[DONE]")

    def write_sse(self, event: str, payload: Any) -> None:
        try:
            self.wfile.write(sse_event(event, payload))
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def extract_chat_delta(self, chunk: dict[str, Any]) -> str:
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else {}
        if not isinstance(delta, dict):
            return ""
        return str(delta.get("content") or delta.get("reasoning_content") or "")

    def upstream_headers(self, *, content_type: bool) -> dict[str, str]:
        headers = {
            "authorization": f"Bearer {self.server.api_key}",
            "accept": "application/json",
        }
        if content_type:
            headers["content-type"] = "application/json"
        return headers

    def upstream_request(
        self,
        *,
        method: str,
        path: str,
        payload: Any | None,
        stream: bool,
    ) -> tuple[int, dict[str, str], bytes]:
        data = json_bytes(payload) if payload is not None else None
        req = urllib.request.Request(
            f"{self.server.upstream_base_url}{path}",
            data=data,
            headers=self.upstream_headers(content_type=payload is not None),
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.server.timeout) as resp:
                return resp.status, dict(resp.headers.items()), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers.items()), exc.read()


class ProxyServer(ThreadingHTTPServer):
    upstream_base_url: str
    api_key: str
    default_model: str
    timeout: int
    verbose: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight Codex Responses to Chat Completions proxy.")
    parser.add_argument("--listen-host", default=os.environ.get("PROXY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT", "38446")))
    parser.add_argument("--upstream-base-url", default=os.environ.get("UPSTREAM_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("UPSTREAM_API_KEY"))
    parser.add_argument("--default-model", default=os.environ.get("UPSTREAM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("PROXY_TIMEOUT", "600")))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.upstream_base_url:
        print("Set --upstream-base-url or UPSTREAM_BASE_URL.", file=sys.stderr)
        return 2
    if not args.api_key:
        print("Set --api-key or UPSTREAM_API_KEY.", file=sys.stderr)
        return 2

    server = ProxyServer((args.listen_host, args.port), ProxyHandler)
    server.upstream_base_url = normalize_base_url(args.upstream_base_url)
    server.api_key = args.api_key
    server.default_model = args.default_model
    server.timeout = args.timeout
    server.verbose = args.verbose
    print(
        f"responses proxy listening on http://{args.listen_host}:{args.port}/v1 "
        f"-> {server.upstream_base_url}/chat/completions "
        f"(api_key={redact(args.api_key)}, default_model={server.default_model})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping proxy", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
