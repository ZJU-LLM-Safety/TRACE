#!/usr/bin/env python3
"""Expose a loopback-only host proxy to Docker bridge containers.

Example:
    python3 tools/tcp_proxy_forwarder.py \
      --listen-host 0.0.0.0 --listen-port 47224 \
      --target-host 127.0.0.1 --target-port 47223

Then containers on the VRAP bridge can use:
    HTTP_PROXY=http://192.168.2.1:47224
    HTTPS_PROXY=http://192.168.2.1:47224
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(64 * 1024)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    *,
    target_host: str,
    target_port: int,
) -> None:
    try:
        target_reader, target_writer = await asyncio.open_connection(target_host, target_port)
    except Exception:
        client_writer.close()
        await client_writer.wait_closed()
        return

    await asyncio.gather(
        pipe(client_reader, target_writer),
        pipe(target_reader, client_writer),
    )


async def run(args: argparse.Namespace) -> None:
    server = await asyncio.start_server(
        lambda reader, writer: handle_client(
            reader,
            writer,
            target_host=args.target_host,
            target_port=args.target_port,
        ),
        args.listen_host,
        args.listen_port,
    )
    sockets = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(
        f"forwarding {sockets} -> {args.target_host}:{args.target_port}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forward TCP connections to a loopback proxy.")
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=47224)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=47223)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nstopping forwarder", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
