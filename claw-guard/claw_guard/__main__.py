"""Entry point for Claw Guard proxy.

Usage:
    python -m claw_guard [--mode passthrough|shadow|active] [--port 18800]
                         [--api-key KEY] [--gemini-url URL]
                         [--log-file FILE] [--metrics-file FILE]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from aiohttp import web

from .config import (
    ClawGuardConfig,
    GEMINI_BASE_URL,
    OperatingMode,
    PROXY_HOST,
    PROXY_PORT,
)
from .proxy import ClawGuardProxy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="claw-guard",
        description="Claw Guard — TPM/RPM guardrail proxy for OpenClaw ↔ Gemini",
    )
    parser.add_argument(
        "--mode",
        choices=["passthrough", "shadow", "active"],
        default="passthrough",
        help="Operating mode (default: passthrough)",
    )
    parser.add_argument(
        "--host",
        default=PROXY_HOST,
        help=f"Bind address (default: {PROXY_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PROXY_PORT,
        help=f"Listen port (default: {PROXY_PORT})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini-url",
        default=GEMINI_BASE_URL,
        help=f"Upstream Gemini base URL (default: {GEMINI_BASE_URL})",
    )
    parser.add_argument(
        "--log-file",
        default="claw_guard.log",
        help="Log file path (default: claw_guard.log)",
    )
    parser.add_argument(
        "--metrics-file",
        default="claw_guard_metrics.jsonl",
        help="Metrics JSONL file path (default: claw_guard_metrics.jsonl)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO)",
    )
    return parser.parse_args(argv)


def setup_logging(log_file: str, log_level: str) -> None:
    """Configure logging to both file and stderr."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level))

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Stderr handler
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(sh)


def build_config(args: argparse.Namespace) -> ClawGuardConfig:
    return ClawGuardConfig(
        mode=OperatingMode(args.mode),
        host=args.host,
        port=args.port,
        gemini_base_url=args.gemini_url.rstrip("/"),
        api_key=args.api_key,
        log_file=args.log_file,
        metrics_file=args.metrics_file,
    )


async def run_server(config: ClawGuardConfig) -> None:
    proxy = ClawGuardProxy(config)
    await proxy.start()

    app = web.Application()

    # Health and status endpoints
    app.router.add_get("/_health", proxy.handle_health)
    app.router.add_get("/_status", proxy.handle_status)

    # Catch-all: proxy everything else to Gemini
    app.router.add_route("*", "/{path_info:.*}", proxy.handle_request)

    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, config.host, config.port)
    await site.start()

    logger = logging.getLogger(__name__)
    logger.info(
        "Claw Guard listening on http://%s:%d (mode=%s)",
        config.host, config.port, config.mode.value,
    )
    logger.info("Upstream: %s", config.gemini_base_url)
    if not config.api_key:
        logger.warning("No API key configured — requests will likely fail")

    # Wait for shutdown signal
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()

    logger.info("Shutting down…")
    await proxy.stop()
    await runner.cleanup()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, args.log_level)
    config = build_config(args)
    asyncio.run(run_server(config))


if __name__ == "__main__":
    main()
