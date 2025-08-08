#!/usr/bin/env python3
"""
OpenRuntime v2 - Main entry point
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for OpenRuntime v2"""
    parser = argparse.ArgumentParser(description="OpenRuntime v2 - Advanced GPU Runtime System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )

    args = parser.parse_args()

    # Set log level
    log_level = getattr(logging, args.log_level.upper())
    logging.getLogger().setLevel(log_level)

    logger.info("Starting OpenRuntime v2...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Log Level: {args.log_level}")

    try:
        if args.reload:
            # Development mode with auto-reload
            uvicorn.run(
                "openruntime.api_v2:app",
                host=args.host,
                port=args.port,
                reload=True,
                log_level=args.log_level,
            )
        else:
            # Production mode
            uvicorn.run(
                "openruntime.api_v2:app",
                host=args.host,
                port=args.port,
                workers=args.workers,
                log_level=args.log_level,
            )
    except KeyboardInterrupt:
        logger.info("Shutting down OpenRuntime v2...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start OpenRuntime v2: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
