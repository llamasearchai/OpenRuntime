#!/usr/bin/env python3
"""
OpenRuntime Main Entry Point
Advanced GPU Runtime System for macOS with MLX Metal integration
"""

import os
import sys
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from openruntime import __author__, __email__, __version__, app

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="OpenRuntime GPU Computing Platform")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--log-level", default="info", choices=["debug", "info", "warning", "error"]
    )

    args = parser.parse_args()

    print("OpenRuntime: Advanced GPU Runtime System")
    print("=" * 50)
    print(f"Version: {__version__}")
    print(f"Author: {__author__} <{__email__}>")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("=" * 50)

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
