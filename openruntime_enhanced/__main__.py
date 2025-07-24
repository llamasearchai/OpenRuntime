#!/usr/bin/env python3
"""
OpenRuntime Enhanced Main Entry Point
Advanced GPU Runtime System with AI Integration for macOS
"""

import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openruntime_enhanced.enhanced import app, __version__, __author__, __email__

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="OpenRuntime Enhanced GPU Computing Platform")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])

    args = parser.parse_args()

    import logging
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info("OpenRuntime Enhanced: Advanced GPU Runtime System with AI Integration")
    logging.info("=" * 60)
    logging.info(f"Version: {__version__}")
    logging.info(f"Author: {__author__} <{__email__}>")
    logging.info(f"Server: http://{args.host}:{args.port}")
    logging.info(f"API Docs: http://{args.host}:{args.port}/docs")
    logging.info("=" * 60)

    uvicorn.run(
        app, host=args.host, port=args.port, reload=args.reload, log_level=args.log_level
    ) 