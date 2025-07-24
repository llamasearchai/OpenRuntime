#!/usr/bin/env python3
import argparse
import uvicorn
from openruntime.core.api import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenRuntime")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "openruntime.core.api:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload
    )
