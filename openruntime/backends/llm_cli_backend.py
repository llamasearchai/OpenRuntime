"""
LLM CLI Backend - Integration with Simon Willison's LLM tool
"""

import asyncio
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class LLMCLIBackend(BaseBackend):
    """Backend for LLM CLI tool integration"""

    def __init__(self, config: Any):
        super().__init__(config)
        self.available_models = {}
        self.available_plugins = {}
        self.embedding_models = {}

    async def initialize(self) -> None:
        """Initialize LLM CLI backend"""
        # Check LLM installation
        if not await self._check_llm():
            raise RuntimeError("LLM CLI not installed. Run: pip install llm")

        # Discover available models and plugins
        await self._discover_models()
        await self._discover_plugins()

        self.initialized = True
        logger.info(f"LLM CLI backend initialized with {len(self.available_models)} models")

    async def _check_llm(self) -> bool:
        """Check if LLM CLI is available"""
        try:
            result = await self._run_command(["llm", "--version"])
            return result["returncode"] == 0
        except Exception:
            return False

    async def _discover_models(self) -> None:
        """Discover available LLM models"""
        try:
            # Get list of models
            result = await self._run_command(["llm", "models", "list", "--json"])
            if result["returncode"] == 0 and result["stdout"]:
                models = json.loads(result["stdout"])
                for model in models:
                    model_id = model.get("model_id", model.get("name"))
                    self.available_models[model_id] = model

            # Check for specific plugin models
            # MLX models
            mlx_result = await self._run_command(["llm", "models", "list", "--plugin", "llm-mlx"])
            if mlx_result["returncode"] == 0:
                logger.info("MLX models available via LLM")

            # Ollama models
            ollama_result = await self._run_command(
                ["llm", "models", "list", "--plugin", "llm-ollama"]
            )
            if ollama_result["returncode"] == 0:
                logger.info("Ollama models available via LLM")

            # Embedding models
            embed_result = await self._run_command(["llm", "embed-models", "list", "--json"])
            if embed_result["returncode"] == 0 and embed_result["stdout"]:
                embed_models = json.loads(embed_result["stdout"])
                for model in embed_models:
                    model_id = model.get("model_id", model.get("name"))
                    self.embedding_models[model_id] = model

        except Exception as e:
            logger.warning(f"Error discovering models: {e}")

    async def _discover_plugins(self) -> None:
        """Discover installed LLM plugins"""
        try:
            result = await self._run_command(["llm", "plugins", "list", "--json"])
            if result["returncode"] == 0 and result["stdout"]:
                plugins = json.loads(result["stdout"])
                for plugin in plugins:
                    plugin_name = plugin.get("name")
                    if plugin_name:
                        self.available_plugins[plugin_name] = plugin

        except Exception as e:
            logger.warning(f"Error discovering plugins: {e}")

    async def _run_command(
        self, cmd: List[str], input_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a command asynchronously"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if input_text else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            if input_text:
                stdout, stderr = await process.communicate(input_text.encode())
            else:
                stdout, stderr = await process.communicate()

            return {
                "returncode": process.returncode,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
            }
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    async def complete(self, payload: Dict[str, Any]) -> Any:
        """Generate completion using LLM CLI"""
        try:
            prompt = payload.get("prompt", "")
            model = payload.get("model", "gpt-3.5-turbo")
            temperature = payload.get("temperature", 0.7)
            max_tokens = payload.get("max_tokens", 2000)
            system = payload.get("system", "")

            cmd = ["llm", "prompt"]

            # Add model
            if model:
                cmd.extend(["-m", model])

            # Add system prompt
            if system:
                cmd.extend(["-s", system])

            # Add options
            cmd.extend(["-o", f"temperature={temperature}", "-o", f"max_tokens={max_tokens}"])

            # Add prompt
            cmd.append(prompt)

            result = await self._run_command(cmd)

            if result["returncode"] == 0:
                return {"content": result["stdout"], "model": model, "error": None}
            else:
                return {"content": None, "model": model, "error": result["stderr"]}

        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            raise

    async def embed(self, payload: Dict[str, Any]) -> Any:
        """Generate embeddings using LLM CLI"""
        try:
            texts = payload.get("texts", [])
            model = payload.get("model", "sentence-transformers/all-MiniLM-L6-v2")

            if isinstance(texts, str):
                texts = [texts]

            # Use llm embed command
            embeddings = []

            for text in texts:
                cmd = ["llm", "embed", "-m", model, "-c", text, "--json"]
                result = await self._run_command(cmd)

                if result["returncode"] == 0 and result["stdout"]:
                    embed_data = json.loads(result["stdout"])
                    embeddings.append(embed_data.get("embedding", []))
                else:
                    embeddings.append([])

            return {
                "embeddings": embeddings,
                "model": model,
                "dimensions": len(embeddings[0]) if embeddings and embeddings[0] else 0,
            }

        except Exception as e:
            logger.error(f"LLM embedding error: {e}")
            raise

    async def run_command(self, payload: Dict[str, Any]) -> Any:
        """Run LLM command"""
        try:
            command_type = payload.get("command_type", "prompt")

            if command_type == "cmd":
                # Use llm-cmd for shell commands
                prompt = payload.get("prompt", "")
                cmd = ["llm", "cmd", prompt]
                result = await self._run_command(cmd)

                return {
                    "command": result["stdout"].strip(),
                    "error": result["stderr"] if result["returncode"] != 0 else None,
                }

            elif command_type == "chat":
                # Interactive chat mode
                messages = payload.get("messages", [])
                model = payload.get("model", "gpt-3.5-turbo")

                # Format messages for LLM
                chat_input = ""
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "system":
                        chat_input = f"System: {content}\n\n" + chat_input
                    elif role == "user":
                        chat_input += f"User: {content}\n"
                    elif role == "assistant":
                        chat_input += f"Assistant: {content}\n"

                cmd = ["llm", "chat", "-m", model, "--continue"]
                result = await self._run_command(cmd, chat_input)

                return {"response": result["stdout"], "model": model}

            elif command_type == "tools":
                # Use with tools
                tool_name = payload.get("tool")
                args = payload.get("args", {})

                # Execute tool via LLM
                cmd = ["llm", "tools", "use", tool_name]
                for key, value in args.items():
                    cmd.extend([f"--{key}", str(value)])

                result = await self._run_command(cmd)

                return {
                    "tool": tool_name,
                    "result": result["stdout"],
                    "error": result["stderr"] if result["returncode"] != 0 else None,
                }

            else:
                return {"error": f"Unknown command type: {command_type}"}

        except Exception as e:
            logger.error(f"LLM command error: {e}")
            raise

    async def use_tool(self, payload: Dict[str, Any]) -> Any:
        """Use LLM tool"""
        tool_name = payload.get("tool")
        args = payload.get("args", {})
        model = payload.get("model", "gpt-4-turbo-preview")

        # Build command
        cmd = ["llm", "prompt", "-m", model, "--tool", tool_name]

        # Add tool arguments
        tool_prompt = json.dumps({"tool": tool_name, "arguments": args})

        cmd.append(tool_prompt)

        result = await self._run_command(cmd)

        if result["returncode"] == 0:
            return {"tool": tool_name, "result": result["stdout"], "model": model}
        else:
            return {"tool": tool_name, "error": result["stderr"], "model": model}

    async def shutdown(self) -> None:
        """Shutdown LLM CLI backend"""
        self.initialized = False
        logger.info("LLM CLI backend shutdown")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics"""
        return {
            "initialized": self.initialized,
            "backend": "LLM_CLI",
            "models": len(self.available_models),
            "plugins": len(self.available_plugins),
            "embedding_models": len(self.embedding_models),
            "available_models": list(self.available_models.keys()),
            "available_plugins": list(self.available_plugins.keys()),
        }
