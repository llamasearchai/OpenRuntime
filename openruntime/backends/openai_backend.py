"""
OpenAI Backend with Agents SDK Integration
"""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

from .base_backend import BaseBackend

logger = logging.getLogger(__name__)


class OpenAIBackend(BaseBackend):
    """OpenAI backend with Agents SDK support"""

    def __init__(self, config: Any):
        super().__init__(config)
        self.client = None
        self.agents = {}
        self.sessions = {}
        self.tools = {}

    async def initialize(self) -> None:
        """Initialize OpenAI client and agents"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = AsyncOpenAI(api_key=api_key)

        # Initialize default agents
        await self._initialize_agents()

        self.initialized = True
        logger.info("OpenAI backend initialized")

    async def _initialize_agents(self) -> None:
        """Initialize default agents"""
        # Developer Agent
        self.agents["developer"] = {
            "name": "Developer Agent",
            "model": "gpt-4-turbo-preview",
            "instructions": """You are a senior software developer. 
            Help with code implementation, debugging, and optimization.
            Focus on clean, efficient, and maintainable code.""",
            "tools": ["code_search", "code_analysis", "test_runner"],
        }

        # Analyst Agent
        self.agents["analyst"] = {
            "name": "Data Analyst Agent",
            "model": "gpt-4-turbo-preview",
            "instructions": """You are a data analyst.
            Analyze data, generate insights, and create visualizations.
            Focus on statistical accuracy and clear communication.""",
            "tools": ["data_query", "visualization", "statistics"],
        }

        # Architect Agent
        self.agents["architect"] = {
            "name": "System Architect Agent",
            "model": "gpt-4-turbo-preview",
            "instructions": """You are a system architect.
            Design system architectures, evaluate trade-offs, and ensure scalability.
            Focus on best practices and design patterns.""",
            "tools": ["system_design", "performance_analysis", "security_audit"],
        }

        # Operations Agent
        self.agents["operations"] = {
            "name": "Operations Agent",
            "model": "gpt-4-turbo-preview",
            "instructions": """You are a DevOps engineer.
            Handle deployment, monitoring, and infrastructure management.
            Focus on reliability, automation, and observability.""",
            "tools": ["deployment", "monitoring", "infrastructure"],
        }

    async def complete(self, payload: Dict[str, Any]) -> Any:
        """Generate completion using OpenAI"""
        try:
            messages = payload.get("messages", [])
            model = payload.get("model", "gpt-4-turbo-preview")
            temperature = payload.get("temperature", 0.7)
            max_tokens = payload.get("max_tokens", 2000)
            stream = payload.get("stream", False)

            if stream:
                return await self._stream_completion(messages, model, temperature, max_tokens)
            else:
                response = await self.client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
                )

                return {
                    "content": response.choices[0].message.content,
                    "usage": response.usage.dict() if response.usage else {},
                    "model": model,
                }

        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise

    async def _stream_completion(
        self, messages: List[Dict], model: str, temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """Stream completion responses"""
        try:
            stream = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise

    async def embed(self, payload: Dict[str, Any]) -> Any:
        """Generate embeddings using OpenAI"""
        try:
            texts = payload.get("texts", [])
            model = payload.get("model", "text-embedding-3-small")

            if isinstance(texts, str):
                texts = [texts]

            response = await self.client.embeddings.create(model=model, input=texts)

            embeddings = [e.embedding for e in response.data]

            return {
                "embeddings": embeddings,
                "model": model,
                "dimensions": len(embeddings[0]) if embeddings else 0,
                "usage": response.usage.dict() if response.usage else {},
            }

        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    async def run_agent(self, payload: Dict[str, Any]) -> Any:
        """Run an agent workflow"""
        try:
            agent_type = payload.get("agent_type", "developer")
            task = payload.get("task", "")
            context = payload.get("context", {})
            session_id = payload.get("session_id", None)

            if agent_type not in self.agents:
                raise ValueError(f"Unknown agent type: {agent_type}")

            agent = self.agents[agent_type]

            # Create or retrieve session
            if session_id and session_id in self.sessions:
                messages = self.sessions[session_id]
            else:
                messages = [{"role": "system", "content": agent["instructions"]}]
                if session_id:
                    self.sessions[session_id] = messages

            # Add user task
            user_message = f"Task: {task}"
            if context:
                user_message += f"\n\nContext: {json.dumps(context, indent=2)}"
            messages.append({"role": "user", "content": user_message})

            # Run completion
            response = await self.client.chat.completions.create(
                model=agent["model"], messages=messages, temperature=0.7, max_tokens=2000
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message.dict())

            # Handle tool calls if present
            result = {
                "agent": agent["name"],
                "response": assistant_message.content,
                "session_id": session_id,
            }

            if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
                tool_results = await self._handle_tool_calls(
                    assistant_message.tool_calls, agent["tools"]
                )
                result["tool_results"] = tool_results

            return result

        except Exception as e:
            logger.error(f"Agent execution error: {e}")
            raise

    async def _handle_tool_calls(
        self, tool_calls: List[Any], available_tools: List[str]
    ) -> List[Dict[str, Any]]:
        """Handle tool calls from agent"""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            if tool_name not in available_tools:
                results.append({"tool": tool_name, "error": f"Tool {tool_name} not available"})
                continue

            # Execute tool (simplified for now)
            tool_result = await self._execute_tool(tool_name, tool_args)
            results.append({"tool": tool_name, "result": tool_result})

        return results

    async def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool"""
        # Tool implementations would go here
        # For now, return mock results
        tool_implementations = {
            "code_search": lambda args: f"Found {args.get('query', '')} in 5 files",
            "code_analysis": lambda args: f"Analyzed {args.get('file', '')}: No issues found",
            "test_runner": lambda args: f"Tests passed: {args.get('test_suite', 'all')}",
            "data_query": lambda args: f"Query returned {args.get('limit', 100)} results",
            "visualization": lambda args: f"Created {args.get('chart_type', 'bar')} chart",
            "statistics": lambda args: f"Mean: {args.get('value', 0)}, Std: 0.1",
            "system_design": lambda args: f"Designed {args.get('component', 'system')}",
            "performance_analysis": lambda args: f"Performance: {args.get('metric', 'good')}",
            "security_audit": lambda args: f"Security scan: {args.get('target', 'system')} secure",
            "deployment": lambda args: f"Deployed to {args.get('environment', 'staging')}",
            "monitoring": lambda args: f"Monitoring {args.get('service', 'all')} services",
            "infrastructure": lambda args: f"Provisioned {args.get('resources', 'servers')}",
        }

        if tool_name in tool_implementations:
            return tool_implementations[tool_name](args)
        else:
            return f"Tool {tool_name} executed with args: {args}"

    async def use_tool(self, payload: Dict[str, Any]) -> Any:
        """Use a specific tool"""
        tool_name = payload.get("tool")
        args = payload.get("args", {})

        result = await self._execute_tool(tool_name, args)

        return {"tool": tool_name, "args": args, "result": result}

    async def run_workflow(self, payload: Dict[str, Any]) -> Any:
        """Run a multi-agent workflow"""
        workflow_type = payload.get("workflow_type", "analysis")
        data = payload.get("data", {})

        workflows = {
            "analysis": ["analyst", "developer"],
            "development": ["architect", "developer", "operations"],
            "deployment": ["developer", "operations"],
            "optimization": ["analyst", "architect", "developer"],
        }

        agents_to_run = workflows.get(workflow_type, ["developer"])
        results = []

        for agent_type in agents_to_run:
            agent_result = await self.run_agent(
                {
                    "agent_type": agent_type,
                    "task": f"Handle {workflow_type} workflow",
                    "context": data,
                }
            )
            results.append(agent_result)

            # Pass results to next agent
            data["previous_results"] = agent_result

        return {"workflow": workflow_type, "agents_used": agents_to_run, "results": results}

    async def shutdown(self) -> None:
        """Shutdown OpenAI backend"""
        if self.client:
            await self.client.close()
        self.initialized = False
        logger.info("OpenAI backend shutdown")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics"""
        return {
            "initialized": self.initialized,
            "backend": "OpenAI",
            "agents": list(self.agents.keys()),
            "active_sessions": len(self.sessions),
            "tools": len(self.tools),
        }
