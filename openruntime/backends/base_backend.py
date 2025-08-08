"""
Base backend interface for OpenRuntime
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional


class BaseBackend(ABC):
    """Base class for all runtime backends"""

    def __init__(self, config: Any):
        self.config = config
        self.initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the backend"""
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the backend"""
        pass

    async def inference(self, payload: Dict[str, Any]) -> Any:
        """Run inference"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support inference")

    async def embed(self, payload: Dict[str, Any]) -> Any:
        """Generate embeddings"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    async def complete(self, payload: Dict[str, Any]) -> Any:
        """Generate completions"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support completions")

    async def run_agent(self, payload: Dict[str, Any]) -> Any:
        """Run agent workflow"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support agents")

    async def use_tool(self, payload: Dict[str, Any]) -> Any:
        """Use tool"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support tools")

    async def run_workflow(self, payload: Dict[str, Any]) -> Any:
        """Run workflow"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support workflows")

    async def run_command(self, payload: Dict[str, Any]) -> Any:
        """Run command"""
        raise NotImplementedError(f"{self.__class__.__name__} does not support commands")

    async def get_metrics(self) -> Dict[str, Any]:
        """Get backend metrics"""
        return {"initialized": self.initialized, "backend": self.__class__.__name__}
