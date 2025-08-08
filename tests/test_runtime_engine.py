"""
Tests for OpenRuntime Engine
"""

import asyncio
import pytest
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from openruntime.runtime_engine import (
    RuntimeEngine,
    RuntimeConfig,
    RuntimeBackend,
    TaskType,
    TaskResult
)


@pytest.fixture
async def runtime_engine():
    """Create runtime engine for testing"""
    config = RuntimeConfig(
        backend=RuntimeBackend.CPU,
        enable_monitoring=False,
        enable_caching=False
    )
    engine = RuntimeEngine(config)
    await engine.initialize()
    yield engine
    await engine.shutdown()


@pytest.fixture
async def mock_openai_backend():
    """Mock OpenAI backend"""
    with patch("openruntime.backends.openai_backend.AsyncOpenAI") as mock_client:
        mock_client.return_value.chat.completions.create = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content="Test response"))],
                usage=MagicMock(dict=lambda: {"total_tokens": 100})
            )
        )
        yield mock_client


class TestRuntimeEngine:
    """Test RuntimeEngine class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, runtime_engine):
        """Test engine initialization"""
        assert runtime_engine._initialized
        assert RuntimeBackend.CPU in runtime_engine.backends
        assert len(runtime_engine.backends) > 0
        
    @pytest.mark.asyncio
    async def test_submit_task(self, runtime_engine):
        """Test task submission"""
        task_id = await runtime_engine.submit_task(
            TaskType.INFERENCE,
            {"inputs": [1, 2, 3]},
            RuntimeBackend.CPU
        )
        
        assert task_id is not None
        assert task_id.startswith("inference_")
        
    @pytest.mark.asyncio
    async def test_cpu_inference(self, runtime_engine):
        """Test CPU backend inference"""
        task_id = await runtime_engine.submit_task(
            TaskType.INFERENCE,
            {
                "inputs": [1, 2, 3, 4, 5],
                "operation": "mean"
            },
            RuntimeBackend.CPU
        )
        
        # Wait for task completion
        await asyncio.sleep(0.5)
        
        # Check metrics
        assert runtime_engine.metrics["tasks_completed"] > 0
        
    @pytest.mark.asyncio
    async def test_cpu_embedding(self, runtime_engine):
        """Test CPU backend embedding generation"""
        task_id = await runtime_engine.submit_task(
            TaskType.EMBEDDING,
            {
                "texts": "Test text for embedding",
                "dimensions": 128
            },
            RuntimeBackend.CPU
        )
        
        await asyncio.sleep(0.5)
        
        # Verify task was processed
        assert task_id not in runtime_engine.active_tasks
        
    @pytest.mark.asyncio
    async def test_status_endpoint(self, runtime_engine):
        """Test status retrieval"""
        status = await runtime_engine.get_status()
        
        assert status["initialized"] is True
        assert "backends" in status
        assert "metrics" in status
        assert "config" in status
        
    @pytest.mark.asyncio
    async def test_backend_selection(self, runtime_engine):
        """Test backend selection logic"""
        # Test with preferred backend
        backend = runtime_engine._select_backend(
            TaskType.INFERENCE,
            RuntimeBackend.CPU
        )
        assert backend is not None
        
        # Test auto-selection
        backend = runtime_engine._select_backend(
            TaskType.INFERENCE,
            None
        )
        assert backend is not None
        
    @pytest.mark.asyncio
    async def test_websocket_registration(self, runtime_engine):
        """Test WebSocket registration"""
        mock_ws = Mock()
        
        await runtime_engine.register_websocket(mock_ws)
        assert mock_ws in runtime_engine._websockets
        
        await runtime_engine.unregister_websocket(mock_ws)
        assert mock_ws not in runtime_engine._websockets
        
    @pytest.mark.asyncio
    async def test_metrics_collection(self, runtime_engine):
        """Test metrics collection"""
        initial_metrics = runtime_engine.metrics.copy()
        
        # Submit a task
        await runtime_engine.submit_task(
            TaskType.INFERENCE,
            {"inputs": [1, 2, 3]},
            RuntimeBackend.CPU
        )
        
        await asyncio.sleep(0.5)
        
        # Check metrics updated
        assert runtime_engine.metrics["tasks_completed"] >= initial_metrics.get("tasks_completed", 0)
        
    @pytest.mark.asyncio
    async def test_concurrent_tasks(self, runtime_engine):
        """Test concurrent task execution"""
        tasks = []
        
        for i in range(5):
            task_id = await runtime_engine.submit_task(
                TaskType.INFERENCE,
                {"inputs": [i, i+1, i+2]},
                RuntimeBackend.CPU
            )
            tasks.append(task_id)
            
        # Wait for all tasks
        await asyncio.sleep(1)
        
        # All tasks should be completed
        for task_id in tasks:
            assert task_id not in runtime_engine.active_tasks
            
    @pytest.mark.asyncio
    async def test_error_handling(self, runtime_engine):
        """Test error handling in task execution"""
        # Submit task with invalid backend
        task_id = await runtime_engine.submit_task(
            TaskType.COMPLETION,
            {"prompt": "test"},
            RuntimeBackend.CPU  # CPU doesn't support completions
        )
        
        await asyncio.sleep(0.5)
        
        # Task should fail gracefully
        assert runtime_engine.metrics["tasks_failed"] > 0


class TestBackends:
    """Test individual backends"""
    
    @pytest.mark.asyncio
    async def test_cpu_backend_operations(self):
        """Test CPU backend operations"""
        from openruntime.backends.cpu_backend import CPUBackend
        
        config = RuntimeConfig()
        backend = CPUBackend(config)
        await backend.initialize()
        
        # Test inference
        result = await backend.inference({
            "inputs": [1, 2, 3, 4, 5],
            "operation": "mean"
        })
        assert result["output"] == 3.0
        
        # Test embedding
        result = await backend.embed({
            "texts": ["test"],
            "dimensions": 64
        })
        assert len(result["embeddings"]) == 1
        assert len(result["embeddings"][0]) == 64
        
        await backend.shutdown()
        
    @pytest.mark.asyncio
    @patch("openruntime.backends.openai_backend.os.getenv")
    async def test_openai_backend_init(self, mock_getenv):
        """Test OpenAI backend initialization"""
        mock_getenv.return_value = "test-api-key"
        
        from openruntime.backends.openai_backend import OpenAIBackend
        
        config = RuntimeConfig()
        backend = OpenAIBackend(config)
        
        with patch("openruntime.backends.openai_backend.AsyncOpenAI"):
            await backend.initialize()
            
        assert backend.initialized
        assert len(backend.agents) > 0
        
    @pytest.mark.asyncio
    async def test_onnx_backend_providers(self):
        """Test ONNX backend provider detection"""
        from openruntime.backends.onnx_backend import ONNXBackend, ONNX_AVAILABLE
        
        if not ONNX_AVAILABLE:
            pytest.skip("ONNX Runtime not available")
            
        config = RuntimeConfig()
        backend = ONNXBackend(config)
        
        providers = backend._get_providers()
        assert "CPUExecutionProvider" in providers
        
    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_llm_cli_backend(self, mock_run):
        """Test LLM CLI backend"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="llm version 0.26.0",
            stderr=""
        )
        
        from openruntime.backends.llm_cli_backend import LLMCLIBackend
        
        config = RuntimeConfig()
        backend = LLMCLIBackend(config)
        
        # Mock the check
        with patch.object(backend, "_check_llm", return_value=True):
            await backend.initialize()
            
        assert backend.initialized