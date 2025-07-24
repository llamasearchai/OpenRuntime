"""
OpenRuntime Enhanced - Advanced GPU runtime with enhanced AI capabilities.

This module provides enhanced functionality on top of the base OpenRuntime system,
including advanced AI workflows, real-time monitoring, and performance optimizations.
"""

from .enhanced import app, __version__, __author__, __email__

# Enhanced features marker
ENHANCED_FEATURES = True

__all__ = [
    "app",
    "__version__",
    "__author__",
    "__email__",
    "ENHANCED_FEATURES"
]