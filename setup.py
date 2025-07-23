#!/usr/bin/env python3
"""
Setup script for OpenRuntime Enhanced
"""

from setuptools import setup, find_packages
from pathlib import Path
import re


# Read version from main module
def get_version():
    version_file = Path("openruntime_enhanced.py")
    if version_file.exists():
        content = version_file.read_text()
        version_match = re.search(r'__version__ = ["\']([^"\']*)["\']', content)
        if version_match:
            return version_match.group(1)
    return "1.0.0"


# Read long description from README
def get_long_description():
    readme_file = Path("README.md")
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ""


# Read requirements
def get_requirements(filename):
    req_file = Path(filename)
    if req_file.exists():
        return req_file.read_text().strip().split("\n")
    return []


setup(
    name="openruntime-enhanced",
    version=get_version(),
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="Advanced GPU Runtime System with AI Integration",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/openruntime/openruntime-enhanced",
    project_urls={
        "Bug Tracker": "https://github.com/openruntime/openruntime-enhanced/issues",
        "Documentation": "https://docs.openruntime.example.com",
        "Source Code": "https://github.com/openruntime/openruntime-enhanced",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=get_requirements("requirements.txt"),
    extras_require={
        "dev": get_requirements("requirements-dev.txt"),
        "gpu": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "torchaudio>=2.0.0",
        ],
        "ai": [
            "openai>=1.0.0",
            "langchain>=0.0.300",
            "shell-gpt>=1.0.0",
        ],
        "monitoring": [
            "prometheus-client>=0.16.0",
            "psutil>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "openruntime=cli_simple:main",
            "openruntime-enhanced=cli_simple:main",
            "openruntime-simple=cli_simple:main",
            "openruntime-cli=cli:cli_main",
            "openruntime-benchmark=scripts.benchmark:main",
            "openruntime-stress=scripts.stress_test:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.md", "*.txt", "*.html", "*.css", "*.js"],
        "static": ["*"],
        "templates": ["*"],
        "monitoring": ["*.yml", "*.yaml", "*.json"],
        "nginx": ["*.conf"],
        "scripts": ["*.sh", "*.py"],
    },
    zip_safe=False,
    keywords=[
        "gpu",
        "computing",
        "ai",
        "machine-learning",
        "performance",
        "runtime",
        "distributed",
        "metal",
        "openai",
        "langchain",
    ],
    platforms=["any"],
    license="MIT",
)
