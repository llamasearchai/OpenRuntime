#!/usr/bin/env python3
"""
OpenRuntime Enhanced - Workflow Testing Script
Validates GitHub workflows, dependencies, and system setup

Author: Nik Jois <nikjois@llamasearch.ai>
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}")


def print_success(text: str):
    """Print success message"""
    print(f"[PASS] {text}")


def print_error(text: str):
    """Print error message"""
    print(f"[FAIL] {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"[WARN] {text}")


def print_info(text: str):
    """Print info message"""
    print(f"[INFO] {text}")


class WorkflowTester:
    """Test GitHub workflows and system setup"""

    def __init__(self):
        self.results = []
        self.errors = []

    def run_command(self, cmd: List[str], timeout: int = 30) -> Dict:
        """Run a command and return results"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=Path.cwd())
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "stdout": "", "stderr": f"Command timed out after {timeout} seconds", "returncode": -1}
        except Exception as e:
            return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}

    def test_python_setup(self) -> bool:
        """Test Python environment setup"""
        print_header("Testing Python Environment")

        # Check Python version
        version_info = sys.version_info
        if version_info.major == 3 and version_info.minor >= 11:
            print_success(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} ✓")
        else:
            print_error(f"Python {version_info.major}.{version_info.minor}.{version_info.micro} - Need 3.11+")
            return False

        # Test imports
        required_modules = ["fastapi", "uvicorn", "pydantic", "openai", "numpy", "yaml", "pytest", "black", "isort", "flake8"]

        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                print_success(f"{module} ✓")
            except ImportError:
                print_error(f"{module} ✗")
                missing_modules.append(module)

        if missing_modules:
            print_error(f"Missing modules: {', '.join(missing_modules)}")
            print_info("Run: pip install -r requirements.txt requirements-dev.txt")
            return False

        return True

    def test_file_structure(self) -> bool:
        """Test file structure completeness"""
        print_header("Testing File Structure")

        required_files = [
            "openruntime_enhanced.py",
            "openruntime.py",
            "cli_simple.py",
            "setup.py",
            "requirements.txt",
            "requirements-dev.txt",
            "README.md",
            "LICENSE",
            "Makefile",
            "pytest.ini",
            ".gitignore",
            "docker-compose.yml",
            "Dockerfile.enhanced",
        ]

        required_dirs = ["tests", "scripts", "monitoring", "nginx", ".github/workflows"]

        missing_files = []
        for file_path in required_files:
            if Path(file_path).exists():
                print_success(f"{file_path} ✓")
            else:
                print_error(f"{file_path} ✗")
                missing_files.append(file_path)

        missing_dirs = []
        for dir_path in required_dirs:
            if Path(dir_path).exists():
                print_success(f"{dir_path}/ ✓")
            else:
                print_error(f"{dir_path}/ ✗")
                missing_dirs.append(dir_path)

        if missing_files or missing_dirs:
            print_error(f"Missing files: {missing_files}")
            print_error(f"Missing directories: {missing_dirs}")
            return False

        return True

    def test_workflow_syntax(self) -> bool:
        """Test GitHub workflow YAML syntax"""
        print_header("Testing GitHub Workflow Syntax")

        workflow_files = [
            ".github/workflows/ci.yml",
            ".github/workflows/ci-cd.yml",
            ".github/workflows/performance.yml",
            ".github/workflows/security.yml",
        ]

        all_valid = True

        for workflow_file in workflow_files:
            if not Path(workflow_file).exists():
                print_error(f"{workflow_file} does not exist")
                all_valid = False
                continue

            try:
                import yaml

                with open(workflow_file, "r") as f:
                    workflow_content = yaml.safe_load(f)

                # Basic validation
                if not isinstance(workflow_content, dict):
                    print_error(f"{workflow_file}: Invalid YAML structure")
                    all_valid = False
                    continue

                # Check required fields
                required_fields = ["name", "on", "jobs"]
                for field in required_fields:
                    if field not in workflow_content:
                        print_error(f"{workflow_file}: Missing required field '{field}'")
                        all_valid = False
                        continue

                # Validate jobs structure
                jobs = workflow_content.get("jobs", {})
                if not isinstance(jobs, dict) or not jobs:
                    print_error(f"{workflow_file}: No jobs defined")
                    all_valid = False
                    continue

                for job_name, job_config in jobs.items():
                    if not isinstance(job_config, dict):
                        print_error(f"{workflow_file}: Job '{job_name}' has invalid structure")
                        all_valid = False
                        continue

                    if "runs-on" not in job_config:
                        print_error(f"{workflow_file}: Job '{job_name}' missing 'runs-on'")
                        all_valid = False
                        continue

                print_success(f"{workflow_file} ✓")

            except yaml.YAMLError as e:
                print_error(f"{workflow_file}: YAML syntax error - {e}")
                all_valid = False
            except Exception as e:
                print_error(f"{workflow_file}: Validation error - {e}")
                all_valid = False

        return all_valid

    def test_linting(self) -> bool:
        """Test code linting"""
        print_header("Testing Code Quality")

        # Test Black formatting
        black_result = self.run_command(["python", "-m", "black", "--check", "--diff", "."])
        if black_result["success"]:
            print_success("Black formatting ✓")
        else:
            print_error("Black formatting issues found")
            print_info("Run: black . to fix formatting")
            return False

        # Test isort imports
        isort_result = self.run_command(["python", "-m", "isort", "--check-only", "--diff", "."])
        if isort_result["success"]:
            print_success("isort imports ✓")
        else:
            print_error("isort import issues found")
            print_info("Run: isort . to fix imports")
            return False

        # Test flake8 linting
        flake8_result = self.run_command(
            ["python", "-m", "flake8", ".", "--count", "--select=E9,F63,F7,F82", "--show-source", "--statistics"]
        )
        if flake8_result["success"]:
            print_success("Flake8 linting ✓")
        else:
            print_error("Flake8 linting issues found")
            print_info(flake8_result["stdout"])
            return False

        return True

    def test_pytest(self) -> bool:
        """Test pytest execution"""
        print_header("Testing Pytest")

        if not Path("tests").exists():
            print_error("tests/ directory not found")
            return False

        # Run pytest with basic options
        pytest_result = self.run_command(["python", "-m", "pytest", "tests/", "-v", "--tb=short", "--maxfail=5"], timeout=120)

        if pytest_result["success"]:
            print_success("Pytest tests ✓")
            print_info(f"Test output:\n{pytest_result['stdout']}")
        else:
            print_error("Pytest tests failed")
            print_info(f"Error output:\n{pytest_result['stderr']}")
            return False

        return True

    def test_docker_build(self) -> bool:
        """Test Docker build"""
        print_header("Testing Docker Build")

        # Check if Docker is available
        docker_check = self.run_command(["docker", "--version"])
        if not docker_check["success"]:
            print_warning("Docker not available - skipping Docker tests")
            return True

        print_success(f"Docker available: {docker_check['stdout']}")

        # Test basic Dockerfile build
        if Path("Dockerfile.enhanced").exists():
            print_info("Building Docker image...")
            build_result = self.run_command(
                ["docker", "build", "-f", "Dockerfile.enhanced", "-t", "openruntime-test", "."], timeout=300
            )

            if build_result["success"]:
                print_success("Docker build ✓")

                # Clean up test image
                self.run_command(["docker", "rmi", "openruntime-test"])
            else:
                print_error("Docker build failed")
                print_info(f"Build error:\n{build_result['stderr']}")
                return False
        else:
            print_warning("Dockerfile.enhanced not found - skipping Docker build test")

        return True

    def test_cli_functionality(self) -> bool:
        """Test CLI functionality"""
        print_header("Testing CLI Functionality")

        # Test simple CLI
        cli_status = self.run_command(["python", "cli_simple.py", "status"])
        if cli_status["success"]:
            print_success("CLI status command ✓")
        else:
            print_error("CLI status command failed")
            print_info(f"Error: {cli_status['stderr']}")
            return False

        # Test benchmark
        cli_benchmark = self.run_command(["python", "cli_simple.py", "benchmark", "--type", "cpu"])
        if cli_benchmark["success"]:
            print_success("CLI benchmark command ✓")
        else:
            print_error("CLI benchmark command failed")
            print_info(f"Error: {cli_benchmark['stderr']}")
            return False

        # Test config generation
        cli_config = self.run_command(["python", "cli_simple.py", "config"])
        if cli_config["success"]:
            print_success("CLI config command ✓")
        else:
            print_error("CLI config command failed")
            print_info(f"Error: {cli_config['stderr']}")
            return False

        return True

    def test_setup_py(self) -> bool:
        """Test setup.py installation"""
        print_header("Testing Setup.py")

        # Test setup.py check
        setup_check = self.run_command(["python", "setup.py", "check"])
        if setup_check["success"]:
            print_success("setup.py check ✓")
        else:
            print_error("setup.py check failed")
            print_info(f"Error: {setup_check['stderr']}")
            return False

        return True

    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        print_header("Running Comprehensive Tests")

        test_results = {
            "python_setup": self.test_python_setup(),
            "file_structure": self.test_file_structure(),
            "workflow_syntax": self.test_workflow_syntax(),
            "linting": self.test_linting(),
            "pytest": self.test_pytest(),
            "docker_build": self.test_docker_build(),
            "cli_functionality": self.test_cli_functionality(),
            "setup_py": self.test_setup_py(),
        }

        # Summary
        print_header("Test Results Summary")

        passed = sum(1 for result in test_results.values() if result)
        total = len(test_results)

        for test_name, result in test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed")

        if passed == total:
            print_success("🎉 All tests passed! System is ready for GitHub workflows.")
        else:
            print_error(f"❌ {total - passed} tests failed. Please fix issues before proceeding.")

        return {"results": test_results, "passed": passed, "total": total, "success": passed == total}

    def run_github_workflow_simulation(self) -> bool:
        """Simulate GitHub workflow steps"""
        print_header("Simulating GitHub Workflow Steps")

        # Simulate CI steps
        steps = [
            ("Checkout", ["echo", "Simulating git checkout"]),
            ("Setup Python", ["python", "--version"]),
            ("Install dependencies", ["python", "-m", "pip", "list"]),
            ("Lint code", ["python", "-m", "flake8", "--version"]),
            ("Run tests", ["python", "-m", "pytest", "--version"]),
            ("Build package", ["python", "setup.py", "check"]),
        ]

        all_passed = True
        for step_name, cmd in steps:
            print_info(f"Running: {step_name}")
            result = self.run_command(cmd)
            if result["success"]:
                print_success(f"{step_name} ✓")
            else:
                print_error(f"{step_name} ✗")
                all_passed = False

        return all_passed


def main():
    """Main test execution"""
    print_header("OpenRuntime Enhanced - Workflow Testing")
    print("Author: Nik Jois <nikjois@llamasearch.ai>")

    tester = WorkflowTester()

    # Run comprehensive tests
    report = tester.generate_report()

    # Run workflow simulation
    workflow_success = tester.run_github_workflow_simulation()

    # Final status
    print_header("Final Status")

    if report["success"] and workflow_success:
        print_success("🚀 System is ready for GitHub workflows!")
        print_info("You can now:")
        print_info("1. Push to GitHub")
        print_info("2. Create pull requests")
        print_info("3. Run CI/CD pipelines")
        print_info("4. Deploy to production")
        return 0
    else:
        print_error("❌ System has issues that need to be resolved")
        print_info("Please fix the failing tests before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
