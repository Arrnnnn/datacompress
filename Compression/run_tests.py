#!/usr/bin/env python3
"""
Test runner script for the compression pipeline.

This script provides a convenient way to run all tests with proper configuration
and generate coverage reports.
"""

import sys
import subprocess
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed!")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"ERROR: Command not found. Make sure pytest is installed.")
        return False


def main():
    """Main test runner function."""
    print("Compression Pipeline Test Runner")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not Path("compression_pipeline").exists():
        print("ERROR: compression_pipeline directory not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if pytest is available
    try:
        subprocess.run(["python", "-m", "pytest", "--version"], 
                      check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: pytest not found!")
        print("Please install pytest: pip install pytest pytest-cov")
        sys.exit(1)
    
    # Test configurations
    test_configs = [
        {
            "cmd": ["python", "-m", "pytest", "tests/", "-v"],
            "description": "Basic test suite"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            "description": "Test suite with short traceback"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/", "-v", 
                   "--cov=compression_pipeline", "--cov-report=term-missing"],
            "description": "Test suite with coverage report"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/test_integration.py", "-v"],
            "description": "Integration tests only"
        },
        {
            "cmd": ["python", "-m", "pytest", "tests/", "-k", "not slow", "-v"],
            "description": "Fast tests only (excluding slow tests)"
        }
    ]
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == "basic":
            configs_to_run = [test_configs[0]]
        elif test_type == "coverage":
            configs_to_run = [test_configs[2]]
        elif test_type == "integration":
            configs_to_run = [test_configs[3]]
        elif test_type == "fast":
            configs_to_run = [test_configs[4]]
        elif test_type == "all":
            configs_to_run = test_configs
        else:
            print(f"Unknown test type: {test_type}")
            print("Available options: basic, coverage, integration, fast, all")
            sys.exit(1)
    else:
        # Default: run basic tests and coverage
        configs_to_run = [test_configs[0], test_configs[2]]
    
    # Run selected test configurations
    results = []
    for config in configs_to_run:
        success = run_command(config["cmd"], config["description"])
        results.append((config["description"], success))
    
    # Summary
    print("\\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for description, success in results:
        status = "PASSED" if success else "FAILED"
        print(f"{description}: {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✅")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED! ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()