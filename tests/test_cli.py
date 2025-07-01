import subprocess
import sys
import pathlib

def test_help():
    """Test that the CLI help command works"""
    result = subprocess.run([sys.executable, "-m", "mlip_platform.cli.main", "--help"],
                            capture_output=True, text=True,
                            env={"PYTHONPATH": "src"})
    assert result.returncode == 0

def test_benchmark_help():
    """Test that the benchmark subcommand help works"""
    result = subprocess.run([sys.executable, "-m", "mlip_platform.cli.main", "benchmark", "--help"],
                            capture_output=True, text=True,
                            env={"PYTHONPATH": "src"})
    assert result.returncode == 0

def test_neb_help():
    """Test that the neb subcommand help works"""
    result = subprocess.run([sys.executable, "-m", "mlip_platform.cli.main", "neb", "--help"],
                            capture_output=True, text=True,
                            env={"PYTHONPATH": "src"})
    assert result.returncode == 0

def test_md_help():
    """Test that the md subcommand help works"""
    result = subprocess.run([sys.executable, "-m", "mlip_platform.cli.main", "md", "--help"],
                            capture_output=True, text=True,
                            env={"PYTHONPATH": "src"})
    assert result.returncode == 0
