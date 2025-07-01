import subprocess, sys, pathlib
import tempfile

def test_help():
    cli = pathlib.Path(__file__).parent.parent / "mlip_bench.py"
    result = subprocess.run([sys.executable, cli, "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0

def test_new_python_path_arguments():
    """Test that the new --mace-py and --sevenn-py arguments are accepted"""
    cli = pathlib.Path(__file__).parent.parent / "mlip_bench.py"
    # Test with help to ensure arguments are parsed correctly
    result = subprocess.run([sys.executable, cli, "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--mace-py" in result.stdout
    assert "--sevenn-py" in result.stdout
    assert "Path to Python interpreter for MACE" in result.stdout
    assert "Path to Python interpreter for SevenNet" in result.stdout

def test_custom_python_paths():
    """Test that custom Python paths are used correctly"""
    cli = pathlib.Path(__file__).parent.parent / "mlip_bench.py"
    poscar = pathlib.Path(__file__).parent / "POSCAR"
    
    # Test with custom Python paths
    result = subprocess.run([
        sys.executable, cli, str(poscar),
        "--mace-py", "/usr/bin/python3",
        "--sevenn-py", "/usr/bin/python3"
    ], capture_output=True, text=True)
    
    # Should show the custom Python paths in output
    assert "Using MACE interpreter: /usr/bin/python3" in result.stdout
    assert "Using Sevenn interpreter: /usr/bin/python3" in result.stdout

def test_default_python_paths():
    """Test that default Python paths are used when not specified"""
    cli = pathlib.Path(__file__).parent.parent / "mlip_bench.py"
    poscar = pathlib.Path(__file__).parent / "POSCAR"
    
    # Test without custom Python paths
    result = subprocess.run([
        sys.executable, cli, str(poscar)
    ], capture_output=True, text=True)
    
    # Should show default Python paths in output
    assert "Using MACE interpreter: python" in result.stdout
    assert "Using Sevenn interpreter: python" in result.stdout