import subprocess, sys, pathlib
def test_help():
    cli = pathlib.Path(__file__).parent.parent / "mlip_bench.py"
    result = subprocess.run([sys.executable, cli, "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0