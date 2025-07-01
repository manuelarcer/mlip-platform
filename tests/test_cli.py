import subprocess
import sys
import pathlib

def test_help():
    cli = pathlib.Path(__file__).parent.parent / "src" / "mlip_platform" / "core" / "mlip_bench.py"
    result = subprocess.run([sys.executable, str(cli), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
