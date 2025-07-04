import sys
import os
import json
import subprocess
import importlib.util

def _call_driver(structure_path: str, mlip_name: str) -> dict:
    """
    Run bench_driver.py as a subprocess and parse its JSON output.
    """
    # Locate bench_driver.py two levels up from this file
    driver_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "bench_driver.py")
    )
    cmd = [sys.executable, driver_path, structure_path, mlip_name]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE, text=True)
        return json.loads(out)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"bench_driver.py failed for MLIP '{mlip_name}':\n{e.stderr}"
        ) from e

def run_benchmark_all(structure_path: str, model: str = None) -> dict:
    """
    Benchmark single-point energy for both MACE and SevenNet via bench_driver.py.

    Parameters
    ----------
    structure_path : str
        Path to the input structure file.
    model : str, optional
        (Ignored by bench_driver; you can extend bench_driver to accept it.)

    Returns
    -------
    dict
        {
          "mace":    {"mlip": "mace",    "energy": ..., "time": ...} or None,
          "sevenn":  {"mlip": "sevenn",  "energy": ..., "time": ...} or None
        }
    """
    results = {}

    # Define the MLIP names as accepted by bench_driver.py
    for mlip in ("mace", "sevenn"):
        # Skip if the MLIP package isn’t even installed
        pkg_name = "mace" if mlip == "mace" else "sevenn"
        if importlib.util.find_spec(pkg_name) is None:
            results[mlip] = None
            continue

        # Call your driver
        results[mlip] = _call_driver(structure_path, mlip)

    return results
