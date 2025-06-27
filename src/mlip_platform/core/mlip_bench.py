import subprocess
import json
import re
from pathlib import Path

# === Update these paths to match your virtual environments ===
PY_MACE = "/Users/leeyuanzhang/Documents/mace-env/bin/python"
PY_SEVENN = "/Users/leeyuanzhang/Documents/sevenn-env/bin/python"


def run_driver(python_exec, structure, mlip_name):
    print(f"\n[INFO] Running {mlip_name} with {python_exec}")
    result = subprocess.run(
        [python_exec, "bench_driver.py", structure, mlip_name],
        capture_output=True,
        text=True
    )

    print(f"\n--- {mlip_name.upper()} STDOUT ---\n{result.stdout}")
    print(f"--- {mlip_name.upper()} STDERR ---\n{result.stderr}")

    if result.returncode != 0:
        print(f"[ERROR] {mlip_name} subprocess failed with exit code {result.returncode}")
        return None

    try:
        match = re.search(r'\{.*?\}', result.stdout, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            print(f"[ERROR] No JSON object found in {mlip_name}.stdout")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to decode JSON from {mlip_name}.stdout\nReason: {e}")
        return None


def run_benchmark(structure: Path):
    structure_path = str(structure)

    mace_result = run_driver(PY_MACE, structure_path, "mace")
    sevenn_result = run_driver(PY_SEVENN, structure_path, "sevenn")

    results = {}
    if mace_result:
        results["mace"] = {
            "energy": mace_result["energy"],
            "time": mace_result["time"],
        }
    else:
        results["mace"] = "Failed"

    if sevenn_result:
        results["sevenn"] = {
            "energy": sevenn_result["energy"],
            "time": sevenn_result["time"],
        }
    else:
        results["sevenn"] = "Failed"

    return results
