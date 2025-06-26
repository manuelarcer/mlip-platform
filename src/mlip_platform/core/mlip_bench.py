import argparse
import subprocess
import json
import re

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
        # Use regex to extract the first JSON object from stdout
        match = re.search(r'\{.*?\}', result.stdout, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            print(f"[ERROR] No JSON object found in {mlip_name}.stdout")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to decode JSON from {mlip_name}.stdout\nReason: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Run MLIP benchmarks on a structure file.")
    parser.add_argument("structure", help="Path to the structure file (e.g., POSCAR, CONTCAR, or *.vasp)")
    args = parser.parse_args()

    structure_path = args.structure
    print(f"\nStructure file to process: {structure_path}")
    print(f"Using MACE interpreter: {PY_MACE}")
    print(f"Using Sevenn interpreter: {PY_SEVENN}")

    mace_result = run_driver(PY_MACE, structure_path, "mace")
    sevenn_result = run_driver(PY_SEVENN, structure_path, "sevenn")

    print("\n=== Results ===")
    if mace_result:
        print(f"MACE   : {mace_result['energy']:.6f} eV  | {mace_result['time']:.2f} s")
    else:
        print("MACE   : Failed")

    if sevenn_result:
        print(f"Sevenn : {sevenn_result['energy']:.6f} eV  | {sevenn_result['time']:.2f} s")
    else:
        print("Sevenn : Failed")

if __name__ == "__main__":
    main()
