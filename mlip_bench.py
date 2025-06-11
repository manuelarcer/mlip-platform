# mlip_bench.py

import argparse

# Hard-coded interpreters for MACE and Sevenn (replace with your actual paths)
PY_MACE = "/Users/leeyuanzhang/Documents/mace-env/bin/python"
PY_SEVENN = "/User/leeyuanzhang/Documents/sevenn/bin/python"

def main():
    parser = argparse.ArgumentParser(description="Run MLIP benchmarks on a structure file.")
    parser.add_argument("structure", help="Path to the structure file (e.g., POSCAR, CONTCAR, or *.vasp)")
    args = parser.parse_args()

    structure_path = args.structure
    print(f"Structure file to process: {structure_path}")
    print(f"Using MACE interpreter: {PY_MACE}")
    print(f"Using Sevenn interpreter: {PY_SEVENN}")

if __name__ == "__main__":
    main()
