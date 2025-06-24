# MLIP Benchmarking Tool

This tool benchmarks different ML interatomic potentials (MLIPs) — like **MACE** and **SevenNet** — on the same structure file. It compares both the **computed energy** and **runtime**, even though each MLIP lives in a separate Python virtual environment.

---

## 🚀 Features

- Runs MLIP calculations in **isolated virtual environments**
- Supports both **MACE** and **SevenNet**
- Reports **energy** and **timing** in a unified summary
- Designed for **research** and **performance comparison**

---

## 📁 Folder Structure

```text
mlip-platform-1/
├── mlip_bench.py           # Main CLI tool (legacy)
├── bench_driver.py         # Worker script for single-point energy timing
├── src/
│   └── mlip_platform/
│       ├── __init__.py
│       ├── cli.py          # New CLI entry point
│       ├── optim.py        # Geometry optimization runner
│       ├── neb.py          # NEB runner
│       └── md.py           # Molecular dynamics runner
├── test/
│   ├── POSCAR              # Example VASP-format structure
│   ├── test_cli.py         # Legacy MLIP bench CLI smoke test
│   └── test_cli_experimental.py # New CLI subcommands smoke test
``` 

---

## 🔧 Setup

### 1. Create and activate two virtual environments:

```bash
# MACE environment
python3 -m venv ~/Documents/mace-env
source ~/Documents/mace-env/bin/activate
pip install mace  # and any dependencies you need

# SevenNet environment
python3 -m venv ~/Documents/sevenn-env
source ~/Documents/sevenn-env/bin/activate
pip install sevenn  # and any dependencies you need
```

### 2. Update `mlip_bench.py` with your interpreter paths:
```python
PY_MACE = "/Users/yourname/Documents/mace-env/bin/python"
PY_SEVENN = "/Users/yourname/Documents/sevenn-env/bin/python"
```

---

## ▶️ Usage

Run the benchmark on any VASP-format structure file:

```bash
python mlip_bench.py test/POSCAR
```

---

## 💻 CLI Interface (experimental)

A new `mlip` command provides subcommands for optimization, NEB, and MD:

```bash
mlip --help
mlip optimize POSCAR --model sevenn-mf-ompa --fmax 0.05
mlip neb initial.vasp final.vasp --model mace --images 9 \
    --fmax 0.05 --interp-fmax 0.1 --interp-steps 1000 --climb
mlip md POSCAR --model sevenn-mf-ompa --temperature 300 \
    --timestep 1 --steps 1000
```

Example output:

```
=== Results ===
MACE   : -3.797862 eV  | 0.27 s
Sevenn : -3.801770 eV  | 0.44 s
```

---

## ✅ Testing

Minimal test (runs `--help` on the CLI):

```bash
pytest tests/test_cli.py
```

---

## ✍️ How It Works

- `mlip_bench.py` is the main script. It:
  - Takes a structure file as input
  - Calls `bench_driver.py` in each virtual env
  - Parses energy + timing from the printed JSON

- `bench_driver.py`:
  - Loads the structure using ASE
  - Initializes the MLIP calculator
  - Times the energy calculation
  - Prints a JSON summary

---

## 📌 Notes

- You can easily add more MLIPs by modifying `bench_driver.py`
- This is useful for publishing **apples-to-apples comparisons** of MLIPs

---

## 🧠 Credits

This benchmarking design was inspired by real research needs where MLIPs can't co-exist due to dependency conflicts. It separates concerns using subprocesses and virtual environments, while staying simple and reproducible.
# mlip-platform

Package to use reported MLIP for atomic simulations (Optimization, MD, TS search, etc)
