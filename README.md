# MLIP Platform

Package to use machine learning interatomic potentials (MLIPs) for atomic simulations including optimization, molecular dynamics, and transition state search.

---

## 🚀 Features

### Benchmarking Tool
- Runs MLIP calculations in **isolated virtual environments**
- Supports both **MACE** and **SevenNet**
- Reports **energy** and **timing** in a unified summary
- Designed for **research** and **performance comparison**

### NEB Test Suite
- Custom `CustomNEB` wrapper class using ASE's NEB tools
- Supports both **SevenNet** and **MACE** models via clean `mlip=` argument
- IDPP interpolation for improved initial path generation
- Modular, testable design with pytest
- Virtual environment separation to avoid dependency clashes

---

## 📁 Folder Structure

```
mlip_platform/
├── src/
│   └── mlip_platform/
│       ├── __init__.py
│       └── neb.py              # Core NEB logic
├── test/
│   ├── POSCAR                  # Example VASP-format structure
│   ├── POSCAR_initial          # Initial structure for NEB
│   ├── POSCAR_final            # Final structure for NEB
│   ├── test_cli.py             # CLI functionality test
│   ├── test_neb_functionality_sevenn.py  # SevenNet NEB test
│   └── test_neb_functionality_mace.py    # MACE NEB test
├── mlip_bench.py               # Main CLI benchmarking tool
├── bench_driver.py             # Worker script for benchmarking
├── README.md
└── pytest.ini
```

---

## 🔧 Setup

### 1. Create and activate virtual environments:

```bash
# MACE environment
python3 -m venv ~/Documents/mace-env
source ~/Documents/mace-env/bin/activate
pip install mace

# SevenNet environment
python3 -m venv ~/Documents/sevenn-env
source ~/Documents/sevenn-env/bin/activate
pip install sevenn
```

### 2. Update paths in scripts as needed for your environment

---

## ▶️ Usage

### Benchmarking Tool
Run the benchmark on any VASP-format structure file:

```bash
python mlip_bench.py test/POSCAR
```

You can specify custom Python interpreters for each MLIP:

```bash
python mlip_bench.py test/POSCAR --mace-py /path/to/mace-env/bin/python --sevenn-py /path/to/sevenn-env/bin/python
```

#### CLI Options:
- `--mace-py`: Path to Python interpreter for MACE (default: 'python')
- `--sevenn-py`: Path to Python interpreter for SevenNet (default: 'python')

Example output:
```
=== Results ===
MACE   : -3.797862 eV  | 0.27 s
Sevenn : -3.801770 eV  | 0.44 s
```

### NEB Tests
Each MLIP is tested with its own virtual environment:

#### Run SevenNet Test
```bash
source ~/Documents/sevenn-env/bin/activate
pytest test/test_neb_functionality_sevenn.py
```

#### Run MACE Test
```bash
source ~/Documents/mace-env/bin/activate
pytest test/test_neb_functionality_mace.py
```

---

## 🧪 Testing

Run all tests:
```bash
pytest test/
```

---

## 💡 Customization

To switch MLIP in `CustomNEB`, change the `mlip` parameter:
```python
mlip="sevenn-mf-ompa"  # or "mace-medium"
```

---

## 🧠 Scientific Purpose

- **Benchmarking**: Compare MLIPs performance on energy calculations
- **NEB**: Estimate minimum energy paths (MEP) between atomic states for:
  - Diffusion processes
  - Chemical reactions  
  - Phase transitions

This setup uses MLIPs for fast approximations compared to DFT calculations.

---

## 🧠 Credits

This design was inspired by real research needs where MLIPs can't co-exist due to dependency conflicts. It separates concerns using subprocesses and virtual environments while staying simple and reproducible.

This repository contains a modular NEB (Nudged Elastic Band) test framework for evaluating machine learning interatomic potentials (MLIPs) using both **SevenNet** and **MACE**. The goal is to benchmark MLIPs on transition path modeling between atomic structures using ASE.

---

## 🔧 Features

- Custom `CustomNEB` wrapper class using ASE's NEB tools
- Supports both **SevenNet** and **MACE** models via clean `mlip=` argument
- IDPP interpolation for improved initial path generation
- Modular, testable design with pytest
- Virtual environment separation to avoid dependency clashes

---

## 📁 File Structure

```
mlip-platform-(NEB)/
├── src/
│ └── milp_platform/
│ ├── init.py
│ └── neb.py # Core NEB logic
├── test/
│ ├── POSCAR_initial # Initial structure
│ ├── POSCAR_final # Final structure
│ ├── test_neb_functionality_sevenn.py # SevenNet NEB test
│ └── test_neb_functionality_mace.py # MACE NEB test
├── README.md
└── pytest.ini

---

## 🧪 Tests

Each MLIP is tested with its own virtual environment:

### ▶️ Run SevenNet Test

Activate your SevenNet env:

```bash
source ~/Documents/sevenn-env/bin/activate
pytest test/test_neb_functionality_sevenn.py
```

### ▶️ Run MACE Test

Activate your MACE env:

```bash
source ~/Documents/mace-env/bin/activate
pytest test/test_neb_functionality_mace.py
```

---

## 💡 Customization

To switch MLIP in `CustomNEB`, change the `mlip` parameter:
```python
mlip="sevenn-mf-ompa"  # or "mace-medium"
```

---

## 🛠️ Virtual Environments

Due to dependency conflicts, SevenNet and MACE are run in **separate Python virtual environments**. Make sure each environment has its respective package installed:

### Example setup:
```bash
# Create and activate
python3.11 -m venv sevenn-env
source sevenn-env/bin/activate

# Inside env
pip install -e /path/to/sevenn
```

Repeat similarly for MACE.

---

## 🧠 Scientific Purpose

NEB is used to estimate the **minimum energy path (MEP)** between two atomic states — relevant for:
- Diffusion
- Reactions
- Phase transitions

This setup uses MLIPs to perform fast approximations compared to DFT.

---

## 📌 TODO / Future

- Add `run()` test to validate force convergence
- Compare energy profiles along NEB path
- Auto-log timing and ΔE summary
