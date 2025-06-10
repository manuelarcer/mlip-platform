# MLIP Platform – Test Suite

This repository provides a minimal test framework to validate machine learning interatomic potentials (MLIPs) using **SevenNet** and **MACE** on a sample structure.

## 📂 Folder Structure

```
mlip-platform/
├── test/
│   ├── test_mlip_single_point_sevenn.py
│   ├── test_mlip_single_point_mace.py
│   └── assets/
│       └── POSCAR                 # Small sample structure for testing
├── pytest.ini                     # Pytest config
└── README.md                      # This file
```

## 🔍 What This Does

- Verifies that both SevenNet and MACE are correctly installed and configured.
- Performs a **single-point energy calculation** on a small test structure (`POSCAR`).
- Ensures that `atoms.get_potential_energy()` returns a valid float.

## 🧪 Test Cases

### ✅ SevenNet python 3.13

Test file: `test/test_mlip_single_point_sevenn.py`

```python
from sevenn.calculator import SevenNetCalculator
atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")
energy = atoms.get_potential_energy()
assert isinstance(energy, float)
```

### ✅ MACE python 3.11

Test file: `test/test_mlip_single_point_mace.py`

```python
from mace.calculators import mace_mp
atoms.calc = mace_mp(model="medium", device="cpu")
energy = atoms.get_potential_energy()
assert isinstance(energy, float)
```

## 🧰 Requirements

Install packages per environment:

### For SevenNet:
```bash
pip install ase
pip install sevenn
```

### For MACE:
```bash
pip install ase
pip install mace-torch
```

> Ensure `POSCAR` exists at `test/assets/POSCAR` before running tests.

## 🏃 Running Tests

```bash
pytest test
```

Use virtual environments (e.g. `venv`, `conda`, `pyenv`) to avoid conflicts between SevenNet and MACE dependencies.

## 🤝 Contributions

Feel free to fork or open issues to add more calculators, force tests, or NEB pathways.

## ⚙️ pytest.ini

The `pytest.ini` file is a minimal configuration to tell `pytest` where to find tests and how to behave. It should look like this:

```ini
[pytest]
testpaths = test
python_files = test_*.py
```

This setup ensures:
- All test files starting with `test_` in the `test/` folder are automatically discovered and run.
- Pytest knows where to look even without additional command-line arguments.

You can run tests simply with:
```bash
pytest
```
