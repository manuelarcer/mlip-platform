# MILP Test Suite

This repository contains a test framework for validating MLIP (Machine-Learned Interatomic Potentials) calculations using **ASE**, **SevenNet**, and **MACE**. It supports structured testing of single-point energy predictions and can be extended to other test scenarios.

## 📁 Project Structure

```
/MILP 1/
│
├── test/
│   └── test_milp_single_point.py   # Pytest script for energy validation
│
├── pytest.ini                      # Config for pytest discovery
├── ...
```

## 🐍 Python & Environment

- Python version: **3.11.13**
- Two virtual environments are created outside of the project folder:
  - `mace-env`: for MACE-related testing
  - `sevenn-env`: for SevenNet-based testing

These environments must be activated before running tests.

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/YuanZhang28/mlip-platform.git
cd mlip-platform/MILP\ 1
```

### 2. Activate one of the environments

For SevenNet:
```bash
source ../sevenn-env/bin/activate
```

For MACE:
```bash
source ../mace-env/bin/activate
```

### 3. Install dependencies

Inside the active environment:

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:
```bash
pip install ase pytest
pip install sevenn  # If using SevenNet
pip install mace-torch==0.3.13 e3nn==0.4.4  # If using MACE
```

---

## 🧪 Running Tests

Tests are run with `pytest`. From the root of `MILP 1`, run:

```bash
pytest
```

Or run a specific test file:
```bash
pytest test/test_milp_single_point.py
```

---

## 🔬 Test Example Description

### `test_milp_single_point.py`

- Loads a `POSCAR` file (VASP format)
- Attaches a machine-learned potential (SevenNet or MACE)
- Validates that an energy value is returned correctly (as `float`)

---

## 📄 pytest.ini

Configured for test discovery inside the `test/` folder:

```ini
[pytest]
testpaths = test
python_files = test_*.py
```

---

## 📦 Recommended `.gitignore`

If you keep your virtual environments outside of the project, ignore them like this:

```gitignore
../mace-env/
../sevenn-env/
__pycache__/
*.pyc
```

---

## 📌 Notes

- Ensure `e3nn` is pinned to `0.4.4` for compatibility with `mace-torch==0.3.13`
- SevenNet requires checkpoint files and may depend on correct model paths
