# MLIP-Platform NEB Test Suite

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
