# 🧠 MLIP Platform

A modular CLI toolkit for evaluating Machine Learning Interatomic Potentials (MLIPs) via:

- **Molecular Dynamics (MD)**
- **Nudged Elastic Band (NEB) simulations**
- **Benchmarking timing**

Supports both **SevenNet** and **MACE** models with isolated virtual environments and streamlined subcommands using [Typer](https://typer.tiangolo.com/).

---

## 🚀 Key Features

✅ Unified CLI: `mlip md`, `mlip neb`, `mlip benchmark`  
🧪 Benchmarking engine with performance comparison  
🔁 IDPP-enhanced NEB interpolation  
🌡️ MD with temperature, timestep, and output logging  
📦 Subcommand-based architecture for easy extensibility  
🧼 Clean separation via subprocess and venvs  
🧪 Full Pytest-based test suite

---

## 📦 Installation

1. **Clone the repo**
```bash
git clone <repo-url>
cd mlip-platform
```

2. **Install package**
```bash
pip install -e .
```

3. **Set up MLIP environments**

You must install SevenNet and MACE in separate virtual environments due to dependency conflicts:

```bash
# SevenNet
python -m venv sevenn-env
source sevenn-env/bin/activate
pip install sevenn

# MACE
python -m venv mace-env
source mace-env/bin/activate
pip install mace
```

---

## 💻 CLI Usage

### Show help
```bash
mlip --help
```

### 🔬 Run MD simulation
```bash
md
```
Prompts for:
- Structure file (.vasp)
- Number of steps
- Temperature (K)
- Timestep (fs)

✔ Outputs: `md.traj`, `md_energy.csv`, plots, and `md_params.txt`

---

### 🧗 Run NEB simulation
```bash
neb
```
Prompts for:
- Initial & final .vasp structures
- Number of NEB images
- IDPP interpolation settings
- Final force convergence criteria

✔ Outputs: `A2B.traj`, `idpp.log`, `neb_data.csv`, `neb_energy.png`, interpolated POSCARs

---

### 📊 Run Benchmark
```bash
benchmark
```
Prompts for structure file, benchmarks SevenNet & MACE via `bench_driver.py`.

✔ Example output:
```json
{
  "mlip": "mace",
  "energy": -3.79,
  "time": 0.27
}
```

---

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run individual model test:
```bash
source sevenn-env/bin/activate
pytest tests/test_neb_sevenn.py
```

---

## 🧠 Scientific Use Cases

- ⚛️ **NEB**: Compute Minimum Energy Pathways (MEP) between atomic states  
- 🧪 **Benchmarking**: Compare speed and accuracy of MLIPs  
- 🌡️ **MD**: Simulate temperature-dependent atomic dynamics  

---

## 🛠️ Developer Notes

- CLI powered by [`typer`](https://typer.tiangolo.com/)
- Entry points (via `setup.py`):
  ```python
  console_scripts=[
    "md = mlip_platform.cli.commands.md:app",
    "neb = mlip_platform.cli.commands.neb:app",
    "benchmark = mlip_platform.cli.commands.benchmark:app"
  ]
  ```
- All outputs/logs are saved next to structure input files
