# 🧠 MLIP Platform

A modular CLI toolkit for evaluating Machine Learning Interatomic Potentials (MLIPs) via:

- **Geometry Optimization**
- **Molecular Dynamics (MD)** with NVE, NVT, and NPT ensembles
- **Nudged Elastic Band (NEB) simulations**
- **Benchmarking timing**

Supports **UMA** (FAIRChem), **SevenNet** (7net), and **MACE** models with automatic detection and streamlined CLI using [Typer](https://typer.tiangolo.com/).

---

## 🚀 Key Features

✅ Unified CLI commands: `md`, `neb`, `optimize`
🤖 Auto-detection of available MLIP models (UMA, SevenNet, MACE)
🌐 UMA model support with multiple task types (OMat, OC20, OMol, ODAC)
🔧 Geometry optimization with multiple optimizers (FIRE, BFGS, LBFGS, etc.)
🌡️ MD with NVE, NVT, and NPT ensembles
🎛️ Configurable thermostats (Langevin, Nosé-Hoover, Berendsen) and barostats
🔁 IDPP-enhanced NEB interpolation
📊 Automated plotting and CSV output for all simulations
🧼 Lazy imports for fast CLI startup
🧪 Full Pytest-based test suite
🚧 Benchmark feature (in development - see roadmap below)

---

## 📦 Installation

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/manuelarcer/mlip-platform.git
cd mlip-platform
```

2. **Install the package**
```bash
pip install -e .
```

3. **Install MLIP models** (choose one or more)

The platform supports three MLIP backends. You can install any combination:

```bash
# UMA models (FAIRChem) - Recommended
pip install fairchem-core

# SevenNet
pip install sevenn

# MACE
pip install mace-torch
```

> **Note**: Models can coexist in the same environment. The CLI will auto-detect available models or you can specify with `--mlip` flag.

### Windows Setup

For detailed Windows installation instructions (Python, virtual environments, Git, and UMA setup), see:
- [Windows Setup Guide](docs/windows_setup_guide.md)

---

## 💻 CLI Usage

All commands support `--help` to see available options.

### 🔧 Geometry Optimization
```bash
optimize run --structure path/to/structure.vasp
```

**Key options:**
- `--mlip`: Model choice (`auto`, `uma-s-1p1`, `uma-m-1p1`, `mace`, `7net-mf-ompa`)
- `--optimizer`: Algorithm (`fire`, `bfgs`, `lbfgs`, `mdmin`, etc.)
- `--fmax`: Force convergence threshold (eV/Å)
- `--max-steps`: Maximum optimization steps

**Example:**
```bash
optimize run --structure POSCAR --mlip uma-s-1p1 --optimizer fire --fmax 0.05
```

**Outputs:** `opt.traj`, `opt.log`, `opt_convergence.csv`, `opt_convergence.png`, `opt_final.vasp`, `opt_params.txt`

---

### 🔬 Molecular Dynamics
```bash
md run --structure path/to/structure.vasp
```

**Key options:**
- `--ensemble`: MD ensemble (`nve`, `nvt`, `npt`)
- `--temperature`: Temperature in K (for NVT/NPT)
- `--pressure`: Pressure in GPa (for NPT)
- `--steps`: Number of MD steps
- `--timestep`: Timestep in fs
- `--thermostat`: For NVT (`langevin`, `nose-hoover`, `berendsen`)
- `--barostat`: For NPT (`npt`, `berendsen`)
- `--mlip`: Model choice (default: `auto`)

**Example (NVT with Langevin thermostat):**
```bash
md run --structure POSCAR --ensemble nvt --temperature 300 --steps 5000 --timestep 1.0 --thermostat langevin
```

**Example (NPT with Berendsen barostat):**
```bash
md run --structure POSCAR --ensemble npt --temperature 300 --pressure 0.0 --steps 10000 --barostat berendsen
```

**Outputs:** `md.traj`, `md_energy.csv`, `md_energy.png`, `md_temperature.png`, `md_pressure.png` (NPT only), `md_volume.png` (NPT only), `md_params.txt`

---

### 🧗 Nudged Elastic Band (NEB)
```bash
neb run --initial path/to/initial.vasp --final path/to/final.vasp
```

**Key options:**
- `--nimages`: Number of intermediate images
- `--fmax`: Force convergence threshold
- `--mlip`: Model choice (default: `auto`)

**Example:**
```bash
neb run --initial POSCAR_A --final POSCAR_B --nimages 7 --fmax 0.05 --mlip uma-s-1p1
```

**Outputs:** `neb.traj`, `neb_data.csv`, `neb_energy.png`, interpolated POSCARs (00/, 01/, ...), `neb_params.txt`

---

### 📊 Benchmark (🚧 In Development)

The benchmark functionality is being redesigned. See [Development Roadmap](#-development-roadmap) below for planned features.

---

## 🧪 Testing

Run all tests:
```bash
pytest
```

Run specific tests:
```bash
pytest tests/test_md_mace.py          # MD with MACE
pytest tests/test_neb_sevenn.py       # NEB with SevenNet
pytest tests/test_cli.py              # CLI interface tests
```

---

## 🧠 Scientific Use Cases

- 🔧 **Optimization**: Relax atomic structures to minimum energy configurations
- 🌡️ **MD**: Simulate temperature and pressure-dependent atomic dynamics
- ⚛️ **NEB**: Compute Minimum Energy Pathways (MEP) and transition barriers between atomic states
- 📊 **Benchmarking**: Compare MLIP model performance (energy, forces, timing)

---

## 🚧 Development Roadmap

### Benchmark Feature Redesign

The benchmark functionality is being redesigned to provide comprehensive MLIP validation against DFT reference data:

**Planned Features:**

1. **Reference Database System**
   - Create structured databases for DFT calculations:
     - `opt_benchmark_db.pkl`: Geometry optimization benchmarks
       - Fields: unique_id, VASP parameters, cell optimization flag, initial config, optimized config, lattice parameters, energy, max_force
     - `neb_benchmark_db.pkl`: NEB pathway benchmarks (future)
   - Dataframe-based storage for easy analysis and filtering

2. **Automated Benchmark Workflow**
   - Extract initial configurations from reference database
   - Generate separate calculation directories for each benchmark case (named by unique_id)
   - Run MLIP calculations in parallel or via job submission system
   - Support for large-scale benchmarking campaigns

3. **Comparison and Analysis**
   - Compare MLIP results vs DFT reference:
     - Energy differences
     - Force field accuracy (RMSE, MAE)
     - Lattice parameter deviations
     - Geometry differences
   - Generate statistical summaries and visualization plots
   - Export comparison results to CSV/JSON

4. **Multi-Model Comparison**
   - Benchmark all available MLIP models (UMA, MACE, SevenNet) against same reference set
   - Side-by-side accuracy and performance metrics

**Use Cases:**
- Validate MLIP model accuracy for specific material systems
- Select optimal MLIP for production calculations
- Track model improvements across versions
- Identify systematic errors or failure modes

---

## 🛠️ Developer Notes

- CLI powered by [`typer`](https://typer.tiangolo.com/)
- Entry points defined in [setup.py](setup.py:16-21):
  ```python
  console_scripts=[
    "md = mlip_platform.cli.commands.md:app",
    "neb = mlip_platform.cli.commands.neb:app",
    "optimize = mlip_platform.cli.commands.optimize:app",
    "benchmark = mlip_platform.cli.commands.benchmark:app"
  ]
  ```
- Lazy imports for fast CLI startup (no heavy dependencies loaded until needed)
- All simulation outputs saved in same directory as input structure
- Automatic plot generation for all trajectory-based calculations

---

## 🙏 Acknowledgments

We gratefully acknowledge the contributions of:

- **Yifan Niu** (National University of Singapore) - Windows setup documentation and testing
- **Lee Yuan Zhang** (National University of Singapore) - Development and testing support

Special thanks to all contributors who have helped improve this platform.
