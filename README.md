# MLIP Platform

A modular CLI toolkit for evaluating Machine Learning Interatomic Potentials (MLIPs) via:

- **Geometry Optimization**
- **Molecular Dynamics (MD)** with NVE, NVT, and NPT ensembles
- **Nudged Elastic Band (NEB) simulations** with restart support
- **AutoNEB** with dynamic image insertion
- **Benchmarking**

Supports **UMA** (FAIRChem), **SevenNet** (7net), and **MACE** models with automatic detection and streamlined CLI using [Typer](https://typer.tiangolo.com/).

---

## Key Features

- Unified CLI commands: `optimize run`, `md run`, `neb run`, `autoneb run`
- Auto-detection of available MLIP models (UMA > SevenNet > MACE)
- UMA model support with multiple task types (OMat, OC20, OMol, ODAC)
- Geometry optimization with multiple optimizers (FIRE, BFGS, LBFGS, BFGSLineSearch, GPMin, MDMin)
- MD with NVE, NVT, and NPT ensembles
- Configurable thermostats (Langevin, Nose-Hoover, Berendsen) and barostats (MTK NPT, Berendsen NPT)
- NEB with IDPP interpolation, restart support, and highly-constrained mode
- AutoNEB with dynamic image insertion
- Automated plotting and CSV output for all simulations
- Lazy imports for fast CLI startup
- Pytest-based test suite with EMT-based unit tests and MLIP integration tests

---

## Installation

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/manuelarcer/mlip-platform.git
cd mlip-platform
```

2. **Install the package** (also installs [asetools](https://github.com/manuelarcer/asetools) dependency)
```bash
pip install -e .
```

3. **Install MLIP models** (choose one or more)

```bash
# UMA models (FAIRChem) - Recommended; gated on Hugging Face
pip install fairchem-core

# SevenNet
pip install sevenn

# MACE
pip install mace-torch

# CHGNet
pip install chgnet
```

> **Note**: Models can coexist in the same environment. With `--mlip auto` (the default), the CLI picks the first available in the order **UMA → SevenNet → MACE → CHGNet**, or you can pass `--mlip <name>` to force a specific one. UMA additionally requires Hugging Face access — see [UMA_USAGE_GUIDE.md](docs/UMA_USAGE_GUIDE.md#2-hugging-face-access).

### Windows Setup

For detailed Windows installation instructions, see: [Windows Setup Guide](docs/windows_setup_guide.md)

---

## CLI Usage

The package installs the following entry points:

- `mlip` — top-level namespace; `mlip --help` lists every subcommand
- `optimize`, `md`, `neb`, `autoneb`, `autoneb-results`, `benchmark` — standalone aliases

`mlip <subcmd>` is equivalent to running `<subcmd>` directly. For example, `mlip md run --structure POSCAR` and `md run --structure POSCAR` do the same thing. The examples below use the standalone form for brevity. All commands support `--help`.

### Geometry Optimization
```bash
optimize run --structure path/to/structure.vasp
```

**Key options:**
- `--mlip`: Model choice (default: `auto`; explicit options include `uma-s-1p2`, `mace`, `7net-mf-ompa`, `chgnet`)
- `--optimizer`: Algorithm (default: `bfgs`; also `fire`, `lbfgs`, `bfgsls`, `gpmin`, `mdmin`)
- `--fmax`: Force convergence threshold in eV/Å (default: `0.05`)
- `--max-steps`: Maximum optimization steps (default: `200`)

**Example:**
```bash
optimize run --structure POSCAR --mlip uma-s-1p2 --optimizer fire --fmax 0.05
```

**Outputs:** `opt.traj`, `opt.log`, `opt_convergence.csv`, `opt_convergence.png`, `opt_final.vasp`, `opt_params.txt`

---

### Molecular Dynamics
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

For the full MD parameter list, defaults, recommended values for solids, and worked examples, see [MD_REFERENCE.md](docs/MD_REFERENCE.md).

**Example (NVT with Langevin thermostat):**
```bash
md run --structure POSCAR --ensemble nvt --temperature 300 --steps 5000 --thermostat langevin
```

**Example (NPT with Berendsen barostat):**
```bash
md run --structure POSCAR --ensemble npt --temperature 300 --pressure 0.0 --steps 10000 --barostat berendsen
```

**Outputs:** `md.traj`, `md_energy.csv`, `md_energy.png`, `md_temperature.png`, `md_pressure.png` (NPT), `md_volume.png` (NPT), `md_params.txt`

---

### Nudged Elastic Band (NEB)
```bash
neb run --initial path/to/initial.vasp --final path/to/final.vasp
```

**Key options:**
- `--num-images`: Number of intermediate images (default: 5)
- `--fmax`: Force convergence threshold (default: 0.05)
- `--mlip`: Model choice (default: `auto`)
- `--k`: Spring constant (default: 0.1)
- `--climb / --no-climb`: Climbing image NEB (default: enabled)
- `--neb-optimizer`: Optimizer for NEB (`fire`, `mdmin`, `bfgs`, `lbfgs`)
- `--neb-max-steps`: Maximum NEB steps
- `--optimize-endpoints / --no-optimize-endpoints`: Pre-optimize endpoints (default: enabled)

**Example:**
```bash
neb run --initial POSCAR_A --final POSCAR_B --num-images 7 --fmax 0.05 --mlip uma-s-1p1
```

**Outputs:** `A2B.traj`, `A2B_full.traj`, `neb.log`, `neb_convergence.csv`, `neb_convergence.png`, `neb_energy.png`, `neb_data.csv`, `neb_parameters.txt`, POSCAR files (`00/`, `01/`, ...)

#### NEB Restart

Resume a previous NEB calculation with optional parameter overrides:

```bash
neb run --restart
neb run --restart --fmax 0.03 --neb-max-steps 1000
neb run --restart --mlip uma-m-1p1  # Warning: changes MLIP model
```

The restart mechanism:
- Loads images from `A2B_full.traj` and parameters from `neb_parameters.txt`
- Creates a timestamped backup of previous results (`bkup_YYYY.MM.DD_HH.MM.SS/`)
- Allows overriding: `--mlip`, `--fmax`, `--k`, `--climb`, `--neb-optimizer`, `--neb-max-steps`
- Forbids changing: `--initial`, `--final`, `--num-images`, `--relax-atoms`, `--optimize-endpoints`

#### Highly-Constrained NEB

For studying diffusion where most atoms should be fixed:

```bash
neb run --initial POSCAR_A --final POSCAR_B --relax-atoms 0,1,5 --no-optimize-endpoints
```

This fixes all atoms except the specified indices and uses linear interpolation (skips IDPP).

---

### AutoNEB
```bash
autoneb run --initial path/to/initial.vasp --final path/to/final.vasp
```

AutoNEB automatically adds intermediate images until `n_max` is reached, making it ideal for complex reaction pathways.

**Key options:**
- `--n-max`: Target number of images including endpoints (default: 9)
- `--n-simul`: Parallel relaxations (default: 1, requires MPI for >1)
- `--fmax`: Force convergence threshold (default: 0.05)
- `--space-energy-ratio`: Geometric vs energy gap preference for image insertion (default: 0.5)
- `--interpolate-method`: `linear` or `idpp` (default: `idpp`)
- `--prefix`: Output file prefix (default: `autoneb`)

**Example:**
```bash
autoneb run --initial POSCAR_A --final POSCAR_B --n-max 11 --fmax 0.03
```

**Outputs:** `autoneb*.traj` files, `AutoNEB_iter/` folder, `autoneb_parameters.txt`

**Note:** Custom convergence plots are not generated in AutoNEB mode.

---

### AutoNEB Results
```bash
autoneb-results results --directory path/to/results
```

Extract and visualize results from a completed AutoNEB calculation:
- Reads final images from `autoneb*.traj` files
- Calculates energy profile and barrier height
- Generates energy profile plot and CSV
- Optionally exports images as VASP POSCAR files (`--export-poscars`)

---

### Benchmark

```bash
benchmark run --structure path/to/structure.vasp
```

Times a single ``get_potential_energy()`` call for each MLIP installed in the current environment (UMA, SevenNet, MACE, CHGNet). Runs in-process — no working-directory or external script dependency.

**Key options:**
- `--models`: Comma-separated MLIP tags to benchmark (default: every installed MLIP).
- `--uma-task`: UMA task head used for `uma-*` models (default: `omat`).
- `--output`: Optional path for a JSON results file.

**Example:**

```bash
benchmark run --structure POSCAR --models mace,uma-s-1p2 --output bench.json
```

A model that fails to load is recorded in the JSON summary with the exception message; other models continue.

---

## Testing

Run all unit tests (no MLIP required):
```bash
pytest -m "not uma and not mace and not sevenn"
```

Run all tests including UMA integration:
```bash
pytest
```

Run specific test categories:
```bash
pytest tests/test_core_optimize.py     # Optimization tests (EMT)
pytest tests/test_core_md.py           # MD tests (EMT)
pytest tests/test_core_neb.py          # NEB tests (EMT)
pytest tests/test_neb_restart.py       # Restart logic tests
pytest tests/test_cli_commands.py      # CLI help/argument tests
pytest -m uma                          # UMA integration tests only
```

---

## Scientific Use Cases

- **Optimization**: Relax atomic structures to minimum energy configurations
- **MD**: Simulate temperature and pressure-dependent atomic dynamics
- **NEB**: Compute Minimum Energy Pathways (MEP) and transition barriers
- **AutoNEB**: Automatically find complex reaction pathways with adaptive image insertion
- **Benchmarking**: Compare MLIP model performance

---

## Python API

The CLI commands are thin wrappers over a small set of public functions and one class. To call them directly from a script or notebook, see [PYTHON_API.md](docs/PYTHON_API.md). It covers `setup_calculator`, `run_optimization`, `run_md` / `setup_dynamics`, the `CustomNEB` class, parameter-file helpers, and small utilities.

---

## Output Files

For a complete reference of every file each command writes — filename, format, and which command produces it — see [OUTPUTS.md](docs/OUTPUTS.md). It also documents the (different) output-directory conventions: `optimize` and `md` write next to the input structure; `neb` and `autoneb` write into the current working directory.

---

## Developer Notes

- CLI powered by [`typer`](https://typer.tiangolo.com/)
- Entry points defined in [setup.py](setup.py):
  ```python
  console_scripts = [
      "md = mlip_platform.cli.commands.md:app",
      "neb = mlip_platform.cli.commands.neb:app",
      "optimize = mlip_platform.cli.commands.optimize:app",
      "autoneb = mlip_platform.cli.commands.autoneb:app",
      "autoneb-results = mlip_platform.cli.commands.autoneb_results:app",
      "benchmark = mlip_platform.cli.commands.benchmark:app",
  ]
  ```
- Lazy imports for fast CLI startup (no heavy dependencies loaded until needed)
- All simulation outputs saved in same directory as input structure
- Automatic plot generation for all trajectory-based calculations
- Shared utilities in `core/utils.py` (fmax calculation, unit conversions)
- Parameter I/O in `core/params_io.py` (reduces duplication across commands)
- Integrates with [asetools](https://github.com/manuelarcer/asetools) for NEB sanity checks

---

## Changelog

Release notes are tracked in [CHANGELOG.md](CHANGELOG.md). Current version: **0.3.0** (`setup.py`).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, test markers, code conventions, and PR expectations.

---

## Acknowledgments

We gratefully acknowledge the contributions of:

- **Yifan Niu** (National University of Singapore) - Windows setup documentation and testing
- **Lee Yuan Zhang** (National University of Singapore) - Development and testing support

Special thanks to all contributors who have helped improve this platform.
