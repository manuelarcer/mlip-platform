# UMA Models Usage Guide

This guide explains how to use UMA (Universal Materials Accelerator) models in the MLIP Platform.

## Supported UMA Models

The platform now supports both UMA model sizes:

| Model | Size | Description |
|-------|------|-------------|
| **uma-s-1p1** | Small | Faster, lower memory, suitable for quick calculations |
| **uma-m-1p1** | Medium | More accurate, higher computational cost |

Both models support the same task types: `omat`, `oc20`, `omol`, `odac`

## Installation

Install the required package:

```bash
pip install fairchem-core
```

Then reinstall the MLIP platform to register the new features:

```bash
pip install -e .
```

## Usage Examples

### 1. Geometry Optimization

#### Auto-detect model (will use uma-s-1p1 if available)
```bash
optimize --structure structure.vasp
```

#### Explicitly specify small UMA model
```bash
optimize --structure structure.vasp --mlip uma-s-1p1 --uma-task omat
```

#### Use larger UMA model for better accuracy
```bash
optimize --structure structure.vasp --mlip uma-m-1p1 --uma-task omat --fmax 0.01
```

#### Different task types
```bash
# Materials (default)
optimize --structure structure.vasp --mlip uma-s-1p1 --uma-task omat

# OC20 dataset
optimize --structure structure.vasp --mlip uma-s-1p1 --uma-task oc20

# Molecular properties
optimize --structure structure.vasp --mlip uma-s-1p1 --uma-task omol

# ODAC dataset
optimize --structure structure.vasp --mlip uma-s-1p1 --uma-task odac
```

#### With different optimizers
```bash
# FIRE optimizer (fast, default)
optimize --structure structure.vasp --mlip uma-m-1p1 --optimizer fire

# BFGS optimizer (small systems)
optimize --structure structure.vasp --mlip uma-m-1p1 --optimizer bfgs

# LBFGS optimizer (large systems)
optimize --structure structure.vasp --mlip uma-m-1p1 --optimizer lbfgs
```

### 2. NEB (Nudged Elastic Band)

#### Auto-detect model
```bash
neb --initial initial.vasp --final final.vasp
```

#### Specify UMA model
```bash
neb --initial initial.vasp --final final.vasp --mlip uma-s-1p1 --uma-task omat
```

#### Use larger model for higher accuracy
```bash
neb --initial initial.vasp --final final.vasp \
    --mlip uma-m-1p1 \
    --uma-task omat \
    --num-images 9 \
    --fmax 0.01
```

### 3. Molecular Dynamics

#### Auto-detect model
```bash
md --structure structure.vasp --steps 1000 --temperature 300
```

#### Specify UMA model
```bash
md --structure structure.vasp \
   --steps 1000 \
   --temperature 300 \
   --timestep 1.0 \
   --mlip uma-s-1p1 \
   --uma-task omat
```

#### High-temperature MD with larger model
```bash
md --structure structure.vasp \
   --steps 5000 \
   --temperature 1000 \
   --timestep 0.5 \
   --mlip uma-m-1p1 \
   --uma-task omat
```

### 4. Benchmarking

```bash
# Small model
python bench_driver.py structure.vasp uma-s-1p1 omat

# Large model
python bench_driver.py structure.vasp uma-m-1p1 omat

# Different task
python bench_driver.py structure.vasp uma-s-1p1 oc20
```

## UMA Task Types

| Task | Description | Recommended Use Case |
|------|-------------|---------------------|
| **omat** | Open Materials | General inorganic materials, crystals, surfaces |
| **oc20** | OC20 Dataset | Catalysis, adsorbates on surfaces |
| **omol** | Open Molecules | Molecular systems, organic molecules |
| **odac** | ODAC Dataset | Specific applications for ODAC dataset |

## Model Selection Guidelines

### When to use **uma-s-1p1** (Small):
- Quick screening calculations
- Large systems (>200 atoms)
- Limited computational resources
- Initial structure relaxations before refinement
- High-throughput workflows

### When to use **uma-m-1p1** (Medium):
- Final production calculations
- Systems requiring high accuracy
- Energy barrier calculations (NEB)
- Small to medium systems (<200 atoms)
- Publication-quality results

## Performance Comparison

Approximate timing (single-point energy, 64 atom system, CPU):

| Model | Time | Relative Accuracy |
|-------|------|-------------------|
| uma-s-1p1 | ~0.3s | Good |
| uma-m-1p1 | ~0.8s | Better |

## Programmatic Usage (Python API)

You can also use UMA models programmatically:

```python
from ase.io import read
from ase.optimize import FIRE
from fairchem.core import pretrained_mlip, FAIRChemCalculator

# Load structure
atoms = read("structure.vasp")

# Setup UMA calculator
predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cpu")
calc = FAIRChemCalculator(predictor, task_name="omat")
atoms.calc = calc

# Run optimization
opt = FIRE(atoms, trajectory="opt.traj")
opt.run(fmax=0.05, steps=200)

# Get results
final_energy = atoms.get_potential_energy()
print(f"Final energy: {final_energy:.6f} eV")
```

### NEB with UMA (Python API)

```python
from ase.io import read
from mlip_platform.core.neb import CustomNEB

# Load structures
initial = read("initial.vasp")
final = read("final.vasp")

# Create NEB calculation
neb = CustomNEB(
    initial=initial,
    final=final,
    num_images=9,
    mlip="uma-m-1p1",
    uma_task="omat",
    fmax=0.05
)

# Run calculation
neb.interpolate_idpp()
neb.run_neb()
df = neb.process_results()
neb.plot_results(df)
```

## Troubleshooting

### Model not found
```
Error: Could not find model 'uma-s-1p1'
```
**Solution**: Install fairchem-core: `pip install fairchem-core`

### Memory issues with uma-m-1p1
**Solution**: Use uma-s-1p1 for large systems, or reduce system size

### Device must be 'cpu'
UMA models in this platform are configured to use CPU only. GPU support would require modifying the calculator setup.

## Output Files

All commands produce standard output files:

**Optimization:**
- `opt.traj` - Trajectory
- `opt_convergence.png` - Convergence plot
- `opt_final.vasp` - Final structure
- `opt_params.txt` - Parameters (includes UMA model and task)

**NEB:**
- `A2B.traj` - NEB path
- `neb_energy.png` - Energy profile
- `neb_data.csv` - Energy data
- POSCARs in subdirectories

**MD:**
- `md.traj` - Trajectory
- `md_energy.csv` - Energy vs time
- `md_energy.png`, `md_temperature.png` - Plots

## Complete Example Workflow

```bash
# 1. Initial relaxation with fast model
optimize --structure initial.vasp \
         --mlip uma-s-1p1 \
         --uma-task omat \
         --fmax 0.1

# 2. Final high-accuracy optimization
optimize --structure opt_final.vasp \
         --mlip uma-m-1p1 \
         --uma-task omat \
         --fmax 0.01 \
         --optimizer bfgs

# 3. NEB calculation between states
neb --initial state_A.vasp \
    --final state_B.vasp \
    --mlip uma-m-1p1 \
    --uma-task omat \
    --num-images 11 \
    --fmax 0.05

# 4. MD simulation at transition state
md --structure neb_saddle.vasp \
   --mlip uma-s-1p1 \
   --uma-task omat \
   --steps 10000 \
   --temperature 500 \
   --timestep 1.0
```

## Model Detection Priority

When using `--mlip auto`, the platform detects models in this order:

1. **UMA** (uma-s-1p1 by default if fairchem-core available)
2. **SevenNet** (7net-mf-ompa if sevenn available)
3. **MACE** (if mace-torch available)

To force a specific model, always use `--mlip <model_name>`.
