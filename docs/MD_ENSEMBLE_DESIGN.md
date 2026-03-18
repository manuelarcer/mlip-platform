# MD Ensemble Design Document

## Overview

Enhanced MD module supporting multiple ensembles (NVE, NVT, NPT) with various thermostats and barostats, optimized for **solid-state materials** simulations.

## Ensemble Selection Strategy

### For Solid-State Materials (Common Use Cases):

| Use Case | Recommended Ensemble | Thermostat/Barostat |
|----------|---------------------|---------------------|
| **Energy conservation test** | NVE | VelocityVerlet |
| **Constant temperature** | NVT | Langevin or Nosé-Hoover |
| **Fast equilibration** | NVT | Berendsen |
| **Lattice relaxation** | NPT | NPT (isotropic) |
| **Crystal structures** | NPT | Berendsen NPT (fast) |

## Implementation Plan

### 1. Supported Ensembles

#### **NVE (Microcanonical Ensemble)**
- **VelocityVerlet** - Most common, energy-conserving
  - No parameters needed
  - Best for: Energy conservation tests, stable simulations

#### **NVT (Canonical Ensemble)**
- **Langevin** - Most popular for MD
  - Parameters: `friction` (1/time units)
  - Best for: General MD, good sampling, realistic dynamics
  - Typical friction: 0.001-0.01 fs⁻¹

- **Nosé-Hoover** - Rigorous thermostat
  - Parameters: `ttime` (characteristic time)
  - Best for: Accurate canonical sampling
  - Typical ttime: 25-100 fs

- **Berendsen** - Fast equilibration (not ergodic)
  - Parameters: `taut` (coupling time)
  - Best for: Initial equilibration only
  - Typical taut: 100-500 fs

#### **NPT (Isothermal-Isobaric Ensemble)**
- **NPT (Isotropic)** - Martyna-Tobias-Klein
  - Parameters: `ttime`, `pfactor`, `pressure`
  - Best for: Lattice constant optimization, isotropic expansion
  - Typical: ttime=25 fs, pfactor=(ttime*75 GPa)²

- **NPTBerendsen** - Fast pressure equilibration
  - Parameters: `taut`, `taup`, `pressure`, `compressibility`
  - Best for: Quick volume relaxation
  - Typical: taut=100 fs, taup=1000 fs

## CLI Interface Design

### Basic Usage

```bash
# NVE - Energy conserving
md --structure structure.vasp --ensemble nve --steps 10000

# NVT - Langevin (default for NVT)
md --structure structure.vasp --ensemble nvt --temperature 300 --steps 10000

# NVT - Langevin with custom friction
md --structure structure.vasp --ensemble nvt --temperature 300 \
   --thermostat langevin --friction 0.01 --steps 10000

# NVT - Nosé-Hoover
md --structure structure.vasp --ensemble nvt --temperature 300 \
   --thermostat nose-hoover --ttime 50 --steps 10000

# NPT - Isotropic pressure
md --structure structure.vasp --ensemble npt --temperature 300 \
   --pressure 0.0 --steps 10000

# NPT - Berendsen (fast equilibration)
md --structure structure.vasp --ensemble npt --temperature 300 \
   --pressure 0.0 --barostat berendsen --taup 1000 --steps 10000
```

### Parameters

#### Required
- `--structure` - Input structure file
- `--ensemble` - Ensemble type: `nve`, `nvt`, `npt`
- `--steps` - Number of MD steps

#### Ensemble-Specific
- `--temperature` - Temperature in K (required for NVT, NPT)
- `--pressure` - Pressure in GPa (required for NPT, default: 0.0)

#### Thermostat Selection (NVT, NPT)
- `--thermostat` - Thermostat type: `langevin`, `nose-hoover`, `berendsen`
  - Default: `langevin` for NVT, built-in for NPT

#### Barostat Selection (NPT only)
- `--barostat` - Barostat type: `npt` (isotropic MTK), `berendsen`
  - Default: `npt`

#### Advanced Parameters
- `--friction` - Langevin friction coefficient (1/fs, default: 0.01)
- `--ttime` - Nosé-Hoover/NPT time constant (fs, default: 25)
- `--taut` - Berendsen temperature coupling time (fs, default: 100)
- `--taup` - Berendsen pressure coupling time (fs, default: 1000)
- `--pfactor` - NPT pressure coupling factor (default: auto-calculated)
- `--compressibility` - Berendsen compressibility (1/GPa, default: 4.57e-5 for water)

#### Other Options
- `--timestep` - Timestep in fs (default: 1.0)
- `--mlip` - MLIP model (default: auto)
- `--uma-task` - UMA task (default: omat)

## Code Architecture

### File Structure

```
src/mlip_platform/core/
├── md.py (enhanced)
│   ├── get_dynamics_class()  # Returns ASE dynamics class
│   ├── setup_dynamics()       # Initialize dynamics with parameters
│   └── run_md()               # Main MD runner
```

### Dynamics Mapping

```python
DYNAMICS_MAP = {
    'nve': {
        'velocityverlet': {
            'class': VelocityVerlet,
            'params': ['timestep'],
        }
    },
    'nvt': {
        'langevin': {
            'class': Langevin,
            'params': ['timestep', 'temperature', 'friction'],
            'defaults': {'friction': 0.01}
        },
        'nose-hoover': {
            'class': NoseHoover,
            'params': ['timestep', 'temperature', 'ttime'],
            'defaults': {'ttime': 25.0}
        },
        'berendsen': {
            'class': Berendsen,
            'params': ['timestep', 'temperature', 'taut'],
            'defaults': {'taut': 100.0}
        }
    },
    'npt': {
        'npt': {
            'class': NPT,
            'params': ['timestep', 'temperature', 'externalstress', 'ttime', 'pfactor'],
            'defaults': {'ttime': 25.0, 'pfactor': None}  # pfactor auto-calculated
        },
        'berendsen': {
            'class': NPTBerendsen,
            'params': ['timestep', 'temperature', 'taut', 'pressure', 'taup', 'compressibility'],
            'defaults': {'taut': 100.0, 'taup': 1000.0, 'compressibility': 4.57e-5}
        }
    }
}
```

## Default Behavior

### When `--ensemble` not specified
- Default: `nvt` with Langevin thermostat
- Reason: Most common for MD simulations

### When `--thermostat` not specified
- NVT: default to `langevin`
- NPT: use thermostat built into barostat

### When `--barostat` not specified
- NPT: default to `npt` (isotropic MTK)

## Output Files

All existing outputs remain:
- `md.traj` - Trajectory
- `md_energy.csv` - Energy, temperature, pressure (if NPT)
- `md_energy.png` - Energy vs time
- `md_temperature.png` - Temperature vs time
- `md_pressure.png` - Pressure vs time (NPT only)
- `md_volume.png` - Volume vs time (NPT only)
- `md_params.txt` - Run parameters

## Parameter Validation

### Ensemble-specific checks:
- NVE: No temperature/pressure needed
- NVT: Require temperature
- NPT: Require temperature and pressure

### Thermostat compatibility:
- `langevin`, `nose-hoover`, `berendsen` → NVT only
- Built-in thermostats → NPT

### Physical ranges:
- Temperature: > 0 K
- Pressure: any (negative = tension)
- Friction: > 0
- Time constants: > 0
- Timestep: 0.5-5.0 fs (typical for solids)

## Examples for Common Scenarios

### 1. Quick Equilibration (Berendsen NVT)
```bash
md --structure structure.vasp \
   --ensemble nvt \
   --thermostat berendsen \
   --temperature 300 \
   --steps 5000 \
   --timestep 1.0
```

### 2. Production MD (Langevin NVT)
```bash
md --structure structure.vasp \
   --ensemble nvt \
   --thermostat langevin \
   --temperature 300 \
   --friction 0.01 \
   --steps 50000 \
   --timestep 1.0
```

### 3. Lattice Constant Optimization (NPT)
```bash
md --structure structure.vasp \
   --ensemble npt \
   --temperature 300 \
   --pressure 0.0 \
   --steps 10000 \
   --timestep 1.0
```

### 4. High Pressure Simulation (NPT)
```bash
md --structure structure.vasp \
   --ensemble npt \
   --temperature 1000 \
   --pressure 50.0 \
   --steps 20000 \
   --timestep 0.5
```

### 5. Energy Conservation Test (NVE)
```bash
md --structure structure.vasp \
   --ensemble nve \
   --steps 10000 \
   --timestep 1.0
```

## Backward Compatibility

The current simple interface remains supported:
```bash
md --structure structure.vasp --steps 1000 --temperature 300
# → Automatically uses NVT with Langevin (new default)
```

Old behavior (NVE with VelocityVerlet) available via:
```bash
md --structure structure.vasp --steps 1000 --ensemble nve
```

## Recommended Defaults for Solids

| Property | Default Value | Reason |
|----------|---------------|--------|
| Ensemble | NVT | Most common use case |
| Thermostat (NVT) | Langevin | Best balance of accuracy/speed |
| Barostat (NPT) | NPT (isotropic) | Rigorous, good for crystals |
| Temperature | 300 K | Room temperature |
| Pressure | 0.0 GPa | Ambient |
| Timestep | 1.0 fs | Safe for most solids |
| Friction | 0.01 fs⁻¹ | Mild damping |

## Implementation Priority

### Phase 1: Core Ensembles (this PR)
1. ✅ NVE with VelocityVerlet
2. ✅ NVT with Langevin (new default)
3. ✅ NVT with Nosé-Hoover
4. ✅ NPT with isotropic MTK

### Phase 2: Additional Methods (future)
5. NVT with Berendsen
6. NPT with Berendsen

### Phase 3: Advanced (future)
7. Anisotropic pressure control
8. Cell shape optimization
9. CSVR thermostat

## Technical Notes

### ASE Imports Required
```python
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nose_hoover import NoseHoover
```

### Pressure Units
- CLI: GPa (standard for solid-state)
- ASE: eV/Å³
- Conversion: 1 GPa = 0.006241509 eV/Å³

### NPT pfactor Calculation
```python
pfactor = (ttime * 75 * units.GPa) ** 2
# Typical: ttime=25 fs → pfactor ≈ 140 eV/Å³
```

### Berendsen Compressibility
- Water: 4.57e-5 GPa⁻¹
- Metals: ~1e-6 GPa⁻¹
- Ceramics: ~1e-7 GPa⁻¹
