# Output Files Reference

Canonical list of every file each CLI command writes. Search this page to find what produces a given filename, what format it's in, and which command's output directory it ends up in.

## Where output goes

| Command | Output directory |
|---------|------------------|
| `optimize run` | Directory containing `--structure` (i.e. `Path(--structure).parent`) |
| `md run` | Directory containing `--structure` |
| `neb run` | Current working directory at invocation |
| `autoneb run` | Current working directory at invocation |
| `autoneb-results results` | The directory passed via `--directory` (default `.`) |
| `benchmark run` | Nothing on disk by default; `--output bench.json` writes a JSON file there |

This is not always the same directory the user is sitting in. `optimize` and `md` write *next to the input structure*; `neb` and `autoneb` write *into the cwd*. Set up the working directory accordingly before running NEB / AutoNEB.

---

## `optimize run`

| File | Format | Contents |
|------|--------|----------|
| `opt.traj` | ASE trajectory (binary) | Every optimizer step |
| `opt.log` | text | ASE optimizer log (step, fmax, energy) |
| `opt_convergence.csv` | CSV | columns: `step`, `energy(eV)`, `fmax(eV/A)` |
| `opt_convergence.png` | PNG | Energy and fmax vs step |
| `opt_final.vasp` | VASP POSCAR (vasp5, direct) | Final relaxed structure |
| `opt_params.txt` | plain text, key/value | Echo of run parameters (MLIP, optimizer, fmax, max_steps, etc.) |

If you change `--logfile <name>.log`, the convergence CSV / PNG and final POSCAR are renamed accordingly: `<name>.log`, `<name>_convergence.csv`, `<name>_convergence.png`, `<name>_final.vasp`. The trajectory filename comes from `--trajectory`. **Two relaxations launched in the same directory will overwrite each other** unless you set `--logfile` and `--trajectory` to different names.

---

## `md run`

Always written:

| File | Format | Contents |
|------|--------|----------|
| `md.traj` | ASE trajectory (binary) | Every `interval` steps |
| `md_energy.csv` | CSV | columns: `step`, `time(fs)`, `temperature(K)`, `total_energy(eV)`, `potential_energy(eV)`, `kinetic_energy(eV)` |
| `md_energy.png` | PNG | Total / potential / kinetic energy vs time |
| `md_temperature.png` | PNG | Temperature vs time, with target line for NVT/NPT |
| `md_params.txt` | plain text | Echo of run parameters |

NPT-only extras:

| File | Format | Contents |
|------|--------|----------|
| `md_pressure.png` | PNG | Pressure vs time, target line shown |
| `md_volume.png` | PNG | Volume vs time |

NPT also adds `pressure(GPa)` and `volume(A^3)` columns to `md_energy.csv`.

---

## `neb run`

Endpoint optimization (only when `--optimize-endpoints` is enabled, the default):

| File | Format | Contents |
|------|--------|----------|
| `initial_opt.traj` / `final_opt.traj` | ASE trajectory | Endpoint relaxations |
| `initial_opt.log` / `final_opt.log` | text | ASE optimizer logs |
| `endpoint_optimization.txt` | plain text | Energies before/after, displacement summary |

IDPP interpolation (skipped in highly-constrained mode):

| File | Format | Contents |
|------|--------|----------|
| `idpp.traj` | ASE trajectory | IDPP iterations |
| `idpp.log` | text | IDPP log |

NEB run:

| File | Format | Contents |
|------|--------|----------|
| `A2B.traj` | ASE trajectory | Final NEB band (one frame per image) |
| `A2B_full.traj` | ASE trajectory | All NEB iteration steps; required for `--restart` |
| `neb.log` (or `--log` value) | text | Per-iteration log: step, fmax, current barrier |
| `neb_convergence.csv` | CSV | columns: `step`, `fmax(eV/A)`, `barrier(eV)` |
| `neb_convergence.png` | PNG | Two-panel: fmax vs step, barrier vs step |
| `neb_data.csv` | CSV | One row per image: index, energy, relative energy, force info |
| `neb_energy.png` | PNG | Smoothed energy profile across the band, barrier annotated |
| `neb_parameters.txt` | plain text | Echo of run parameters; required for `--restart` |
| `00/POSCAR`, `01/POSCAR`, ... | VASP POSCAR | One directory per image, including endpoints |

Restart side-effect:

| File / folder | Notes |
|---------------|-------|
| `bkup_YYYY.MM.DD_HH.MM.SS/` | Created when `--restart` is used. The previous run's outputs (everything above plus `0N/` POSCAR folders) are *moved* into this folder; `neb_parameters.txt` is *copied* so the original record is preserved. |

---

## `autoneb run`

Endpoint optimization (only when `--optimize-endpoints` is enabled, the default): same files as for `neb run` (`initial_opt.*`, `final_opt.*`, `endpoint_optimization.txt`).

AutoNEB run:

| File / folder | Format | Contents |
|---------------|--------|----------|
| `autoneb000.traj`, `autoneb001.traj`, ... | ASE trajectory | One file per image; suffix is the image index (3-digit padded). The `autoneb` prefix can be changed with `--prefix`. |
| `AutoNEB_iter/` | folder of trajectories | Per-iteration history written by ASE's AutoNEB |
| `autoneb_parameters.txt` | plain text | Echo of run parameters |

**Note:** `autoneb run` does *not* produce `neb_convergence.csv` / `neb_convergence.png` / `neb_data.csv` / `neb_energy.png`. To get an energy profile after AutoNEB finishes, run `autoneb-results results` (next section).

---

## `autoneb-results results`

| File | Format | Contents |
|------|--------|----------|
| `<prefix>_energy_profile.csv` | CSV | columns: `image`, `energy`, `rel_energy` |
| `<prefix>_energy_profile.png` | PNG | Spline-smoothed energy profile with TS annotated |
| `image_00/POSCAR`, `image_01/POSCAR`, ... | VASP POSCAR | Only when `--export-poscars` is set |

`<prefix>` defaults to `autoneb` and matches the prefix of the trajectory files this command reads.

---

## `benchmark run`

Stdout:

- A summary block per model (energy in eV, time in seconds), plus a final JSON summary block. Failed models are recorded as a string starting with `failed:` and the loop continues.

Optional file (only when `--output` is set):

| File | Format | Contents |
|------|--------|----------|
| `<--output path>` | JSON | `{model_tag: {"energy_eV": ..., "time_s": ...} or "failed: ..."}` |

---

## Parameter file conventions

`*_params.txt` and `*_parameters.txt` are written by `mlip_platform.core.params_io.write_parameters_file`. Same two-column layout (`{key:<23}{value}`) for every command. The keys include their trailing colon. These files are plain text and intended to be diffed across runs.

`endpoint_optimization.txt` (NEB / AutoNEB) is written by `write_endpoint_results` from the same module and has its own structured layout — see [PYTHON_API.md](PYTHON_API.md#parameter-file-io) if you need to consume it programmatically.
