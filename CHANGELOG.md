# Changelog

All notable changes to this project are documented here. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `optimize batch`: relax a series of structures in one process, loading the MLIP model **only once** and reusing it across every relaxation (avoids the per-run model-load cost). Discovers one input structure per immediate subdirectory of `--parent` (default `--input-name '*.vasp'`; the platform's own `*_final.vasp` outputs are ignored), relaxes each in place, continues past failures, and writes a `batch_summary.csv` into `--parent`. Supports `--skip-existing` to resume a partial batch.
- `build_calculator()` in `mlip_platform.cli.utils`: builds and returns an ASE calculator without attaching it, so the loaded model can be reused across many structures. `setup_calculator()` is now a thin build-and-attach wrapper around it.
- `optimize run` / `optimize batch` now also write a fixed-name `CONTCAR` (copy of the final relaxed structure) so a follow-up DFT run managed by asetools can restart from the directory.
- `optimize run` / `optimize batch` `--no-plot` flag: skip the per-structure `*_convergence.png` figure (the `*_convergence.csv` is still written). The matplotlib figure/save is per-structure IO that dominates short relaxations, so disabling it materially speeds up large batches — measured ~3x on frozen-surface site scans (BFGS, single-adsorbate), where the plot cost more per job than the model load. Threaded through `run_optimization(plot=...)`.
- `neb run --device {cpu|cuda}`: run NEB on GPU. Defaults to `cpu` (unlike `optimize`/`md`, which default to `auto`); pass `--device cuda` for GPU runs. The device is recorded in `neb_parameters.txt` and honored on `--restart`.
- `neb run` support for multi-head MACE foundation models via `--mace-head` (same head values as the other commands); the resolved MACE head is logged in the NEB parameter file.
- CI: `tests.yml` GitHub Actions workflow runs the unit suite with a diff-coverage gate (>= 90% on changed lines) on every push/PR, plus a `weekly.yml` scheduled quality-check run. New baseline characterization tests (golden outputs + physics invariants) guard against regressions.

### Changed

- NEB now uses ASE's `improvedtangent` tangent method instead of the older default (`aseneb`), which improves convergence on curved reaction paths.
- All parameter files (`opt_params.txt`, `md_params.txt`, `neb_parameters.txt`) now record the resolved MLIP head/task, so a run's provenance is fully captured in its parameter file.
- `md_energy.png` plots kinetic energy on a twin right axis for readability alongside total/potential energy.

## [0.3.0] - 2026-05-08

### Added

- `mlip` parent CLI exposing every subcommand under one namespace (`mlip optimize`, `mlip md`, `mlip neb`, `mlip autoneb`, `mlip autoneb-results`, `mlip benchmark`). The standalone aliases (`optimize`, `md`, …) continue to work.
- `md run --resume`: continue an MD run from the last frame of `md.traj`. Preserves momenta (no Maxwell-Boltzmann re-init), appends to `md.traj` and `md_energy.csv`, treats `--steps` as additional steps. `setup_dynamics` now accepts `set_velocities`; `run_md` accepts `resume`.
- `MLIP_HELP` and `UMA_TASK_HELP` constants in `mlip_platform.cli.utils` — single source of truth for the model / task help wording, imported by all four CLI commands.
- `--models tag1,tag2` flag on `benchmark run` for explicit model selection.
- `--output bench.json` flag on `benchmark run` to persist results to a JSON file.
- New documentation: `docs/PYTHON_API.md` (public Python API reference), `docs/OUTPUTS.md` (canonical output-files reference), `docs/MD_REFERENCE.md` (MD CLI reference, replacing the old design doc), `CHANGELOG.md`, and `CONTRIBUTING.md`.

### Changed

- `mlip_platform.core.optimize.run_optimization`: default `optimizer` flipped from `"fire"` to `"bfgs"`. The CLI `optimize run` already defaulted to `bfgs`; the function default now matches.
- `benchmark run` rewritten as an in-process loop. Replaces the previous subprocess shell-out to `bench_driver.py` (which only worked from the repo root). Now auto-detects every installed MLIP (UMA, SevenNet, MACE, CHGNet) and is independent of the working directory.
- README and Windows setup guide updated to describe the `mlip` namespace, list CHGNet alongside the other MLIPs, and document the auto-detection priority (UMA → SevenNet → MACE → CHGNet) explicitly.
- `docs/UMA_USAGE_GUIDE.md` rewritten end-to-end: all CLI examples now use the `<command> run` subcommand pattern, `uma-s-1p2` is documented as the only current model, and a new "Hugging Face access" section covers gated-repo setup.
- `tests/README.md` replaced — previous content described an unrelated project ("MILP 1") and was wholly inaccurate.
- Validator error message in `validate_mlip` reworded to reflect the wildcard `uma-*` behavior instead of pretending only `uma-s-1p1` and `uma-m-1p1` are accepted.
- Help strings in all four CLI commands now state that any `uma-*` tag is forwarded to FAIRChem unchanged.

### Removed

- `src/mlip_platform/core/mlip_bench.py` — dead module (109 lines), not imported by anything live.
- `docs/MD_ENSEMBLE_DESIGN.md` — replaced by `docs/MD_REFERENCE.md`. The implementation-plan / Phase 1–3 / file-architecture sections were stale; the implementation has been complete for some time.
- `bench_driver.py` (root) — superseded by the rewritten in-process `benchmark run`. The "Benchmarking with bench_driver.py" subsection in `UMA_USAGE_GUIDE.md` now points at `benchmark run` instead.
- `docs/Python_MLIP_UMA_Setup_Guide.docx` and `.pdf` moved to `docs/legacy/` — `windows_setup_guide.md` is the canonical Markdown version.

### Fixed

- README example commands no longer reference `uma-s-1p1` after the project default switched to `uma-s-1p2`.
- README and CLI help no longer claim a `mlip` command exists when it doesn't (now actually wired).
- `md run --resume` step accounting: the previous draft called `dyn.run(prior_steps + steps)` after `dyn.nsteps = prior_steps`, which double-counted because ASE's `dyn.run(N)` runs `N` *additional* steps. A resume of `--steps 50` from step 50 now correctly ends at step 100, not step 150.

## [0.2.0] - 2026-03-13

### Added

- Shared utility modules: `mlip_platform.cli.utils` (MLIP detection / validation, calculator setup, relax-atoms parsing) and `mlip_platform.core.utils` (`calc_fmax`, GPa conversion).
- Parameter-file I/O helpers in `mlip_platform.core.params_io` (`write_parameters_file`, `write_endpoint_results`).
- AutoNEB support: new `autoneb` and `autoneb-results` CLI commands, `CustomNEB.run_autoneb`, MIC-aware path interpolation.
- NEB `--restart` flag with `bkup_<timestamp>/` automatic backup folder, parameter override / lock rules, image-count consistency check.
- Highly-constrained NEB mode: `--relax-atoms 0,1,5` keeps only the listed atoms mobile (others fixed); skips IDPP automatically.
- Automatic endpoint optimization (default-on) for both `neb` and `autoneb`, with similarity check between input and relaxed endpoints.
- BFGS and L-BFGS as NEB optimizers (`--neb-optimizer`).
- MD ensemble support: NVE, NVT (Langevin / Nose-Hoover / Berendsen), NPT (MTK / Berendsen). New `--ensemble`, `--thermostat`, `--barostat`, `--friction`, `--ttime`, `--taut`, `--taup` flags on `md run`.
- `--maximum-steps` limit for NEB optimization.
- Detailed iteration logging with custom log-file option for NEB.
- UMA-1.2 (`uma-s-1p2`) added as the new default UMA model; old `uma-s-1p1` still loads if requested.
- Windows setup guide (`docs/windows_setup_guide.md`).
- Comprehensive pytest suite — 17 test files, EMT-based unit tests plus marker-gated MLIP integration tests (`uma`, `mace`, `sevenn`, `slow`).

### Changed

- Default UMA model changed from `uma-s-1p1` to `uma-s-1p2`.
- Default NEB optimizer changed from MDMin to FIRE.
- Default `optimize` optimizer changed from FIRE to BFGS.
- NEB output now goes to the current working directory (was previously inconsistent across commands).
- NEB convergence plot shows barrier height instead of absolute max energy.
- NEB energy reference is now the initial structure (was previously the band minimum).
- NEB force logging uses actual NEB forces instead of cached image forces.
- Lazy imports throughout the CLI — heavy MLIP packages are loaded on demand only, eliminating PyTorch warnings and slow startup.
- `optimize` convergence filenames use the `--logfile` stem so two runs with different logs don't collide.

### Fixed

- NEB restart: parameter file is no longer overwritten before backup; copied to backup, then a fresh restart-tagged file is written.
- NEB restart: handles `None` overrides correctly when reusing loaded parameters.
- NEB restart: skips re-creation of the NEB instance when loading from restart.
- NEB endpoint optimization: `FixAtoms` constraints now correctly applied when `--relax-atoms` is set.
- NEB output directory respects symlinks.
- Endpoint optimization: convergence check uses the converged criterion, not the optimizer's exit code.
- AutoNEB: initial-image energies now computed before AutoNEB starts (avoids crash).
- AutoNEB: `n_simul` defaults to 1 with a warning when `>1` is requested without MPI.
- AutoNEB: stale `*.traj` and `AutoNEB_iter/` from prior runs are cleaned up before starting a new run.
- F-string syntax error in restart validation.
- Removed `egg-info` from git tracking (was a build artifact).

## [0.1.0]

Initial public iteration.

### Added

- Standalone CLI commands: `optimize`, `md`, `neb`, `benchmark`.
- Single-point energy + timing benchmark via `bench_driver.py` (called as a subprocess from `benchmark run`).
- Initial UMA model support alongside MACE and SevenNet.
- Initial pytest scaffolding for MACE and SevenNet single-point energy validation.
- Project README and `.gitignore`.

The 0.1 series predates this CHANGELOG; entries are reconstructed from the git history. There was no formal 0.1.0 release tag; `setup.py` jumped from project inception to `0.2.0` during the 2026-03-13 refactor.

[Unreleased]: https://github.com/manuelarcer/mlip-platform/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/manuelarcer/mlip-platform/releases/tag/v0.3.0
[0.2.0]: https://github.com/manuelarcer/mlip-platform/releases/tag/v0.2.0
[0.1.0]: https://github.com/manuelarcer/mlip-platform/releases/tag/v0.1.0
