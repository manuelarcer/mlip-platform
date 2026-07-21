# Unified JSON run record

**Date:** 2026-07-21
**Status:** Approved, awaiting implementation plan

## Problem

Run parameters are recorded inconsistently, and library callers get no record at all.

The observed symptom was a relaxation directory
(`.../Ni3Fe/adsorption/111_t0/CH2/basin_00`) containing no mliprun parameter
file, only a four-key `compcat_run.json`. The cause is a layering defect, not a
difference between one-off and batch runs:

- `core/optimize.py:run_optimization` writes the trajectory, log, convergence
  CSV, `opt_final.vasp` and `CONTCAR` — but **no parameter record**.
- The parameter record is written only by the CLI layer, in two duplicated
  copies: inline in `cli/commands/optimize.py:run` (line 124) and in
  `_write_params` (line 313) for `batch`. Both emit an identical
  `opt_params.txt`, so `run` and `batch` do **not** diverge from each other.
- `batch_relax.py` in the separate `compcatalysis-skills` repo imports
  `run_optimization` as a library function and bypasses the CLI entirely. It
  therefore received no record, and invented its own minimal
  `compcat_run.json` holding only `mlip`, `uma_task`, `fmax` and `optimizer`.

Two further defects follow from the same root cause:

1. Four ad-hoc text formats exist across commands — `opt_params.txt`,
   `md_params.txt`, `neb_parameters.txt`, `autoneb_parameters.txt`.
2. No record distinguishes a value the user chose from one inherited from a
   default, so a directory cannot answer whether `max_steps: 200` was a
   decision or an accident — a question that becomes unanswerable once a
   default changes between versions.

`neb_parameters.txt` is load-bearing rather than merely informational:
`core/neb.py:_parse_parameters_file` reads it back to reconstruct a NEB for
restart.

## Goals

Write one canonical, machine- and human-readable JSON record per run
directory, from a layer every caller passes through, capturing the complete
resolved parameter set with the provenance of each value.

## Non-goals

- Removing the existing `.txt` files. They stay; see Decision 2.
- Changing anything in the `compcatalysis-skills` repo. That is a separate
  change; see Decision 6.
- An MD equilibration-window detector. Deferred; see Decision 5.
- New golden reference files. Nothing here is a numerical baseline.

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | The record is written from the **core layer** | Every caller gets it, including `batch_relax.py` with no changes to that repo. The CLI layer is precisely the layer library callers skip. |
| 2 | Existing `.txt` files are **kept**; JSON is canonical | NEB restart parses `neb_parameters.txt`; existing run directories stay valid; existing tests keep passing. Cost is one extra file per run. |
| 3 | Each parameter records **value + source** | Distinguishes a deliberate value from an inherited default, and survives a future change to the defaults. |
| 4 | One filename, `mliprun_run.json`, for every command | A single glob finds every run; a `command` field inside says which. |
| 5 | MD records **raw statistics only** | Means, stds, drift and block means are unambiguous. Equilibration-window detection is a methodological choice, deferred to its own change so no algorithm is silently baked into a provenance file. |
| 6 | Scope is **mliprun only** | `batch_relax.py` starts producing records automatically; opting it into the batch fields and retiring `compcat_run.json` is a separate PR against a separate repo. |

## Architecture

A new module `src/mliprun/core/run_record.py` owns the record. Nothing outside
it knows the file format.

It exposes a `RunContext` dataclass carrying what core cannot determine on its
own: the command name, whether the run is one-off or part of a batch, the batch
identity, and the source of each parameter value. Every core entry point
(`run_optimization`, `run_md`, the NEB runners) gains one optional
`run_context` argument and calls the writer itself.

### Two-phase write

The record is written twice: once immediately **before** the run starts, with
inputs, parameters and provenance filled in and `"status": "running"`; once
**after** it finishes, rewritten with outcome and timings.

This is deliberate. `cli/commands/md.py` already writes its parameter file up
front (see the comment at line 154) so a 500k-step job that dies still leaves
evidence. A killed job now leaves a record saying `running` rather than nothing.

### Parameter sources

The CLI derives sources from Typer's context via `ctx.get_parameter_source(name)`,
which returns a `click.core.ParameterSource` — verified available as
`COMMANDLINE`, `DEFAULT`, `ENVIRONMENT`, `PROMPT` under the installed
typer 0.20.0 / click 8.2.1. These map to `user`, `default`, `env`, `prompt`.

When a library caller supplies no `RunContext`, core tags every parameter
`unspecified`. It does **not** infer sources by comparing against signature
defaults: a caller that explicitly passes `fmax=0.05` is indistinguishable from
one that omitted it, and guessing would mislabel the record.

### Where the record is written

The record goes in the command's existing output directory — the same
directory that already receives that command's other outputs, per
`docs/OUTPUTS.md`. It therefore lands next to the input structure for
`optimize` and `md`, and in the invocation cwd for `neb` and `autoneb`.

`optimize batch` writes one record per subdirectory, alongside each
relaxation. It writes **no** record into `--parent`; the existing
`batch_summary.csv` remains the parent-level artifact, and the shared
`batch_id` is what links the per-subdirectory records back together.

## Schema

```json
{
  "schema_version": 1,
  "command": "optimize",
  "status": "converged",
  "run": {
    "mode": "batch",
    "batch": {
      "batch_id": "20260721T142233-a1b2c3",
      "driver": "mliprun optimize batch",
      "argv": ["mliprun", "optimize", "batch", "--parent", ".", "--fmax", "0.02"],
      "root": "/abs/path/to/parent",
      "config_file": null
    }
  },
  "inputs": {
    "structure": "input.vasp",
    "structure_abspath": "/abs/.../basin_00/input.vasp",
    "n_atoms": 96,
    "formula": "CH2Fe24Ni72"
  },
  "parameters": {
    "fmax":       {"value": 0.02,   "source": "user"},
    "max_steps":  {"value": 200,    "source": "default"},
    "optimizer":  {"value": "bfgs", "source": "default"},
    "uma_task":   {"value": "oc20", "source": "user"},
    "relax_cell": {"value": false,  "source": "default"}
  },
  "provenance": {
    "mliprun_version": "0.4.0",
    "ase_version": "3.26.0",
    "mlip_package": {"name": "fairchem-core", "version": "2.x"},
    "mlip_model": "uma-s-1p2",
    "device_requested": "auto",
    "device_resolved": "cuda",
    "python_version": "3.12.4",
    "hostname": "cosrnode01g",
    "started_at": "2026-07-21T14:22:33+08:00",
    "finished_at": "2026-07-21T14:24:36+08:00",
    "walltime_s": 123.4
  },
  "stages": [
    {
      "index": 0,
      "kind": "optimize",
      "status": "converged",
      "started_at": "2026-07-21T14:22:33+08:00",
      "walltime_s": 123.4,
      "steps": 47,
      "results": {
        "converged": true,
        "final_energy_eV": -1234.567890,
        "final_fmax_eV_per_A": 0.0181
      }
    }
  ]
}
```

Field notes:

- `mode` is `one-off` or `batch`; the whole `batch` object is `null` for
  one-off runs.
- Top-level `status` reflects the **latest** stage: `running` while a stage is
  in flight, otherwise that stage's terminal status. Per-stage `status` is
  never rewritten once terminal, so a converged stage 0 followed by a failed
  stage 1 yields a record whose top-level status is `failed` while stage 0
  still reads `converged`.
- `kind` is one of `optimize`, `md`, `md-resume`, `neb`, `neb-restart`,
  `autoneb`. It names what the stage did, so a reader need not infer it from
  the parameters that changed.
- `source` is one of `user`, `default`, `env`, `prompt`, `unspecified`.
- `device_requested` vs `device_resolved` are separate because `auto` is what
  was typed and `cuda` is what ran. `mlip_model` likewise records the resolved
  model name, never `auto`.
- `schema_version` exists so a future field change is detectable by readers
  rather than silently misparsed.

### Stages

`stages` is an array, and a restart or resume **appends** to it rather than
overwriting. A one-shot run simply has one stage.

This is required, not decorative. mliprun has no two-stage NEB command —
`climb` is a single bool passed to `run_neb`. A plain-then-CI-NEB workflow is
therefore *two invocations in the same directory* through the restart path
(`cli/commands/neb.py:_handle_restart`, `core/neb.py:load_from_restart` with a
`climb` override). MD `--resume` has the same shape, which is why
`md_params.txt` is opened in append mode today. A flat outcome block would let
the second invocation erase the first.

Top-level `parameters` holds what is fixed for the directory; per-stage
`parameters` holds only what that stage changed.

#### NEB stages

```json
"stages": [
  {"index": 0, "kind": "neb", "status": "converged",
   "parameters": {"climb": {"value": false, "source": "user"},
                  "fmax":  {"value": 0.1,   "source": "user"}},
   "steps": 118, "walltime_s": 840.2,
   "results": {"forward_barrier_eV": 0.853, "reverse_barrier_eV": 1.204,
               "reaction_energy_eV": -0.351, "ts_image_index": 4,
               "n_images": 9, "ts_at_endpoint": false,
               "final_fmax_eV_per_A": 0.094}},
  {"index": 1, "kind": "neb-restart", "status": "converged",
   "parameters": {"climb": {"value": true, "source": "user"},
                  "fmax":  {"value": 0.03, "source": "user"}},
   "steps": 61, "walltime_s": 512.7,
   "results": {"forward_barrier_eV": 0.911, "reverse_barrier_eV": 1.262,
               "reaction_energy_eV": -0.351, "ts_image_index": 4,
               "n_images": 9, "ts_at_endpoint": false,
               "final_fmax_eV_per_A": 0.028}}
]
```

Forward **and** reverse barriers are recorded because the current code reports
only `max(E) - E[0]` (`core/neb.py:624`), which is half the answer.

`ts_at_endpoint` is a physical sanity flag: a maximum sitting on image 0 or
`n_images - 1` means no saddle was bracketed. Nothing warns about this today.

#### MD stages

Each `--resume` appends a stage. Per-stage `results` carry unambiguous raw
statistics:

- mean and std of temperature, total energy, potential energy
- mean and std of volume and pressure (NPT only)
- total-energy drift per atom per ps — the quantity that indicates whether an
  NVE run is trustworthy
- decile block means, so equilibration is eyeballable without reopening
  `md_energy.csv`

No equilibration window and no production average; see Decision 5.

## Failure behavior

The governing rule: **the record must never be able to kill a run.** A
six-hour trajectory must not be lost because a provenance file failed to
serialize. Every record write is wrapped so a failure warns and the run
continues.

- **Atomic writes.** Write a temp file in the same directory, then
  `os.replace`. The file is written at least twice per run, so a crash
  mid-write would otherwise leave truncated JSON where a valid record was.
- **Killed jobs** keep `"status": "running"` with `finished_at: null` — a
  positive signal that the job started and never reported back.
- **Unparseable existing record** on restart: back it up to
  `mliprun_run.json.corrupt-<timestamp>`, start fresh, warn. Never crash a new
  run over a damaged old file.
- **Pre-existing directories** have no record. A restart there begins one whose
  first stage is explicitly marked as having unknown prior history, rather than
  implying the current stage is the whole story.
- **Serialization**: `Path`, numpy scalars and non-finite floats are coerced at
  the boundary. `NaN` and `Inf` are not valid JSON, so a diverged run producing
  `NaN` energy must not corrupt its own record.

## Testing

Following the repo's existing split: core tests exercise the writer directly,
CLI tests go through the Typer runner.

- **Schema** — a run produces a record parsing as JSON with the documented keys
  and `schema_version: 1`.
- **Source tags** — invoking with `--fmax 0.02` while omitting `--max-steps`
  tags one `user` and the other `default`. This is the central claim of the
  feature.
- **Library caller with no context** — record is produced with sources tagged
  `unspecified`, proving the `batch_relax.py` path works unchanged.
- **Batch identity** — every subdirectory of one `optimize batch` shares a
  `batch_id`, and each records its own `structure`.
- **Stage append** — a NEB restart and an MD resume each yield
  `len(stages) == 2` with stage 0 intact. This is the regression a flat outcome
  block would have caused.
- **Failure isolation** — with the output directory unwritable, the run still
  completes and returns its normal result.
- **Atomicity** — a simulated crash between temp-write and rename leaves the
  prior valid record in place.

Existing tests are unaffected: `tests/test_optimize_batch.py` asserts files
*exist* (lines 85, 184, 198) rather than enumerating directory contents, and
the `.txt` files are retained.

## Documentation

`docs/OUTPUTS.md` is the canonical list of files each command writes and must
gain `mliprun_run.json` for every command, plus a section documenting the
schema.
