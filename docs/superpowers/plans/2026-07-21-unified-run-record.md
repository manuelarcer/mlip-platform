# Unified JSON Run Record Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write one canonical `mliprun_run.json` per run directory from the core layer, capturing every resolved parameter with the provenance of its value, so CLI and library callers alike get a complete, machine-readable record.

**Architecture:** A new `src/mliprun/core/run_record.py` owns the file format; nothing else knows it. Core entry points (`run_optimization`, `run_md`, `run_neb`, `run_autoneb`) gain one optional `run_context` argument and write the record themselves — two phases, `running` before the work and the terminal status after. A `stages` array appends on NEB restart and MD resume instead of overwriting.

**Tech Stack:** Python 3.10+, ASE 3.26, Typer 0.20 / Click 8.2 (for `ctx.get_parameter_source`), pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-07-21-unified-run-record-design.md`

## Global Constraints

- **Never regenerate golden reference files** (`tests/goldens/*.json`). No task here adds or modifies a golden. On a golden failure, report the numerical delta only.
- **The record must never kill a run.** Every public entry point in `run_record.py` swallows its own exceptions, logs a warning, and returns. A failed record write must never propagate into a running simulation.
- **The existing `.txt` parameter files stay.** `opt_params.txt`, `md_params.txt`, `neb_parameters.txt`, `autoneb_parameters.txt` keep being written, byte-for-byte as today. `core/neb.py:_parse_parameters_file` still parses `neb_parameters.txt` for restart; do not repoint it at the JSON.
- **Filename is `mliprun_run.json`** for every command, in the command's existing output directory.
- **`schema_version` is `1`** and is written on every record.
- **Parameter sources** are exactly one of `user`, `default`, `env`, `prompt`, `unspecified`.
- Every test asserts a concrete value or invariant. No test may assert only "file exists" without also asserting content.
- Run tests with `pytest` from the repo root (`pytest.ini` sets `pythonpath = src`). Use the `/Users/juar/venv/fairchem/bin/python -m pytest` interpreter if `pytest` is not on PATH.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/mliprun/core/run_record.py` | **Create.** Owns the record: dataclasses, provenance collection, JSON coercion, atomic write, stage append. The only module that knows the file format. |
| `src/mliprun/cli/utils.py` | **Modify.** Add `param_sources_from_ctx()` mapping Click's `ParameterSource` to our source strings; re-export the relocated `resolve_device`. |
| `src/mliprun/core/utils.py` | **Modify.** Receives `resolve_device` (moved from `cli/utils.py`) so core can record the resolved device without importing from the CLI layer. |
| `src/mliprun/core/optimize.py` | **Modify.** `run_optimization` gains `run_context`; begins/completes a record. |
| `src/mliprun/cli/commands/optimize.py` | **Modify.** `run` and `batch` build a `RunContext`; `batch` mints one shared `batch_id`. Deduplicate the two copies of the `.txt` writer. |
| `src/mliprun/core/md.py` | **Modify.** `run_md` gains `run_context`; appends a stage on resume; computes raw statistics. |
| `src/mliprun/cli/commands/md.py` | **Modify.** Builds a `RunContext`. |
| `src/mliprun/core/neb.py` | **Modify.** `run_neb` / `run_autoneb` gain `run_context`; record barrier metrics and the endpoint sanity flag. |
| `src/mliprun/cli/commands/neb.py`, `autoneb.py` | **Modify.** Build a `RunContext`; restart appends a stage. |
| `tests/test_run_record.py` | **Create.** Unit tests for the module in isolation. |
| `tests/test_run_record_integration.py` | **Create.** Record produced through real core + CLI paths. |
| `docs/OUTPUTS.md` | **Modify.** Document the new file and its schema. |

Tasks 1–2 build the foundation, 3–4 deliver `optimize` (the path that produced the empty `basin_00`), 5 delivers `md`, 6–7 deliver NEB/AutoNEB, 8 documents. Each task after 1 leaves the tree working and testable.

---

### Task 1: The `run_record` module

**Files:**
- Create: `src/mliprun/core/run_record.py`
- Test: `tests/test_run_record.py`

**Interfaces:**
- Consumes: nothing (leaf module, no mliprun imports).
- Produces:
  - `RECORD_FILENAME: str = "mliprun_run.json"`, `SCHEMA_VERSION: int = 1`
  - `new_batch_id() -> str`
  - `BatchInfo(batch_id: str, driver: str, argv: list[str], root: str, config_file: str | None = None)`
  - `RunContext(command: str, mode: str = "one-off", batch: BatchInfo | None = None, param_sources: dict[str, str] | None = None)`
  - `collect_provenance(*, mlip_model: str, device_requested: str, device_resolved: str) -> dict`
  - `RunRecord.begin(output_dir, *, command, stage_kind, parameters, inputs, provenance, run_context=None, stage_parameters=None, append=False) -> RunRecord`
  - `RunRecord.complete(*, status: str, steps: int | None = None, results: dict | None = None) -> None`
  - `RunRecord.path -> Path`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run_record.py`:

```python
"""Unit tests for the canonical JSON run record."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mliprun.core.run_record import (
    RECORD_FILENAME,
    SCHEMA_VERSION,
    BatchInfo,
    RunContext,
    RunRecord,
    collect_provenance,
    new_batch_id,
)


def _read(tmp_path: Path) -> dict:
    return json.loads((tmp_path / RECORD_FILENAME).read_text())


def _begin(tmp_path, **kwargs):
    defaults = dict(
        command="optimize",
        stage_kind="optimize",
        parameters={"fmax": 0.02, "max_steps": 200},
        inputs={"structure": "init.vasp", "n_atoms": 4},
        provenance={"mliprun_version": "0.4.0"},
    )
    defaults.update(kwargs)
    return RunRecord.begin(tmp_path, **defaults)


class TestPhaseOne:
    def test_writes_running_status_before_completion(self, tmp_path):
        _begin(tmp_path)
        data = _read(tmp_path)
        assert data["status"] == "running"
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["command"] == "optimize"
        assert data["provenance"]["finished_at"] is None
        assert data["stages"][0]["status"] == "running"

    def test_unspecified_source_without_context(self, tmp_path):
        _begin(tmp_path)
        params = _read(tmp_path)["parameters"]
        assert params["fmax"] == {"value": 0.02, "source": "unspecified"}
        assert params["max_steps"] == {"value": 200, "source": "unspecified"}

    def test_sources_applied_from_context(self, tmp_path):
        ctx = RunContext(command="optimize",
                         param_sources={"fmax": "user", "max_steps": "default"})
        _begin(tmp_path, run_context=ctx)
        params = _read(tmp_path)["parameters"]
        assert params["fmax"]["source"] == "user"
        assert params["max_steps"]["source"] == "default"

    def test_extra_inputs_merged_from_context(self, tmp_path):
        ctx = RunContext(command="optimize",
                         extra_inputs={"structure": "init.vasp",
                                       "structure_abspath": "/tmp/init.vasp"})
        _begin(tmp_path, run_context=ctx)
        inputs = _read(tmp_path)["inputs"]
        assert inputs["structure"] == "init.vasp"
        assert inputs["structure_abspath"] == "/tmp/init.vasp"
        assert inputs["n_atoms"] == 4  # core-supplied fact survives the merge

    def test_one_off_has_null_batch(self, tmp_path):
        _begin(tmp_path)
        run = _read(tmp_path)["run"]
        assert run["mode"] == "one-off"
        assert run["batch"] is None

    def test_batch_fields_recorded(self, tmp_path):
        batch = BatchInfo(batch_id="20260721T000000-abc123",
                          driver="mliprun optimize batch",
                          argv=["mliprun", "optimize", "batch"],
                          root="/tmp/parent")
        ctx = RunContext(command="optimize", mode="batch", batch=batch)
        _begin(tmp_path, run_context=ctx)
        run = _read(tmp_path)["run"]
        assert run["mode"] == "batch"
        assert run["batch"]["batch_id"] == "20260721T000000-abc123"
        assert run["batch"]["root"] == "/tmp/parent"
        assert run["batch"]["config_file"] is None


class TestPhaseTwo:
    def test_complete_sets_status_and_results(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", steps=47,
                     results={"converged": True, "final_energy_eV": -1.5})
        data = _read(tmp_path)
        assert data["status"] == "converged"
        assert data["provenance"]["finished_at"] is not None
        assert data["provenance"]["walltime_s"] >= 0
        stage = data["stages"][0]
        assert stage["status"] == "converged"
        assert stage["steps"] == 47
        assert stage["results"]["final_energy_eV"] == -1.5

    def test_parameters_survive_completion(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", results={})
        assert _read(tmp_path)["parameters"]["fmax"]["value"] == 0.02


class TestStages:
    def test_append_adds_stage_preserving_first(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", steps=10, results={"barrier_eV": 0.85})

        rec2 = _begin(tmp_path, stage_kind="neb-restart", append=True,
                      stage_parameters={"climb": True})
        rec2.complete(status="converged", steps=5, results={"barrier_eV": 0.91})

        stages = _read(tmp_path)["stages"]
        assert len(stages) == 2
        assert stages[0]["index"] == 0
        assert stages[0]["results"]["barrier_eV"] == 0.85
        assert stages[0]["status"] == "converged"
        assert stages[1]["index"] == 1
        assert stages[1]["kind"] == "neb-restart"
        assert stages[1]["parameters"]["climb"]["value"] is True

    def test_top_level_status_reflects_latest_stage(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", results={})
        rec2 = _begin(tmp_path, stage_kind="neb-restart", append=True)
        rec2.complete(status="failed", results={})
        data = _read(tmp_path)
        assert data["status"] == "failed"
        assert data["stages"][0]["status"] == "converged"

    def test_append_to_missing_record_marks_unknown_history(self, tmp_path):
        rec = _begin(tmp_path, stage_kind="md-resume", append=True)
        rec.complete(status="converged", results={})
        data = _read(tmp_path)
        assert data["stages"][0]["prior_history_unknown"] is True

    def test_append_to_fresh_record_does_not_mark_unknown(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", results={})
        rec2 = _begin(tmp_path, append=True)
        rec2.complete(status="converged", results={})
        stages = _read(tmp_path)["stages"]
        assert "prior_history_unknown" not in stages[1]


class TestRobustness:
    def test_corrupt_record_is_backed_up_not_fatal(self, tmp_path):
        (tmp_path / RECORD_FILENAME).write_text("{not valid json")
        rec = _begin(tmp_path, append=True)
        rec.complete(status="converged", results={})
        data = _read(tmp_path)
        assert data["status"] == "converged"
        backups = list(tmp_path.glob(f"{RECORD_FILENAME}.corrupt-*"))
        assert len(backups) == 1
        assert backups[0].read_text() == "{not valid json"

    def test_non_finite_floats_coerced_to_null(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="failed",
                     results={"final_energy_eV": float("nan"),
                              "final_fmax_eV_per_A": float("inf")})
        raw = (tmp_path / RECORD_FILENAME).read_text()
        assert "NaN" not in raw and "Infinity" not in raw
        results = _read(tmp_path)["stages"][0]["results"]
        assert results["final_energy_eV"] is None
        assert results["final_fmax_eV_per_A"] is None

    def test_numpy_and_path_values_serialized(self, tmp_path):
        rec = _begin(tmp_path, parameters={"fmax": np.float64(0.02),
                                           "steps": np.int64(7),
                                           "out": Path("/tmp/x")})
        rec.complete(status="converged", results={})
        params = _read(tmp_path)["parameters"]
        assert params["fmax"]["value"] == 0.02
        assert params["steps"]["value"] == 7
        assert params["out"]["value"] == "/tmp/x"

    def test_unwritable_directory_does_not_raise(self, tmp_path):
        target = tmp_path / "nope"
        target.mkdir()
        target.chmod(0o500)  # read+execute, no write
        try:
            rec = _begin(target)
            rec.complete(status="converged", results={})  # must not raise
        finally:
            target.chmod(0o700)

    def test_no_partial_file_left_by_failed_serialization(self, tmp_path):
        rec = _begin(tmp_path)
        rec.complete(status="converged", results={})
        before = (tmp_path / RECORD_FILENAME).read_text()

        class Boom:
            def __repr__(self):
                raise RuntimeError("boom")

        rec2 = _begin(tmp_path, append=True)
        rec2.complete(status="converged", results={"bad": Boom()})
        # Prior valid record is intact or validly replaced -- never truncated.
        text = (tmp_path / RECORD_FILENAME).read_text()
        json.loads(text)
        assert len(list(tmp_path.glob(f"{RECORD_FILENAME}.tmp*"))) == 0


class TestHelpers:
    def test_batch_ids_are_unique(self):
        assert new_batch_id() != new_batch_id()

    def test_provenance_records_requested_and_resolved_device(self):
        prov = collect_provenance(mlip_model="uma-s-1p2",
                                  device_requested="auto",
                                  device_resolved="cuda")
        assert prov["device_requested"] == "auto"
        assert prov["device_resolved"] == "cuda"
        assert prov["mlip_model"] == "uma-s-1p2"
        assert prov["mlip_package"]["name"] == "fairchem-core"
        assert prov["mliprun_version"]
        assert prov["ase_version"]
        assert prov["hostname"]
        assert prov["started_at"] is None

    @pytest.mark.parametrize("model,expected", [
        ("uma-s-1p2", "fairchem-core"),
        ("mace", "mace-torch"),
        ("mace-mh-1", "mace-torch"),
        ("7net-mf-ompa", "sevenn"),
        ("chgnet", "chgnet"),
        ("something-else", None),
    ])
    def test_mlip_package_mapping(self, model, expected):
        prov = collect_provenance(mlip_model=model, device_requested="cpu",
                                  device_resolved="cpu")
        assert prov["mlip_package"]["name"] == expected
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_run_record.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'mliprun.core.run_record'`

- [ ] **Step 3: Write the module**

Create `src/mliprun/core/run_record.py`:

```python
"""Canonical JSON run record.

One ``mliprun_run.json`` per run directory, written from the core layer so
every caller gets one -- CLI commands and direct library callers alike. This
module is the only place that knows the file format.

Design: docs/superpowers/specs/2026-07-21-unified-run-record-design.md

The governing rule is that a record failure must never kill a run: every
public entry point swallows its own exceptions and logs a warning. A six-hour
trajectory must not be lost because a provenance file could not be written.
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import secrets
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

RECORD_FILENAME = "mliprun_run.json"
SCHEMA_VERSION = 1

#: Model-tag prefix -> installed distribution name. Longest prefix wins, so
#: ``mace-mh-1`` resolves before the bare ``mace`` entry.
_MLIP_PACKAGES = (
    ("uma-", "fairchem-core"),
    ("mace-mh-", "mace-torch"),
    ("mace", "mace-torch"),
    ("7net", "sevenn"),
    ("chgnet", "chgnet"),
)


def _now_iso() -> str:
    """Timezone-aware local timestamp, ISO 8601."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def new_batch_id() -> str:
    """Return an identifier shared by every run of one batch."""
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def _jsonable(obj: Any) -> Any:
    """Coerce a value into something ``json.dumps`` accepts.

    NaN and Inf become ``None``: they are not valid JSON, and Python's encoder
    would otherwise emit bare ``NaN``/``Infinity`` tokens that strict parsers
    reject. A diverged run must not corrupt its own record.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    # numpy scalars and anything else array-like exposing .item()
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:  # noqa: BLE001 -- fall through to repr
            pass
    return repr(obj)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Serialize fully, then replace the target in one step.

    Serialization happens before the target is touched, so a payload that
    cannot be encoded leaves the previous record intact. ``os.replace`` is
    atomic within a directory, so a crash never leaves truncated JSON where a
    valid record used to be.
    """
    text = json.dumps(payload, indent=2, allow_nan=False)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_existing(path: Path) -> Optional[dict]:
    """Return the record at ``path``, or None if absent or unusable.

    An unparseable record is moved aside rather than deleted: it may be the
    only evidence of what a prior run did.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        backup = path.with_suffix(
            path.suffix + f".corrupt-{datetime.now().strftime('%Y%m%dT%H%M%S')}"
        )
        try:
            os.replace(path, backup)
            logger.warning("Unparseable %s moved to %s", path.name, backup.name)
        except OSError:
            logger.warning("Unparseable %s could not be backed up", path.name)
        return None
    return data if isinstance(data, dict) else None


def _mlip_package(mlip_model: str) -> dict:
    """Resolve a model tag to its installed distribution name and version."""
    name = None
    for prefix, dist in _MLIP_PACKAGES:
        if mlip_model.startswith(prefix):
            name = dist
            break
    if name is None:
        return {"name": None, "version": None}
    try:
        return {"name": name, "version": version(name)}
    except PackageNotFoundError:
        return {"name": name, "version": None}


def collect_provenance(*, mlip_model: str, device_requested: str,
                        device_resolved: str) -> dict:
    """Gather environment and version facts for the record.

    ``device_requested`` and ``device_resolved`` are kept apart because
    ``auto`` is what was typed and ``cuda`` is what ran. ``started_at`` and
    ``finished_at`` are filled in by :class:`RunRecord`.
    """
    try:
        mliprun_version = version("mliprun")
    except PackageNotFoundError:
        mliprun_version = None
    try:
        from ase import __version__ as ase_version
    except Exception:  # noqa: BLE001 -- ASE absence must not break the record
        ase_version = None
    return {
        "mliprun_version": mliprun_version,
        "ase_version": ase_version,
        "mlip_package": _mlip_package(mlip_model),
        "mlip_model": mlip_model,
        "device_requested": device_requested,
        "device_resolved": device_resolved,
        "python_version": platform.python_version(),
        "hostname": socket.gethostname(),
        "started_at": None,
        "finished_at": None,
        "walltime_s": None,
    }


@dataclass
class BatchInfo:
    """Identity of the batch a run belongs to."""

    batch_id: str
    driver: str
    argv: list[str] = field(default_factory=list)
    root: str = ""
    config_file: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "batch_id": self.batch_id,
            "driver": self.driver,
            "argv": list(self.argv),
            "root": self.root,
            "config_file": self.config_file,
        }


@dataclass
class RunContext:
    """What core cannot determine on its own.

    ``param_sources`` maps a parameter name to one of ``user``, ``default``,
    ``env`` or ``prompt``. Any parameter absent from the map is tagged
    ``unspecified`` -- core never guesses by comparing against signature
    defaults, because a caller that explicitly passes the default value is
    indistinguishable from one that omitted it.
    """

    command: str
    mode: str = "one-off"
    batch: Optional[BatchInfo] = None
    param_sources: Optional[dict[str, str]] = None
    #: Input facts only the caller knows -- the structure filename and path.
    #: Core sees an ``Atoms`` object, which carries no provenance of its own.
    extra_inputs: Optional[dict] = None


def _tag(parameters: dict, sources: Optional[dict]) -> dict:
    sources = sources or {}
    return {
        str(k): {"value": _jsonable(v), "source": sources.get(k, "unspecified")}
        for k, v in parameters.items()
    }


class RunRecord:
    """A record being written. Obtain one from :meth:`begin`."""

    def __init__(self, path: Path, payload: dict, stage_index: int, t0: float):
        self.path = path
        self._payload = payload
        self._stage_index = stage_index
        self._t0 = t0

    @classmethod
    def begin(cls, output_dir, *, command: str, stage_kind: str,
              parameters: dict, inputs: dict, provenance: dict,
              run_context: Optional[RunContext] = None,
              stage_parameters: Optional[dict] = None,
              append: bool = False) -> "RunRecord":
        """Write phase one and return a handle for :meth:`complete`.

        Never raises. On failure a handle is still returned, so callers need
        no error handling of their own; the subsequent ``complete`` is simply
        a no-op.
        """
        t0 = time.perf_counter()
        path = Path(output_dir) / RECORD_FILENAME
        try:
            sources = run_context.param_sources if run_context else None
            existing = _load_existing(path) if append else None
            stages = list(existing.get("stages", [])) if existing else []

            stage = {
                "index": len(stages),
                "kind": stage_kind,
                "status": "running",
                "started_at": _now_iso(),
                "walltime_s": None,
                "steps": None,
                "results": {},
            }
            if stage_parameters:
                stage["parameters"] = _tag(stage_parameters, sources)
            if append and existing is None:
                # Resuming a directory that predates the record, or whose
                # record was unreadable. Say so rather than implying this
                # stage is the whole story.
                stage["prior_history_unknown"] = True
            stages.append(stage)

            if existing:
                payload = existing
                payload["stages"] = stages
                payload["status"] = "running"
            else:
                prov = dict(provenance)
                prov["started_at"] = _now_iso()
                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "command": command,
                    "status": "running",
                    "run": {
                        "mode": run_context.mode if run_context else "one-off",
                        "batch": (run_context.batch.as_dict()
                                  if run_context and run_context.batch else None),
                    },
                    "inputs": _jsonable({
                        **inputs,
                        **((run_context.extra_inputs or {}) if run_context else {}),
                    }),
                    "parameters": _tag(parameters, sources),
                    "provenance": _jsonable(prov),
                    "stages": stages,
                }

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            _atomic_write_json(path, payload)
            return cls(path, payload, stage["index"], t0)
        except Exception as exc:  # noqa: BLE001 -- never kill the run
            logger.warning("Could not write %s: %s", RECORD_FILENAME, exc)
            return cls(path, {}, -1, t0)

    def complete(self, *, status: str, steps: Optional[int] = None,
                 results: Optional[dict] = None) -> None:
        """Write phase two: terminal status, timings and results.

        Never raises.
        """
        if self._stage_index < 0:
            return
        try:
            stage = self._payload["stages"][self._stage_index]
            stage["status"] = status
            stage["walltime_s"] = round(time.perf_counter() - self._t0, 3)
            stage["steps"] = _jsonable(steps)
            stage["results"] = _jsonable(results or {})

            self._payload["status"] = status
            prov = self._payload["provenance"]
            prov["finished_at"] = _now_iso()
            prov["walltime_s"] = round(time.perf_counter() - self._t0, 3)

            _atomic_write_json(self.path, self._payload)
        except Exception as exc:  # noqa: BLE001 -- never kill the run
            logger.warning("Could not finalize %s: %s", RECORD_FILENAME, exc)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_run_record.py -v`
Expected: PASS, all tests.

If `test_unwritable_directory_does_not_raise` fails because the suite runs as root (root ignores mode bits), skip it rather than weakening it: add `@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses mode bits")`.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/run_record.py tests/test_run_record.py
git commit -m "feat(core): canonical JSON run record module

Owns mliprun_run.json: two-phase write, appending stages, atomic
replace, and coercion of Path/numpy/non-finite values. Every entry
point swallows its own exceptions so a record failure can never kill
a running simulation."
```

---

### Task 2: Parameter sources, and `resolve_device` moved to core

**Files:**
- Modify: `src/mliprun/cli/utils.py`
- Modify: `src/mliprun/core/utils.py`
- Test: `tests/test_cli_utils.py`

**Interfaces:**
- Consumes: `RunContext` from Task 1 (only as the eventual consumer of this map).
- Produces:
  - `param_sources_from_ctx(ctx) -> dict[str, str]` mapping each parameter name to `user`, `default`, `env` or `prompt`.
  - `mliprun.core.utils.resolve_device(device: str) -> str` — the existing
    `cli/utils.py:_resolve_device`, relocated. `CustomNEB` stores `self.device`
    and builds its own provenance inside core, so core needs this; core must
    not import from the CLI layer. `cli/utils.py` re-exports it under the old
    private name so its existing call site (line 408) is untouched.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_utils.py`:

```python
class TestParamSourcesFromCtx:
    """Click records where each parameter value came from; we relabel it."""

    def test_maps_click_sources_to_record_vocabulary(self):
        import typer
        from typer.testing import CliRunner

        from mliprun.cli.utils import param_sources_from_ctx

        seen = {}
        app = typer.Typer()

        @app.command()
        def go(ctx: typer.Context,
               fmax: float = typer.Option(0.05),
               max_steps: int = typer.Option(200)):
            seen.update(param_sources_from_ctx(ctx))

        result = CliRunner().invoke(app, ["--fmax", "0.02"])
        assert result.exit_code == 0, result.output
        assert seen["fmax"] == "user"
        assert seen["max_steps"] == "default"

    def test_returns_empty_dict_for_none(self):
        from mliprun.cli.utils import param_sources_from_ctx

        assert param_sources_from_ctx(None) == {}


class TestResolveDeviceRelocation:
    def test_explicit_device_passes_through_from_core(self):
        from mliprun.core.utils import resolve_device

        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda") == "cuda"

    def test_auto_resolves_to_a_concrete_device(self):
        from mliprun.core.utils import resolve_device

        assert resolve_device("auto") in {"cuda", "cpu"}

    def test_cli_alias_still_points_at_the_same_function(self):
        """cli/utils.py:408 still calls _resolve_device; keep it working."""
        from mliprun.cli.utils import _resolve_device
        from mliprun.core.utils import resolve_device

        assert _resolve_device is resolve_device
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_cli_utils.py -k "ParamSources or ResolveDevice" -v`
Expected: FAIL with `ImportError: cannot import name 'param_sources_from_ctx'`
and `cannot import name 'resolve_device'`

- [ ] **Step 3: Relocate `resolve_device`**

Cut the `_resolve_device` function from `src/mliprun/cli/utils.py` (lines
363-375) and paste it into `src/mliprun/core/utils.py` under its public name,
unchanged apart from the leading underscore:

```python
def resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` if a CUDA device is present, else ``"cpu"``.

    A passed-through ``"cuda"`` or ``"cpu"`` is returned unchanged. Lives in
    core because the run record needs the resolved value and core must not
    import from the CLI layer.
    """
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
```

In `src/mliprun/cli/utils.py`, replace the removed definition with a re-export
so the existing call site at line 408 keeps working:

```python
from mliprun.core.utils import resolve_device as _resolve_device
```

- [ ] **Step 4: Implement `param_sources_from_ctx`**

Add to `src/mliprun/cli/utils.py` (imports at top, function at the end):

```python
from click.core import ParameterSource
```

```python
#: Click's provenance vocabulary -> the record's. DEFAULT_MAP means a config
#: file supplied the value; we report it as "user" because a human wrote it.
_SOURCE_LABELS = {
    ParameterSource.COMMANDLINE: "user",
    ParameterSource.ENVIRONMENT: "env",
    ParameterSource.PROMPT: "prompt",
    ParameterSource.DEFAULT: "default",
    ParameterSource.DEFAULT_MAP: "user",
}


def param_sources_from_ctx(ctx) -> dict:
    """Map each CLI parameter name to where its value came from.

    Returns an empty dict when no context is available, which the record
    module then reports as ``unspecified`` rather than guessing.
    """
    if ctx is None:
        return {}
    sources = {}
    for name in getattr(ctx, "params", {}):
        try:
            src = ctx.get_parameter_source(name)
        except Exception:  # noqa: BLE001 -- provenance is best-effort
            continue
        if src is not None:
            sources[name] = _SOURCE_LABELS.get(src, "unspecified")
    return sources
```

- [ ] **Step 5: Run to verify it passes**

Run: `pytest tests/test_cli_utils.py -v`
Expected: PASS, including the pre-existing tests in that file — the
`_resolve_device` alias keeps every current caller working.

- [ ] **Step 6: Commit**

```bash
git add src/mliprun/cli/utils.py src/mliprun/core/utils.py tests/test_cli_utils.py
git commit -m "feat(cli): map Click parameter sources to record vocabulary

Distinguishes a value the user typed from one inherited from a default,
which is the fact the run record exists to preserve.

Also relocates _resolve_device to core.utils as resolve_device: the NEB
record is built inside core, which must not import from the CLI layer.
cli.utils re-exports the old private name."
```

---

### Task 3: Wire the record into `run_optimization`

**Files:**
- Modify: `src/mliprun/core/optimize.py` (signature ~line 40-52; body after line 156)
- Test: `tests/test_run_record_integration.py` (create)

**Interfaces:**
- Consumes: `RunRecord`, `RunContext`, `collect_provenance` from Task 1.
- Produces: `run_optimization(..., run_context: RunContext | None = None, device_requested: str = "auto", device_resolved: str = "auto")` writing `mliprun_run.json` in `output_dir`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_record_integration.py`:

```python
"""The record as produced through real core and CLI code paths."""
import json

from ase.build import bulk
from ase.calculators.emt import EMT

from mliprun.core.optimize import run_optimization
from mliprun.core.run_record import RECORD_FILENAME, RunContext


def _cu(a=3.7):
    atoms = bulk("Cu", "fcc", a=a)
    atoms.calc = EMT()
    return atoms


def _record(d):
    return json.loads((d / RECORD_FILENAME).read_text())


class TestOptimizeRecord:
    def test_library_caller_gets_a_complete_record(self, tmp_path):
        """A caller that bypasses the CLI still gets a record -- the defect
        that left basin_00 with no parameter file at all."""
        run_optimization(atoms=_cu(), optimizer="bfgs", fmax=0.5, max_steps=5,
                         output_dir=tmp_path, model_name="emt", verbose=False)
        data = _record(tmp_path)
        assert data["command"] == "optimize"
        assert data["status"] in {"converged", "not_converged"}
        assert data["parameters"]["fmax"]["value"] == 0.5
        assert data["parameters"]["fmax"]["source"] == "unspecified"
        assert data["stages"][0]["kind"] == "optimize"
        assert isinstance(data["stages"][0]["steps"], int)

    def test_records_outcome_numbers(self, tmp_path):
        run_optimization(atoms=_cu(), optimizer="bfgs", fmax=0.5, max_steps=5,
                         output_dir=tmp_path, model_name="emt", verbose=False)
        results = _record(tmp_path)["stages"][0]["results"]
        assert isinstance(results["converged"], bool)
        assert isinstance(results["final_energy_eV"], float)
        assert results["final_fmax_eV_per_A"] >= 0.0

    def test_sources_honoured_when_context_supplied(self, tmp_path):
        ctx = RunContext(command="optimize",
                         param_sources={"fmax": "user", "max_steps": "default"})
        run_optimization(atoms=_cu(), optimizer="bfgs", fmax=0.5, max_steps=5,
                         output_dir=tmp_path, model_name="emt", verbose=False,
                         run_context=ctx)
        params = _record(tmp_path)["parameters"]
        assert params["fmax"]["source"] == "user"
        assert params["max_steps"]["source"] == "default"

    def test_inputs_describe_the_structure(self, tmp_path):
        run_optimization(atoms=_cu(), optimizer="bfgs", fmax=0.5, max_steps=5,
                         output_dir=tmp_path, model_name="emt", verbose=False)
        inputs = _record(tmp_path)["inputs"]
        assert inputs["n_atoms"] == 1
        assert inputs["formula"] == "Cu"

    def test_failed_run_still_leaves_a_record(self, tmp_path):
        """An exception mid-run must not swallow the evidence."""
        class Exploding(EMT):
            def calculate(self, *args, **kwargs):
                raise RuntimeError("calculator exploded")

        atoms = bulk("Cu", "fcc", a=3.7)
        atoms.calc = Exploding()
        try:
            run_optimization(atoms=atoms, optimizer="bfgs", fmax=0.5,
                             max_steps=5, output_dir=tmp_path,
                             model_name="emt", verbose=False)
        except Exception:
            pass
        data = _record(tmp_path)
        assert data["status"] == "failed"
        assert data["stages"][0]["status"] == "failed"
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_run_record_integration.py -v`
Expected: FAIL — `FileNotFoundError` on `mliprun_run.json`

- [ ] **Step 3: Implement**

In `src/mliprun/core/optimize.py`, add the import after `from mliprun.core.utils import calc_fmax`:

```python
from mliprun.core.run_record import RunContext, RunRecord, collect_provenance
```

Extend the signature (currently lines 40-52) with three new keyword arguments, keeping every existing default unchanged:

```python
def run_optimization(
    atoms,
    optimizer: str = "bfgs",
    fmax: float = 0.05,
    max_steps: int = 200,
    trajectory: str = "opt.traj",
    logfile: str = "opt.log",
    output_dir: str | Path = ".",
    model_name: str = "mlip",
    verbose: bool = True,
    relax_cell: bool = False,
    plot: bool = False,
    run_context: Optional[RunContext] = None,
    device_requested: str = "auto",
    device_resolved: str = "auto",
) -> bool:
```

Add to the docstring's Parameters section, before `Returns`:

```
    run_context : RunContext, optional
        Declares the command, batch identity, and where each parameter value
        came from. When omitted the record still gets written, with every
        parameter tagged ``unspecified``.
    device_requested : str
        The device as asked for (e.g. ``'auto'``), recorded for provenance.
    device_resolved : str
        The device actually used (e.g. ``'cuda'``).
```

Immediately after the optimizer-name validation block (after line 120, `OptimizerClass = OPTIMIZER_MAP[optimizer_name]`), open the record:

```python
    record = RunRecord.begin(
        output_path,
        command="optimize",
        stage_kind="optimize",
        parameters={
            "optimizer": optimizer_name,
            "fmax": fmax,
            "max_steps": max_steps,
            "relax_cell": relax_cell,
            "trajectory": trajectory,
            "logfile": logfile,
            "plot": plot,
            "verbose": verbose,
        },
        inputs={
            "n_atoms": len(atoms),
            "formula": atoms.get_chemical_formula(),
        },
        provenance=collect_provenance(
            mlip_model=model_name,
            device_requested=device_requested,
            device_resolved=device_resolved,
        ),
        run_context=run_context,
    )
```

Wrap the optimization itself so a mid-run exception is recorded before it
propagates. Replace lines 140-156 (the `if verbose:` block through the
`logger.info` call) with:

```python
    try:
        if verbose:
            opt = OptimizerClass(opt_target, trajectory=str(traj_file), logfile=str(log_file))
            opt.attach(log_convergence, interval=1)
            logger.info("Starting optimization with %s (fmax=%.4f, max_steps=%d, relax_cell=%s)",
                        optimizer.upper(), fmax, max_steps, relax_cell)
            converged = opt.run(fmax=fmax, steps=max_steps)
        else:
            with open(log_file, "w") as lf:
                opt = OptimizerClass(opt_target, trajectory=str(traj_file), logfile=lf)
                opt.attach(log_convergence, interval=1)
                converged = opt.run(fmax=fmax, steps=max_steps)

        final_energy = atoms.get_potential_energy()
        final_fmax = calc_fmax(opt_target.get_forces())
    except Exception as exc:
        record.complete(status="failed", results={"error": str(exc)})
        raise

    logger.info("Optimization complete (converged=%s, steps=%d, energy=%.6f eV, fmax=%.6f eV/Ang)",
                converged, opt.nsteps, final_energy, final_fmax)

    record.complete(
        status="converged" if converged else "not_converged",
        steps=int(opt.nsteps),
        results={
            "converged": bool(converged),
            "final_energy_eV": float(final_energy),
            "final_fmax_eV_per_A": float(final_fmax),
        },
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_run_record_integration.py tests/test_core_optimize.py -v`
Expected: PASS. `test_core_optimize.py` must pass unchanged — the new arguments are keyword-only with defaults, so no existing caller breaks.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/optimize.py tests/test_run_record_integration.py
git commit -m "feat(optimize): write the run record from the core layer

Library callers that bypass the CLI (batch_relax.py in
compcatalysis-skills) previously got no parameter record at all. A
mid-run exception now records status=failed before propagating."
```

---

### Task 4: `optimize` CLI — context, batch identity, and deduplication

**Files:**
- Modify: `src/mliprun/cli/commands/optimize.py` (`run` ~line 46, `batch` ~line 167, `_write_params` ~line 313)
- Test: `tests/test_run_record_integration.py`

**Interfaces:**
- Consumes: `param_sources_from_ctx` (Task 2), `run_optimization(run_context=...)` (Task 3), `new_batch_id`, `BatchInfo`, `RunContext` (Task 1).
- Produces: `optimize run` and `optimize batch` each write `mliprun_run.json`; every subdirectory of one batch shares a `batch_id`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_record_integration.py`:

```python
import pytest
from ase.io import write
from typer.testing import CliRunner

from mliprun.cli.commands import optimize as opt_cmd
from mliprun.cli.commands.optimize import app as optimize_app

runner = CliRunner()


@pytest.fixture
def emt_patched(monkeypatch):
    """Swap the real model load for EMT so the CLI runs offline."""
    monkeypatch.setattr(opt_cmd, "build_calculator", lambda *a, **k: EMT())
    monkeypatch.setattr(opt_cmd, "validate_mlip", lambda *a, **k: None)

    def fake_setup(atoms, *a, **k):
        atoms.calc = EMT()
        return atoms

    monkeypatch.setattr(opt_cmd, "setup_calculator", fake_setup)


class TestOptimizeCliRecord:
    def test_run_marks_one_off_and_tags_sources(self, tmp_path, emt_patched):
        struct = tmp_path / "init.vasp"
        write(str(struct), bulk("Cu", "fcc", a=3.7), format="vasp")

        result = runner.invoke(optimize_app, [
            "--structure", str(struct), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert result.exit_code == 0, result.output

        data = _record(tmp_path)
        assert data["run"]["mode"] == "one-off"
        assert data["run"]["batch"] is None
        assert data["parameters"]["fmax"]["source"] == "user"
        assert data["parameters"]["optimizer"]["source"] == "default"
        assert data["inputs"]["structure"] == "init.vasp"

    def test_batch_shares_one_id_across_subdirs(self, tmp_path, emt_patched):
        for name in ("a", "b"):
            d = tmp_path / name
            d.mkdir()
            write(str(d / "init.vasp"), bulk("Cu", "fcc", a=3.7), format="vasp")

        result = runner.invoke(optimize_app, [
            "batch", "--parent", str(tmp_path), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert result.exit_code == 0, result.output

        rec_a = _record(tmp_path / "a")
        rec_b = _record(tmp_path / "b")
        assert rec_a["run"]["mode"] == "batch"
        assert rec_a["run"]["batch"]["batch_id"] == rec_b["run"]["batch"]["batch_id"]
        assert rec_a["run"]["batch"]["root"] == str(tmp_path.resolve())
        assert rec_a["inputs"]["structure"] == "init.vasp"
        assert "batch" in rec_a["run"]["batch"]["driver"]

    def test_batch_writes_no_record_into_parent(self, tmp_path, emt_patched):
        d = tmp_path / "a"
        d.mkdir()
        write(str(d / "init.vasp"), bulk("Cu", "fcc", a=3.7), format="vasp")
        runner.invoke(optimize_app, [
            "batch", "--parent", str(tmp_path), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert not (tmp_path / RECORD_FILENAME).exists()
        assert (tmp_path / "batch_summary.csv").exists()

    def test_txt_file_still_written(self, tmp_path, emt_patched):
        """The .txt files are retained; the JSON supplements, not replaces."""
        struct = tmp_path / "init.vasp"
        write(str(struct), bulk("Cu", "fcc", a=3.7), format="vasp")
        runner.invoke(optimize_app, [
            "--structure", str(struct), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert "Geometry Optimization Parameters" in (tmp_path / "opt_params.txt").read_text()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_run_record_integration.py -k OptimizeCli -v`
Expected: FAIL — `mliprun_run.json` absent, and `run` has no `ctx` parameter.

- [ ] **Step 3: Implement**

In `src/mliprun/cli/commands/optimize.py`, extend the imports:

```python
import sys

from mliprun.core.run_record import BatchInfo, RunContext, new_batch_id
from mliprun.cli.utils import _resolve_device, param_sources_from_ctx
```

`_resolve_device` is the helper relocated to `core/utils.py` in Task 2 and
re-exported from `cli/utils.py` under its old name; it turns `auto` into `cuda`
or `cpu`. Recording the resolved value is the entire reason the record
separates `device_requested` from `device_resolved`; echoing `auto` into both
would make the pair pointless.

Add `ctx: typer.Context` as the **first** parameter of both `run` and `batch`
(Typer injects it and does not expose it as a CLI flag):

```python
@app.command()
def run(
    ctx: typer.Context,
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
```

```python
@app.command()
def batch(
    ctx: typer.Context,
    parent: Path = typer.Option(..., prompt=True,
        help="Parent directory; each immediate subdirectory holds one input structure."),
```

In `run`, immediately before the `run_optimization(` call (currently line 109),
build the context and pass it through:

```python
    run_context = RunContext(
        command="optimize",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
```

Then add these arguments to the `run_optimization(...)` call:

```python
        run_context=run_context,
        device_requested=device,
        device_resolved=_resolve_device(device),
```

Core sees only an `Atoms` object, so the structure filename has to come from
the CLI via `extra_inputs` (defined on `RunContext` in Task 1):

```python
    run_context.extra_inputs = {
        "structure": structure.name,
        "structure_abspath": str(structure.resolve()),
    }
```

In `batch`, mint one id before the subdirectory loop (after the `subdirs`
list is built, around line 224):

```python
    batch_info = BatchInfo(
        batch_id=new_batch_id(),
        driver="mliprun optimize batch",
        argv=list(sys.argv),
        root=str(parent.resolve()),
    )
    batch_sources = param_sources_from_ctx(ctx)
```

Inside the loop, after `structure` is resolved and before `run_optimization`:

```python
            run_context = RunContext(
                command="optimize",
                mode="batch",
                batch=batch_info,
                param_sources=batch_sources,
            )
            run_context.extra_inputs = {
                "structure": structure.name,
                "structure_abspath": str(structure.resolve()),
            }
```

and add the same three arguments to the `run_optimization(...)` call inside the
loop:

```python
                run_context=run_context,
                device_requested=device,
                device_resolved=_resolve_device(device),
```

Finally, remove the duplication: `run` writes its `.txt` inline (lines 124-140)
while `batch` calls `_write_params` (line 313) with identical output. Delete the
inline block in `run` and call the shared helper instead:

```python
    # Save parameters
    _write_params(output_dir / "opt_params.txt", mlip, uma_task, mace_head,
                  device, relax_cell, structure.name, optimizer, fmax,
                  max_steps, converged, output_dir)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_run_record_integration.py tests/test_optimize_batch.py tests/test_cli_commands.py -v`
Expected: PASS. `test_optimize_batch.py` lines 85, 184 and 198 assert on `opt_params.txt`, which is unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/cli/commands/optimize.py src/mliprun/core/run_record.py tests/test_run_record_integration.py
git commit -m "feat(optimize): record run mode, batch identity and param sources

Every subdirectory of one batch shares a batch_id, so sibling runs can
be grouped after the fact. Also collapses the two duplicate copies of
the opt_params.txt writer into the existing shared helper."
```

---

### Task 5: `md` — stage per resume and raw statistics

**Files:**
- Modify: `src/mliprun/core/md.py` (`run_md` signature ~line 163; body ~line 231-316)
- Modify: `src/mliprun/cli/commands/md.py` (~line 19, ~line 201)
- Test: `tests/test_run_record_integration.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `run_md(..., run_context=None, device_requested="auto", device_resolved="auto")`; a resume appends a stage; per-stage `results` carry the raw statistics named in the spec.

**Two traps specific to this file:**
1. `run_md` has an **early `return`** when `plot` is false (just after `df = pd.DataFrame(log_data)`). The record must be completed *before* that return, or a non-plotting run never finalizes.
2. On resume, `log_data` is pre-seeded with the prior CSV's rows (lines 240-246), so the final `df` is the **whole history**. Stage statistics must be computed over the new rows only — slice from the prior row count.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_record_integration.py`:

```python
from mliprun.core.md import run_md


class TestMdRecord:
    def _run(self, tmp_path, **kwargs):
        opts = dict(atoms=_cu(), ensemble="nve", steps=20, timestep=1.0,
                    log_interval=2, traj_interval=10, output_dir=tmp_path,
                    model_name="emt", plot=False)
        opts.update(kwargs)
        return run_md(**opts)

    def test_records_raw_statistics(self, tmp_path):
        self._run(tmp_path)
        results = _record(tmp_path)["stages"][0]["results"]
        assert results["mean_temperature_K"] >= 0.0
        assert results["std_temperature_K"] >= 0.0
        assert isinstance(results["mean_total_energy_eV"], float)
        assert isinstance(results["mean_potential_energy_eV"], float)
        assert isinstance(results["total_energy_drift_eV_per_atom_per_ps"], float)
        assert len(results["decile_mean_total_energy_eV"]) == 10

    def test_completes_even_without_plotting(self, tmp_path):
        """run_md returns early when plot=False; the record must still close."""
        self._run(tmp_path, plot=False)
        assert _record(tmp_path)["status"] == "converged"
        assert _record(tmp_path)["stages"][0]["walltime_s"] is not None

    def test_resume_appends_a_stage(self, tmp_path):
        self._run(tmp_path)
        self._run(tmp_path, resume=True, steps=10)
        stages = _record(tmp_path)["stages"]
        assert len(stages) == 2
        assert stages[0]["kind"] == "md"
        assert stages[1]["kind"] == "md-resume"
        assert stages[0]["results"]["mean_temperature_K"] >= 0.0

    def test_resume_statistics_cover_only_new_rows(self, tmp_path):
        """Stage 1 must describe the resumed segment, not the whole history."""
        self._run(tmp_path, steps=20)
        self._run(tmp_path, resume=True, steps=10)
        stages = _record(tmp_path)["stages"]
        assert stages[1]["results"]["n_samples"] < stages[0]["results"]["n_samples"]

    def test_npt_records_pressure_and_volume(self, tmp_path):
        self._run(tmp_path, ensemble="npt", temperature=300, pressure=0.0,
                  steps=10)
        results = _record(tmp_path)["stages"][0]["results"]
        assert "mean_pressure_GPa" in results
        assert "mean_volume_A3" in results
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_run_record_integration.py -k MdRecord -v`
Expected: FAIL — no `mliprun_run.json`.

- [ ] **Step 3: Implement**

In `src/mliprun/core/md.py`, add the import:

```python
from mliprun.core.run_record import RunContext, RunRecord, collect_provenance
```

Extend the `run_md` signature with the same three keyword arguments used in
Task 3 (`run_context`, `device_requested`, `device_resolved`), each defaulting
as before.

Add this statistics helper at module level, above `run_md`:

```python
def _md_statistics(df, n_atoms: int) -> dict:
    """Raw, assumption-free summary of one MD segment.

    Deliberately excludes any equilibration-window detection: choosing a
    production region is a methodological decision, and MD samples are
    autocorrelated, so a naive standard-error criterion would truncate too
    early and understate the uncertainty. Decile block means are provided so
    equilibration can be judged by eye without reopening the CSV.
    """
    if df.empty:
        return {"n_samples": 0}

    total = df["total_energy(eV)"]
    stats = {
        "n_samples": int(len(df)),
        "mean_temperature_K": float(df["temperature(K)"].mean()),
        "std_temperature_K": float(df["temperature(K)"].std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_total_energy_eV": float(total.mean()),
        "std_total_energy_eV": float(total.std(ddof=1)) if len(df) > 1 else 0.0,
        "mean_potential_energy_eV": float(df["potential_energy(eV)"].mean()),
        "std_potential_energy_eV": float(df["potential_energy(eV)"].std(ddof=1)) if len(df) > 1 else 0.0,
    }

    # Drift per atom per ps: the number that says whether an NVE run is
    # trustworthy. Time is logged in fs.
    span_fs = float(df["time(fs)"].iloc[-1] - df["time(fs)"].iloc[0])
    if span_fs > 0 and n_atoms > 0:
        drift = float(total.iloc[-1] - total.iloc[0])
        stats["total_energy_drift_eV_per_atom_per_ps"] = (
            drift / n_atoms / (span_fs / 1000.0)
        )
    else:
        stats["total_energy_drift_eV_per_atom_per_ps"] = 0.0

    # Ten equal blocks over the segment, so equilibration is eyeballable.
    n = len(df)
    stats["decile_mean_total_energy_eV"] = [
        float(total.iloc[(i * n) // 10:((i + 1) * n) // 10].mean())
        if ((i + 1) * n) // 10 > (i * n) // 10 else float(total.iloc[-1])
        for i in range(10)
    ]

    if "pressure(GPa)" in df:
        stats["mean_pressure_GPa"] = float(df["pressure(GPa)"].mean())
        stats["mean_volume_A3"] = float(df["volume(A^3)"].mean())
    return stats
```

In `run_md`, capture the prior row count where the resume branch already reads
the CSV (inside `if resume:`, after `prior_df` is loaded):

```python
    n_prior_rows = 0
    if resume:
        prior_df = pd.read_csv(csv_file)
        n_prior_rows = len(prior_df)
```

(keep the existing `for col in log_data:` extension and `prior_steps` lines that
follow.)

Open the record immediately after `n_prior_rows` is known and before
`setup_dynamics` is called:

```python
    record = RunRecord.begin(
        output_path,
        command="md",
        stage_kind="md-resume" if resume else "md",
        parameters={
            "ensemble": ensemble, "steps": steps, "temperature": temperature,
            "pressure": pressure, "timestep": timestep,
            "thermostat": thermostat, "barostat": barostat,
            "friction": friction, "ttime": ttime, "taut": taut, "taup": taup,
            "log_interval": log_interval, "traj_interval": traj_interval,
        },
        inputs={"n_atoms": len(atoms), "formula": atoms.get_chemical_formula()},
        provenance=collect_provenance(
            mlip_model=model_name,
            device_requested=device_requested,
            device_resolved=device_resolved,
        ),
        run_context=run_context,
        append=resume,
    )
```

Wrap the dynamics run and finalize **before** the `if not plot: return`.
Replace the block from `dyn.run(steps)` through `df = pd.DataFrame(log_data)`
with:

```python
    try:
        dyn.run(steps)
    except Exception as exc:
        traj_writer.close()
        _flush_csv()
        record.complete(status="failed", results={"error": str(exc)})
        raise
    traj_writer.close()
    _flush_csv()  # final tail of buffered rows

    df = pd.DataFrame(log_data)

    # Statistics describe THIS segment only: on resume, log_data was seeded
    # with the prior run's rows, so df holds the whole history.
    record.complete(
        status="converged",
        steps=int(dyn.get_number_of_steps()),
        results=_md_statistics(df.iloc[n_prior_rows:], len(atoms)),
    )
```

In `src/mliprun/cli/commands/md.py`, add `ctx: typer.Context` as the first
parameter of `run`, add the imports

```python
from mliprun.core.run_record import RunContext
from mliprun.cli.utils import _resolve_device, param_sources_from_ctx
```

and pass a context into `run_md` (before the existing call at line 201):

```python
    run_context = RunContext(
        command="md",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
    run_context.extra_inputs = {
        "structure": structure.name,
        "structure_abspath": str(structure.resolve()),
    }
```

with these arguments added to the `run_md(...)` call:

```python
        run_context=run_context,
        device_requested=device,
        device_resolved=_resolve_device(device),
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_run_record_integration.py tests/test_core_md.py -v`
Expected: PASS, including `test_completes_even_without_plotting` and
`test_resume_statistics_cover_only_new_rows`.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/md.py src/mliprun/cli/commands/md.py tests/test_run_record_integration.py
git commit -m "feat(md): run record with per-segment statistics, stage per resume

Statistics are computed over the resumed segment only (log_data is
seeded with prior rows on resume) and the record is finalized before
the plot=False early return. No equilibration-window detection: that is
a methodological choice, deferred deliberately."
```

---

### Task 6: NEB — barrier metrics and restart stages

**Files:**
- Modify: `src/mliprun/core/neb.py` (`run_neb` ~line 574-660)
- Modify: `src/mliprun/cli/commands/neb.py` (~line 336, ~line 390)
- Test: `tests/test_run_record_integration.py`

**Interfaces:**
- Consumes: Tasks 1-2.
- Produces: `run_neb(..., run_context=None, append=False)`; stage `results` carry `forward_barrier_eV`, `reverse_barrier_eV`, `reaction_energy_eV`, `ts_image_index`, `n_images`, `ts_at_endpoint`, `final_fmax_eV_per_A`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_record_integration.py`:

```python
from mliprun.core.run_record import RunRecord


class TestNebBarrierMetrics:
    """The barrier arithmetic, isolated from the cost of a real NEB."""

    def test_forward_and_reverse_barriers(self):
        from mliprun.core.neb import summarize_neb_path

        # Energies along a path with a clear interior maximum.
        s = summarize_neb_path([0.0, 0.4, 0.9, 0.3, -0.35])
        assert s["forward_barrier_eV"] == pytest.approx(0.9)
        assert s["reverse_barrier_eV"] == pytest.approx(1.25)
        assert s["reaction_energy_eV"] == pytest.approx(-0.35)
        assert s["ts_image_index"] == 2
        assert s["n_images"] == 5
        assert s["ts_at_endpoint"] is False

    def test_maximum_at_final_image_is_flagged(self):
        from mliprun.core.neb import summarize_neb_path

        s = summarize_neb_path([0.0, 0.2, 0.5, 0.8])
        assert s["ts_at_endpoint"] is True
        assert s["ts_image_index"] == 3

    def test_maximum_at_first_image_is_flagged(self):
        from mliprun.core.neb import summarize_neb_path

        s = summarize_neb_path([0.5, 0.2, 0.0])
        assert s["ts_at_endpoint"] is True

    def test_single_image_degenerates_safely(self):
        from mliprun.core.neb import summarize_neb_path

        s = summarize_neb_path([1.0])
        assert s["n_images"] == 1
        assert s["forward_barrier_eV"] == 0.0


class TestNebStageAppend:
    def test_restart_appends_second_stage(self, tmp_path):
        """A plain NEB then a CI-NEB in the same directory is two stages."""
        rec = RunRecord.begin(
            tmp_path, command="neb", stage_kind="neb",
            parameters={"climb": False}, inputs={"n_images": 5},
            provenance={"mliprun_version": "test"},
            stage_parameters={"climb": False},
        )
        rec.complete(status="converged", steps=100,
                     results={"forward_barrier_eV": 0.85})

        rec2 = RunRecord.begin(
            tmp_path, command="neb", stage_kind="neb-restart",
            parameters={"climb": True}, inputs={"n_images": 5},
            provenance={"mliprun_version": "test"},
            stage_parameters={"climb": True}, append=True,
        )
        rec2.complete(status="converged", steps=40,
                      results={"forward_barrier_eV": 0.91})

        stages = _record(tmp_path)["stages"]
        assert [s["kind"] for s in stages] == ["neb", "neb-restart"]
        assert stages[0]["results"]["forward_barrier_eV"] == 0.85
        assert stages[1]["parameters"]["climb"]["value"] is True
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_run_record_integration.py -k "Neb" -v`
Expected: FAIL — `ImportError: cannot import name 'summarize_neb_path'`

- [ ] **Step 3: Implement**

In `src/mliprun/core/neb.py`, add the import and a module-level helper above
the `CustomNEB` class:

```python
from mliprun.core.run_record import RunContext, RunRecord, collect_provenance
from mliprun.core.utils import resolve_device
```

(`resolve_device` is the helper relocated into core in Task 2. `CustomNEB`
stores `self.device` from its constructor, so the NEB record can report both
the requested and the resolved device without core reaching into the CLI.)

```python
def summarize_neb_path(energies) -> dict:
    """Barrier metrics for one converged NEB path.

    Both directions are reported: the existing convergence log records only
    ``max(E) - E[0]``, which is half the answer for any reaction you might
    want to run backwards.

    ``ts_at_endpoint`` flags a maximum sitting on the first or last image,
    which means no saddle was bracketed -- the path is monotonic and the
    "barrier" is not a transition state.
    """
    energies = [float(e) for e in energies]
    n = len(energies)
    if n == 0:
        return {"n_images": 0, "forward_barrier_eV": 0.0,
                "reverse_barrier_eV": 0.0, "reaction_energy_eV": 0.0,
                "ts_image_index": None, "ts_at_endpoint": False}
    peak = max(energies)
    idx = energies.index(peak)
    return {
        "n_images": n,
        "forward_barrier_eV": peak - energies[0],
        "reverse_barrier_eV": peak - energies[-1],
        "reaction_energy_eV": energies[-1] - energies[0],
        "ts_image_index": idx,
        "ts_at_endpoint": n > 1 and idx in (0, n - 1),
    }
```

Extend `run_neb`'s signature with `run_context: Optional[RunContext] = None`
and `append: bool = False`.

Open the record at the top of `run_neb`, before the `NEB(...)` construction:

```python
        record = RunRecord.begin(
            self.output_dir,
            command="neb",
            stage_kind="neb-restart" if append else "neb",
            parameters={"fmax": self.fmax, "num_images": self.num_images,
                        "uma_task": self.uma_task,
                        "interp_fmax": self.interp_fmax,
                        "interp_steps": self.interp_steps},
            inputs={"n_images": len(self.images),
                    "n_atoms": len(self.images[0]) if self.images else 0},
            provenance=collect_provenance(
                mlip_model=self.mlip,
                device_requested=self.device,
                device_resolved=resolve_device(self.device),
            ),
            run_context=run_context,
            # k, climb, max_steps and the optimizer are arguments of this
            # call, not instance state, so they belong to the stage: a
            # CI-NEB restart changes them without changing the directory.
            stage_parameters={"climb": climb, "max_steps": max_steps, "k": k,
                              "optimizer": getattr(optimizer, "__name__", str(optimizer))},
            append=append,
        )
```

Capture convergence from the optimizer (line 640 currently discards it) and
finalize after the final energies are computed. Replace `opt.run(fmax=self.fmax, steps=max_steps)` with:

```python
            try:
                converged = opt.run(fmax=self.fmax, steps=max_steps)
            except Exception as exc:
                record.complete(status="failed", results={"error": str(exc)})
                raise
```

Then after the existing "Ensure final energies are computed" loop, and before
the final `return self.images`:

```python
        final_energies = [img.get_potential_energy() for img in self.images]
        results = summarize_neb_path(final_energies)
        results["final_fmax_eV_per_A"] = (
            float(log_data["fmax(eV/A)"][-1]) if log_data["fmax(eV/A)"] else None
        )
        record.complete(
            status="converged" if converged else "not_converged",
            steps=int(opt.nsteps),
            results=results,
        )
```

In `src/mliprun/cli/commands/neb.py`, add `ctx: typer.Context` as the first
parameter of the `run` command, import `RunContext` and
`param_sources_from_ctx`, and pass them into the `neb_obj.run_neb(...)` call at
line 390 — setting `append` from whether this invocation is a restart:

```python
    run_context = RunContext(
        command="neb",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
    neb_obj.run_neb(optimizer=neb_opt, climb=climb_val, max_steps=max_steps,
                    plot=plot, run_context=run_context, append=restart)
```

(`restart` is the existing boolean the command already uses to choose the
restart path.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_run_record_integration.py tests/test_core_neb.py tests/test_neb_restart.py -v`
Expected: PASS. `test_neb_restart.py` must pass unchanged — `neb_parameters.txt`
and its parser are untouched.

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/neb.py src/mliprun/cli/commands/neb.py tests/test_run_record_integration.py
git commit -m "feat(neb): record barrier metrics and append a stage per restart

Records forward and reverse barriers (the log kept only the forward
one) and flags a maximum sitting on an endpoint, which means no saddle
was bracketed. A plain-then-CI-NEB workflow is two stages, so the
second invocation no longer erases the first."
```

---

### Task 7: AutoNEB

**Files:**
- Modify: `src/mliprun/core/neb.py` (`run_autoneb` ~line 666)
- Modify: `src/mliprun/cli/commands/autoneb.py` (~line 84, ~line 110)
- Test: `tests/test_run_record_integration.py`

**Interfaces:**
- Consumes: Tasks 1, 2, 6 (`summarize_neb_path`).
- Produces: `run_autoneb(..., run_context=None)` writing a record with `command: "autoneb"` and `stage_kind: "autoneb"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_run_record_integration.py`:

```python
class TestAutonebRecord:
    def test_autoneb_records_command_and_stage_kind(self, tmp_path):
        """AutoNEB shares the barrier summary but is its own command."""
        rec = RunRecord.begin(
            tmp_path, command="autoneb", stage_kind="autoneb",
            parameters={"n_max": 9, "climb": True},
            inputs={"n_images": 5},
            provenance={"mliprun_version": "test"},
        )
        from mliprun.core.neb import summarize_neb_path
        rec.complete(status="converged", steps=250,
                     results=summarize_neb_path([0.0, 0.6, 1.1, 0.2]))
        data = _record(tmp_path)
        assert data["command"] == "autoneb"
        assert data["stages"][0]["kind"] == "autoneb"
        assert data["stages"][0]["results"]["forward_barrier_eV"] == pytest.approx(1.1)
        assert data["stages"][0]["results"]["ts_at_endpoint"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_run_record_integration.py -k Autoneb -v`
Expected: PASS only if Tasks 1 and 6 are complete; this test exercises the
shared machinery. If `summarize_neb_path` is missing, FAIL with ImportError.

- [ ] **Step 3: Implement**

In `run_autoneb`, add `run_context: Optional[RunContext] = None` to the
signature and open the record before the AutoNEB construction:

```python
        record = RunRecord.begin(
            self.output_dir,
            command="autoneb",
            stage_kind="autoneb",
            parameters={"n_max": n_max, "climb": climb, "fmax": self.fmax,
                        "k": k, "maxsteps": maxsteps,
                        "uma_task": self.uma_task},
            inputs={"n_images": len(self.images),
                    "n_atoms": len(self.images[0]) if self.images else 0},
            provenance=collect_provenance(
                mlip_model=self.mlip,
                device_requested=self.device,
                device_resolved=resolve_device(self.device),
            ),
            run_context=run_context,
        )
```

Wrap the AutoNEB execution so a failure is recorded, and finalize afterwards
using the same summary helper:

```python
        try:
            autoneb.run()
        except Exception as exc:
            record.complete(status="failed", results={"error": str(exc)})
            raise

        final_energies = [img.get_potential_energy() for img in self.images]
        record.complete(status="converged", results=summarize_neb_path(final_energies))
```

In `src/mliprun/cli/commands/autoneb.py`, add `ctx: typer.Context` as the first
parameter of the command, import `RunContext` and `param_sources_from_ctx`, and
pass a context into the `run_autoneb(...)` call:

```python
    run_context = RunContext(
        command="autoneb",
        mode="one-off",
        param_sources=param_sources_from_ctx(ctx),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_run_record_integration.py tests/test_cli_commands.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/mliprun/core/neb.py src/mliprun/cli/commands/autoneb.py tests/test_run_record_integration.py
git commit -m "feat(autoneb): write the run record

Reuses the NEB barrier summary so autoneb and neb report identical
metrics under the same keys."
```

---

### Task 8: Document the record

**Files:**
- Modify: `docs/OUTPUTS.md`

**Interfaces:**
- Consumes: the schema as implemented in Tasks 1-7.
- Produces: no code.

- [ ] **Step 1: Add `mliprun_run.json` to every command's output table**

`docs/OUTPUTS.md` has one table per command. Add this row to the table for
`optimize run`, `optimize batch` (per-subdirectory), `md run`, `neb run` and
`autoneb run`:

```markdown
| `mliprun_run.json` | JSON | Canonical run record: every resolved parameter with the source of its value, provenance (versions, device, host, timings), and per-stage outcome. See [The run record](#the-run-record). |
```

- [ ] **Step 2: Add the schema section**

Append to `docs/OUTPUTS.md`:

```markdown
---

## The run record

Every command writes `mliprun_run.json` into its output directory. Unlike the
`*_params.txt` files (which are kept, and which NEB restart still parses), this
one file has the same schema for every command and is written by the core
layer — so a script that calls `run_optimization` directly gets one too.

### Top-level keys

| Key | Meaning |
|-----|---------|
| `schema_version` | Currently `1`. Check it before parsing. |
| `command` | `optimize`, `md`, `neb` or `autoneb`. |
| `status` | Status of the **latest** stage: `running`, `converged`, `not_converged` or `failed`. A record left saying `running` means the job died without reporting back. |
| `run.mode` | `one-off` or `batch`. |
| `run.batch` | `null` for one-off runs; otherwise `batch_id`, `driver`, `argv`, `root`, `config_file`. Every run of one batch shares a `batch_id`. |
| `inputs` | Structure filename and absolute path, atom count, formula. |
| `parameters` | Every resolved parameter as `{"value": ..., "source": ...}`. |
| `provenance` | Versions (mliprun, ASE, the MLIP package), model, requested vs resolved device, Python, hostname, timestamps, wall time. |
| `stages` | One entry per invocation in this directory. A NEB restart or MD resume **appends**. |

### Parameter sources

`source` is one of:

| Value | Meaning |
|-------|---------|
| `user` | Given on the command line (or in a config file). |
| `default` | Not given; the command's default applied. |
| `env` | Taken from an environment variable. |
| `prompt` | Typed at an interactive prompt. |
| `unspecified` | A library caller supplied no context. mliprun does not guess: a caller passing `fmax=0.05` explicitly is indistinguishable from one that omitted it. |

### Stages

`stages` is an array because a workflow can be several invocations in one
directory — most commonly a plain NEB followed by a CI-NEB restart, or an MD
run extended with `--resume`. Each stage records its own `kind`
(`optimize`, `md`, `md-resume`, `neb`, `neb-restart`, `autoneb`), `status`,
`steps`, `walltime_s`, any `parameters` that stage changed, and its `results`.
A stage's terminal status is never rewritten, so a converged stage 0 followed
by a failed stage 1 keeps both facts.

A stage carrying `"prior_history_unknown": true` was appended to a directory
with no readable prior record — an older run, or one whose record was damaged.

### Results by command

**optimize** — `converged`, `final_energy_eV`, `final_fmax_eV_per_A`.

**md** — `n_samples`, mean and std of temperature, total energy and potential
energy, `total_energy_drift_eV_per_atom_per_ps`, and
`decile_mean_total_energy_eV` (ten block means over the segment, for judging
equilibration by eye). NPT adds `mean_pressure_GPa` and `mean_volume_A3`.
Statistics describe **that stage's segment only**, not the whole trajectory.

There is deliberately no equilibration-window detection and no production
average: choosing a production region is a methodological decision, and MD
samples are autocorrelated, so a naive standard-error criterion truncates too
early and understates uncertainty.

**neb / autoneb** — `forward_barrier_eV`, `reverse_barrier_eV`,
`reaction_energy_eV`, `ts_image_index`, `n_images`, `final_fmax_eV_per_A`, and
`ts_at_endpoint`. The last is a sanity flag: `true` means the energy maximum
sits on the first or last image, so no saddle was bracketed and the "barrier"
is not a transition state.

### Failure behavior

A record write can never abort a run. Writes are atomic (serialize, then
replace), so a crash never leaves truncated JSON. An unparseable record found
on restart is moved to `mliprun_run.json.corrupt-<timestamp>` and a fresh one
started. `NaN` and `Inf` are written as `null`, since neither is valid JSON.
```

- [ ] **Step 3: Verify the document is consistent with the code**

Run: `grep -o '"[a-z_]*_eV[a-zA-Z_]*"' docs/OUTPUTS.md | sort -u`
Cross-check each key against `summarize_neb_path` and `_md_statistics`. Every
key named in the doc must exist in the code, and vice versa.

- [ ] **Step 4: Commit**

```bash
git add docs/OUTPUTS.md
git commit -m "docs(outputs): document mliprun_run.json and its schema"
```

---

## Verification

After Task 8, run the full suite and confirm nothing regressed:

```bash
pytest -q
```

Expected: all previously-passing tests still pass. Specifically confirm
`tests/test_optimize_batch.py`, `tests/test_neb_restart.py`,
`tests/test_core_md.py` and `tests/test_characterization_golden.py` are
unchanged — no golden was touched, and every `.txt` file is still written.

Then confirm the original defect is fixed, using the real library entry point
that `batch_relax.py` uses:

```bash
python - <<'PY'
from pathlib import Path
import json, tempfile
from ase.build import bulk
from ase.calculators.emt import EMT
from mliprun.core.optimize import run_optimization

d = Path(tempfile.mkdtemp())
atoms = bulk("Cu", "fcc", a=3.7); atoms.calc = EMT()
run_optimization(atoms=atoms, fmax=0.5, max_steps=5, output_dir=d,
                 model_name="emt", verbose=False)
print(json.dumps(json.loads((d / "mliprun_run.json").read_text()), indent=2))
PY
```

Expected: a complete record with `command`, `status`, `parameters` (all tagged
`unspecified`), `provenance` and one stage — the file that `basin_00` did not
have.
