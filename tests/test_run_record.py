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
