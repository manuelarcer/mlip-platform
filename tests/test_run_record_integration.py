"""The record as produced through real core and CLI code paths."""
import csv
import json

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from typer.testing import CliRunner

from mliprun.core.optimize import run_optimization
from mliprun.core.run_record import RECORD_FILENAME, RunContext
from mliprun.cli.commands import optimize as opt_cmd
from mliprun.cli.commands.optimize import app as optimize_app

runner = CliRunner()


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
            "run", "--structure", str(struct), "--mlip", "mace",
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

        summary_path = tmp_path / "batch_summary.csv"
        assert summary_path.exists()
        with open(summary_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["subdir", "status", "converged", "steps",
                            "energy_eV", "walltime_s", "detail"]
        assert any(row[0] == "a" for row in rows[1:])

    def test_txt_file_still_written(self, tmp_path, emt_patched):
        """The .txt files are retained; the JSON supplements, not replaces."""
        struct = tmp_path / "init.vasp"
        write(str(struct), bulk("Cu", "fcc", a=3.7), format="vasp")
        runner.invoke(optimize_app, [
            "run", "--structure", str(struct), "--mlip", "mace",
            "--fmax", "0.5", "--max-steps", "5",
        ])
        assert "Geometry Optimization Parameters" in (tmp_path / "opt_params.txt").read_text()
