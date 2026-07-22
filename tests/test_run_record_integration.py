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
