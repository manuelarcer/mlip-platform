"""Tests for the `optimize batch` command: model-reuse and directory iteration."""
import pytest
from pathlib import Path
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write
from typer.testing import CliRunner

from mlip_platform.cli.commands import optimize as opt_cmd
from mlip_platform.cli.commands.optimize import _find_input_structure, app

runner = CliRunner()


def _make_tree(parent: Path, names, input_name="init.vasp"):
    """Create subdirectories each holding one Cu structure as `input_name`."""
    for name in names:
        d = parent / name
        d.mkdir()
        atoms = bulk("Cu", "fcc", a=3.7)
        write(str(d / input_name), atoms, format="vasp")


class TestFindInputStructure:
    def test_single_match(self, tmp_path):
        write(str(tmp_path / "init.vasp"), bulk("Cu", "fcc", a=3.6), format="vasp")
        assert _find_input_structure(tmp_path, "*.vasp").name == "init.vasp"

    def test_ignores_final_output(self, tmp_path):
        # An input plus a previous run's output should still resolve to the input.
        write(str(tmp_path / "init.vasp"), bulk("Cu", "fcc", a=3.6), format="vasp")
        write(str(tmp_path / "opt_final.vasp"), bulk("Cu", "fcc", a=3.6), format="vasp")
        assert _find_input_structure(tmp_path, "*.vasp").name == "init.vasp"

    def test_zero_matches_raises(self, tmp_path):
        with pytest.raises(ValueError, match="no input structure"):
            _find_input_structure(tmp_path, "*.vasp")

    def test_multiple_matches_raises(self, tmp_path):
        write(str(tmp_path / "a.vasp"), bulk("Cu", "fcc", a=3.6), format="vasp")
        write(str(tmp_path / "b.vasp"), bulk("Cu", "fcc", a=3.6), format="vasp")
        with pytest.raises(ValueError, match="multiple structures"):
            _find_input_structure(tmp_path, "*.vasp")


class TestBatchCommand:
    def _patch_model(self, monkeypatch):
        """Replace the real model load with a single EMT calculator and count loads."""
        calls = {"n": 0}

        def fake_build(*args, **kwargs):
            calls["n"] += 1
            return EMT()

        monkeypatch.setattr(opt_cmd, "build_calculator", fake_build)
        monkeypatch.setattr(opt_cmd, "validate_mlip", lambda *a, **k: None)
        return calls

    def test_model_loaded_once_for_many_dirs(self, tmp_path, monkeypatch):
        calls = self._patch_model(monkeypatch)
        _make_tree(tmp_path, ["s1", "s2", "s3"])

        result = runner.invoke(app, [
            "batch", "--parent", str(tmp_path), "--mlip", "mace",
            "--fmax", "0.1", "--max-steps", "50",
        ])

        assert result.exit_code == 0, result.output
        # The whole point: one model load for the entire batch.
        assert calls["n"] == 1

    def test_outputs_written_per_subdir(self, tmp_path, monkeypatch):
        self._patch_model(monkeypatch)
        _make_tree(tmp_path, ["s1", "s2"])

        result = runner.invoke(app, [
            "batch", "--parent", str(tmp_path), "--mlip", "mace",
            "--fmax", "0.1", "--max-steps", "50",
        ])

        assert result.exit_code == 0, result.output
        for name in ["s1", "s2"]:
            assert (tmp_path / name / "opt_final.vasp").exists()
            assert (tmp_path / name / "CONTCAR").exists()
            assert (tmp_path / name / "opt_params.txt").exists()
        assert (tmp_path / "batch_summary.csv").exists()

    def test_continue_on_failure(self, tmp_path, monkeypatch):
        self._patch_model(monkeypatch)
        _make_tree(tmp_path, ["good"])
        # A subdir with no input structure should be logged, not abort the batch.
        (tmp_path / "empty").mkdir()

        result = runner.invoke(app, [
            "batch", "--parent", str(tmp_path), "--mlip", "mace",
            "--fmax", "0.1", "--max-steps", "50",
        ])

        assert result.exit_code == 0, result.output
        assert (tmp_path / "good" / "CONTCAR").exists()
        summary = (tmp_path / "batch_summary.csv").read_text()
        assert "no_input" in summary

    def test_skip_existing(self, tmp_path, monkeypatch):
        self._patch_model(monkeypatch)
        _make_tree(tmp_path, ["done"])
        # Pretend this subdir was already relaxed.
        write(str(tmp_path / "done" / "CONTCAR"), bulk("Cu", "fcc", a=3.6),
              format="vasp")

        result = runner.invoke(app, [
            "batch", "--parent", str(tmp_path), "--mlip", "mace",
            "--fmax", "0.1", "--max-steps", "50", "--skip-existing",
        ])

        assert result.exit_code == 0, result.output
        assert "skipping" in result.output
        summary = (tmp_path / "batch_summary.csv").read_text()
        assert "skipped" in summary
