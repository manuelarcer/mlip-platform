"""Tests for NEB restart logic."""
import pytest
from pathlib import Path

from mlip_platform.core.neb import CustomNEB


class TestParseParametersFile:
    def test_parses_standard_file(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            uma-s-1p1\n"
            "UMA task:              omat\n"
            "Initial:               initial.vasp\n"
            "Final:                 final.vasp\n"
            "Intermediate images:   5\n"
            "Total images:          7\n"
            "IDPP fmax:             0.1\n"
            "IDPP steps:            100\n"
            "Final fmax:            0.05\n"
            "Spring constant (k):   0.1\n"
            "Climb:                 True\n"
            "NEB optimizer:         fire\n"
            "NEB max steps:         600\n"
            "Optimize endpoints:    True\n"
            "Log file:              neb.log\n"
            "Output dir:            /tmp/test\n"
        )

        params = CustomNEB._parse_parameters_file(params_file)
        assert params["mlip"] == "uma-s-1p1"
        assert params["uma_task"] == "omat"
        assert params["num_images"] == 5
        assert params["total_images"] == 7
        assert params["fmax"] == 0.05
        assert params["k"] == 0.1
        assert params["climb"] is True
        assert params["neb_optimizer"] == "fire"
        assert params["neb_max_steps"] == 600

    def test_parses_none_values(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
            "Spring constant (k):   None\n"
            "NEB max steps:         None\n"
        )
        params = CustomNEB._parse_parameters_file(params_file)
        assert params["k"] is None
        assert params["neb_max_steps"] is None

    def test_parses_relax_atoms(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
            "Relax atoms:           [0, 1, 2]\n"
        )
        params = CustomNEB._parse_parameters_file(params_file)
        assert params["relax_atoms"] == [0, 1, 2]

    def test_missing_required_fields_raises(self, tmp_path):
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
        )
        with pytest.raises(ValueError, match="missing required fields"):
            CustomNEB._parse_parameters_file(params_file)


class TestLoadFromRestart:
    def test_missing_trajectory_raises(self, tmp_path):
        # Create only the params file, not the trajectory
        params_file = tmp_path / "neb_parameters.txt"
        params_file.write_text(
            "NEB Run Parameters\n"
            "===================\n"
            "MLIP model:            mace\n"
            "Intermediate images:   3\n"
            "Total images:          5\n"
            "Final fmax:            0.05\n"
        )
        with pytest.raises(FileNotFoundError, match="A2B_full.traj"):
            CustomNEB.load_from_restart(output_dir=tmp_path)

    def test_missing_params_raises(self, tmp_path):
        # Create a dummy trajectory but no params file
        (tmp_path / "A2B_full.traj").touch()
        with pytest.raises(FileNotFoundError, match="neb_parameters.txt"):
            CustomNEB.load_from_restart(output_dir=tmp_path)


class TestCreateBackupFolder:
    def test_creates_backup(self, tmp_path):
        from mlip_platform.cli.commands.neb import create_backup_folder

        # Create some dummy files
        (tmp_path / "A2B.traj").touch()
        (tmp_path / "neb.log").touch()
        (tmp_path / "neb_parameters.txt").write_text("test")
        poscar_dir = tmp_path / "00"
        poscar_dir.mkdir()
        (poscar_dir / "POSCAR").touch()

        backup_dir, moved = create_backup_folder(tmp_path)

        assert backup_dir.exists()
        assert backup_dir.name.startswith("bkup_")
        assert "A2B.traj" in moved
        assert "neb.log" in moved
        assert "00/" in moved
        assert "neb_parameters.txt (copied)" in moved

        # Original neb_parameters.txt should still exist (copied, not moved)
        assert (tmp_path / "neb_parameters.txt").exists()
        # But A2B.traj should be moved
        assert not (tmp_path / "A2B.traj").exists()
