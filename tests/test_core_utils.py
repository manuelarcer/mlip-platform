"""Tests for mlip_platform.core.utils."""
import numpy as np
import pytest
from pathlib import Path

from mlip_platform.core.utils import calc_fmax, GPA_TO_EV_PER_ANG3


class TestCalcFmax:
    def test_known_forces(self):
        forces = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
        assert calc_fmax(forces) == pytest.approx(5.0)

    def test_single_atom(self):
        forces = np.array([[1.0, 0.0, 0.0]])
        assert calc_fmax(forces) == pytest.approx(1.0)

    def test_zero_forces(self):
        forces = np.zeros((5, 3))
        assert calc_fmax(forces) == pytest.approx(0.0)

    def test_negative_forces(self):
        forces = np.array([[-3.0, -4.0, 0.0]])
        assert calc_fmax(forces) == pytest.approx(5.0)

    def test_list_input(self):
        forces = [[0.0, 0.0, 2.0], [0.0, 0.0, 0.0]]
        assert calc_fmax(forces) == pytest.approx(2.0)


class TestConstants:
    def test_gpa_conversion(self):
        assert GPA_TO_EV_PER_ANG3 == pytest.approx(0.006241509)


class TestWriteParametersFile:
    def test_writes_correct_format(self, tmp_path):
        from mlip_platform.core.params_io import write_parameters_file

        path = tmp_path / "params.txt"
        write_parameters_file(path, "Test Parameters", {
            "MLIP model:": "uma-s-1p1",
            "fmax:": 0.05,
            "Steps:": 200,
        })

        content = path.read_text()
        assert "Test Parameters" in content
        assert "uma-s-1p1" in content
        assert "0.05" in content
        assert "200" in content

    def test_empty_params(self, tmp_path):
        from mlip_platform.core.params_io import write_parameters_file

        path = tmp_path / "empty.txt"
        write_parameters_file(path, "Empty", {})
        content = path.read_text()
        assert "Empty" in content


class TestWriteEndpointResults:
    def test_writes_all_sections(self, tmp_path):
        from mlip_platform.core.params_io import write_endpoint_results

        path = tmp_path / "endpoints.txt"
        results = {
            "initial": {
                "energy_before": -10.0,
                "energy_after": -10.5,
                "energy_change": -0.5,
                "steps": 15,
                "converged": True,
            },
            "final": {
                "energy_before": -9.0,
                "energy_after": -9.8,
                "energy_change": -0.8,
                "steps": 20,
                "converged": True,
            },
            "reaction_energy": 0.7,
            "similarity": {
                "avg_displacement": 1.5,
                "max_displacement": 2.0,
                "max_disp_atom": 3,
                "min_displacement": 0.1,
                "energy_diff": 0.7,
                "is_similar": False,
                "warning_reasons": [],
            },
        }

        write_endpoint_results(path, results)
        content = path.read_text()

        assert "Initial Structure:" in content
        assert "Final Structure:" in content
        assert "Reaction energy:" in content
        assert "Similarity Check:" in content
        assert "-10.500000" in content

    def test_writes_warning_reasons(self, tmp_path):
        from mlip_platform.core.params_io import write_endpoint_results

        path = tmp_path / "endpoints_warn.txt"
        results = {
            "initial": {
                "energy_before": -10.0,
                "energy_after": -10.0,
                "energy_change": 0.0,
                "steps": 0,
                "converged": True,
            },
            "final": {
                "energy_before": -10.0,
                "energy_after": -10.0,
                "energy_change": 0.0,
                "steps": 0,
                "converged": True,
            },
            "reaction_energy": 0.0,
            "similarity": {
                "avg_displacement": 0.01,
                "max_displacement": 0.02,
                "max_disp_atom": 0,
                "min_displacement": 0.001,
                "energy_diff": 0.001,
                "is_similar": True,
                "warning_reasons": ["energy too close", "geometry too similar"],
            },
        }

        write_endpoint_results(path, results)
        content = path.read_text()
        assert "energy too close" in content
        assert "geometry too similar" in content
