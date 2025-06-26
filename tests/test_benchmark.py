import pytest
from pathlib import Path
from mlip_platform.core.benchmark import run

# Define the full path to your test structure files
STRUCTURES = Path("tests/fixtures/structures")
POSCAR = STRUCTURES / "POSCAR"
FRAGMENT_INITIAL = STRUCTURES / "fragment_initial.vasp"
FRAGMENT_FINAL = STRUCTURES / "fragment_final.vasp"
LI_INITIAL = STRUCTURES / "li_initial.vasp"
LI_FINAL = STRUCTURES / "li_final.vasp"

# List of models to test
MLIP_METHODS = ["mace", "sevenn"]

@pytest.mark.parametrize("method", MLIP_METHODS)
def test_single_point(method):
    result = run(
        structure=str(POSCAR),
        model_name=None,
        method=method,
        run_type="single_point"
    )
    assert "energy" in result, f"{method} did not return energy"
    assert "time" in result
    assert isinstance(result["energy"], float)
    assert result["time"] > 0

@pytest.mark.parametrize("method", MLIP_METHODS)
def test_neb(method):
    neb_structures = f"{LI_INITIAL},{LI_FINAL}"
    result = run(
        structure=str(neb_structures),
        model_name=None,
        method=method,
        run_type="neb"
    )
    assert "energy" in result, f"{method} NEB run failed"
    assert "time" in result
    assert result["time"] > 0

@pytest.mark.parametrize("method", MLIP_METHODS)
def test_md(method):
    result = run(
        structure=str(FRAGMENT_INITIAL),
        model_name=None,
        method=method,
        run_type="md"
    )
    assert "energy" in result, f"{method} MD run failed"
    assert "time" in result
    assert result["time"] > 0
