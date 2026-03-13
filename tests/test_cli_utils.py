"""Tests for mlip_platform.cli.utils."""
import pytest
from unittest.mock import patch

import typer
from click.exceptions import Exit as ClickExit

from mlip_platform.cli.utils import (
    detect_mlip,
    validate_mlip,
    resolve_mlip,
    parse_relax_atoms,
    FAIRCHEM_AVAILABLE,
    SEVENN_AVAILABLE,
    MACE_AVAILABLE,
)


class TestDetectMlip:
    def test_returns_string(self):
        # Should return a string regardless of what's available
        try:
            result = detect_mlip()
            assert isinstance(result, str)
        except SystemExit:
            # No MLIP installed — that's also a valid outcome
            pass

    @patch("mlip_platform.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_prefers_uma(self):
        assert detect_mlip() == "uma-s-1p1"

    @patch("mlip_platform.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mlip_platform.cli.utils.SEVENN_AVAILABLE", True)
    def test_falls_back_to_sevenn(self):
        assert detect_mlip() == "7net-mf-ompa"

    @patch("mlip_platform.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mlip_platform.cli.utils.SEVENN_AVAILABLE", False)
    @patch("mlip_platform.cli.utils.MACE_AVAILABLE", True)
    def test_falls_back_to_mace(self):
        assert detect_mlip() == "mace"

    @patch("mlip_platform.cli.utils.FAIRCHEM_AVAILABLE", False)
    @patch("mlip_platform.cli.utils.SEVENN_AVAILABLE", False)
    @patch("mlip_platform.cli.utils.MACE_AVAILABLE", False)
    def test_none_available_raises(self):
        with pytest.raises((SystemExit, ClickExit)):
            detect_mlip()


class TestValidateMlip:
    def test_auto_passes(self):
        validate_mlip("auto")  # should not raise

    def test_unknown_mlip_raises(self):
        with pytest.raises((SystemExit, ClickExit)):
            validate_mlip("nonexistent-model")

    @patch("mlip_platform.cli.utils.MACE_AVAILABLE", False)
    def test_mace_unavailable_raises(self):
        with pytest.raises((SystemExit, ClickExit)):
            validate_mlip("mace")

    @patch("mlip_platform.cli.utils.SEVENN_AVAILABLE", False)
    def test_sevenn_unavailable_raises(self):
        with pytest.raises((SystemExit, ClickExit)):
            validate_mlip("7net-mf-ompa")

    @patch("mlip_platform.cli.utils.FAIRCHEM_AVAILABLE", False)
    def test_uma_unavailable_raises(self):
        with pytest.raises((SystemExit, ClickExit)):
            validate_mlip("uma-s-1p1")


class TestResolveMlip:
    @patch("mlip_platform.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_auto_resolves(self):
        result = resolve_mlip("auto")
        assert isinstance(result, str)
        assert result != "auto"

    @patch("mlip_platform.cli.utils.FAIRCHEM_AVAILABLE", True)
    def test_explicit_passes_through(self):
        result = resolve_mlip("uma-s-1p1")
        assert result == "uma-s-1p1"


class TestParseRelaxAtoms:
    def test_valid_input(self):
        result = parse_relax_atoms("0,1,5", num_atoms=10)
        assert result == [0, 1, 5]

    def test_single_atom(self):
        result = parse_relax_atoms("3", num_atoms=10)
        assert result == [3]

    def test_with_spaces(self):
        result = parse_relax_atoms("0, 1, 5", num_atoms=10)
        assert result == [0, 1, 5]

    def test_invalid_format_raises(self):
        with pytest.raises((SystemExit, ClickExit)):
            parse_relax_atoms("a,b,c", num_atoms=10)

    def test_out_of_range_raises(self):
        with pytest.raises((SystemExit, ClickExit)):
            parse_relax_atoms("0,1,100", num_atoms=10)

    def test_negative_index_raises(self):
        with pytest.raises((SystemExit, ClickExit)):
            parse_relax_atoms("-1,0,1", num_atoms=10)
