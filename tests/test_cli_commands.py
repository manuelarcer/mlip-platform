"""Tests for CLI command invocations."""
import subprocess
import pytest


def _run_cli(cmd):
    """Run a CLI command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True)


class TestHelpCommands:
    """Test that --help works for all CLI entry points."""

    @pytest.mark.parametrize("command", [
        ["optimize", "run", "--help"],
        ["md", "run", "--help"],
        ["neb", "run", "--help"],
        ["autoneb", "run", "--help"],
        ["autoneb-results", "results", "--help"],
        ["benchmark", "run", "--help"],
    ])
    def test_help_exits_cleanly(self, command):
        result = _run_cli(command)
        assert result.returncode == 0
        assert "Usage" in result.stdout or "Options" in result.stdout


class TestMissingArgs:
    """Test that missing required arguments produce errors."""

    def test_optimize_missing_structure(self):
        result = _run_cli(["optimize", "run", "--no-input"])
        # Should fail because --structure is required
        assert result.returncode != 0

    def test_md_missing_structure(self):
        result = _run_cli(["md", "run", "--no-input"])
        assert result.returncode != 0

    def test_neb_missing_initial_final(self):
        result = _run_cli(["neb", "run"])
        # Should fail: neither --initial/--final nor --restart provided
        assert result.returncode != 0
