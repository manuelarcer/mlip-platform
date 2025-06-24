import subprocess
import sys
import pathlib


def test_cli_help():
    """Ensure the new CLI group shows all subcommands."""
    cli_script = pathlib.Path(__file__).parent.parent / "src" / "mlip_platform" / "cli.py"
    result = subprocess.run(
        [sys.executable, str(cli_script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout
    # The group help should list our three commands
    assert "optimize" in out
    assert "neb" in out
    assert "md" in out


def test_subcommands_help():
    """Smoke-test each subcommand help text."""
    cli_script = pathlib.Path(__file__).parent.parent / "src" / "mlip_platform" / "cli.py"
    for cmd in ("optimize", "neb", "md"):
        result = subprocess.run(
            [sys.executable, str(cli_script), cmd, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Help for '{cmd}' failed"