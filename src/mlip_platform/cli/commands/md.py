import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.md import run_md

# Optional calculator imports
try:
    from mace.calculators import mace_mp
except ImportError:
    mace_mp = None

try:
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    SevenNetCalculator = None

app = typer.Typer(help="""
Run molecular dynamics simulations using MACE or SevenNet.

Examples:
  Run MD with MACE for 100 steps at 300K:
    mlip-platform md run --structure structure.vasp --model mace --steps 100

  Run MD with SevenNet:
    mlip-platform md run --structure structure.vasp --model 7net-mf-ompa
""")

@app.command("run")
def run(
    structure: Path = typer.Option(..., help="Structure file (.vasp)"),
    model: str = typer.Option("mace", help="MLIP model: 'mace' or '7net-mf-ompa'"),
    steps: int = typer.Option(100, help="Number of MD steps"),
    temperature: float = typer.Option(300.0, help="Temperature in K"),
    log: str = typer.Option("md.log", help="Output log file")
):
    """Run MD simulation using specified MLIP model."""
    atoms = read(structure)

    if model.lower() == "mace":
        if not mace_mp:
            raise typer.Exit("MACE is not installed.")
        atoms.calc = mace_mp(model="medium", device="cpu")
    elif model.lower() in ["sevenn", "7net-mf-ompa"]:
        if not SevenNetCalculator:
            raise typer.Exit("SevenNet is not installed.")
        atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")
    else:
        raise typer.Exit(f"Unsupported model: {model}")

    typer.echo(f"Running MD with {model} for {steps} steps at {temperature}K...")
    run_md(atoms, log_path=log, temperature_K=temperature, steps=steps)
    typer.echo(f"MD simulation complete. Log written to: {log}")
