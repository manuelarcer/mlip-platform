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

app = typer.Typer(help="Run molecular dynamics simulations.")

def detect_model():
    if mace_mp:
        return "mace"
    elif SevenNetCalculator:
        return "7net-mf-ompa"
    else:
        raise typer.Exit("No supported MLIP model found in the environment (MACE or SevenNet).")

@app.command()
def run(
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
    steps: int = typer.Option(100, prompt=True, help="Number of MD steps"),
    temperature: float = typer.Option(300.0, prompt=True, help="Temperature in K"),
    log: str = typer.Option("md.log", prompt=True, help="Output log file")
):
    """Run MD simulation using available MLIP model."""
    atoms = read(structure)
    model = detect_model()

    if model == "mace":
        typer.echo("Using MACE calculator.")
        atoms.calc = mace_mp(model="medium", device="cpu")
    elif model == "7net-mf-ompa":
        typer.echo("Using SevenNet calculator.")
        atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")

    typer.echo(f"Running MD for {steps} steps at {temperature}K...")
    run_md(atoms, log_path=log, temperature_K=temperature, steps=steps)
    typer.echo(f"MD simulation complete. Log written to: {log}")
