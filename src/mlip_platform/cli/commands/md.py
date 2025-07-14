import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.md import run_md

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
        raise typer.Exit("❌ No supported MLIP model found (MACE or SevenNet).")

@app.command()
def run(
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
    steps: int = typer.Option(100, prompt=True, help="Number of MD steps"),
    temperature: float = typer.Option(298, prompt=True, help="Temperature in K"),
    timestep: float = typer.Option(2.0, prompt=True, help="Timestep in fs")
):
    """Run MD simulation using an available MLIP model."""
    atoms = read(structure)
    model = detect_model()

    typer.echo(f"Detected MLIP: {model}")
    typer.echo(f"Running MD for {steps} steps at {temperature} K, timestep {timestep} fs...")

    if model == "mace":
        typer.echo(" Using MACE calculator.")
        atoms.calc = mace_mp(model="medium", device="cpu")
    elif model == "7net-mf-ompa":
        typer.echo("Using SevenNet calculator.")
        atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")

    typer.echo("Starting MD simulation...")
    run_md(
        atoms,
        log_path=None,
        temperature_K=temperature,
        timestep_fs=timestep,
        steps=steps,
        interval=1,
        output_dir="md_result",
        model_name=model
    )

    output_dir = Path("md_result") / model
    typer.echo("💾 Processing results...")
    typer.echo("✅ MD complete. Output written to:")
    for fname in ["md.traj", "md_energy.csv", "md_energy.png", "md_temperature.png"]:
        typer.echo(f" - {output_dir / fname}")

if __name__ == "__main__":
    app()
