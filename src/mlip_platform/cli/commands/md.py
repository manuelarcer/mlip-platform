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
    """Run MD simulation using a supported MLIP model."""
    atoms = read(structure)
    model = detect_model()

    typer.echo(f"🧠 Detected MLIP: {model}")
    typer.echo(f"Running MD for {steps} steps at {temperature} K, timestep {timestep} fs")

    # Assign calculator
    if model == "mace":
        typer.echo("Using MACE calculator.")
        atoms.calc = mace_mp(model="medium", device="cpu")
    elif model == "7net-mf-ompa":
        typer.echo("Using SevenNet calculator.")
        atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")

    output_dir = structure.parent

    run_md(
        atoms=atoms,
        temperature_K=temperature,
        timestep_fs=timestep,
        steps=steps,
        interval=1,
        output_dir=output_dir,
        model_name=model
    )

    # Save parameters
    param_file = output_dir / "md_params.txt"
    with open(param_file, "w") as f:
        f.write("MD Run Parameters\n")
        f.write("===================\n")
        f.write(f"MLIP model:        {model}\n")
        f.write(f"Structure:         {structure.name}\n")
        f.write(f"Number of steps:   {steps}\n")
        f.write(f"Temperature (K):   {temperature}\n")
        f.write(f"Timestep (fs):     {timestep}\n")
        f.write(f"Output dir:        {output_dir.resolve()}\n")

    typer.echo("✅ MD complete. Output written to:")
    for file in ["md.traj", "md_energy.csv", "md_energy.png", "md_temperature.png", "md_params.txt"]:
        typer.echo(f" - {(output_dir / file).resolve()}")

if __name__ == "__main__":
    app()
