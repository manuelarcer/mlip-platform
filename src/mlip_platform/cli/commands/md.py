import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.md import run_md
from mlip_platform.cli.utils import detect_mlip, validate_mlip, setup_calculator

app = typer.Typer(help="Run molecular dynamics simulations.")

@app.command()
def run(
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
    steps: int = typer.Option(100, prompt=True, help="Number of MD steps"),
    temperature: float = typer.Option(298, prompt=True, help="Temperature in K"),
    timestep: float = typer.Option(2.0, prompt=True, help="Timestep in fs"),
    mlip: str = typer.Option("auto", help="MLIP model: 'uma-s-1p1', 'uma-m-1p1', 'mace', '7net-mf-ompa', or 'auto'"),
    uma_task: str = typer.Option("omat", help="UMA task name: 'omat', 'oc20', 'omol', or 'odac' (only for UMA models)")
):
    """Run MD simulation using a supported MLIP model."""
    atoms = read(structure)

    # Detect or use specified model
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
    else:
        validate_mlip(mlip)
        typer.echo(f"🧠 Using MLIP: {mlip}")

    if mlip.startswith("uma-"):
        typer.echo(f"   UMA task: {uma_task}")

    typer.echo(f"Running MD for {steps} steps at {temperature} K, timestep {timestep} fs")

    # Assign calculator
    typer.echo(f"⚙️  Attaching {mlip} calculator...")
    atoms = setup_calculator(atoms, mlip, uma_task)

    output_dir = structure.parent

    run_md(
        atoms=atoms,
        temperature_K=temperature,
        timestep_fs=timestep,
        steps=steps,
        interval=1,
        output_dir=output_dir,
        model_name=mlip
    )

    # Save parameters
    param_file = output_dir / "md_params.txt"
    with open(param_file, "w") as f:
        f.write("MD Run Parameters\n")
        f.write("===================\n")
        f.write(f"MLIP model:        {mlip}\n")
        if mlip.startswith("uma-"):
            f.write(f"UMA task:          {uma_task}\n")
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
