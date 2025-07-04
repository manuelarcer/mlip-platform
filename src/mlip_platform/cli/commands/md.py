import typer
from ase.io import read
from mlip_platform.core.md import run_md

app = typer.Typer(
    name="md",
    invoke_without_command=True,
    no_args_is_help=False,
    help="Run a molecular dynamics simulation using the available MLIP (auto-detected)."
)

@app.callback()
def run(
    structure: str = typer.Option(
        ..., prompt="Enter the path to your structure file (e.g., POSCAR, .traj)"
    ),
    temperature: float = typer.Option(
        300.0, prompt="Enter temperature in Kelvin"
    ),
    timestep: float = typer.Option(
        2.0, prompt="Enter timestep in femtoseconds"
    ),
    steps: int = typer.Option(
        1000, prompt="Enter number of MD steps"
    ),
    log: str = typer.Option(
    "", prompt="Enter log file path (leave blank for no file logging)"
)

):
    """
    Executes MD simulation with given parameters.
    MLIP model is automatically selected based on environment (MACE or SevenNet).
    """
    atoms = read(structure)

    run_md(
        atoms,
        log_path=log,
        temperature_K=temperature,
        timestep_fs=timestep,
        steps=steps
    )

    typer.secho("✅ MD simulation completed!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
