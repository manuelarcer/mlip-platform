import typer
from ase.io import read
from mlip_platform.core.md import run_md

app = typer.Typer(help="Run Molecular Dynamics simulations")

VALID_MODELS = {
    "mace": "mace",
    "sevennet": "sevenn-mf-ompa",
}

@app.command("run")
def md_command(
    initial: str = typer.Option(..., prompt="Structure file path"),
    model: str = typer.Option(..., prompt="Choose model [MACE/SevenNet]"),
    temperature: float = typer.Option(300.0, prompt=True),
    timestep: float = typer.Option(1.0, prompt=True),
    steps: int = typer.Option(1000, prompt=True),
):
    model_key = model.strip().lower()
    if model_key not in VALID_MODELS:
        typer.echo("Invalid model. Choose 'MACE' or 'SevenNet'.")
        raise typer.Exit(1)

    atoms = read(initial)
    run_md(
        atoms=atoms,
        model=VALID_MODELS[model_key],
        temperature=temperature,
        timestep=timestep,
        steps=steps,
    )
