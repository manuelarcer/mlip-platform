import typer
from ase.io import read
from mlip_platform.core.mlip_bench import run_benchmark

app = typer.Typer(help="Run MLIP benchmark calculations")

VALID_MODELS = {
    "mace": "mace",
    "sevennet": "sevenn-mf-ompa",
}

@app.command("run")
def benchmark_command(
    structure: str = typer.Option(..., prompt="Structure file path"),
    model: str = typer.Option(..., prompt="Choose model [MACE/SevenNet]"),
):
    model_key = model.strip().lower()
    if model_key not in VALID_MODELS:
        typer.echo("Invalid model. Choose 'MACE' or 'SevenNet'.")
        raise typer.Exit(1)

    atoms = read(structure)
    run_benchmark(
        atoms=atoms,
        model=VALID_MODELS[model_key]
    )
