import typer
from ase.io import read
from mlip_platform.core.neb import CustomNEB

app = typer.Typer(help="Run NEB calculations")

VALID_MODELS = {
    "mace": "mace",
    "sevennet": "sevenn-mf-ompa",
}

@app.command("run")
def neb_command(
    initial: str = typer.Option(..., prompt="Initial structure path"),
    final: str = typer.Option(..., prompt="Final structure path"),
    model: str = typer.Option(..., prompt="Choose model [MACE/SevenNet]"),
    images: int = typer.Option(9),
    fmax: float = typer.Option(0.05),
    interp_fmax: float = typer.Option(0.1),
    interp_steps: int = typer.Option(1000),
    climb: bool = typer.Option(False),
):
    model_key = model.strip().lower()
    if model_key not in VALID_MODELS:
        raise typer.Exit(code=1)

    ini = read(initial)
    fin = read(final)

    runner = CustomNEB(
        initial=ini,
        final=fin,
        num_images=images,
        interp_fmax=interp_fmax,
        interp_steps=interp_steps,
        fmax=fmax,
        mlip=VALID_MODELS[model_key],
    )

    runner.run_neb(climb=climb)
