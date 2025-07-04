import typer
from ase.io import read
from mlip_platform.core.neb import CustomNEB

app = typer.Typer(
    name="neb",
    invoke_without_command=True,
    no_args_is_help=False,
    help="Run a NEB calculation using the available MLIP (auto-detected)."
)

@app.callback()
def run(
    initial_path: str = typer.Option(
        ..., prompt="Enter the path to the initial structure file (e.g., initial.vasp)"
    ),
    final_path: str = typer.Option(
        ..., prompt="Enter the path to the final structure file (e.g., final.vasp)"
    ),
    num_images: int = typer.Option(
        9, prompt="Enter the number of images"
    ),
    interp_fmax: float = typer.Option(
        0.1, prompt="Enter interpolation fmax"
    ),
    interp_steps: int = typer.Option(
        1000, prompt="Enter interpolation steps"
    ),
    fmax: float = typer.Option(
        0.05, prompt="Enter the NEB convergence fmax"
    ),
    optimizer: str = typer.Option(
        "BFGS", prompt="Choose optimizer [BFGS/MDMin/FIRE]"
    ),
    model: str = typer.Option(
        "", prompt="Enter MLIP model name (leave blank to use default)"
    ),
):
    """
    Run NEB interpolation and relaxation using the provided inputs.
    """
    initial = read(initial_path)
    final = read(final_path)
    mlip_model = model or None

    job = CustomNEB(
        initial=initial,
        final=final,
        num_images=num_images,
        interp_fmax=interp_fmax,
        interp_steps=interp_steps,
        fmax=fmax,
        model=mlip_model,
    )
    job.run(optimizer=optimizer)
    typer.secho("✅ NEB calculation completed!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
