import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.neb import CustomNEB

app = typer.Typer()

@app.command("run")
def run(
    initial: Path = typer.Option(..., help="Initial structure (.vasp)"),
    final: Path = typer.Option(..., help="Final structure (.vasp)"),
    mlip: str = typer.Option("mace", help="MLIP model to use: 'mace' or '7net-mf-ompa'")
):
    """Run NEB interpolation and relaxation with specified MLIP."""
    atoms_initial = read(initial, format="vasp")
    atoms_final = read(final, format="vasp")

    neb = CustomNEB(
        initial=atoms_initial,
        final=atoms_final,
        num_images=5,
        interp_fmax=0.1,
        interp_steps=100,
        fmax=0.05,
        mlip=mlip
    )

    typer.echo(f"Running NEB with {mlip}")
    neb.interpolate_idpp()
    neb.run_neb()
    typer.echo("NEB complete.")
