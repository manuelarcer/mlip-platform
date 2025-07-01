"""
Run NEB (nudged elastic band) interpolation and relaxation using MLIP models.

Examples:
  Run NEB with MACE:
    mlip-platform neb run --initial a.vasp --final b.vasp --mlip mace

  Run NEB with SevenNet:
    mlip-platform neb run --initial a.vasp --final b.vasp --mlip 7net-mf-ompa

This command performs both IDPP interpolation and full NEB relaxation.
"""

import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.neb import CustomNEB

app = typer.Typer()

@app.command("run")
def run(
    initial: Path = typer.Option(..., help="Initial structure file (.vasp)"),
    final: Path = typer.Option(..., help="Final structure file (.vasp)"),
    mlip: str = typer.Option("mace", help="MLIP model to use: 'mace' or '7net-mf-ompa'")
):
    """Run NEB interpolation and relaxation with the specified MLIP model."""
    atoms_initial = read(initial, format="vasp")
    atoms_final = read(final, format="vasp")

    typer.echo(f"Running NEB with {mlip}")
    neb = CustomNEB(
        initial=atoms_initial,
        final=atoms_final,
        num_images=5,
        interp_fmax=0.1,
        interp_steps=100,
        fmax=0.05,
        mlip=mlip
    )

    neb.interpolate_idpp()
    neb.run_neb()
    typer.echo("NEB complete.")
