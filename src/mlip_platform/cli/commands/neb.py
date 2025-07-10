from pathlib import Path
import typer
from ase.io import read
from mlip_platform.core.neb import CustomNEB
import matplotlib.pyplot as plt
import pandas as pd

app = typer.Typer()

def detect_mlip() -> str:
    """Detect installed MLIP model automatically."""
    try:
        import sevenn
        return "7net-mf-ompa"
    except ImportError:
        try:
            import mace
            return "mace"
        except ImportError:
            raise typer.Exit("❌ No supported MLIP found. Please install SevenNet or MACE.")

@app.command()
def neb(
    initial: Path = typer.Option(..., prompt=True, help="Initial structure file (.vasp)"),
    final: Path = typer.Option(..., prompt=True, help="Final structure file (.vasp)")
):
    """Run NEB interpolation and relaxation using an automatically detected MLIP model."""

    atoms_initial = read(initial, format="vasp")
    atoms_final = read(final, format="vasp")

    if len(atoms_initial) != len(atoms_final):
        typer.echo("❌ Error: Initial and final structures must have the same number of atoms.")
        raise typer.Exit(code=1)

    mlip = detect_mlip()
    typer.echo(f"Detected MLIP: {mlip}")
    output_dir = Path("neb_result") / mlip
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    neb = CustomNEB(
        initial=atoms_initial,
        final=atoms_final,
        num_images=5,
        interp_fmax=0.1,
        interp_steps=100,
        fmax=0.05,
        mlip=mlip,
        output_dir="neb_result"
    )

    typer.echo("Interpolating with IDPP...")
    neb.interpolate_idpp()

    typer.echo("Running NEB optimization...")
    neb.run_neb()

    typer.echo("Writing images and results...")
    neb.write_images(subdir="images")
    df = neb.process_results()
    neb.plot_results(df)

    typer.echo("✅ NEB complete. Output written to:")
    for file in ["A2B.traj", "idpp.traj", "idpp.log", "neb_data.csv", "neb_energy.png"]:
        typer.echo(f" - {output_dir / file}")
    for f in sorted(images_dir.glob("*.vasp")):
        typer.echo(f" - {f}")

if __name__ == "__main__":
    app()
