import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.neb import CustomNEB

# Optional calculator imports (just to check environment)
try:
    from mace.calculators import mace_mp
except ImportError:
    mace_mp = None

try:
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    SevenNetCalculator = None

app = typer.Typer(help="Run NEB calculations between two structures.")

def detect_model():
    if mace_mp:
        return "mace"
    elif SevenNetCalculator:
        return "7net-mf-ompa"
    else:
        raise typer.Exit("No supported MLIP model found in the environment (MACE or SevenNet).")

@app.command()
def neb(
    initial: Path = typer.Option(..., prompt=True, help="Initial structure file (.vasp)"),
    final: Path = typer.Option(..., prompt=True, help="Final structure file (.vasp)")
):
    """Run NEB interpolation and relaxation using available MLIP."""
    mlip = detect_model()
    typer.echo(f"Using model: {mlip}")

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

    typer.echo("Running NEB interpolation and optimization...")
    neb.interpolate_idpp()
    neb.run_neb()
    typer.echo("✅ NEB calculation complete.")
