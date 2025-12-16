from pathlib import Path
import typer
from ase.io import read
from ase.optimize import FIRE
from mlip_platform.core.neb import CustomNEB
from mlip_platform.cli.utils import detect_mlip, validate_mlip

app = typer.Typer()

@app.command()
def autoneb(
    initial: Path = typer.Option(..., prompt=True, help="Initial structure file (.vasp)"),
    final: Path = typer.Option(..., prompt=True, help="Final structure file (.vasp)"),
    n_max: int = typer.Option(9, help="Maximum number of images (including endpoints)"),
    n_simul: int = typer.Option(4, help="Number of parallel relaxations"),
    fmax: float = typer.Option(0.05, help="Force convergence threshold (eV/Å)"),
    mlip: str = typer.Option("auto", help="MLIP model: 'uma-s-1p1', 'uma-m-1p1', 'mace', '7net-mf-ompa', or 'auto'"),
    uma_task: str = typer.Option("omat", help="UMA task name: 'omat', 'oc20', 'omol', or 'odac' (only for UMA models)"),
    climb: bool = typer.Option(True, help="Enable climbing image NEB"),
    k: float = typer.Option(0.1, help="Spring constant"),
    space_energy_ratio: float = typer.Option(0.5, help="Preference for geometric (1.0) vs energy (0.0) gaps"),
    interpolate_method: str = typer.Option("idpp", help="Interpolation method: 'linear' or 'idpp'"),
    maxsteps: int = typer.Option(10000, help="Maximum steps per relaxation"),
    prefix: str = typer.Option("autoneb", help="Prefix for output files"),
    relax_atoms: str = typer.Option(None, help="⚠️ WARNING: Comma-separated atom indices to relax. May not work well with AutoNEB!")
):
    """
    Run AutoNEB calculation with dynamic image insertion.

    AutoNEB automatically adds intermediate images until n_max is reached,
    making it ideal for complex reaction pathways where the optimal number
    of images is unclear.

    IMPORTANT NOTES:
    - AutoNEB uses file-based I/O (creates prefix000.traj, prefix001.traj, etc.)
    - Results stored in AutoNEB_iter/ folder
    - Custom convergence plots (CSV/PNG) are NOT generated
    - Highly-constrained mode (--relax-atoms) may not work properly
    - For simple transitions, consider using 'neb' command instead
    """
    atoms_initial = read(initial, format="vasp")
    atoms_final = read(final, format="vasp")

    if len(atoms_initial) != len(atoms_final):
        typer.echo("❌ Error: Initial and final structures must have the same number of atoms.")
        raise typer.Exit(code=1)

    # Detect or use specified model
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
    else:
        validate_mlip(mlip)
        typer.echo(f"🧠 Using MLIP: {mlip}")

    if mlip.startswith("uma-"):
        typer.echo(f"   UMA task: {uma_task}")

    output_dir = Path.cwd()

    relax_indices = None
    if relax_atoms:
        try:
            relax_indices = [int(i.strip()) for i in relax_atoms.split(",")]
            num_atoms = len(atoms_initial)
            invalid_indices = [i for i in relax_indices if i < 0 or i >= num_atoms]
            if invalid_indices:
                typer.echo(f"❌ Error: Invalid atom indices {invalid_indices}. Must be between 0 and {num_atoms-1}.")
                raise typer.Exit(code=1)
            typer.echo(f"⚠️  WARNING: Highly-constrained mode with AutoNEB may not work as expected!")
            typer.echo(f"   Relaxing only atoms: {relax_indices}")
        except ValueError:
            typer.echo("❌ Error: --relax-atoms must be a comma-separated list of integers.")
            raise typer.Exit(code=1)

    typer.echo(f"\n⚙️ Running AutoNEB with:")
    typer.echo(f"   n_max:              {n_max} (target images including endpoints)")
    typer.echo(f"   n_simul:            {n_simul} (parallel relaxations)")
    typer.echo(f"   fmax:               {fmax} eV/Å")
    typer.echo(f"   climb:              {climb}")
    typer.echo(f"   k:                  {k}")
    typer.echo(f"   space_energy_ratio: {space_energy_ratio}")
    typer.echo(f"   interpolate_method: {interpolate_method}")
    typer.echo(f"   maxsteps:           {maxsteps}")
    typer.echo(f"   prefix:             {prefix}")
    typer.echo(f"   output_dir:         {output_dir}\n")

    # Save parameters
    with open(output_dir / "autoneb_parameters.txt", "w") as f:
        f.write("AutoNEB Run Parameters\n")
        f.write("======================\n")
        f.write(f"MLIP model:            {mlip}\n")
        if mlip.startswith("uma-"):
            f.write(f"UMA task:              {uma_task}\n")
        f.write(f"Initial:               {initial}\n")
        f.write(f"Final:                 {final}\n")
        f.write(f"n_max:                 {n_max}\n")
        f.write(f"n_simul:               {n_simul}\n")
        f.write(f"fmax:                  {fmax}\n")
        f.write(f"climb:                 {climb}\n")
        f.write(f"k:                     {k}\n")
        f.write(f"space_energy_ratio:    {space_energy_ratio}\n")
        f.write(f"interpolate_method:    {interpolate_method}\n")
        f.write(f"maxsteps:              {maxsteps}\n")
        f.write(f"prefix:                {prefix}\n")
        f.write(f"output_dir:            {output_dir}\n")
        if relax_indices:
            f.write(f"relax_atoms:           {relax_indices}\n")

    # Create CustomNEB instance
    # Note: num_images is not used by AutoNEB, but required by __init__
    neb = CustomNEB(
        initial=atoms_initial,
        final=atoms_final,
        num_images=5,  # Dummy value, not used by AutoNEB
        fmax=fmax,
        mlip=mlip,
        uma_task=uma_task,
        output_dir=output_dir,
        relax_atoms=relax_indices
    )

    # Run AutoNEB
    neb.run_autoneb(
        n_simul=n_simul,
        n_max=n_max,
        k=k,
        climb=climb,
        optimizer=FIRE,
        space_energy_ratio=space_energy_ratio,
        interpolate_method=interpolate_method,
        maxsteps=maxsteps,
        prefix=prefix
    )

    typer.echo("\n📁 AutoNEB output files:")
    typer.echo(f"   {output_dir / f'{prefix}*.traj'} - Individual image trajectories")
    typer.echo(f"   {output_dir / 'AutoNEB_iter'} - Iteration history folder")
    typer.echo(f"   {output_dir / 'autoneb_parameters.txt'} - Run parameters")
    typer.echo("\nℹ️  Note: Custom convergence plots (CSV/PNG) are not generated in AutoNEB mode.")
    typer.echo("   Check AutoNEB_iter/ folder for detailed iteration history.")

if __name__ == "__main__":
    app()
