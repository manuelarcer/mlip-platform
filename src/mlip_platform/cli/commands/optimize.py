import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.optimize import run_optimization, OPTIMIZER_MAP
from mlip_platform.cli.utils import detect_mlip, validate_mlip, setup_calculator

app = typer.Typer(help="Run geometry optimization on structures.")


@app.command()
def run(
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
    mlip: str = typer.Option("auto", help="MLIP model: 'uma-s-1p1', 'uma-m-1p1', 'mace', '7net-mf-ompa', or 'auto'"),
    uma_task: str = typer.Option("omat", help="UMA task name: 'omat', 'oc20', 'omol', or 'odac' (only for UMA models)"),
    optimizer: str = typer.Option("fire", help=f"Optimizer algorithm: {', '.join(OPTIMIZER_MAP.keys())}"),
    fmax: float = typer.Option(0.05, help="Force convergence threshold (eV/Å)"),
    max_steps: int = typer.Option(200, help="Maximum optimization steps"),
    trajectory: str = typer.Option("opt.traj", help="Trajectory filename"),
    logfile: str = typer.Option("opt.log", help="Log filename"),
    verbose: bool = typer.Option(True, help="Show optimization progress table (forces, energies)"),
):
    """
    Run geometry optimization using a supported MLIP model.

    Optimizes atomic positions to minimize forces until fmax convergence is reached.
    """
    # Read structure
    atoms = read(structure)
    typer.echo(f"📂 Loaded structure: {structure.name}")
    typer.echo(f"   Atoms: {len(atoms)}, Formula: {atoms.get_chemical_formula()}")

    # Detect or validate MLIP
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
    else:
        validate_mlip(mlip)
        typer.echo(f"🧠 Using MLIP: {mlip}")

    # Validate optimizer
    if optimizer.lower() not in OPTIMIZER_MAP:
        typer.echo(f"❌ Unknown optimizer: {optimizer}")
        typer.echo(f"   Available: {', '.join(OPTIMIZER_MAP.keys())}")
        raise typer.Exit(1)

    # Assign calculator
    typer.echo(f"⚙️  Attaching {mlip} calculator...")
    if mlip.startswith("uma-"):
        typer.echo(f"   UMA task: {uma_task}")
    atoms = setup_calculator(atoms, mlip, uma_task)

    # Output directory
    output_dir = structure.parent

    # Run optimization
    typer.echo(f"\n🔧 Optimizer: {optimizer.upper()}")
    typer.echo(f"   fmax = {fmax} eV/Å")
    typer.echo(f"   max_steps = {max_steps}")
    typer.echo(f"   Output dir: {output_dir.resolve()}\n")

    converged = run_optimization(
        atoms=atoms,
        optimizer=optimizer,
        fmax=fmax,
        max_steps=max_steps,
        trajectory=trajectory,
        logfile=logfile,
        output_dir=output_dir,
        model_name=mlip,
        verbose=verbose
    )

    # Save parameters
    param_file = output_dir / "opt_params.txt"
    with open(param_file, "w") as f:
        f.write("Geometry Optimization Parameters\n")
        f.write("=================================\n")
        f.write(f"MLIP model:        {mlip}\n")
        f.write(f"Structure:         {structure.name}\n")
        f.write(f"Optimizer:         {optimizer.upper()}\n")
        f.write(f"fmax (eV/Å):       {fmax}\n")
        f.write(f"Max steps:         {max_steps}\n")
        f.write(f"Converged:         {converged}\n")
        f.write(f"Output dir:        {output_dir.resolve()}\n")

    # Print output summary
    typer.echo("\n✅ Optimization complete. Output files:")
    output_files = [
        trajectory,
        logfile,
        "opt_convergence.csv",
        "opt_convergence.png",
        "opt_final.vasp",
        "opt_params.txt"
    ]
    for file in output_files:
        typer.echo(f"   📄 {(output_dir / file).resolve()}")

    if not converged:
        typer.echo("\n⚠️  Warning: Optimization did not converge. Consider:")
        typer.echo("   - Increasing max_steps")
        typer.echo("   - Relaxing fmax threshold")
        typer.echo("   - Trying a different optimizer")


if __name__ == "__main__":
    app()
