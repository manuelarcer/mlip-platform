import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.optimize import run_optimization, OPTIMIZER_MAP

try:
    from mace.calculators import mace_mp
except ImportError:
    mace_mp = None

try:
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    SevenNetCalculator = None

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    fairchem_available = True
except ImportError:
    fairchem_available = False

app = typer.Typer(help="Run geometry optimization on structures.")


def detect_mlip():
    """Detect available MLIP model in order of preference: UMA > SevenNet > MACE"""
    if fairchem_available:
        return "uma-s-1p1"
    elif SevenNetCalculator:
        return "7net-mf-ompa"
    elif mace_mp:
        return "mace"
    else:
        raise typer.Exit("❌ No supported MLIP model found (UMA, SevenNet, or MACE).")


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
        # Validate user-provided MLIP
        if mlip == "mace" and not mace_mp:
            raise typer.Exit("❌ MACE not available. Install with: pip install mace-torch")
        elif mlip == "7net-mf-ompa" and not SevenNetCalculator:
            raise typer.Exit("❌ SevenNet not available. Install with: pip install sevenn")
        elif mlip.startswith("uma-") and not fairchem_available:
            raise typer.Exit("❌ UMA not available. Install with: pip install fairchem-core")
        elif not (mlip in ["mace", "7net-mf-ompa"] or mlip.startswith("uma-")):
            raise typer.Exit(f"❌ Unknown MLIP: {mlip}. Use 'uma-s-1p1', 'mace', or '7net-mf-ompa'.")
        typer.echo(f"🧠 Using MLIP: {mlip}")

    # Validate optimizer
    if optimizer.lower() not in OPTIMIZER_MAP:
        typer.echo(f"❌ Unknown optimizer: {optimizer}")
        typer.echo(f"   Available: {', '.join(OPTIMIZER_MAP.keys())}")
        raise typer.Exit(1)

    # Assign calculator
    if mlip == "mace":
        typer.echo("⚙️  Attaching MACE calculator...")
        atoms.calc = mace_mp(model="medium", device="cpu")
    elif mlip == "7net-mf-ompa":
        typer.echo("⚙️  Attaching SevenNet calculator...")
        atoms.calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")
    elif mlip.startswith("uma-"):
        typer.echo(f"⚙️  Attaching UMA calculator ({mlip}, task={uma_task})...")
        predictor = pretrained_mlip.get_predict_unit(mlip, device="cpu")
        atoms.calc = FAIRChemCalculator(predictor, task_name=uma_task)

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
