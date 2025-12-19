from pathlib import Path
import typer
from ase.io import read
from ase.optimize import FIRE, MDMin
from mlip_platform.core.neb import CustomNEB
from mlip_platform.cli.utils import detect_mlip, validate_mlip

app = typer.Typer()

@app.command()
def neb(
    initial: Path = typer.Option(..., prompt=True, help="Initial structure file (.vasp)"),
    final: Path = typer.Option(..., prompt=True, help="Final structure file (.vasp)"),
    num_images: int = typer.Option(5, help="Number of intermediate images (excluding initial and final)"),
    interp_fmax: float = typer.Option(0.1, help="IDPP interpolation fmax"),
    interp_steps: int = typer.Option(100, help="IDPP interpolation steps"),
    fmax: float = typer.Option(0.05, help="Final NEB force threshold"),
    mlip: str = typer.Option("auto", help="MLIP model: 'uma-s-1p1', 'uma-m-1p1', 'mace', '7net-mf-ompa', or 'auto'"),
    uma_task: str = typer.Option("omat", help="UMA task name: 'omat', 'oc20', 'omol', or 'odac' (only for UMA models)"),
    relax_atoms: str = typer.Option(None, help="Comma-separated list of atom indices to relax (e.g. '0,1,5'). If set, others are fixed."),
    log: str = typer.Option("neb.log", help="Name for the NEB iteration log file (default: neb.log)"),
    k: float = typer.Option(0.1, help="Spring constant for NEB"),
    climb: bool = typer.Option(True, help="Enable climbing image NEB"),
    neb_optimizer: str = typer.Option("fire", help="NEB optimizer: 'fire' or 'mdmin'"),
    optimize_endpoints: bool = typer.Option(True, help="Optimize initial and final structures before NEB"),
    endpoint_fmax: float = typer.Option(0.01, help="Force threshold for endpoint optimization (eV/Å)"),
    endpoint_optimizer: str = typer.Option("bfgs", help="Optimizer for endpoints: 'bfgs', 'lbfgs', 'fire'"),
    endpoint_max_steps: int = typer.Option(200, help="Maximum steps for endpoint optimization")
):
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
            # Validate indices are within range
            num_atoms = len(atoms_initial)
            invalid_indices = [i for i in relax_indices if i < 0 or i >= num_atoms]
            if invalid_indices:
                typer.echo(f"❌ Error: Invalid atom indices {invalid_indices}. Must be between 0 and {num_atoms-1}.")
                raise typer.Exit(code=1)
            typer.echo(f"🔒 HIGHLY CONSTRAINED MODE: Relaxing only atoms: {relax_indices}")
        except ValueError:
            typer.echo("❌ Error: --relax-atoms must be a comma-separated list of integers.")
            raise typer.Exit(code=1)


    total_images = num_images + 2  # intermediate + initial + final
    typer.echo(f"⚙️ Running NEB with:")
    typer.echo(f" - Intermediate images: {num_images}")
    typer.echo(f" - Total images:        {total_images} (including initial and final)")
    typer.echo(f" - interp_fmax:         {interp_fmax}")
    typer.echo(f" - interp_steps:        {interp_steps}")
    typer.echo(f" - final fmax:          {fmax}")
    typer.echo(f" - spring constant (k): {k}")
    typer.echo(f" - climb:               {climb}")
    typer.echo(f" - NEB optimizer:       {neb_optimizer}")
    typer.echo(f" - optimize_endpoints:  {optimize_endpoints}")
    if optimize_endpoints:
        typer.echo(f" - endpoint_fmax:       {endpoint_fmax}")
        typer.echo(f" - endpoint_optimizer:  {endpoint_optimizer}")
    typer.echo(f" - output_dir:          {output_dir}")

    with open(output_dir / "neb_parameters.txt", "w") as f:
        f.write("NEB Run Parameters\n")
        f.write("===================\n")
        f.write(f"MLIP model:            {mlip}\n")
        if mlip.startswith("uma-"):
            f.write(f"UMA task:              {uma_task}\n")
        f.write(f"Initial:               {initial}\n")
        f.write(f"Final:                 {final}\n")
        f.write(f"Intermediate images:   {num_images}\n")
        f.write(f"Total images:          {total_images}\n")
        f.write(f"IDPP fmax:             {interp_fmax}\n")
        f.write(f"IDPP steps:            {interp_steps}\n")
        f.write(f"Final fmax:            {fmax}\n")
        f.write(f"Spring constant (k):   {k}\n")
        f.write(f"Climb:                 {climb}\n")
        f.write(f"NEB optimizer:         {neb_optimizer}\n")
        f.write(f"Optimize endpoints:    {optimize_endpoints}\n")
        if optimize_endpoints:
            f.write(f"Endpoint fmax:         {endpoint_fmax}\n")
            f.write(f"Endpoint optimizer:    {endpoint_optimizer}\n")
            f.write(f"Endpoint max steps:    {endpoint_max_steps}\n")
        f.write(f"Log file:              {log}\n")
        f.write(f"Output dir:            {output_dir}\n")
        if relax_indices:
            f.write(f"Relax atoms:           {relax_indices}\n")


    neb = CustomNEB(
        initial=atoms_initial,
        final=atoms_final,
        num_images=num_images,
        interp_fmax=interp_fmax,
        interp_steps=interp_steps,
        fmax=fmax,
        mlip=mlip,
        uma_task=uma_task,
        output_dir=output_dir,
        relax_atoms=relax_indices,
        logfile=log
    )

    # Optimize endpoints if requested
    if optimize_endpoints:
        endpoint_results = neb.optimize_endpoints(
            endpoint_fmax=endpoint_fmax,
            optimizer=endpoint_optimizer,
            max_steps=endpoint_max_steps
        )
        # Re-setup NEB images with optimized endpoints
        neb.images = neb.setup_neb()

        # Save endpoint optimization results to file
        with open(output_dir / "endpoint_optimization.txt", "w") as f:
            f.write("Endpoint Optimization Results\n")
            f.write("==============================\n\n")
            f.write("Initial Structure:\n")
            f.write(f"  Energy before: {endpoint_results['initial']['energy_before']:.6f} eV\n")
            f.write(f"  Energy after:  {endpoint_results['initial']['energy_after']:.6f} eV\n")
            f.write(f"  Energy change: {endpoint_results['initial']['energy_change']:.6f} eV\n")
            f.write(f"  Steps:         {endpoint_results['initial']['steps']}\n")
            f.write(f"  Converged:     {endpoint_results['initial']['converged']}\n\n")
            f.write("Final Structure:\n")
            f.write(f"  Energy before: {endpoint_results['final']['energy_before']:.6f} eV\n")
            f.write(f"  Energy after:  {endpoint_results['final']['energy_after']:.6f} eV\n")
            f.write(f"  Energy change: {endpoint_results['final']['energy_change']:.6f} eV\n")
            f.write(f"  Steps:         {endpoint_results['final']['steps']}\n")
            f.write(f"  Converged:     {endpoint_results['final']['converged']}\n\n")
            f.write(f"Reaction energy: {endpoint_results['reaction_energy']:.6f} eV\n\n")

            # Similarity check
            sim = endpoint_results['similarity']
            f.write("Similarity Check:\n")
            f.write(f"  Average displacement: {sim['avg_displacement']:.3f} Å\n")
            f.write(f"  Max displacement:     {sim['max_displacement']:.3f} Å (atom {sim['max_disp_atom']})\n")
            f.write(f"  Min displacement:     {sim['min_displacement']:.3f} Å\n")
            f.write(f"  Energy difference:    {sim['energy_diff']:.6f} eV\n")
            f.write(f"  Structures similar:   {sim['is_similar']}\n")
            if sim['warning_reasons']:
                f.write(f"  Warning reasons:\n")
                for reason in sim['warning_reasons']:
                    f.write(f"    - {reason}\n")

    typer.echo(" Interpolating with IDPP...")
    neb.interpolate_idpp()

    # Select NEB optimizer
    optimizer_map = {'fire': FIRE, 'mdmin': MDMin}
    neb_opt = optimizer_map.get(neb_optimizer.lower(), FIRE)

    typer.echo(f" Running NEB optimization (optimizer={neb_optimizer.upper()}, climb={climb})...")
    neb.run_neb(optimizer=neb_opt, climb=climb)

    typer.echo(" Processing results...")
    df = neb.process_results()
    neb.plot_results(df)

    typer.echo("Exporting POSCARs...")
    neb.export_poscars()

    typer.echo("✅ NEB complete. Output written to:")
    for file in [log, "neb_convergence.csv", "neb_convergence.png", "A2B.traj", "A2B_full.traj", "idpp.traj", "idpp.log", "neb_data.csv", "neb_energy.png", "neb_parameters.txt"]:
        typer.echo(f" - {output_dir / file}")
    for i in range(total_images):
        typer.echo(f" - {output_dir / f'{i:02d}' / 'POSCAR'}")

if __name__ == "__main__":
    app()
