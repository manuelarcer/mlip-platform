from pathlib import Path
import typer
from ase.io import read
from ase.optimize import FIRE, MDMin, BFGS, LBFGS
from mlip_platform.core.neb import CustomNEB
from mlip_platform.cli.utils import detect_mlip, validate_mlip

app = typer.Typer()


def create_backup_folder(output_dir):
    """
    Create timestamped backup folder and move all NEB output files.

    Parameters
    ----------
    output_dir : Path
        Current output directory

    Returns
    -------
    Path
        Path to created backup folder
    list
        List of moved files/folders
    """
    from datetime import datetime
    import shutil
    import glob

    # Create timestamp: YYYY.MM.DD_HH.mm.ss
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    backup_dir = output_dir / f"bkup_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    # List of files/folders to backup
    files_to_backup = [
        'A2B.traj',
        'A2B_full.traj',
        'neb.log',
        'neb_convergence.csv',
        'neb_convergence.png',
        'neb_energy.png',
        'neb_data.csv',
        'idpp.traj',
        'idpp.log',
        'endpoint_optimization.txt',
        'initial_opt.traj',
        'initial_opt.log',
        'final_opt.traj',
        'final_opt.log',
    ]

    moved_files = []

    # Move individual files
    for filename in files_to_backup:
        filepath = output_dir / filename
        if filepath.exists():
            shutil.move(str(filepath), str(backup_dir / filename))
            moved_files.append(filename)

    # Move POSCAR folders (00/, 01/, 02/, ...)
    poscar_dirs = glob.glob(str(output_dir / '[0-9][0-9]'))
    for poscar_dir in poscar_dirs:
        poscar_dir = Path(poscar_dir)
        if poscar_dir.is_dir():
            shutil.move(str(poscar_dir), str(backup_dir / poscar_dir.name))
            moved_files.append(poscar_dir.name + '/')

    # DO NOT move neb_parameters.txt - it's needed for restart

    return backup_dir, moved_files


@app.command()
def neb(
    restart: bool = typer.Option(False, "--restart", help="Restart from previous NEB calculation"),
    initial: Path = typer.Option(None, help="Initial structure file (.vasp)"),
    final: Path = typer.Option(None, help="Final structure file (.vasp)"),
    num_images: int = typer.Option(None, help="Number of intermediate images (excluding initial and final)"),
    interp_fmax: float = typer.Option(None, help="IDPP interpolation fmax"),
    interp_steps: int = typer.Option(None, help="IDPP interpolation steps"),
    fmax: float = typer.Option(None, help="Final NEB force threshold"),
    mlip: str = typer.Option(None, help="MLIP model: 'uma-s-1p1', 'uma-m-1p1', 'mace', '7net-mf-ompa', or 'auto'"),
    uma_task: str = typer.Option(None, help="UMA task name: 'omat', 'oc20', 'omol', or 'odac' (only for UMA models)"),
    relax_atoms: str = typer.Option(None, help="Comma-separated list of atom indices to relax (e.g. '0,1,5'). If set, others are fixed."),
    log: str = typer.Option(None, help="Name for the NEB iteration log file (default: neb.log)"),
    k: float = typer.Option(None, help="Spring constant for NEB"),
    climb: bool = typer.Option(None, help="Enable climbing image NEB"),
    neb_optimizer: str = typer.Option(None, help="NEB optimizer: 'fire', 'mdmin', 'bfgs', or 'lbfgs'"),
    neb_max_steps: int = typer.Option(None, help="Maximum steps for NEB optimization"),
    optimize_endpoints: bool = typer.Option(None, help="Optimize initial and final structures before NEB"),
    endpoint_fmax: float = typer.Option(None, help="Force threshold for endpoint optimization (eV/Å)"),
    endpoint_optimizer: str = typer.Option(None, help="Optimizer for endpoints: 'bfgs', 'lbfgs', 'fire'"),
    endpoint_max_steps: int = typer.Option(None, help="Maximum steps for endpoint optimization")
):
    output_dir = Path.cwd()

    # ===== RESTART MODE =====
    if restart:
        typer.echo("🔄 RESTART MODE")

        # 1. Validate forbidden parameters
        forbidden_params = {
            'initial': initial,
            'final': final,
            'num_images': num_images,
            'relax_atoms': relax_atoms,
            'optimize_endpoints': optimize_endpoints,
        }

        provided_forbidden = [k for k, v in forbidden_params.items() if v is not None]
        if provided_forbidden:
            # Convert parameter names to CLI format (e.g., 'num_images' -> '--num-images')
            param_names = ', '.join(f'--{k.replace("_", "-")}' for k in provided_forbidden)
            typer.echo(f"❌ Error: Cannot specify {param_names} with --restart")
            typer.echo("   These parameters are loaded from neb_parameters.txt")
            raise typer.Exit(code=1)

        # 2. Load from restart files
        typer.echo(f"📁 Loading restart files from: {output_dir}")

        try:
            # Warn if MLIP is being changed
            if mlip is not None:
                typer.echo(f"   ⚠️  MLIP override detected: {mlip}")

            neb_instance, loaded_params = CustomNEB.load_from_restart(
                output_dir=output_dir,
                mlip=mlip,
                uma_task=uma_task,
                fmax=fmax,
                logfile=log,
                k=k,
                climb=climb,
                neb_optimizer=neb_optimizer,
                neb_max_steps=neb_max_steps
            )

        except (FileNotFoundError, ValueError, RuntimeError) as e:
            typer.echo(f"❌ {e}")
            raise typer.Exit(code=1)

        # 3. Show loaded parameters
        typer.echo("\n📋 Parameters loaded from neb_parameters.txt:")
        typer.echo(f"   - MLIP model:          {loaded_params['mlip']}")
        if loaded_params.get('uma_task'):
            typer.echo(f"   - UMA task:            {loaded_params['uma_task']}")
        typer.echo(f"   - Intermediate images: {loaded_params['num_images']}")
        typer.echo(f"   - Total images:        {loaded_params['total_images']}")
        if loaded_params.get('relax_atoms'):
            typer.echo(f"   - Relax atoms:         {loaded_params['relax_atoms']}")

        # 4. Show override parameters (if any)
        overrides = []
        if mlip is not None:
            overrides.append(f"MLIP: {mlip}")
        if fmax is not None:
            overrides.append(f"fmax: {fmax}")
        if k is not None:
            overrides.append(f"k: {k}")
        if climb is not None:
            overrides.append(f"climb: {climb}")
        if neb_optimizer is not None:
            overrides.append(f"optimizer: {neb_optimizer}")
        if neb_max_steps is not None:
            overrides.append(f"max_steps: {neb_max_steps}")

        if overrides:
            typer.echo(f"\n🔧 Parameter overrides: {', '.join(overrides)}")

        # 5. Validate image count consistency
        from ase.io import read as ase_read
        import time

        full_traj_path = output_dir / 'A2B_full.traj'
        all_frames = ase_read(str(full_traj_path), index=':')
        expected_total_images = loaded_params['total_images']

        # Check if the number of frames is divisible by expected total_images
        if len(all_frames) % expected_total_images != 0:
            typer.echo(f"\n⚠️  WARNING: Image count mismatch!")
            typer.echo(f"   Expected multiples of {expected_total_images} images, but found {len(all_frames)} frames in trajectory.")
            typer.echo(f"   This suggests the previous run may have been interrupted mid-step.")
            typer.echo(f"\n   Press Ctrl+C within 15 seconds to cancel, or wait to continue...")

            # 15-second countdown
            for i in range(15, 0, -1):
                typer.echo(f"   Continuing in: {i}...", nl=False)
                time.sleep(1)
                typer.echo("\r" + " " * 50 + "\r", nl=False)  # Clear the line

            typer.echo("   Proceeding with restart...\n")

        # 6. Create backup folder
        typer.echo("💾 Creating backup of previous results...")
        backup_dir, moved_files = create_backup_folder(output_dir)
        typer.echo(f"   ✓ Backup created: {backup_dir.name}")
        typer.echo(f"   ✓ Moved {len(moved_files)} files/folders")

        # 7. Set parameters for continuation (use overrides or loaded values)
        mlip = mlip or loaded_params['mlip']
        uma_task = uma_task or loaded_params.get('uma_task', 'omat')
        fmax = fmax if fmax is not None else loaded_params['fmax']
        k = k if k is not None else loaded_params.get('k', 0.1)
        climb = climb if climb is not None else loaded_params.get('climb', True)
        neb_optimizer = neb_optimizer or loaded_params.get('neb_optimizer', 'fire')
        neb_max_steps = neb_max_steps if neb_max_steps is not None else loaded_params.get('neb_max_steps', 600)
        log = log or loaded_params.get('log', 'neb.log')
        num_images = loaded_params['num_images']
        total_images = loaded_params['total_images']

        # Skip endpoint optimization on restart
        optimize_endpoints = False

        # Assign loaded NEB instance
        neb = neb_instance

        # Skip to NEB optimization (after interpolation)
        skip_to_optimization = True

    # ===== NORMAL MODE =====
    else:
        skip_to_optimization = False

        # Validate required parameters
        if initial is None or final is None:
            typer.echo("❌ Error: --initial and --final are required for new NEB calculation")
            typer.echo("   Use --restart to continue from previous calculation")
            raise typer.Exit(code=1)

        # Set defaults for normal mode
        num_images = num_images or 5
        interp_fmax = interp_fmax or 0.1
        interp_steps = interp_steps or 100
        fmax = fmax or 0.05
        mlip = mlip or "auto"
        uma_task = uma_task or "omat"
        log = log or "neb.log"
        k = k or 0.1
        climb = climb if climb is not None else True
        neb_optimizer = neb_optimizer or "fire"
        neb_max_steps = neb_max_steps or 600
        optimize_endpoints = optimize_endpoints if optimize_endpoints is not None else True
        endpoint_fmax = endpoint_fmax or 0.01
        endpoint_optimizer = endpoint_optimizer or "bfgs"
        endpoint_max_steps = endpoint_max_steps or 200

        # Continue with existing logic
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

    # Handle relax_atoms only in normal mode (in restart mode it's loaded from parameters)
    relax_indices = None
    if not skip_to_optimization and relax_atoms:
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
    typer.echo(f" - NEB max steps:       {neb_max_steps}")
    typer.echo(f" - optimize_endpoints:  {optimize_endpoints}")
    if optimize_endpoints:
        typer.echo(f" - endpoint_fmax:       {endpoint_fmax}")
        typer.echo(f" - endpoint_optimizer:  {endpoint_optimizer}")
    typer.echo(f" - output_dir:          {output_dir}")

    # Write parameter file only in normal mode (not restart mode where it already exists)
    if not skip_to_optimization:
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
            f.write(f"NEB max steps:         {neb_max_steps}\n")
            f.write(f"Optimize endpoints:    {optimize_endpoints}\n")
            if optimize_endpoints:
                f.write(f"Endpoint fmax:         {endpoint_fmax}\n")
                f.write(f"Endpoint optimizer:    {endpoint_optimizer}\n")
                f.write(f"Endpoint max steps:    {endpoint_max_steps}\n")
            f.write(f"Log file:              {log}\n")
            f.write(f"Output dir:            {output_dir}\n")
            if relax_indices:
                f.write(f"Relax atoms:           {relax_indices}\n")

    # Create NEB instance (skip in restart mode where it's already loaded)
    if not skip_to_optimization:
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

    # Skip interpolation and endpoint optimization in restart mode
    if not skip_to_optimization:
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
    else:
        typer.echo(" ⏭️  Skipping interpolation (loaded from restart)")

    # Select NEB optimizer
    optimizer_map = {'fire': FIRE, 'mdmin': MDMin, 'bfgs': BFGS, 'lbfgs': LBFGS}
    neb_opt = optimizer_map.get(neb_optimizer.lower(), FIRE)

    typer.echo(f" Running NEB optimization (optimizer={neb_optimizer.upper()}, climb={climb}, max_steps={neb_max_steps})...")
    neb.run_neb(optimizer=neb_opt, climb=climb, max_steps=neb_max_steps)

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
