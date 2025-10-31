import typer
from pathlib import Path
from ase.io import read
from mlip_platform.core.md import run_md
from mlip_platform.cli.utils import detect_mlip, validate_mlip, setup_calculator

app = typer.Typer(help="Run molecular dynamics simulations.")

@app.command()
def run(
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp)"),
    ensemble: str = typer.Option("nvt", help="Ensemble: 'nve', 'nvt', 'npt'"),
    steps: int = typer.Option(1000, help="Number of MD steps"),
    temperature: float = typer.Option(300, help="Temperature in K (required for NVT, NPT)"),
    pressure: float = typer.Option(0.0, help="Pressure in GPa (required for NPT)"),
    timestep: float = typer.Option(1.0, help="Timestep in fs"),

    # Thermostat/Barostat selection
    thermostat: str = typer.Option("langevin", help="Thermostat for NVT: 'langevin', 'nose-hoover', 'berendsen'"),
    barostat: str = typer.Option("npt", help="Barostat for NPT: 'npt' (MTK), 'berendsen'"),

    # Advanced thermostat/barostat parameters
    friction: float = typer.Option(0.01, help="Langevin friction coefficient (1/fs)"),
    ttime: float = typer.Option(25.0, help="Nosé-Hoover/NPT time constant (fs)"),
    taut: float = typer.Option(100.0, help="Berendsen temperature coupling time (fs)"),
    taup: float = typer.Option(1000.0, help="Berendsen pressure coupling time (fs)"),

    # MLIP options
    mlip: str = typer.Option("auto", help="MLIP model: 'uma-s-1p1', 'uma-m-1p1', 'mace', '7net-mf-ompa', or 'auto'"),
    uma_task: str = typer.Option("omat", help="UMA task name: 'omat', 'oc20', 'omol', or 'odac' (only for UMA models)")
):
    """
    Run molecular dynamics simulation using a supported MLIP model.

    Supports NVE, NVT, and NPT ensembles with various thermostats and barostats.
    """
    atoms = read(structure)
    ensemble = ensemble.lower()

    # Validate ensemble
    if ensemble not in ['nve', 'nvt', 'npt']:
        raise typer.Exit(f"❌ Unknown ensemble: {ensemble}. Use 'nve', 'nvt', or 'npt'.")

    # Validate parameters for each ensemble
    if ensemble in ['nvt', 'npt'] and temperature <= 0:
        raise typer.Exit(f"❌ Temperature must be > 0 for {ensemble.upper()} ensemble.")

    if ensemble == 'npt' and pressure is None:
        raise typer.Exit("❌ Pressure must be specified for NPT ensemble.")

    # Detect or use specified model
    if mlip == "auto":
        mlip = detect_mlip()
        typer.echo(f"🧠 Auto-detected MLIP: {mlip}")
    else:
        validate_mlip(mlip)
        typer.echo(f"🧠 Using MLIP: {mlip}")

    if mlip.startswith("uma-"):
        typer.echo(f"   UMA task: {uma_task}")

    # Display ensemble information
    typer.echo(f"\n🔬 MD Simulation Setup:")
    typer.echo(f"   Ensemble:    {ensemble.upper()}")
    if ensemble == 'nvt':
        typer.echo(f"   Thermostat:  {thermostat}")
        if thermostat == 'langevin':
            typer.echo(f"   Friction:    {friction} fs⁻¹")
        elif thermostat == 'nose-hoover':
            typer.echo(f"   Time const:  {ttime} fs")
        elif thermostat == 'berendsen':
            typer.echo(f"   Tau T:       {taut} fs")
    elif ensemble == 'npt':
        typer.echo(f"   Barostat:    {barostat}")
        typer.echo(f"   Pressure:    {pressure} GPa")
        if barostat == 'npt':
            typer.echo(f"   Time const:  {ttime} fs")
        elif barostat == 'berendsen':
            typer.echo(f"   Tau T:       {taut} fs")
            typer.echo(f"   Tau P:       {taup} fs")

    if ensemble in ['nvt', 'npt']:
        typer.echo(f"   Temperature: {temperature} K")
    typer.echo(f"   Steps:       {steps}")
    typer.echo(f"   Timestep:    {timestep} fs")

    # Assign calculator
    typer.echo(f"\n⚙️  Attaching {mlip} calculator...")
    atoms = setup_calculator(atoms, mlip, uma_task)

    output_dir = structure.parent

    # Run MD
    run_md(
        atoms=atoms,
        ensemble=ensemble,
        thermostat=thermostat,
        barostat=barostat,
        temperature=temperature,
        pressure=pressure,
        timestep=timestep,
        friction=friction,
        ttime=ttime,
        taut=taut,
        taup=taup,
        steps=steps,
        interval=1,
        output_dir=output_dir,
        model_name=mlip
    )

    # Save parameters
    param_file = output_dir / "md_params.txt"
    with open(param_file, "w") as f:
        f.write("MD Run Parameters\n")
        f.write("===================\n")
        f.write(f"MLIP model:        {mlip}\n")
        if mlip.startswith("uma-"):
            f.write(f"UMA task:          {uma_task}\n")
        f.write(f"Structure:         {structure.name}\n")
        f.write(f"Ensemble:          {ensemble.upper()}\n")

        if ensemble == 'nvt':
            f.write(f"Thermostat:        {thermostat}\n")
            if thermostat == 'langevin':
                f.write(f"Friction (1/fs):   {friction}\n")
            elif thermostat == 'nose-hoover':
                f.write(f"Time constant (fs): {ttime}\n")
            elif thermostat == 'berendsen':
                f.write(f"Tau T (fs):        {taut}\n")

        elif ensemble == 'npt':
            f.write(f"Barostat:          {barostat}\n")
            f.write(f"Pressure (GPa):    {pressure}\n")
            if barostat == 'npt':
                f.write(f"Time constant (fs): {ttime}\n")
            elif barostat == 'berendsen':
                f.write(f"Tau T (fs):        {taut}\n")
                f.write(f"Tau P (fs):        {taup}\n")

        if ensemble in ['nvt', 'npt']:
            f.write(f"Temperature (K):   {temperature}\n")
        f.write(f"Number of steps:   {steps}\n")
        f.write(f"Timestep (fs):     {timestep}\n")
        f.write(f"Output dir:        {output_dir.resolve()}\n")

    # List output files
    typer.echo("\n✅ MD complete. Output written to:")
    output_files = ["md.traj", "md_energy.csv", "md_energy.png", "md_temperature.png"]

    if ensemble == 'npt':
        output_files.extend(["md_pressure.png", "md_volume.png"])

    output_files.append("md_params.txt")

    for file in output_files:
        typer.echo(f"   📄 {(output_dir / file).resolve()}")

if __name__ == "__main__":
    app()
