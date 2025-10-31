from pathlib import Path
import typer
from ase.io import read
from mlip_platform.core.neb import CustomNEB
from mlip_platform.cli.utils import detect_mlip, validate_mlip

app = typer.Typer()

@app.command()
def neb(
    initial: Path = typer.Option(..., prompt=True, help="Initial structure file (.vasp)"),
    final: Path = typer.Option(..., prompt=True, help="Final structure file (.vasp)"),
    num_images: int = typer.Option(5, prompt="Number of NEB images"),
    interp_fmax: float = typer.Option(0.1, prompt="IDPP interpolation fmax"),
    interp_steps: int = typer.Option(100, prompt="IDPP interpolation steps"),
    fmax: float = typer.Option(0.05, prompt="Final NEB force threshold"),
    mlip: str = typer.Option("auto", help="MLIP model: 'uma-s-1p1', 'uma-m-1p1', 'mace', '7net-mf-ompa', or 'auto'"),
    uma_task: str = typer.Option("omat", help="UMA task name: 'omat', 'oc20', 'omol', or 'odac' (only for UMA models)")
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

    base_dir = initial.resolve().parent
    output_dir = base_dir  

    typer.echo(f"⚙️ Running NEB with:")
    typer.echo(f" - num_images:    {num_images}")
    typer.echo(f" - interp_fmax:   {interp_fmax}")
    typer.echo(f" - interp_steps:  {interp_steps}")
    typer.echo(f" - final fmax:    {fmax}")
    typer.echo(f" - output_dir:    {output_dir}")

    with open(output_dir / "neb_parameters.txt", "w") as f:
        f.write("NEB Run Parameters\n")
        f.write("===================\n")
        f.write(f"MLIP model:        {mlip}\n")
        if mlip.startswith("uma-"):
            f.write(f"UMA task:          {uma_task}\n")
        f.write(f"Initial:           {initial}\n")
        f.write(f"Final:             {final}\n")
        f.write(f"Number of images:  {num_images}\n")
        f.write(f"IDPP fmax:         {interp_fmax}\n")
        f.write(f"IDPP steps:        {interp_steps}\n")
        f.write(f"Final fmax:        {fmax}\n")
        f.write(f"Output dir:        {output_dir}\n")

    neb = CustomNEB(
        initial=atoms_initial,
        final=atoms_final,
        num_images=num_images,
        interp_fmax=interp_fmax,
        interp_steps=interp_steps,
        fmax=fmax,
        mlip=mlip,
        uma_task=uma_task,
        output_dir=output_dir
    )

    typer.echo(" Interpolating with IDPP...")
    neb.interpolate_idpp()

    typer.echo(" Running NEB optimization...")
    neb.run_neb()

    typer.echo(" Processing results...")
    df = neb.process_results()
    neb.plot_results(df)

    typer.echo("Exporting POSCARs...")
    neb.export_poscars()

    typer.echo("✅ NEB complete. Output written to:")
    for file in ["A2B.traj", "A2B_full.traj", "idpp.traj", "idpp.log", "neb_data.csv", "neb_energy.png", "neb_parameters.txt"]:
        typer.echo(f" - {output_dir / file}")
    for i in range(num_images):
        typer.echo(f" - {output_dir / f'{i:02d}' / 'POSCAR'}")

if __name__ == "__main__":
    app()
