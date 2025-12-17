from pathlib import Path
import typer
from ase.io import read
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from ase.io.vasp import write_vasp

app = typer.Typer()

@app.command()
def results(
    directory: Path = typer.Option(".", help="Directory containing AutoNEB results"),
    prefix: str = typer.Option("autoneb", help="Prefix used for AutoNEB files"),
    export_poscars: bool = typer.Option(False, help="Export each image as VASP POSCAR"),
):
    """
    Extract and visualize results from completed AutoNEB calculation.

    This command:
    - Reads final images from autoneb*.traj files
    - Calculates energy profile and barrier height
    - Generates energy profile plot
    - Exports data to CSV
    - Optionally exports images to VASP POSCAR format
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        typer.echo(f"❌ Error: Directory {directory} does not exist.")
        raise typer.Exit(code=1)

    # Find all trajectory files
    import glob
    traj_files = sorted(glob.glob(str(directory / f"{prefix}*.traj")))

    if not traj_files:
        typer.echo(f"❌ Error: No {prefix}*.traj files found in {directory}")
        raise typer.Exit(code=1)

    typer.echo(f"\n📊 Processing AutoNEB results from: {directory}")
    typer.echo(f"   Found {len(traj_files)} image files\n")

    # Read final images (last frame from each trajectory)
    images = []
    for traj_file in traj_files:
        # Read last frame from trajectory
        atoms = read(traj_file, index=-1)
        images.append(atoms)

    typer.echo(f"✅ Loaded {len(images)} final images")

    # Extract energies
    results = {'image': [], 'energy': []}
    for i, image in enumerate(images):
        try:
            energy = image.get_potential_energy()
            results['image'].append(i)
            results['energy'].append(energy)
        except Exception as e:
            typer.echo(f"⚠️  Warning: Could not get energy for image {i}: {e}")

    # Create dataframe
    df = pd.DataFrame(results)
    df['rel_energy'] = df['energy'] - df['energy'].min()

    # Calculate barrier height
    initial_energy = df['energy'].iloc[0]
    final_energy = df['energy'].iloc[-1]
    max_energy = df['energy'].max()
    barrier_forward = max_energy - initial_energy
    barrier_reverse = max_energy - final_energy
    ts_index = df['energy'].idxmax()

    typer.echo(f"\n📈 Energy Analysis:")
    typer.echo(f"   Initial energy:      {initial_energy:.6f} eV")
    typer.echo(f"   Final energy:        {final_energy:.6f} eV")
    typer.echo(f"   TS energy:           {max_energy:.6f} eV")
    typer.echo(f"   TS image index:      {ts_index}")
    typer.echo(f"   Forward barrier:     {barrier_forward:.6f} eV")
    typer.echo(f"   Reverse barrier:     {barrier_reverse:.6f} eV")
    typer.echo(f"   Reaction energy:     {final_energy - initial_energy:.6f} eV")

    # Save to CSV
    csv_path = directory / f"{prefix}_energy_profile.csv"
    df.to_csv(csv_path, index=False)
    typer.echo(f"\n💾 Saved energy data to: {csv_path}")

    # Plot energy profile
    fig_path = directory / f"{prefix}_energy_profile.png"
    x = df['image']
    y = df['rel_energy']

    # Create smooth spline if we have enough points
    plt.figure(figsize=(8, 6))
    if len(x) >= 4:
        x_smooth = np.linspace(x.min(), x.max(), 200)
        spline = make_interp_spline(x, y, k=min(3, len(x)-1))
        y_smooth = spline(x_smooth)
        plt.plot(x_smooth, y_smooth, label="NEB Path", linewidth=2, color='steelblue')
    else:
        plt.plot(x, y, '-', label="NEB Path", linewidth=2, color='steelblue')

    plt.scatter(x, y, color="red", zorder=5, s=80, edgecolors='black', linewidths=1.5)
    plt.xlabel("Image Index", fontsize=12)
    plt.ylabel("Relative Energy (eV)", fontsize=12)
    plt.title(f"AutoNEB Energy Profile", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Annotate barrier
    barrier_index = y.idxmax()
    barrier = y.max()
    plt.annotate(f"TS: {barrier:.3f} eV",
                 xy=(x[barrier_index], barrier),
                 xytext=(x[barrier_index], barrier + 0.1),
                 arrowprops=dict(arrowstyle="->", lw=1.5),
                 fontsize=10,
                 ha='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    typer.echo(f"📊 Saved energy profile plot to: {fig_path}")

    # Export POSCARs if requested
    if export_poscars:
        typer.echo(f"\n📁 Exporting images to VASP POSCAR format...")
        for i, image in enumerate(images):
            folder = directory / f"image_{i:02d}"
            folder.mkdir(exist_ok=True)
            write_vasp(folder / "POSCAR", image, direct=True, vasp5=True, sort=True)
        typer.echo(f"   Exported {len(images)} POSCARs to image_XX/ directories")

    # Summary
    typer.echo(f"\n✅ AutoNEB results extraction complete!")
    typer.echo(f"\n📁 Output files:")
    typer.echo(f"   {csv_path.name} - Energy data")
    typer.echo(f"   {fig_path.name} - Energy profile plot")
    if export_poscars:
        typer.echo(f"   image_XX/ - VASP POSCAR files for each image")

if __name__ == "__main__":
    app()
