#!/usr/bin/env python3
"""Command-line interface for mlip_platform: optimization, NEB, and MD."""
import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """mlip-platform: run MLIP-based optimizations, NEB, and MD."""
    pass


@cli.command()
@click.argument("initial", type=click.Path(exists=True))
@click.option("--model", "-m", default="sevenn-mf-ompa",
              type=click.Choice(["sevenn-mf-ompa", "mace", "chgnet"]),
              help="Which MLIP model to use")
@click.option("--fmax", default=0.05,
              help="Maximum force (eV/Å) convergence criterion")
@click.option("--optimizer", default="BFGS",
              type=click.Choice(["BFGS", "FIRE", "MDMin"]),
              help="Which ASE optimizer to use")
def optimize(initial, model, fmax, optimizer):
    """Geometry optimization of INITIAL structure."""
    from ase.io import read
    from mlip_platform.optim import Optimizer

    atoms = read(initial)
    runner = Optimizer(atoms, model=model, fmax=float(fmax), method=optimizer)
    traj_file, energy = runner.run()
    click.echo(f"Optimization finished: final energy = {energy:.6f} eV")
    click.echo(f"Trajectory written to {traj_file}")


@cli.command()
@click.argument("initial", type=click.Path(exists=True))
@click.argument("final", type=click.Path(exists=True))
@click.option("--model", "-m", default="sevenn-mf-ompa",
              type=click.Choice(["sevenn-mf-ompa", "mace", "chgnet"]),
              help="Which MLIP model to use")
@click.option("--images", "-n", default=9, help="Number of NEB images")
@click.option("--fmax", default=0.05, help="Force convergence for NEB")
@click.option("--interp-fmax", default=0.1, help="Interpolation fmax")
@click.option("--interp-steps", default=1000, help="IDPP interpolation steps")
@click.option("--climb/--no-climb", default=False, help="Enable CI-NEB")
def neb(initial, final, model, images, fmax, interp_fmax, interp_steps, climb):
    """Run NEB from INITIAL → FINAL."""
    from ase.io import read
    from mlip_platform.neb import CustomNEB

    ini = read(initial)
    fin = read(final)
    runner = CustomNEB(
        initial=ini,
        final=fin,
        num_images=int(images),
        interp_fmax=float(interp_fmax),
        interp_steps=int(interp_steps),
        fmax=float(fmax),
        mlip=model,
    )
    runner.run_neb(climb=climb)
    df = runner.process_results()
    df.to_csv("neb_results.csv", index=False)
    runner.write_images(path="neb_images")
    click.echo("NEB done. Energies saved to neb_results.csv; images in neb_images/")


@cli.command()
@click.argument("initial", type=click.Path(exists=True))
@click.option("--model", "-m", default="sevenn-mf-ompa",
              type=click.Choice(["sevenn-mf-ompa", "mace", "chgnet"]),
              help="Which MLIP model to use")
@click.option("--temperature", "-T", default=300.0, help="Temperature in K")
@click.option("--timestep", "-dt", default=1.0, help="MD timestep in fs")
@click.option("--steps", "-n", default=1000, help="Number of MD steps")
def md(initial, model, temperature, timestep, steps):
    """Run molecular dynamics on INITIAL structure."""
    from ase.io import read
    from mlip_platform.md import MdRunner

    atoms = read(initial)
    runner = MdRunner(
        atoms,
        model=model,
        temperature=float(temperature),
        timestep=float(timestep),
        steps=int(steps),
    )
    traj_file, energies = runner.run()
    click.echo(f"MD finished. Trajectory: {traj_file}")
    click.echo(f"Average energy: {sum(energies)/len(energies):.6f} eV")


if __name__ == "__main__":
    cli()