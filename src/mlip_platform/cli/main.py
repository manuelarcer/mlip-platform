import typer
from mlip_platform.cli.commands import md, neb, benchmark

app = typer.Typer(help="MLIP Platform CLI")

# Register subcommands
app.add_typer(md.app, name="md", help="Run Molecular Dynamics simulations")
app.add_typer(neb.app, name="neb", help="Run NEB interpolation and relaxation")
app.add_typer(benchmark.app, name="benchmark", help="Run MLIP benchmark calculations")

if __name__ == "__main__":
    app()