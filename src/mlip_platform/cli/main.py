import typer
from mlip_platform.cli.commands import benchmark, neb

app = typer.Typer()

# Register subcommands
app.add_typer(benchmark.app, name="benchmark", help="Run MLIP benchmark on a structure")
app.add_typer(neb.app, name="neb", help="Run NEB interpolation and relaxation")

if __name__ == "__main__":
    app()
