import typer
from mlip_platform.cli.commands import benchmark, neb, autoneb, autoneb_results, md

app = typer.Typer()

# Register subcommands
app.add_typer(benchmark.app, name="benchmark", help="Run MLIP benchmark on a structure" \
)
app.add_typer(neb.app, name="neb", help="Run NEB interpolation and relaxation "
)
app.add_typer(autoneb.app, name="autoneb", help="Run AutoNEB with dynamic image insertion"
)
app.add_typer(autoneb_results.app, name="autoneb-results", help="Extract and visualize AutoNEB results"
)
app.add_typer(md.app, name="md", help="Run MD simulations "
)  

if __name__ == "__main__":
    app()
