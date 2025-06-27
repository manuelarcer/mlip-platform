import typer
from pathlib import Path
from mlip_platform.core import benchmark

app = typer.Typer()

@app.command()
def run(structure: Path):
    """Run MLIP benchmark on a structure file."""
    result = benchmark.run_benchmark(structure)
    typer.echo(result)
