import typer
from pathlib import Path
from mlip_platform.core.mlip_bench import run_benchmark 

app = typer.Typer()

@app.command("run")
def run(structure: Path):
    """Run MLIP benchmark on a structure file."""
    result = run_benchmark(structure)
    typer.echo(result)
