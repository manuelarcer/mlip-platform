import typer
from pathlib import Path
from mlip_platform.core.mlip_bench import run_benchmark 

app = typer.Typer()

@app.command("run")
def run(structure: Path):
    """Run MLIP benchmark on a structure file. PYTHONPATH=src python -m mlip_platform.cli.main benchmark run tests/fixtures/structures/POSCAR"""
    result = run_benchmark(structure)
    typer.echo(result)
