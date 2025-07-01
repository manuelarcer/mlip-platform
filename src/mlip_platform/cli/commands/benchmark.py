import typer
from pathlib import Path
from mlip_platform.core.mlip_bench import run_benchmark 

app = typer.Typer(help="""
Run MLIP benchmark on a structure using a preconfigured model.

Examples:
  Run benchmark on a VASP file:
    mlip-platform benchmark run tests/fixtures/structures/POSCAR

This command compares MLIP predictions to DFT reference values (if available),
and outputs performance metrics such as MAE and energy errors.
""")

@app.command("run")
def run(structure: Path):
    """Run MLIP benchmark on a structure file. PYTHONPATH=src python -m mlip_platform.cli.main benchmark run tests/fixtures/structures/POSCAR"""
    result = run_benchmark(structure)
    typer.echo(result)
