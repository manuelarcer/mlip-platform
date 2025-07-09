import typer
import subprocess
from pathlib import Path

app = typer.Typer(help="Run benchmarks using MACE and SevenNet via subprocess.")

@app.command()
def benchmark(
    structure: Path = typer.Option(..., prompt=True, help="Structure file (.vasp) for benchmarking")
):
    """Run benchmark using all available MLIP models."""
    models = ["mace", "sevenn"]
    successful = []

    for model in models:
        typer.echo(f"\n▶ Running benchmark with {model}...")
        try:
            subprocess.run(
                ["python", "bench_driver.py", str(structure), model],
                check=True
            )
            successful.append(model)
        except subprocess.CalledProcessError:
            typer.echo(f"⚠️  {model} benchmark failed.")
        except FileNotFoundError:
            typer.echo("❌ 'bench_driver.py' not found. Make sure you're running from the project root.")

    if successful:
        typer.echo(f"\n✅ Benchmarks completed for: {', '.join(successful)}")
    else:
        typer.echo("\n❌ No models completed successfully.")
