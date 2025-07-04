import typer
from mlip_platform.core.mlip_bench import run_benchmark_all

app = typer.Typer(
    name="benchmark",
    invoke_without_command=True,
    no_args_is_help=False,
    help="Run single-point energy benchmarks using the available MLIPs."
)

@app.callback()
def run(
    structure: str = typer.Option(
        ..., prompt="Enter the path to your structure file (e.g., POSCAR, .traj)"
    ),
    model: str = typer.Option(
        "", prompt="Enter MLIP model name (leave blank to run all available)"
    ),
):
    """
    Benchmark the structure against the selected or all available MLIP models.
    """
    model_name = model or None
    results = run_benchmark_all(structure, model_name)

    for mlip_name, res in results.items():
        if res is None:
            typer.secho(
                f"{mlip_name}: not available in this environment.",
                fg=typer.colors.YELLOW
            )
        else:
            energy = res["energy"]
            time_s = res["time"]
            typer.secho(
                f"{mlip_name}: energy = {energy:.6f} eV, time = {time_s:.4f} s",
                fg=typer.colors.GREEN
            )

    typer.secho("✅ Benchmarking complete!", fg=typer.colors.GREEN)

if __name__ == "__main__":
    app()
