"""Benchmark CLI command: time a single-point energy call per available MLIP."""
import json
import time
from pathlib import Path

import typer
from ase.io import read

from mlip_platform.cli.utils import (
    FAIRCHEM_AVAILABLE,
    MACE_AVAILABLE,
    SEVENN_AVAILABLE,
    CHGNET_AVAILABLE,
    UMA_TASK_HELP,
    setup_calculator,
)

app = typer.Typer(help="Benchmark single-point energy + timing across available MLIPs.")


def _available_models() -> list[str]:
    """Return the default list of MLIP tags installed in this environment."""
    models: list[str] = []
    if FAIRCHEM_AVAILABLE:
        models.append("uma-s-1p2")
    if SEVENN_AVAILABLE:
        models.append("7net-mf-ompa")
    if MACE_AVAILABLE:
        models.append("mace")
    if CHGNET_AVAILABLE:
        models.append("chgnet")
    return models


@app.command()
def run(
    structure: Path = typer.Option(..., prompt=True, help="Structure file"),
    models: str = typer.Option(
        None,
        help="Comma-separated MLIP tags to benchmark. Default: every MLIP installed in the current environment.",
    ),
    uma_task: str = typer.Option("omat", help=UMA_TASK_HELP),
    output: Path = typer.Option(None, help="Optional path for a JSON results file."),
):
    """Run a single-point energy + wall-time benchmark for each MLIP in turn.

    Loads the structure once, then for each model: attaches the calculator,
    times one ``get_potential_energy()`` call, and reports the result. Failed
    models are recorded but do not abort the run.
    """
    atoms = read(structure)
    typer.echo(f"Structure: {structure} ({len(atoms)} atoms, {atoms.get_chemical_formula()})")

    if models:
        model_list = [m.strip() for m in models.split(",") if m.strip()]
    else:
        model_list = _available_models()
        if not model_list:
            typer.echo("No MLIP installed. Install one of fairchem-core, sevenn, mace-torch, chgnet.")
            raise typer.Exit(code=1)

    typer.echo(f"Benchmarking: {', '.join(model_list)}\n")

    results: dict[str, dict | str] = {}
    for model in model_list:
        typer.echo(f"--- {model} ---")
        bench_atoms = atoms.copy()
        try:
            setup_calculator(bench_atoms, model, uma_task)
            t0 = time.perf_counter()
            energy = bench_atoms.get_potential_energy()
            elapsed = time.perf_counter() - t0
            results[model] = {"energy_eV": energy, "time_s": elapsed}
            typer.echo(f"  energy = {energy:.6f} eV")
            typer.echo(f"  time   = {elapsed:.3f} s\n")
        except Exception as exc:
            results[model] = f"failed: {type(exc).__name__}: {exc}"
            typer.echo(f"  FAILED: {exc}\n")

    typer.echo("Summary:")
    typer.echo(json.dumps(results, indent=2))

    if output:
        output.write_text(json.dumps(results, indent=2))
        typer.echo(f"\nResults written to {output}")


if __name__ == "__main__":
    app()
