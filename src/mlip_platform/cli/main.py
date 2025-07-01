import typer
from mlip_platform.cli.commands import benchmark, neb, md

app = typer.Typer(help="""
MLIP Platform CLI

This command-line tool lets you run machine-learning interatomic potential tasks:
• Run molecular dynamics simulations
• Perform NEB (nudged elastic band) calculations
• Benchmark models on structures

Examples:
  Run MD with MACE:
    mlip-platform md run --structure structure.vasp --model mace --steps 100

  Run NEB between two structures:
    mlip-platform neb run --initial initial.vasp --final final.vasp --mlip mace

  Run a benchmark on a structure:
    mlip-platform benchmark run --structure structure.vasp --model mace
""")

# Register subcommands
app.add_typer(benchmark.app, name="benchmark", help="Run MLIP benchmark on a structure")
app.add_typer(neb.app, name="neb", help="Run NEB interpolation and relaxation")
app.add_typer(md.app, name="md", help="Run molecular dynamics simulations")

if __name__ == "__main__":
    app()
