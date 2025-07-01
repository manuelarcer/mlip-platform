import typer
from mlip_platform.cli.commands import benchmark, neb, md  

app = typer.Typer()

# Register subcommands
app.add_typer(benchmark.app, name="benchmark", help="Run MLIP benchmark on a structure" \
"PYTHONPATH=src python -m mlip_platform.cli.main benchmark run tests/fixtures/structures/POSCAR")
app.add_typer(neb.app, name="neb", help="Run NEB interpolation and relaxation " \
"PYTHONPATH=src python -m mlip_platform.cli.main neb run \
  --initial tests/fixtures/structures/fragment_initial.vasp \
  --final tests/fixtures/structures/fragment_final.vasp \
  --mlip 7net-mf-ompa or mace ")
app.add_typer(md.app, name="md", help="Run MD simulations "
"PYTHONPATH=src python -m mlip_platform.cli.main md run \
  --structure tests/fixtures/structures/fragment_initial.vasp \
  --model mace \
  --steps 10 \
  --temperature 300 \
  --log mace_md.log or --log seven_md.log")  

if __name__ == "__main__":
    app()
