import typer
from mlip_platform.cli.commands import md as md_cmd
from mlip_platform.cli.commands import neb as neb_cmd
from mlip_platform.cli.commands import benchmark as bench_cmd

app = typer.Typer(
    help="MLIP Platform CLI: run MD, NEB, or Benchmark using MACE or SevenNet."
)

# Register top-level commands (flattened, no subcommand nesting)
app.add_typer(md_cmd.app)
app.add_typer(neb_cmd.app)
app.add_typer(bench_cmd.app)

if __name__ == "__main__":
    app()
