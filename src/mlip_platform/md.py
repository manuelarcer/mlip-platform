"""Molecular dynamics runner for mlip_platform."""
from ase.md.langevin import Langevin
from ase import units


def get_calculator(model):
    """Return an ASE calculator instance for the given MLIP model."""
    if model == "sevenn-mf-ompa":
        from sevenn.calculator import SevenNetCalculator

        return SevenNetCalculator("7net-mf-ompa", modal="mpa")
    elif model == "mace":
        from mace.calculators import mace_mp

        return mace_mp(model="medium", device="cpu")
    elif model == "chgnet":
        from chgnet.model import CHGNetCalculator

        return CHGNetCalculator()
    else:
        raise ValueError(f"Unknown model: {model}")


class MdRunner:
    def __init__(self, atoms, model="sevenn-mf-ompa", temperature=300.0, timestep=1.0, steps=1000):
        self.atoms = atoms
        self.model = model
        self.temperature = temperature
        self.timestep = timestep
        self.steps = steps

    def run(self):
        """Run MD and return (trajectory_file, [energies])."""
        calc = get_calculator(self.model)
        self.atoms.calc = calc

        dyn = Langevin(self.atoms, timestep=self.timestep * units.fs,
                       temperature_K=self.temperature)
        traj_file = "md.traj"
        energies = []

        def record_energy(a=self.atoms):
            energies.append(a.get_potential_energy())

        dyn.attach(record_energy, interval=1)
        dyn.run(self.steps)
        return traj_file, energies