"""Geometry optimization runner for mlip_platform."""
from ase.optimize import BFGS, FIRE, MDMin


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


class Optimizer:
    def __init__(self, atoms, model="sevenn-mf-ompa", fmax=0.05, method="BFGS"):
        self.atoms = atoms
        self.model = model
        self.fmax = fmax
        self.method = method

    def run(self):
        """Run geometry optimization and return (trajectory_file, final_energy)."""
        calc = get_calculator(self.model)
        self.atoms.calc = calc

        optimizer_cls = {"BFGS": BFGS, "FIRE": FIRE, "MDMin": MDMin}[self.method]
        traj_file = "opt.traj"
        opt = optimizer_cls(self.atoms, trajectory=traj_file)
        opt.run(fmax=self.fmax)

        energy = self.atoms.get_potential_energy()
        return traj_file, energy