from ase.io import read, write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import BFGS, MDMin, FIRE

from .utils import load_calculator

class CustomNEB:
    def __init__(
        self,
        initial,
        final,
        num_images: int = 9,
        interp_fmax: float = 0.1,
        interp_steps: int = 1000,
        fmax: float = 0.05,
        model: str = None,
    ):
        self.initial = initial
        final.set_cell(initial.get_cell(), scale_atoms=True)
        self.final = final
        self.num_images = num_images
        self.interp_fmax = interp_fmax
        self.interp_steps = interp_steps
        self.fmax = fmax
        self.model = model
        self.images = self._setup_images()

    def _setup_calculator(self):
        """Return the appropriate MLIP calculator based on the active environment."""
        return load_calculator(self.model)

    def _setup_images(self):
        """Interpolate images and attach calculators to each."""
        # Create initial, intermediate, and final images
        images = [self.initial.copy()]
        for _ in range(self.num_images - 2):
            images.append(self.initial.copy())
        images.append(self.final.copy())

        # Interpolate with IDPP
        idpp_interpolate(images, fmax=self.interp_fmax, steps=self.interp_steps)

        # Assign calculators
        for img in images:
            img.calc = self._setup_calculator()

        # Build NEB object
        neb = NEB(images)
        return neb

    def run(self, optimizer: str = "BFGS"):
        """
        Run the NEB optimization.

        Parameters
        ----------
        optimizer : {"BFGS", "MDMin", "FIRE"}
            Choice of optimizer class.
        """
        opt_cls = {"BFGS": BFGS, "MDMin": MDMin, "FIRE": FIRE}.get(optimizer, BFGS)
        opt = opt_cls(self.images, trajectory="neb.traj")
        opt.run(fmax=self.fmax)
