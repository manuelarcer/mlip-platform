from ase.io import read, write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import FIRE
import pandas as pd

class CustomNEB():
    def __init__(self, initial, final, num_images=9, interp_fmax=0.1, interp_steps=1000, fmax=0.05, mlip='sevenn-mf-ompa'):
        self.initial = initial
        final.set_cell(initial.get_cell(), scale_atoms=True)
        self.final = final
        self.num_images = num_images
        self.interp_fmax = interp_fmax
        self.interp_steps = interp_steps
        self.fmax = fmax
        self.mlip = mlip
        self.images = self.setup_neb()

    def setup_calculator(self, model):
        if model == 'sevenn-mf-ompa':
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator('7net-mf-ompa', modal='mpa')
        elif model == 'mace-medium':
            from mace.calculators import mace_mp
            return mace_mp(model='medium', device='cpu')
        else:
            raise ValueError(f"Unsupported MLIP model: {model}")

    def setup_neb(self):
        images = [self.initial]
        for _ in range(self.num_images - 2):
            images.append(self.initial.copy())
        images.append(self.final)

        calc = self.setup_calculator(self.mlip)
        for image in images:
            image.calc = calc

        self.neb = NEB(images)
        return images

    def interpolate_idpp(self):
        """Interpolate NEB images using IDPP (Improved Dimer Projection Path)."""
        idpp_interpolate(self.images)

    def run(self):
        """Run NEB optimization using the FIRE optimizer."""
        optimizer = FIRE(self.neb)
        optimizer.run(fmax=self.fmax)
