from ase.io import read, write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate  
from ase.optimize import BFGS, MDMin, FIRE
from sevenn.calculator import SevenNetCalculator
import pandas as pd

class CustomNEB():
    def __init__(self, initial, final, num_images=9, interp_fmax=0.1, interp_steps=1000, fmax=0.05, mlip='sevenn-mf-ompa'):
        self.initial = initial
        # Resize final to match initial
        final.set_cell(initial.get_cell(), scale_atoms=True)
        self.final = final
        self.num_images = num_images
        self.interp_fmax = interp_fmax
        self.interp_steps = interp_steps
        self.fmax = fmax
        self.mlip = mlip
        self.images = self.setup_neb()

    def setup_calculator(self, model='sevenn-mf-ompa'):
        if model == 'sevenn-mf-ompa':
            calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')
            return calc
        else:
            raise ValueError(f"Unknown MLIP model: {model}")

    def setup_neb(self):
        # Generate images
        images = [self.initial]
        for _ in range(self.num_images - 2):
            images.append(self.initial.copy())
        images.append(self.final)

        # Assign calculator to each image
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
