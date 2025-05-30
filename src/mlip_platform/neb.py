from ase.io import read, write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import BFGS, MDMin, FIRE
from sevenn.calculator import SevenNetCalculator
#from chgnet.model import CHGNetCalculator
#from chgnet.model import CHGNet
import pandas as pd

class CustomNEB():
    def __init__(self, initial, final, num_images=9, interp_fmax=0.1, interp_steps=1000, fmax=0.05, mlip='sevenn-mf-ompa'):
        self.initial = initial
        # initial and final may have different cell sizes. Resize final state to initial, scale atoms accordingly
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
        elif model == 'chgnet':
            calc = CHGNetCalculator()
        else:
            print(f"Unknown model: {model}. Using default.")
            calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')
        return calc

    def setup_neb(self):
        images = [self.initial] + [self.initial.copy() for _ in range(self.num_images - 2)] + [self.final]
        idpp_interpolate(images, traj='idpp.traj', log='idpp.log', fmax=self.interp_fmax, mic=True, steps=self.interp_steps)
        return images
    
    def run_neb(self, optimizer=MDMin, trajectory='A2B.traj', climb=False):
        # Set up NEB
        neb = NEB(self.images, climb=climb)
        for image in self.images:
            image.calc = self.setup_calculator(self.mlip)
        optimizer = optimizer(neb, trajectory=trajectory)
        optimizer.run(fmax=self.fmax)
        return self.images
    
    def process_results(self):
        results = {'i': [], 'e': []}
        for i, image in enumerate(self.images):
            results['i'].append(i)
            results['e'].append(image.get_potential_energy())
        df = pd.DataFrame(results)
        df['rel_e'] = df['e'] - df['e'].min()
        return df

    def write_images(self, path='images'):
        for i, image in enumerate(self.images):
            write(f'{path}/{i:02d}.vasp', image, format='vasp')
