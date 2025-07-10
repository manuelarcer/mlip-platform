from pathlib import Path
from ase.io import read, write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import MDMin
import pandas as pd
import matplotlib.pyplot as plt

class CustomNEB:
    def __init__(self, initial, final, num_images=9, interp_fmax=0.1, interp_steps=1000,
                 fmax=0.05, mlip='7net-mf-ompa', output_dir='neb_result'):
        self.initial = initial
        final.set_cell(initial.get_cell(), scale_atoms=True)
        self.final = final
        self.num_images = num_images
        self.interp_fmax = interp_fmax
        self.interp_steps = interp_steps
        self.fmax = fmax
        self.mlip = mlip
        self.output_dir = Path(output_dir) / mlip  # Save by model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images = self.setup_neb()

    def setup_calculator(self, model=None):
        model = model or self.mlip
        if model == '7net-mf-ompa':
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator(model, modal='mpa')
        elif model == 'mace':
            from mace.calculators import mace_mp
            return mace_mp(model="medium", device="cpu")
        else:
            raise ValueError(f"Unknown model: {model}")

    def setup_neb(self):
        return [self.initial] + [self.initial.copy() for _ in range(self.num_images - 2)] + [self.final]

    def interpolate_idpp(self):
        traj_path = self.output_dir / 'idpp.traj'
        log_path = self.output_dir / 'idpp.log'
        idpp_interpolate(self.images, traj=str(traj_path), log=str(log_path),
                         fmax=self.interp_fmax, mic=True, steps=self.interp_steps)

    def run_neb(self, optimizer=MDMin, trajectory='A2B.traj', climb=False):
        neb = NEB(self.images, climb=climb)
        for image in self.images:
            image.calc = self.setup_calculator()
        traj_path = self.output_dir / trajectory
        opt = optimizer(neb, trajectory=str(traj_path))
        opt.run(fmax=self.fmax)
        return self.images

    def process_results(self):
        results = {'i': [], 'e': []}
        for i, image in enumerate(self.images):
            results['i'].append(i)
            results['e'].append(image.get_potential_energy())
        df = pd.DataFrame(results)
        df['rel_e'] = df['e'] - df['e'].min()
        csv_path = self.output_dir / "neb_data.csv"
        df.to_csv(csv_path, index=False)
        return df

    def write_images(self, subdir='images'):
        image_dir = self.output_dir / subdir
        image_dir.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(self.images):
            write(image_dir / f'{i:02d}.vasp', image, format='vasp')

    def plot_results(self, df):
        fig_path = self.output_dir / "neb_energy.png"
        plt.plot(df['i'], df['rel_e'], marker='o')
        plt.xlabel("Image index")
        plt.ylabel("Relative Energy (eV)")
        plt.title(f"NEB Energy Profile ({self.mlip})")
        plt.grid(True)
        plt.savefig(fig_path)
        plt.close()
