from pathlib import Path
from ase.io import write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.optimize import MDMin
import pandas as pd
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from scipy.interpolate import make_interp_spline
import numpy as np

class CustomNEB:
    def __init__(self, initial, final, num_images=9, interp_fmax=0.1, interp_steps=1000,
                 fmax=0.05, mlip='7net-mf-ompa', uma_task='omat', output_dir='.'):
        self.initial = initial
        final.set_cell(initial.get_cell(), scale_atoms=True)
        self.final = final
        self.num_images = num_images
        self.interp_fmax = interp_fmax
        self.interp_steps = interp_steps
        self.fmax = fmax
        self.mlip = mlip
        self.uma_task = uma_task
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images = self.setup_neb()

    def setup_calculator(self, model=None, uma_task=None):
        """
        Setup calculator for the given MLIP model.

        Parameters
        ----------
        model : str, optional
            Model name. For UMA, use format "uma-s-1p1" or "uma-m-1p1".
        uma_task : str, optional
            Task name for UMA models: "omat", "oc20", "omol", or "odac" (default: uses self.uma_task)
        """
        model = model or self.mlip
        uma_task = uma_task or self.uma_task
        if model == '7net-mf-ompa':
            from sevenn.calculator import SevenNetCalculator
            return SevenNetCalculator(model, modal='mpa')
        elif model == 'mace':
            from mace.calculators import mace_mp
            return mace_mp(model="medium", device="cpu")
        elif model.startswith('uma-'):
            from fairchem.core import pretrained_mlip, FAIRChemCalculator
            predictor = pretrained_mlip.get_predict_unit(model, device="cpu")
            return FAIRChemCalculator(predictor, task_name=uma_task)
        else:
            raise ValueError(f"Unknown model: {model}")

    def setup_neb(self):
        return [self.initial] + [self.initial.copy() for _ in range(self.num_images - 2)] + [self.final]

    def interpolate_idpp(self):
        traj_path = self.output_dir / 'idpp.traj'
        log_path = self.output_dir / 'idpp.log'
        idpp_interpolate(self.images, traj=str(traj_path), log=str(log_path),
                         fmax=self.interp_fmax, mic=True, steps=self.interp_steps)

    def run_neb(self, optimizer=MDMin, trajectory='A2B.traj', full_traj='A2B_full.traj', climb=False):
        neb = NEB(self.images, climb=climb)
        for image in self.images:
            image.calc = self.setup_calculator()

        full_traj_path = self.output_dir / full_traj
        traj_writer = Trajectory(str(full_traj_path), 'w')

        def log_images():
            for img in neb.images:
                img.get_potential_energy()
                traj_writer.write(img)

        opt = optimizer(neb)
        opt.attach(log_images)
        opt.run(fmax=self.fmax)
        traj_writer.close()

        for image in self.images:
            image.get_potential_energy()

        traj_path = self.output_dir / trajectory
        with Trajectory(str(traj_path), 'w') as traj:
            for img in self.images:
                traj.write(img)

        return self.images

    def process_results(self):
        results = {'i': [], 'e': []}
        for i, image in enumerate(self.images):
            results['i'].append(i)
            results['e'].append(image.get_potential_energy())
        df = pd.DataFrame(results)
        df['rel_e'] = df['e'] - df['e'].min()
        df.to_csv(self.output_dir / "neb_data.csv", index=False)
        return df

    def plot_results(self, df):
        fig_path = self.output_dir / "neb_energy.png"
        x = df['i']
        y = df['rel_e']

        x_smooth = np.linspace(x.min(), x.max(), 200)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_smooth)

        plt.figure(figsize=(6, 4))
        plt.plot(x_smooth, y_smooth, label="NEB Path", linewidth=2)
        plt.scatter(x, y, color="blue", zorder=5)
        plt.xlabel("Reaction Coordinate (Image Index)")
        plt.ylabel("Relative Energy (eV)")
        plt.title(f"NEB Energy Profile ({self.mlip})")
        plt.grid(True)

        barrier = y.max()
        barrier_index = y.idxmax()
        plt.annotate(f"Barrier: {barrier:.3f} eV",
                     xy=(x[barrier_index], barrier),
                     xytext=(x[barrier_index], barrier + 0.1),
                     arrowprops=dict(arrowstyle="->"),
                     fontsize=9)

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()

    def export_poscars(self):
        from ase.io.vasp import write_vasp
        for i, image in enumerate(self.images):
            folder = self.output_dir / f"{i:02d}"
            folder.mkdir(exist_ok=True)
            write_vasp(folder / "POSCAR", image, direct=True, vasp5=True, sort=True)
