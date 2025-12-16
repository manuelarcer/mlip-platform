from pathlib import Path
from ase.io import write
from ase.mep import NEB
from ase.mep.neb import idpp_interpolate
from ase.mep.autoneb import AutoNEB
from ase.optimize import MDMin, FIRE
import pandas as pd
import matplotlib.pyplot as plt
from ase.io.trajectory import Trajectory
from scipy.interpolate import make_interp_spline
import numpy as np

class CustomNEB:
    def __init__(self, initial, final, num_images=9, interp_fmax=0.1, interp_steps=1000,
                 fmax=0.05, mlip='7net-mf-ompa', uma_task='omat', output_dir='.', relax_atoms=None,
                 logfile='neb.log'):
        """
        Initialize NEB calculation.

        Parameters
        ----------
        initial : ase.Atoms
            Initial structure
        final : ase.Atoms
            Final structure
        num_images : int
            Number of INTERMEDIATE images (excluding initial and final)
            Total images = num_images + 2
        interp_fmax : float
            Force threshold for IDPP interpolation
        interp_steps : int
            Maximum steps for IDPP interpolation
        fmax : float
            Force convergence threshold for NEB optimization
        mlip : str
            MLIP model name
        uma_task : str
            Task name for UMA models
        output_dir : str or Path
            Output directory for results
        relax_atoms : list of int, optional
            List of atom indices to relax. If provided, all other atoms are fixed
            at their linearly interpolated positions. IDPP is skipped.
        logfile : str
            Name of the log file for NEB iteration logging (default: 'neb.log')
        """
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
        self.relax_atoms = relax_atoms
        self.logfile = logfile
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
        """Setup NEB image list."""
        # num_images is the number of INTERMEDIATE images (not including initial/final)
        from ase.constraints import FixAtoms

        images = [self.initial]
        images += [self.initial.copy() for _ in range(self.num_images)]
        images += [self.final]

        # Store FixAtoms constraint from initial (to restore after IDPP)
        self._fix_atoms_constraint = None
        for constraint in self.initial.constraints:
            if isinstance(constraint, FixAtoms):
                self._fix_atoms_constraint = constraint
                break

        # Remove FixAtoms from intermediate images to allow interpolation
        # Keep all other constraints (Hookean, etc.)
        for img in images[1:-1]:
            non_fix_constraints = [c for c in img.constraints if not isinstance(c, FixAtoms)]
            img.set_constraint(non_fix_constraints)

        # Use ASE's built-in interpolate method (handles MIC automatically)
        neb_temp = NEB(images)
        neb_temp.interpolate(method='linear', mic=True)

        # If relax_atoms is specified, we want "linear interpolation" + "fixed atoms"
        # Since we just did linear interpolation above, now we fix the NON-relaxed atoms.
        if self.relax_atoms is not None:
             fix_indices = [i for i in range(len(self.initial)) if i not in self.relax_atoms]
             constraint = FixAtoms(indices=fix_indices)
             for img in images[1:-1]:
                 # We likely want to add to existing constraints (Hookean etc), but we removed FixAtoms earlier.
                 # So we append this new FixAtoms constraint.
                 current_constraints = list(img.constraints)
                 current_constraints.append(constraint)
                 img.set_constraint(current_constraints)

        return images

    def interpolate_idpp(self):
        """Run IDPP interpolation and restore FixAtoms constraints after."""
        if self.relax_atoms is not None:
            print("Skipping IDPP interpolation because relax_atoms is specified (highly-constraint mode).")
            return

        traj_path = self.output_dir / 'idpp.traj'
        log_path = self.output_dir / 'idpp.log'
        idpp_interpolate(self.images, traj=str(traj_path), log=str(log_path),
                         fmax=self.interp_fmax, mic=True, steps=self.interp_steps)

        # Restore FixAtoms constraints to intermediate images after IDPP
        if self._fix_atoms_constraint is not None:
            for img in self.images[1:-1]:  # Skip initial and final
                # Add FixAtoms back to existing constraints
                current_constraints = list(img.constraints)
                current_constraints.append(self._fix_atoms_constraint)
                img.set_constraint(current_constraints)

    def run_neb(self, optimizer=MDMin, trajectory='A2B.traj', full_traj='A2B_full.traj', climb=False):
        neb = NEB(self.images, climb=climb)
        for image in self.images:
            image.calc = self.setup_calculator()

        full_traj_path = self.output_dir / full_traj
        traj_writer = Trajectory(str(full_traj_path), 'w')

        # Setup log file
        log_path = self.output_dir / self.logfile
        log_file = open(log_path, 'w')

        # Data collection for convergence tracking
        log_data = {
            "step": [],
            "fmax(eV/A)": [],
            "barrier(eV)": [],
        }

        def log_iteration():
            """Callback to log each NEB iteration"""
            step = opt.nsteps
            # Get the NEB forces (this includes spring forces and is what the optimizer sees)
            # We need to get the forces from the NEB object's get_forces() method
            neb_forces = neb.get_forces()
            # Calculate fmax from the NEB forces (not individual image forces)
            fmax = (neb_forces**2).sum(axis=1).max()**0.5

            # Get energies
            energies = [img.get_potential_energy() for img in neb.images]
            initial_energy = energies[0]
            max_energy = max(energies)
            barrier_height = max_energy - initial_energy

            # Write to log file
            log_file.write(f"Step {step:4d}  Fmax: {fmax:.6f} eV/A  Barrier: {barrier_height:.6f} eV\n")
            log_file.flush()

            # Store data
            log_data["step"].append(step)
            log_data["fmax(eV/A)"].append(fmax)
            log_data["barrier(eV)"].append(barrier_height)

            # Write trajectory
            for img in neb.images:
                traj_writer.write(img)

        opt = optimizer(neb, logfile=log_file)
        opt.attach(log_iteration, interval=1)
        opt.run(fmax=self.fmax)

        log_file.close()
        traj_writer.close()

        # Save convergence data to CSV
        csv_path = self.output_dir / 'neb_convergence.csv'
        df_conv = pd.DataFrame(log_data)
        df_conv.to_csv(csv_path, index=False)

        # Plot convergence
        self._plot_convergence(df_conv)

        for image in self.images:
            image.get_potential_energy()

        traj_path = self.output_dir / trajectory
        with Trajectory(str(traj_path), 'w') as traj:
            for img in self.images:
                traj.write(img)

        return self.images

    def run_autoneb(self, n_simul=1, n_max=9, k=0.1, climb=True,
                    optimizer=FIRE, space_energy_ratio=0.5,
                    interpolate_method='idpp', maxsteps=10000, prefix='autoneb'):
        """
        Run AutoNEB calculation.

        Parameters
        ----------
        n_simul : int
            Number of parallel relaxations (default: 1). Requires MPI for n_simul > 1
        n_max : int
            Target number of images including endpoints (default: 9)
        k : float
            Spring constant (default: 0.1)
        climb : bool
            Enable climbing image NEB (default: True)
        optimizer : ASE optimizer class
            Optimizer to use (default: FIRE)
        space_energy_ratio : float
            Preference for geometric (1.0) vs energy (0.0) gaps when inserting images (default: 0.5)
        interpolate_method : str
            Interpolation method: 'linear' or 'idpp' (default: 'idpp')
        maxsteps : int
            Maximum number of steps per relaxation (default: 10000)
        prefix : str
            Prefix for AutoNEB output files (default: 'autoneb')

        Returns
        -------
        None
            AutoNEB manages files directly. Check output_dir for results.

        Notes
        -----
        - AutoNEB uses its own file-based I/O with prefix naming (prefix000.traj, prefix001.traj, etc.)
        - Custom convergence tracking (CSV/PNG) is not available in AutoNEB mode
        - Results are stored in AutoNEB_iter/ folder
        - Highly-constrained mode (relax_atoms) may not be fully compatible with AutoNEB
        """
        import os

        if self.relax_atoms is not None:
            print("⚠️  WARNING: AutoNEB with relax_atoms (highly-constrained mode) may not work as expected.")
            print("   Consider using standard NEB for constrained calculations.")

        # Change to output directory so AutoNEB writes files there
        original_cwd = os.getcwd()
        os.chdir(self.output_dir)

        # Clean up any existing AutoNEB files to avoid conflicts
        import glob
        for old_file in glob.glob(f"{prefix}*.traj"):
            os.remove(old_file)
        if os.path.exists("AutoNEB_iter"):
            import shutil
            shutil.rmtree("AutoNEB_iter")

        try:
            # Create calculator attachment function
            def attach_calculators(images):
                """Attach calculators to all images"""
                for image in images:
                    image.calc = self.setup_calculator()

            # Prepare initial images (just start and end for AutoNEB)
            # AutoNEB will dynamically add intermediate images
            initial_images = [self.initial, self.final]

            # Attach calculators and calculate energies for initial images
            # This is required so AutoNEB doesn't crash with NaN energies
            for img in initial_images:
                img.calc = self.setup_calculator()
                img.get_potential_energy()  # Force energy calculation

            # Write initial images to trajectory files
            from ase.io import write as ase_write
            for i, img in enumerate(initial_images):
                ase_write(f"{prefix}{i:03d}.traj", img)

            print(f"\n🤖 Starting AutoNEB calculation")
            print(f"   n_max: {n_max} images (including endpoints)")
            print(f"   n_simul: {n_simul} parallel relaxations")
            print(f"   fmax: {self.fmax} eV/Å")
            print(f"   climb: {climb}")
            print(f"   optimizer: {optimizer.__name__}")
            print(f"   space_energy_ratio: {space_energy_ratio}")
            print(f"   interpolate_method: {interpolate_method}")
            print(f"   Output directory: {self.output_dir.resolve()}\n")

            # Create AutoNEB object
            autoneb = AutoNEB(
                attach_calculators=attach_calculators,
                prefix=prefix,
                n_simul=n_simul,
                n_max=n_max,
                fmax=self.fmax,
                maxsteps=maxsteps,
                k=k,
                climb=climb,
                optimizer=optimizer,
                space_energy_ratio=space_energy_ratio,
                interpolate_method=interpolate_method
            )

            # Run AutoNEB
            autoneb.run()

            print("\n✅ AutoNEB calculation complete")
            print(f"   Output files in: {self.output_dir.resolve()}")
            print(f"   - {prefix}*.traj: Individual image trajectories")
            print(f"   - AutoNEB_iter/: Iteration history folder")

        finally:
            # Return to original directory
            os.chdir(original_cwd)

    def _plot_convergence(self, df):
        """Plot NEB convergence (force and max energy vs steps)"""
        fig_path = self.output_dir / "neb_convergence.png"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # Force convergence
        ax1.plot(df["step"], df["fmax(eV/A)"], marker='o', markersize=4, linewidth=1.5, color='orange')
        ax1.axhline(y=self.fmax, color='r', linestyle='--', label=f'fmax target = {self.fmax}')
        ax1.set_xlabel("NEB Step")
        ax1.set_ylabel("Max Force (eV/Å)")
        ax1.set_title("NEB Force Convergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')

        # Barrier height evolution
        ax2.plot(df["step"], df["barrier(eV)"], marker='o', markersize=4, linewidth=1.5)
        ax2.set_xlabel("NEB Step")
        ax2.set_ylabel("Barrier Height (eV)")
        ax2.set_title("Barrier Height Evolution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

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
