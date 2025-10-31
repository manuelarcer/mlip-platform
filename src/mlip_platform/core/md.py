import sys
import numpy as np
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase import units
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Try to import Nosé-Hoover (may not be available in all ASE versions)
try:
    from ase.md.nose_hoover import NoseHoover
    NOSEHOOVER_AVAILABLE = True
except ImportError:
    NOSEHOOVER_AVAILABLE = False


# Ensemble/thermostat/barostat mapping
DYNAMICS_MAP = {
    'nve': {
        'velocityverlet': VelocityVerlet,
    },
    'nvt': {
        'langevin': Langevin,
        'nose-hoover': NoseHoover if NOSEHOOVER_AVAILABLE else None,
        'berendsen': NVTBerendsen,
    },
    'npt': {
        'npt': NPT,  # Isotropic MTK
        'berendsen': NPTBerendsen,
    }
}


def setup_dynamics(atoms, ensemble='nvt', thermostat='langevin', barostat='npt',
                   temperature=300, pressure=0.0, timestep=1.0,
                   friction=0.01, ttime=25.0, pfactor=None,
                   taut=100.0, taup=1000.0, compressibility=4.57e-5):
    """
    Setup MD dynamics with specified ensemble and parameters.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with calculator attached
    ensemble : str
        Ensemble type: 'nve', 'nvt', 'npt'
    thermostat : str
        Thermostat for NVT: 'langevin', 'nose-hoover', 'berendsen'
    barostat : str
        Barostat for NPT: 'npt' (isotropic MTK), 'berendsen'
    temperature : float
        Temperature in Kelvin
    pressure : float
        Pressure in GPa (for NPT)
    timestep : float
        Timestep in fs
    friction : float
        Langevin friction coefficient (1/fs)
    ttime : float
        Nosé-Hoover/NPT time constant (fs)
    pfactor : float or None
        NPT pressure coupling factor (auto-calculated if None)
    taut : float
        Berendsen temperature coupling time (fs)
    taup : float
        Berendsen pressure coupling time (fs)
    compressibility : float
        Berendsen compressibility (1/GPa)

    Returns
    -------
    dyn : ASE dynamics object
    """
    ensemble = ensemble.lower()

    # Initialize velocities for temperature-based ensembles
    if ensemble in ['nvt', 'npt'] and temperature > 0:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

    timestep_ase = timestep * units.fs

    # NVE ensemble
    if ensemble == 'nve':
        return VelocityVerlet(atoms, timestep=timestep_ase)

    # NVT ensemble
    elif ensemble == 'nvt':
        thermostat = thermostat.lower()

        if thermostat == 'langevin':
            return Langevin(
                atoms,
                timestep=timestep_ase,
                temperature_K=temperature,
                friction=friction / units.fs  # Convert to ASE units
            )

        elif thermostat == 'nose-hoover':
            if not NOSEHOOVER_AVAILABLE:
                raise ImportError("Nosé-Hoover thermostat not available in this ASE version")
            return NoseHoover(
                atoms,
                timestep=timestep_ase,
                temperature_K=temperature,
                ttime=ttime * units.fs
            )

        elif thermostat == 'berendsen':
            return NVTBerendsen(
                atoms,
                timestep=timestep_ase,
                temperature_K=temperature,
                taut=taut * units.fs
            )

        else:
            raise ValueError(f"Unknown thermostat: {thermostat}. Use 'langevin', 'nose-hoover', or 'berendsen'")

    # NPT ensemble
    elif ensemble == 'npt':
        barostat = barostat.lower()

        # Convert pressure from GPa to ASE units (eV/Å³)
        # 1 GPa = 0.006241509 eV/Å³
        pressure_ase = pressure * 0.006241509
        externalstress = pressure_ase * np.ones(6)  # Isotropic pressure

        if barostat == 'npt':
            # Auto-calculate pfactor if not provided
            if pfactor is None:
                pfactor = (ttime * 75 * units.GPa) ** 2

            return NPT(
                atoms,
                timestep=timestep_ase,
                temperature_K=temperature,
                externalstress=externalstress,
                ttime=ttime * units.fs,
                pfactor=pfactor
            )

        elif barostat == 'berendsen':
            return NPTBerendsen(
                atoms,
                timestep=timestep_ase,
                temperature_K=temperature,
                taut=taut * units.fs,
                pressure_au=pressure_ase,
                taup=taup * units.fs,
                compressibility_au=compressibility / units.GPa
            )

        else:
            raise ValueError(f"Unknown barostat: {barostat}. Use 'npt' or 'berendsen'")

    else:
        raise ValueError(f"Unknown ensemble: {ensemble}. Use 'nve', 'nvt', or 'npt'")


def run_md(atoms, ensemble='nvt', thermostat='langevin', barostat='npt',
           temperature=300, pressure=0.0, timestep=1.0,
           friction=0.01, ttime=25.0, pfactor=None,
           taut=100.0, taup=1000.0, compressibility=4.57e-5,
           steps=1000, interval=1, log_path=None, output_dir=".", model_name="mlip"):

    """
    Run molecular dynamics simulation.

    Parameters
    ----------
    atoms : ase.Atoms
        Atoms object with calculator attached
    ensemble : str
        Ensemble type: 'nve', 'nvt', 'npt'
    ... (other parameters as in setup_dynamics)
    steps : int
        Number of MD steps
    interval : int
        Logging interval
    log_path : str or None
        Path to log file
    output_dir : str or Path
        Output directory
    model_name : str
        MLIP model name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    traj_file = output_path / "md.traj"
    csv_file = output_path / "md_energy.csv"
    energy_plot = output_path / "md_energy.png"
    temp_plot = output_path / "md_temperature.png"

    # Setup dynamics
    dyn = setup_dynamics(
        atoms, ensemble=ensemble, thermostat=thermostat, barostat=barostat,
        temperature=temperature, pressure=pressure, timestep=timestep,
        friction=friction, ttime=ttime, pfactor=pfactor,
        taut=taut, taup=taup, compressibility=compressibility
    )

    # Trajectory writing
    traj_writer = Trajectory(str(traj_file), 'w', atoms)
    dyn.attach(traj_writer.write, interval=interval)

    # Terminal logger (with stress for NPT)
    stress_log = (ensemble == 'npt')
    dyn.attach(MDLogger(dyn, atoms, sys.stdout, header=True, stress=stress_log), interval=interval)
    if log_path:
        dyn.attach(MDLogger(dyn, atoms, log_path, header=True, stress=stress_log), interval=interval)

    # Data for CSV and plots
    log_data = {
        "step": [],
        "time(fs)": [],
        "temperature(K)": [],
        "total_energy(eV)": [],
        "potential_energy(eV)": [],
        "kinetic_energy(eV)": []
    }

    # Add pressure and volume tracking for NPT
    if ensemble == 'npt':
        log_data["pressure(GPa)"] = []
        log_data["volume(A^3)"] = []

    def log_properties():
        step = dyn.get_number_of_steps()
        log_data["step"].append(step)
        log_data["time(fs)"].append(step * timestep)
        log_data["temperature(K)"].append(atoms.get_temperature())
        log_data["total_energy(eV)"].append(atoms.get_total_energy())
        log_data["potential_energy(eV)"].append(atoms.get_potential_energy())
        log_data["kinetic_energy(eV)"].append(atoms.get_kinetic_energy())

        if ensemble == 'npt':
            # Get pressure from stress tensor
            stress = atoms.get_stress(voigt=False)  # 3x3 tensor
            pressure_ase = -stress.trace() / 3.0  # eV/Å³
            pressure_gpa = pressure_ase / 0.006241509  # Convert to GPa
            log_data["pressure(GPa)"].append(pressure_gpa)
            log_data["volume(A^3)"].append(atoms.get_volume())

    dyn.attach(log_properties, interval=interval)
    dyn.run(steps)
    traj_writer.close()

    # Save to CSV
    df = pd.DataFrame(log_data)
    df.to_csv(csv_file, index=False)

    # Plot energy
    plt.figure(figsize=(8, 5))
    plt.plot(df["time(fs)"], df["total_energy(eV)"], label="Total", linewidth=1.5)
    plt.plot(df["time(fs)"], df["potential_energy(eV)"], label="Potential", alpha=0.7)
    plt.plot(df["time(fs)"], df["kinetic_energy(eV)"], label="Kinetic", alpha=0.7)
    plt.xlabel("Time (fs)")
    plt.ylabel("Energy (eV)")
    plt.title(f"MD Energy ({ensemble.upper()})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(energy_plot, dpi=150)
    plt.close()

    # Plot temperature
    plt.figure(figsize=(8, 5))
    plt.plot(df["time(fs)"], df["temperature(K)"], color="orange", linewidth=1.5)
    if ensemble in ['nvt', 'npt']:
        plt.axhline(y=temperature, color='r', linestyle='--', label=f'Target: {temperature} K', alpha=0.5)
    plt.xlabel("Time (fs)")
    plt.ylabel("Temperature (K)")
    plt.title(f"MD Temperature ({ensemble.upper()})")
    if ensemble in ['nvt', 'npt']:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(temp_plot, dpi=150)
    plt.close()

    # Plot pressure and volume for NPT
    if ensemble == 'npt':
        pressure_plot = output_path / "md_pressure.png"
        volume_plot = output_path / "md_volume.png"

        # Pressure plot
        plt.figure(figsize=(8, 5))
        plt.plot(df["time(fs)"], df["pressure(GPa)"], color="blue", linewidth=1.5)
        plt.axhline(y=pressure, color='r', linestyle='--', label=f'Target: {pressure} GPa', alpha=0.5)
        plt.xlabel("Time (fs)")
        plt.ylabel("Pressure (GPa)")
        plt.title("MD Pressure (NPT)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pressure_plot, dpi=150)
        plt.close()

        # Volume plot
        plt.figure(figsize=(8, 5))
        plt.plot(df["time(fs)"], df["volume(A^3)"], color="green", linewidth=1.5)
        plt.xlabel("Time (fs)")
        plt.ylabel("Volume (Å³)")
        plt.title("MD Volume (NPT)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(volume_plot, dpi=150)
        plt.close()
