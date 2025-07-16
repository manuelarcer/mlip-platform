import sys
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md import MDLogger
from ase.io.trajectory import Trajectory
from ase import units
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def initialize_md(atoms, temperature_K=298, timestep_fs=2.0):
    """Initialize velocity and integrator."""
    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
    dyn = VelocityVerlet(atoms, timestep=timestep_fs * units.fs)
    return dyn

def run_md(atoms, log_path=None, temperature_K=298, timestep_fs=2.0, steps=10,
           interval=1, stress=False, output_dir="md_result", model_name="mlip"):
    
    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    traj_file = output_path / "md.traj"
    csv_file = output_path / "md_energy.csv"
    energy_plot = output_path / "md_energy.png"
    temp_plot = output_path / "md_temperature.png"

    dyn = initialize_md(atoms, temperature_K=temperature_K, timestep_fs=timestep_fs)

    # Attach trajectory writer
    traj_writer = Trajectory(str(traj_file), 'w', atoms)
    dyn.attach(traj_writer.write, interval=interval)

    # Attach logger to terminal and optional file
    dyn.attach(MDLogger(dyn, atoms, sys.stdout, header=True, stress=stress), interval=interval)
    if log_path:
        dyn.attach(MDLogger(dyn, atoms, log_path, header=True, stress=stress), interval=interval)

    # Collect data for CSV and plot
    log_data = {"step": [], "time(fs)": [], "temperature(K)": [], "total_energy(eV)": []}

    def log_properties():
        step = dyn.get_number_of_steps()
        time_fs = step * timestep_fs
        temp = atoms.get_temperature()
        energy = atoms.get_total_energy()
        log_data["step"].append(step)
        log_data["time(fs)"].append(time_fs)
        log_data["temperature(K)"].append(temp)
        log_data["total_energy(eV)"].append(energy)

    dyn.attach(log_properties, interval=interval)

    dyn.run(steps)

    # Save CSV
    df = pd.DataFrame(log_data)
    df.to_csv(csv_file, index=False)

    # Plot energy vs time
    plt.figure()
    plt.plot(df["time(fs)"], df["total_energy(eV)"], label="Energy (eV)")
    plt.xlabel("Time (fs)")
    plt.ylabel("Total Energy (eV)")
    plt.title("MD: Energy vs. Time")
    plt.grid(True)
    plt.savefig(energy_plot)
    plt.close()

    # Plot temperature vs time
    plt.figure()
    plt.plot(df["time(fs)"], df["temperature(K)"], label="Temperature (K)", color="orange")
    plt.xlabel("Time (fs)")
    plt.ylabel("Temperature (K)")
    plt.title("MD: Temperature vs. Time")
    plt.grid(True)
    plt.savefig(temp_plot)
    plt.close()
