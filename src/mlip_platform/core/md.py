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

    # Write to trajectory file
    traj_writer = Trajectory(str(traj_file), 'w', atoms)
    dyn.attach(traj_writer.write, interval=interval)

    # Live logging to terminal
    dyn.attach(MDLogger(dyn, atoms, sys.stdout, header=True, stress=stress), interval=interval)

    # Optional logging to file
    if log_path:
        dyn.attach(MDLogger(dyn, atoms, log_path, header=True, stress=stress), interval=interval)

    # Custom logging for CSV + plot
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

    # Run simulation
    dyn.run(steps)

    # Save data to CSV
    df = pd.DataFrame(log_data)
    df.to_csv(csv_file, index=False)

    # Plot energy
    plt.figure()
    plt.plot(df["step"], df["total_energy(eV)"], label="Total Energy (eV)")
    plt.xlabel("Step")
    plt.ylabel("Energy (eV)")
    plt.title("MD: Energy vs. Step")
    plt.grid(True)
    plt.savefig(energy_plot)
    plt.close()

    # Plot temperature
    plt.figure()
    plt.plot(df["step"], df["temperature(K)"], label="Temperature (K)", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Temperature (K)")
    plt.title("MD: Temperature vs. Step")
    plt.grid(True)
    plt.savefig(temp_plot)
    plt.close()
