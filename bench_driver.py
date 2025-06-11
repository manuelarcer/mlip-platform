# bench_driver.py

import json, sys, time
from ase.io import read

mlip_name = sys.argv[2]
structure = sys.argv[1]

if mlip_name == "mace":
    from mace.calculators import mace_mp
    calc = mace_mp(model="medium", device="cpu")

elif mlip_name == "sevenn":
    from sevenn.calculator import SevenNetCalculator
    calc = SevenNetCalculator("7net-mf-ompa", modal="mpa")

else:
    raise ValueError(f"Unknown MLIP: {mlip_name}")

atoms = read(structure)
atoms.calc = calc

t0 = time.perf_counter()
energy = atoms.get_potential_energy()
t1 = time.perf_counter()

print(json.dumps({
    "mlip": mlip_name,
    "energy": energy,
    "time": t1 - t0
}))
