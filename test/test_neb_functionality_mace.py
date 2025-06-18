from ase.io import read
from mlip_platform.neb import CustomNEB

def test_neb_run_mace():
    initial_path = "/Users/leeyuanzhang/Documents/mlip-platform-(NEB)/test/POSCAR_initial"
    final_path = "/Users/leeyuanzhang/Documents/mlip-platform-(NEB)/test/POSCAR_final"

    initial = read(initial_path, format="vasp")
    final = read(final_path, format="vasp")

    neb = CustomNEB(
        initial=initial,
        final=final,
        num_images=5,
        interp_fmax=0.1,
        interp_steps=100,
        fmax=0.05,
        mlip="mace"
    )

    assert len(neb.images) == 5

    # Interpolate images with IDPP
    neb.interpolate_idpp()

    for img in neb.images[1:-1]:
        assert img.get_positions().shape == initial.get_positions().shape
        assert not any(p is None for p in img.get_positions().flatten())

    # Run NEB to assign calculator and relax path
    neb.run_neb()

    # Validate energy extraction
    for img in neb.images:
        energy = img.get_potential_energy()
        assert isinstance(energy, float)
