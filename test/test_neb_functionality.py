from ase.io import read
from ase.mep.neb import idpp_interpolate  # optional fallback
from mlip_platform.neb import CustomNEB

def test_neb_run():
    # Absolute paths to your POSCAR files
    initial_path = "/Users/leeyuanzhang/Documents/mlip-platform-(NEB)/test/POSCAR_initial"
    final_path = "/Users/leeyuanzhang/Documents/mlip-platform-(NEB)/test/POSCAR_final"

    # Read initial and final structures
    initial = read(initial_path, format="vasp")
    final = read(final_path, format="vasp")

    # Set up NEB calculation with 5 images
    neb = CustomNEB(
        initial=initial,
        final=final,
        num_images=5,
        interp_fmax=0.1,
        interp_steps=100,
        fmax=0.05,
        mlip="sevenn-mf-ompa"
    )

    # Check number of images (should be 5)
    assert len(neb.images) == 5

    # Run IDPP interpolation
    neb.interpolate_idpp()

    # Sanity check: positions are valid and same shape
    for img in neb.images[1:-1]:
        assert img.get_positions().shape == initial.get_positions().shape
        assert not any(p is None for p in img.get_positions().flatten())
