

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from mlip_platform.core.neb import CustomNEB

def test_highly_constrained_neb():
    # 1. Create simple initial and final structures (3 atoms)
    # 0: Fixed at 0,0,0 -> 1,1,1
    # 1: Relaxed (User specified)
    # 2: Fixed at 2,2,2 -> 3,3,3
    
    initial = Atoms('H3', positions=[[0, 0, 0], [0, 0, 0], [2, 2, 2]], cell=[10, 10, 10])
    final = Atoms('H3', positions=[[1, 1, 1], [0, 0, 0], [3, 3, 3]], cell=[10, 10, 10])
    
    # We want to relax atom 1 ONLY.
    relax_atoms = [1]
    
    # Initialize CustomNEB with relax_atoms
    # num_images=1 means total 3 images: Initial -> Image 1 -> Final
    neb = CustomNEB(initial, final, num_images=1, mlip='mace', relax_atoms=relax_atoms)
    
    # 2. Verify Interpolation (Linear)
    # Image 1 should be exactly halfway between Initial and Final for ALL atoms initially
    # Initial: [0,0,0], [0,0,0], [2,2,2]
    # Final:   [1,1,1], [0,0,0], [3,3,3]
    # Expected Image 1: [0.5, 0.5, 0.5], [0,0,0], [2.5, 2.5, 2.5]
    
    image1 = neb.images[1]
    positions = image1.get_positions()
    expected_pos_0 = np.array([0.5, 0.5, 0.5])
    expected_pos_2 = np.array([2.5, 2.5, 2.5])
    
    assert np.allclose(positions[0], expected_pos_0), f"Atom 0 pos {positions[0]} != {expected_pos_0}"
    assert np.allclose(positions[2], expected_pos_2), f"Atom 2 pos {positions[2]} != {expected_pos_2}"
    
    # 3. Verify Constraints
    # Atoms 0 and 2 should be fixed. Atom 1 should NOT be fixed.
    constraints = image1.constraints
    assert len(constraints) > 0
    
    # We expect one FixAtoms constraint covering indices [0, 2]
    found_correct_constraint = False
    for c in constraints:
        if isinstance(c, FixAtoms):
            indices = c.get_indices()
            if set(indices) == {0, 2}:
                found_correct_constraint = True
                break
    
    assert found_correct_constraint, "Did not find FixAtoms constraint on indices [0, 2]"
    
    # 4. Verify IDPP Skipping
    # We call interpolate_idpp and expect positions strictly NOT to change from linear
    # (Mocking logging/print might be hard, but we can check positions remain linear)
    neb.interpolate_idpp()
    
    positions_after = image1.get_positions()
    assert np.allclose(positions_after, positions), "IDPP should have been skipped, but positions changed!"
    
    print("test_highly_constrained_neb passed!")

if __name__ == "__main__":
    test_highly_constrained_neb()
