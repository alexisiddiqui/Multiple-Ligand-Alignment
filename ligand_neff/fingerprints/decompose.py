from rdkit import Chem
from rdkit.Chem import AllChem
import jax.numpy as jnp
from jaxtyping import Float, Array
import chex


class AtomDecomposition:
    """Contains atom to bit mapping logic for a single molecule."""
    
    def __init__(self, mol: Chem.Mol, radius: int, fp_size: int, info: dict[int, tuple[tuple[int, int], ...]]):
        self.mol = mol
        self.radius = radius
        self.fp_size = fp_size
        self.info = info

    def build_atom_bit_mask(self) -> Float[Array, "n_atoms fp_size"]:
        """
        Builds the Boolean mask representing which atoms contributed to which fingerprint bits.
        Returns a float array to avoid casting inside the JIT'd `per_atom_neff` kernels.
        """
        import numpy as np
        
        n_atoms = self.mol.GetNumAtoms()
        # Initialize as float32 to match JAX computation dtype expectations
        # Use simple numpy since this is precomputation on the CPU 
        mask = np.zeros((n_atoms, self.fp_size), dtype=np.float32)

        for bit, envs in self.info.items():
            for env in envs:
                center_atom_idx = env[0]
                # We could set bits for all atoms in the environment (if we calculate shortest paths),
                # but standard Morgan atom-attribution maps bits to their center atom.
                mask[center_atom_idx, bit] = 1.0
                
        return mask


def decompose(mol: Chem.Mol, radius: int, fp_size: int = 2048, use_chirality: bool = False, use_features: bool = False) -> AtomDecomposition:
    """Decompose molecule into an atom-to-bit mapping at the given Morgan radius."""
    info = {}
    
    # Generate Morgan fingerprint with bit info
    _ = AllChem.GetMorganFingerprintAsBitVect(
        mol, 
        radius=radius,
        nBits=fp_size,
        useChirality=use_chirality,
        useFeatures=use_features,
        bitInfo=info
    )
    
    return AtomDecomposition(mol, radius, fp_size, info)
