from rdkit import Chem
from pathlib import Path

def load_query(path: str | Path) -> Chem.Mol:
    """Load query molecule from SDF."""
    suppl = Chem.SDMolSupplier(str(path))
    for mol in suppl:
        if mol is not None:
            return mol
    raise ValueError(f"No valid molecule found in {path}")
