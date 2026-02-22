from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import io
import PIL.Image


def plot_atom_neff(
    mol: Chem.Mol,
    confidence: np.ndarray,          # (n_atoms,) — ALWAYS normalised [0, 1]
    size: tuple[int, int] = (600, 400),
    cmap_name: str = "RdYlGn",
    show_values: bool = False,
) -> PIL.Image.Image:
    """
    Render 2D depiction with atoms coloured by confidence score.

    IMPORTANT: This function takes confidence (0-1), not raw Neff.
    This guarantees:
    - Consistent colour scaling across molecules and databases
    - No outlier sensitivity (raw Neff of 500 vs 10 would break colour maps)
    - Intuitive interpretation: red = low confidence, green = high

    Uses RDKit's atom highlight-based colouring for clean rendering.
    """
    # Use matplotlib's new colormap API
    cmap = plt.get_cmap(cmap_name)

    # Build atom colour map: atom_idx → (r, g, b)
    atom_colors = {}
    for i in range(mol.GetNumAtoms()):
        rgba = cmap(float(confidence[i]))
        atom_colors[i] = tuple(rgba[:3])

    # Build radius map proportional to confidence for visual weight
    atom_radii = {}
    for i in range(mol.GetNumAtoms()):
        atom_radii[i] = 0.3 + 0.3 * float(confidence[i])  # 0.3 to 0.6

    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    # Set atom indices if requested
    if show_values:
        opts = drawer.drawOptions()
        opts.addAtomIndices = True

    drawer.DrawMoleculeWithHighlights(
        mol,
        "",
        atom_colors,
        {},       # bond colours
        atom_radii,
        {},       # bond radii
    )
    drawer.FinishDrawing()

    bio = io.BytesIO(drawer.GetDrawingText())
    return PIL.Image.open(bio)


def plot_confidence_bar(
    mol: Chem.Mol,
    confidence: np.ndarray,
    neff_per_radius: dict[int, np.ndarray] | None = None,
) -> plt.Figure:
    """
    Bar chart of per-atom confidence with optional radius breakdown.

    Shows each heavy atom on x-axis with its element symbol and index,
    stacked/grouped bars for each radius if neff_per_radius provided.
    Useful for detailed analysis beyond the 2D depiction.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    n_atoms = mol.GetNumAtoms()
    indices = np.arange(n_atoms)
    
    labels = [f"{mol.GetAtomWithIdx(i).GetSymbol()}{i}" for i in range(n_atoms)]
    
    if neff_per_radius is None:
        ax.bar(indices, confidence, color="skyblue")
    else:
        # Simplistic grouped bar implementation for demo purposes
        num_radii = len(neff_per_radius)
        width = 0.8 / num_radii
        for i, (r, vals) in enumerate(sorted(neff_per_radius.items())):
            ax.bar(indices + i * width - 0.4 + width/2, vals, width=width, label=f"Radius {r}")
        ax.legend()
        
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Confidence / Neff")
    ax.set_title("Per-Atom Confidence")
    fig.tight_layout()
    return fig
