from dataclasses import dataclass
from typing import Literal
import yaml
from pathlib import Path

VALID_FP_SIZES = (2048, 4096, 8192)

@dataclass
class NeffConfig:
    """All tunable parameters for a Neff computation."""

    # ── Fingerprint ──────────────────────────────────────────────
    fp_radii: tuple[int, ...] = (1, 2, 3)
    fp_size: int = 2048                         # Must be in VALID_FP_SIZES
    use_chirality: bool = False
    use_features: bool = False                  # FCFP-style if True

    # ── Reference filtering ──────────────────────────────────────
    tanimoto_inclusion: float = 0.3
    max_references: int = 10_000                # STATIC shape for JAX

    # ── Weighting ────────────────────────────────────────────────
    cluster_threshold: float = 0.7
    weighting: Literal["inverse_degree", "none"] = "inverse_degree"
    inverse_degree_chunk_size: int = 2048             # Rows per chunk in pairwise

    # ── Per-atom Neff ────────────────────────────────────────────
    coverage_metric: Literal["overlap", "binary"] = "overlap"
    min_overlap: float = 0.5

    # ── Aggregation ──────────────────────────────────────────────
    aggregation: Literal["geometric", "minimum", "mean"] = "geometric"
    radius_weights: tuple[float, ...] = (0.2, 0.5, 0.3)

    # ── Normalisation ────────────────────────────────────────────
    atom_norm: Literal["none", "q_length", "sqrt(q_length)"] = "sqrt(q_length)"
    lambda_mode: Literal["fixed", "adaptive"] = "adaptive"
    lambda_fixed: float = 10.0                  # Only used if lambda_mode="fixed"
    lambda_quantile: float = 0.5                # Median of global Neff for adaptive

    def __post_init__(self):
        if self.fp_size not in VALID_FP_SIZES:
            raise ValueError(
                f"fp_size must be one of {VALID_FP_SIZES}, got {self.fp_size}. "
                f"Larger sizes reduce bit collisions but increase memory. "
                f"2048 ≈ ECFP standard, 4096 recommended for per-atom work, "
                f"8192 for maximum precision."
            )
        if len(self.radius_weights) != len(self.fp_radii):
            raise ValueError(
                f"radius_weights length ({len(self.radius_weights)}) must match "
                f"fp_radii length ({len(self.fp_radii)})"
            )
        if self.max_references > 50_000:
            import warnings
            warnings.warn(
                f"max_references={self.max_references} will allocate "
                f"~{self.max_references**2 * 4 / 1e9:.1f} GB for Inverse Degree "
                f"pairwise matrix. Consider reducing or using weighting='none'."
            )

def load_config(path: str | Path) -> NeffConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    
    # Flatten nested YAML structure to match flat dataclass fields
    flat_data = {}
    for section in data.values():
        if isinstance(section, dict):
            flat_data.update(section)
        else:
            flat_data.update(data)
            break
            
    if "fp_radii" in flat_data:
        flat_data["fp_radii"] = tuple(flat_data["fp_radii"])
    if "radius_weights" in flat_data:
        flat_data["radius_weights"] = tuple(flat_data["radius_weights"])
        
    return NeffConfig(**flat_data)
