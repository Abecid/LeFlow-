"""Training-free, native-LeFlow latent-path refinement.

Importing this package requires only PyTorch, not simulator/training dependencies.
"""
from .core import RefinementConfig, Work, sample_interiors

__all__ = ["RefinementConfig", "Work", "sample_interiors"]
