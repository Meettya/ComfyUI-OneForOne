from numpy import ndarray
from typing import Tuple


def get_latent_size(latent: dict, scale_factor: int = 8) -> Tuple[int, int]:
    """Convert latent dimensions to pixel dimensions using scale factor."""
    return (latent["samples"].shape[2] * scale_factor, latent["samples"].shape[3] * scale_factor)


def get_image_size(image: ndarray) -> Tuple[int, int]:
    """Get image height and width from numpy array."""
    return (image.shape[1], image.shape[2])
