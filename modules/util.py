from numpy import ndarray
from typing import Union, Tuple

# latent to pixel scale factor
SCALE = 8


def get_sizes(target: Union[dict, ndarray]) -> Tuple[int, int]:
    if isinstance(target, dict) and "samples" in target:
        # Latent height, width
        return (target["samples"].shape[2] * SCALE, target["samples"].shape[3] * SCALE)
    else:
        # Image height, width
        return (target.shape[1], target.shape[2])
