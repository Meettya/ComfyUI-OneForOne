from numpy import ndarray
from typing import Union, Tuple

scale = 8

def get_sizes(target: Union[dict, ndarray]) -> Tuple[int, int]:
    if isinstance(target, dict) and "samples" in target:
        # Latent height, width
        return (
            target["samples"].shape[2] * scale,
            target["samples"].shape[3] * scale
        )
    else:
        # Image height, width
        return (
            target.shape[1],
            target.shape[2]
        )
