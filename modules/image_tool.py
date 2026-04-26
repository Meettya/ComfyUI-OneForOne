from numpy import ndarray
from typing import Union

from .util import get_latent_size, get_image_size

SCALE_8X = "SD, Qwen (8x)"
SCALE_16X = "Flux2 (16x)"
SCALE_OPTIONS = [SCALE_8X, SCALE_16X]
SCALE_FACTOR_MAP = {
    SCALE_8X: 8,
    SCALE_16X: 16,
}


class OFOImageFit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "image": ("IMAGE",),
                "latent_scale": (SCALE_OPTIONS, {"default": SCALE_8X}),
            },
            "optional": {
                "width_position": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "number",
                        "lazy": True,
                    },
                ),
                "height_position": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.001,
                        "display": "number",
                        "lazy": True,
                    },
                ),
                "scale": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                        "display": "number",
                        "lazy": True,
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "IMAGE_WIDTH",
        "IMAGE_HEIGHT",
        "PAD_TOP",
        "PAD_RIGHT",
        "PAD_BOTTOM",
        "PAD_LEFT",
    )

    FUNCTION = "calculate"
    CATEGORY = "OneForOne"

    def calculate(
        self,
        latent: Union[dict, ndarray],
        image: ndarray,
        latent_scale: str,
        width_position: float = 0.5,
        height_position: float = 0.5,
        scale: float = 1.0,
    ):
        """
        Calculate new dimensions and paddings for image, based on latent and given parameters.

        Args:
            latent (np.ndarray or dict): Latent representation or dictionary containing latent samples.
            image (np.ndarray): Input image to be resized and padded.
            latent_scale (str): Latent scale factor option. "SD, Qwen (8x)" or "Flux2 (16x)".
            width_position (float, optional): Horizontal position where the image will be placed within the latent space. Defaults to 0.5.
            height_position (float, optional): Vertical position where the image will be placed within the latent space. Defaults to 0.5.
            scale (float, optional): Scaling factor applied to the image before resizing. Defaults to 1.0.

        Returns:
            tuple: A tuple containing six integers representing the new width, height, top, right, bottom, and left padding values.
        """
        scale_factor = SCALE_FACTOR_MAP[latent_scale]

        latent_height, latent_width = get_latent_size(latent, scale_factor)
        image_height, image_width = get_image_size(image)

        # Compute base scaling (fit inside latent)
        scale_w = latent_width / image_width
        scale_h = latent_height / image_height
        min_scale = min(scale_w, scale_h) * scale

        # Initial candidate dimensions (floored to avoid exceeding latent)
        raw_width = int(image_width * min_scale)
        raw_height = int(image_height * min_scale)

        # Ensure raw dimensions do not exceed latent bounds (safety)
        raw_width = min(raw_width, latent_width)
        raw_height = min(raw_height, latent_height)

        # ----- Adjust to multiples of 8 -----
        # We prefer rounding down to the nearest multiple of 8 (keep inside latent).
        # But we also need to ensure we don't lose too much size.
        # Let's find the largest multiple of 8 <= raw_width and <= raw_height.
        target_width = (raw_width // 8) * 8
        target_height = (raw_height // 8) * 8

        # However, if the difference is too large (more than 7 pixels), we could try rounding up if still fits.
        # But to be safe and keep exact fit, we just round down.
        # If rounding down reduces size too much, the user can increase `scale`.
        new_image_width = target_width
        new_image_height = target_height

        # ----- Recompute paddings based on final aligned sizes -----
        # Horizontal padding
        remaining_width = latent_width - new_image_width
        padding_left = int(remaining_width * width_position)
        padding_right = remaining_width - padding_left

        # Vertical padding
        remaining_height = latent_height - new_image_height
        padding_top = int(remaining_height * height_position)
        padding_bottom = remaining_height - padding_top

        return (
            new_image_width,
            new_image_height,
            padding_top,
            padding_right,
            padding_bottom,
            padding_left,
        )
