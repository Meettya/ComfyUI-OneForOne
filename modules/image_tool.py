from numpy import ndarray
from typing import Union

from .util import get_sizes


class OFOImageFit:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"latent": ("LATENT",), "image": ("IMAGE",)},
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
        width_position: float = 0.5,
        height_position: float = 0.5,
        scale: float = 1.0,
    ):
        """
        Calculate new dimensions and paddings for image, based on latent and given parameters.

        Args:
            latent (np.ndarray or dict): Latent representation or dictionary containing latent samples.
            image (np.ndarray): Input image to be resized and padded.
            width_position (float, optional): Horizontal position where the image will be placed within the latent space. Defaults to 0.5.
            height_position (float, optional): Vertical position where the image will be placed within the latent space. Defaults to 0.5.
            scale (float, optional): Scaling factor applied to the image before resizing. Defaults to 1.0.

        Returns:
            tuple: A tuple containing six integers representing the new width, height, top, right, bottom, and left padding values.
        """
        # get sizes
        latent_height, latent_width = get_sizes(latent)
        image_height, image_width = get_sizes(image)

        # get minimum scale
        scale_w = latent_width / image_width
        scale_h = latent_height / image_height
        min_scale = min(scale_w, scale_h) * scale

        # scale image
        new_image_width = round(image_width * min_scale)
        new_image_height = round(image_height * min_scale)

        # calculate padding
        padding_left = round((latent_width - new_image_width) * width_position)
        padding_right = latent_width - new_image_width - padding_left
        padding_bottom = round((latent_height - new_image_height) * height_position)
        padding_top = latent_height - new_image_height - padding_bottom

        return (
            new_image_width,
            new_image_height,
            padding_top,
            padding_right,
            padding_bottom,
            padding_left,
        )
