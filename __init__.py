from .modules.image_tool import OFOImageFit

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "OFO Image Fit":OFOImageFit
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OFO Image Fit" : "Image Fit Calculator"
}
