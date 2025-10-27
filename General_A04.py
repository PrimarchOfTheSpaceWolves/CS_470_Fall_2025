###############################################################################
# IMPORTS
###############################################################################

import os
from enum import Enum

class LBP_LABEL_TYPES(Enum):  
    UNIFORM = "Uniform"
    UNIFORM_ROT = "UniformRot"
    FULL = "Full"    

base_dir = "assign04"
image_dir = os.path.join(base_dir, "images")
ground_dir = os.path.join(base_dir, "ground")
out_dir = os.path.join(base_dir, "output")

inputImageFilenames = [
    "Image0.png",
    "Image1.png",
    "Image2.png",
    "Image3.png",
    "Image4.png",
    "Image5.png",
    "Image6.png"
]

