#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This is an example of finding the primitive unit cell vectors using msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors function

This file should be run from the terminal in examples directory.

The image featured is grian_1_cropped.png in the ./images directory

The vectors are plotted in <current directory>/grain_1_optimized.png and saved in <current directory>/grain_1_vectors.npz
"""

from msiplib.unit_cell_from_real_space import get_primitive_unit_cell_vectors
from msiplib.io import read_image

img = read_image("./images/grain_1_cropped.png")
img_name = "grain_1"
path = "./"
v1, v2 = get_primitive_unit_cell_vectors(
    img, path, img_name, write_to_file=False, plot_final_vectors=True, imaging_mode="BF"
)
print(v1, v2)
