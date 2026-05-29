#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An example of calculating the angle between two neighboring grains (images/left_grain.png and images/right_grain.png) using get_angle_between_similar_grains function from msiplib.grain_angle
"""

# run this example in msiplib/examples directory
from msiplib.grain_angle import get_angle_between_similar_grains
from msiplib.io import read_image

img_name = "experimental_data"
path = "./"
left_segment = read_image("./images/left_grain.png")
right_segment = read_image("./images/right_grain.png")
get_angle_between_similar_grains(left_segment, right_segment, path, img_name, accuracy=2, read_vectors=False)
