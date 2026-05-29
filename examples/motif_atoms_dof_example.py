#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The is an example of the usage of met_motif_atoms_dof

run this code in the msiplib/examples folder and be aware that some files will be created
"""

from msiplib.io import read_image
from msiplib.motif_atoms_dof import get_motif_atoms_dof
from skimage.exposure import rescale_intensity
import jax

np_mod = jax.numpy
f = rescale_intensity(read_image("./images/right_grain.png"), out_range=(0.0, 1.0))
name = "right_grain"
n = 2
compute_uv = True
erase_inf_radius = 8
initial_diameter = 6
image_path = "./"
output_dir = "./"
v1, v2 = None, None
num_sigma = 4.0
separation = None


get_motif_atoms_dof(
    f,
    name,
    output_dir,
    n,
    v1,
    v2,
    compute_uv,
    np_mod,
    num_sigma,
    initial_diameter,
    erase_inf_radius,
    separation,
    plot=False,
    max_iter=50000,
)[0]
