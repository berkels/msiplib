#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples of the crystalline motif extraction using :py:meth:`get_motif <msiplib.motif.get_motif>`

run this code in the msiplib/examples directory

output files will be written in the msiplib/examples directory
"""

from msiplib.motif import get_motif
import numpy as np
from msiplib.emic import generate_crystal_image
from msiplib.io import read_image

# example 1
f1 = generate_crystal_image(np.array([[20.5, 15], [0, 22.8]]))
name1 = "gt_"
get_motif(f1, name1)

# example 2
f2 = bumps3 = read_image("./images/bumps3.nc")
name2 = "bumps3_"
get_motif(f2, name2)
