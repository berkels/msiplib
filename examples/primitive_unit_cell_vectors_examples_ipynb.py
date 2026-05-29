# %%
"""
This notebok shows examples of finding the primitive unit cell vectors using msiplib.find_real_space_vectors function

The images featured here are from ./images directory

Run this notebook in the msiplib/examples directory

The output files will be written in msiplib/examples directory
"""

# %%
import imageio
from msiplib.unit_cell_from_real_space import get_primitive_unit_cell_vectors
from msiplib.io import read_image
from msiplib.emic import generate_crystal_image
import numpy as np

msip_dir = "./"
output_dir = "./"

# %%
"""### Example 1: ### 
find and view the primitive unit cell vectors"""

# %%
bumps3 = read_image(msip_dir + "images/bumps3.nc")
get_primitive_unit_cell_vectors(bumps3, output_dir, "bumps3", write_to_file=False, plot_final_vectors=True)

# %%
"""## Example 2: ### 
minimal working example"""

# %%
v = np.array([[20.5, 0], [0, 28.8]])
im = generate_crystal_image(v)
v1, v2 = get_primitive_unit_cell_vectors(im, output_dir, "im")
v1, v2

# %%
"""### Example 3: ### 
Bright field mode"""

# %%
img = read_image(msip_dir + "images/grain_1_cropped.png")
img_name = "grain_1"
v1, v2 = get_primitive_unit_cell_vectors(
    img, output_dir, img_name, write_to_file=False, imaging_mode="BF", plot_final_vectors=True
)
print(v1, v2)

# %%
"""### Example 4: ### 
input image with .nc format """

# %%
im4 = read_image(msip_dir + "images/Nc3Nm.nc")
get_primitive_unit_cell_vectors(im4, output_dir, "Nc3Nm", write_to_file=False, plot_final_vectors=False)
v1, v2

# %%
"""### Example 5: ### 
Latex output file"""

# %%
im6 = read_image(msip_dir + "images/hexVacancy.nc")
get_primitive_unit_cell_vectors(im6, output_dir, "hexVacancy", write_to_file=True)
