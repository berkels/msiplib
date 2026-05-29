# %%
'''The is an example of the usage of get_motif_atoms_dof

run this code in the msiplib/examples folder and be aware that some files will be created'''

# %%
from msiplib.io import read_image
from msiplib.motif_atoms_dof import get_motif_atoms_dof
from skimage.exposure import rescale_intensity
import jax
from pdf2image import convert_from_path

# %%
np_mod = jax.numpy
f = rescale_intensity(read_image("./images/right_grain.png"), out_range=(0.0, 1.0))
name = "right_grain"
n = 2
compute_uv = True
erase_inf_radius = 20
initial_diameter = 16
image_path = "./"
output_dir = "./"
v1, v2 = None,None
num_sigma = 4.
separation = None

# %%
g,v = get_motif_atoms_dof(f, name, output_dir, n, v1, v2, compute_uv, np_mod, num_sigma, initial_diameter, erase_inf_radius, separation,plot=False, max_iter=50000)

# %%
img = convert_from_path("/home/amel/msiplib/examples/right_grain_motif_results.pdf")

# %%
''' ### The results: '''

# %%
''' Input image'''

# %%
img[0]

# %%
''' Model image from motif (reconstruction)'''

# %%
img[1]

# %%
''' Normalized reconstruction'''

# %%
img[2]

# %%
''' The input image (background) with the extracted motif (red parallelogram) and the periodic atomic positions'''

# %%
img[3]

# %%
''' The primitive unit cell vectors'''

# %%
v

# %%
''' The atom positions'''

# %%
g[:,:2]

# %%
''' Motif atoms Gaussian fitting parameters'''

# %%
g
