#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
import jax
import argparse
from datetime import datetime
import getpass
import json
from pathlib import Path, PurePath
from skimage.exposure import rescale_intensity
from skimage import restoration
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from msiplib.motif_atoms_dof import get_motif_atoms_dof, project_positions_to_unit_cell
from msiplib.unit_cell_from_real_space import reduce_uc_vectors
from msiplib.misc import eval_param, StandardLogger
from msiplib.motif import plot_motif_on_image
from msiplib.io import readArrayFromNetCDF,read_image
from msiplib.io.emic import get_nm_per_pixel_from_attributes

from msiplib.config_loader import Config


parser = argparse.ArgumentParser(description="Motif extraction.")
parser.add_argument("-p", "--param_file", default="get_motif_atoms.yaml", type=str, help="Parameter file name.")
args = parser.parse_args()

save = False # Whether to save converted config file

# Load config file with sections and validate
if args.param_file is not None:
    ext = os.path.splitext(args.param_file)[1].lower()
    
    if ext == ".toml":
        # Load TOML with sections
        print(f"Loading TOML config: {args.param_file}")
        config = Config.from_toml(args.param_file)
    elif ext in (".yaml", ".yml"):
        # Load YAML (flat -> auto-converted to sections)
        print(f"Loading YAML config: {args.param_file}")
        config = Config.from_yaml(args.param_file)
        # Save converted config as TOML next to original YAML
        if save:
            toml_path = os.path.splitext(args.param_file)[0] + ".toml"
            config.save_as_toml(toml_path)
    else:
        raise ValueError(f"Unsupported config file format: {ext}")

else:
    raise ValueError("No parameter file provided. Use -p/--param_file.")

#Load parameter sections from config
input_params = config.get_input_params()
preproc = config.get_preprocessing_params()
motif_params = config.get_motif_atoms_params()
background_params = config.get_background_params()
visualization_params = config.get_visualization_params()

output_dir = os.path.expandvars(input_params.get("save_directory"))
# Set up output directory and name prefix for consistent file naming
output_path = Path(output_dir)
output_path.mkdir(parents=True, exist_ok=True)

name = input_params.get("save_name")
name_prefix = "" if name is None else f"{name}_"
with StandardLogger(output_dir, params=config.to_dict(), param_file=args.param_file):
    input_file = input_params["input_file"]
    f = rescale_intensity(read_image(input_file, as_gray=not input_file.endswith(".nc")), out_range=(0.0, 1.0))
    if preproc.get("rotation_nr") is not None:
        f = np.rot90(f, k=preproc.get("rotation_nr"))
    crop_start = preproc.get("crop_start")
    crop_size = preproc.get("crop_size")
    if crop_start is not None and crop_size is not None:
        f_full = f
        f = f[crop_start[0] : (crop_start[0] + crop_size[0]), crop_start[1] : (crop_start[1] + crop_size[1])]
    plot = motif_params.get("plot")
    if background_params: 
        radius = background_params.get("rolling_ball_radius")
        intensity = background_params.get("rolling_ball_intensity")
        background = None
        if radius is not None and intensity is not None:
            kernel = restoration.ellipsoid_kernel((radius, radius), intensity)
            background = restoration.rolling_ball(f, kernel=kernel)

        if plot and background is not None:
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(f, cmap='gray')
            ax[0].set_title('Input image')
            ax[0].axis('off')
            ax[1].imshow(background, cmap='gray')
            ax[1].set_title('Background')
            ax[1].axis('off')
            ax[2].imshow(f - background, cmap='gray')
            ax[2].set_title('Result')
            ax[2].axis('off')
            plt.show()
        if background is not None:
            f = f - background
    
    #Get v1 and v2 in correct format
    v1 = motif_params.get("v1",None)
    if v1 is not None:
        v1 = [eval_param(x) for x in v1]
    v2 = motif_params.get("v2",None)
    if v2 is not None:
        v2 = [eval_param(x) for x in v2]
    reduce_v1v2 = motif_params.get("reduce_v1v2", False)
   
    # Get nm_per_pixel from input file if not given in config
    nm_per_pixel = eval_param(motif_params.get("nm_per_pixel",None))
    if nm_per_pixel is None and input_file.endswith(".nc"):
        attributes = readArrayFromNetCDF(os.path.expanduser(os.path.expandvars(input_file)), return_attributes=True)[1]
        try:
            nm_per_pixel = get_nm_per_pixel_from_attributes(attributes)
            print(f"Using nm_per_pixel={nm_per_pixel} from input file metadata")
        except ValueError:
            print("Cannot determine nm_per_pixel from input file metadata")

    fit_to_reconstruction = motif_params.get("fit_to_reconstruction", False)
    
    # Get UCE_config from [uce_config] section (keys are auto-renamed)
    UCE_config = config.get_uce_config()

    if reduce_v1v2:
        v1, v2 = reduce_uc_vectors(v1, v2)

    if visualization_params: 
        plt.rc("legend", fontsize=visualization_params.get("legend_fontsize"))

    g, v = get_motif_atoms_dof(
        f,
        name,
        output_dir,
        motif_params.get("num_motif_atoms",None),
        v1,
        v2,
        motif_params.get("compute_uv", True),
        jax.numpy,
        motif_params.get("num_sigma", 4),
        motif_params.get("initial_diameter",None),
        motif_params.get("erase_inf_radius",None),
        motif_params.get("separation",None),
        nm_per_pixel,
        50000,
        motif_params.get("plot", False),
        motif_params.get("energy_exclusion_factor", 1.15),
        motif_params.get("show_bad_energies", False),
        motif_params.get("height_std_factor", 2.5),
        show_motif_legend=motif_params.get("show_motif_legend", True),
        min_center_value=motif_params.get("min_center_value",None),
        UCE_config=UCE_config,
        align_to_y_axis=motif_params.get("align_to_y_axis", False),
        fit_to_reconstruction=fit_to_reconstruction,
        use_kmedoids=motif_params.get("use_kmedoids", False),
    )

    if crop_start is not None:
        shift = project_positions_to_unit_cell(v, crop_start[::-1], np_mod=np)
        fig, dpi = plot_motif_on_image(
            None,
            v,
            f_full,
            show_outline=False,
            motif_atoms=g[:, :2] + shift,
            extend_motif_atoms=True,
            show_plot=False,
            dot_size=int((10 * max(f_full.shape)) / 512),
            show_motif_legend=True,
        )

        rect = Rectangle(
            crop_start[::-1],
            crop_size[1],
            crop_size[0],
            fc="none",
            ec="g",
            lw=1.5,
        )
        plt.gca().add_patch(rect)
        fig.savefig(output_path / f"{name_prefix}motif_extended.pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)

