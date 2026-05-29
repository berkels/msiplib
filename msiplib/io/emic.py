'''
    Functions for input and output of electron microscopy related tasks.
'''
import pandas as pd
import numpy as np
from datetime import datetime
from msiplib.io import readArrayFromNetCDF


def get_nm_per_pixel_from_attributes(attributes):
    if "nm_per_pixel" in attributes.keys():
        return attributes["nm_per_pixel"]
    elif set(("x-axis units", "y-axis units", "x-axis scale", "y-axis scale")) <= attributes.keys():
        if attributes["x-axis units"] == "nm" and attributes["y-axis units"] == "nm":
            x_scale = attributes["x-axis scale"]
            y_scale = attributes["y-axis scale"]
            if x_scale == y_scale:
                return x_scale
    elif set(("pixelSize", "pixelUnit")) <= attributes.keys():
        if attributes["pixelUnit"][0] == "m" and attributes["pixelUnit"][1] == "m":
            if attributes["pixelSize"][0] == attributes["pixelSize"][1]:
                return 1e9 * attributes["pixelSize"][0]

    raise ValueError("Can't determine scale.")


