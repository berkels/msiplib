""" Collection of general useful functions """

import os
import numpy as np
import numexpr
from subprocess import PIPE, run
from io import StringIO
import logging
from skimage.feature import match_template
import pandas as pd
import matplotlib.pyplot as plt
from .io import Tee
from pathlib import Path


def row_norms(mat, squared=False):
    """
    computes (squared) euclidean norm for every row in the array
    Args:
        mat: an array with vectors as entries
    Returns:
        an array containing the norms of the vectors of the input array
    """
    if mat.ndim == 1:
        m = mat[np.newaxis]
    elif mat.ndim > 2:
        raise NotImplementedError("Unsupported dimension.")
    else:
        m = mat

    m_sq_sum = np.sum(np.square(m), axis=1)
    if squared:
        return m_sq_sum
    else:
        return np.sqrt(m_sq_sum)


def normalize(vec, order=2):
    """
    normalize a vector with respect to given norm
    order accepts the same values as the ord argument for the numpy norm
    """
    if vec.ndim != 1:
        raise NotImplementedError("Unsupported dimension.")

    return vec / np.linalg.norm(vec, order)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def rint(x):
    """round to nearest integer, always rounding .5 up
    (different from numpy.rint which rounds .5 to the nearest even number)"""
    return np.rint(np.nextafter(x, x + 1)).astype(int)


def rotation_matrix(alpha):
    return np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::

        >>> angle_between((1, 0, 0), (0, 1, 0))
        1.5707963267948966
        >>> angle_between((1, 0, 0), (1, 0, 0))
        0.0
        >>> angle_between((1, 0, 0), (-1, 0, 0))
        3.141592653589793

    Adapted from https://stackoverflow.com/a/13849249
    """
    v1_normed = v1 / np.linalg.norm(v1)
    v2_normed = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_normed, v2_normed), -1.0, 1.0))


def eval_param(x):
    return numexpr.evaluate(x) if isinstance(x, str) else x


def is_float(element: any) -> bool:
    # https://stackoverflow.com/a/20929881
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def plot_image_pixel_precise(im, vmin=None, vmax=None,cmap="gray"):
    # This uses ideas from https://stackoverflow.com/a/34769840 to render the
    # image exactly with its pixel resolution.
    dpi = 80
    figsize = im.shape[1] / float(dpi), im.shape[0] / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(im, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.axis("off")
    return fig, ax, dpi


def plot_image_pixel_precise_to_pdf(im, pdf, vmin=None, vmax=None):
    fig, ax, dpi = plot_image_pixel_precise(im, vmin=vmin, vmax=vmax)
    fig.savefig(pdf, format="pdf", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()




def get_conda_package_info():
    try:
        result = run(["conda", "list"], stdout=PIPE, stderr=PIPE, universal_newlines=True)
        if result.returncode == 0:
            string_io = StringIO(result.stdout)
            # The output of "conda list" starts with three comment lines. The third one contains the column names
            # for the table that follows
            string_io.readline()
            string_io.readline()
            string_io.read(2)
            df = pd.read_csv(string_io, header=0, sep=r"\s+", index_col=None)
            return df
        else:
            print(f"'conda list' returned the error {result.stderr}.")
            return None
    except FileNotFoundError:
        print("'conda' not found.")
        return None


def dump_conda_package_info(save_directory: str | os.PathLike):
    save_directory = Path(save_directory)
    packages = get_conda_package_info()
    if packages is not None:
        packages.to_csv(save_directory / "packages-dump.csv", index=False)
    else:
        print("Could not determine package info. Skipping the package dump.")


def dump_parameters(params, param_file, save_directory:str | os.PathLike):
    save_directory = Path(save_directory)
    with open(save_directory / "parameter-dump.yml", "w") as outfile:
        if param_file is not None:
            outfile.write("# Initialized from file " + param_file + "\n")
        import os
        import yaml
        import git
        import platform
        uname = platform.uname()
        outfile.write(f"# Computer name : {uname.node}\n")
        outfile.write(f"# OS            : {uname.system}\n")
        outfile.write(f"# Release       : {uname.release}\n")
        outfile.write(f"# Version       : {uname.version}\n")
        outfile.write(f"# Architecture  : {uname.machine}\n")
        try:
            repo = git.Repo(path=os.path.dirname(os.path.abspath(__file__)), search_parent_directories=True)
            outfile.write("# Generated with msiplib git commit: " + repo.head.object.hexsha + "\n")
            status = repo.git.status()
            for line in status.splitlines():
                outfile.write(f"# {line}\n")
        except git.InvalidGitRepositoryError:
            outfile.write("# " + __file__ + " is not in a git repository.\n")
        if params is not None:
            yaml.dump(params, outfile, default_flow_style=False)
        else:
            outfile.write("# No parameters specified.\n")


def dump_reproducibility_info(params, param_file, save_directory:str | os.PathLike):
    dump_parameters(params, param_file, save_directory)
    dump_conda_package_info(save_directory)


class StandardLogger(Tee):
    def __init__(self, save_directory: str | os.PathLike, params=None, param_file=None, debug_log=False):
        self.save_directory = Path(save_directory)
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        dump_reproducibility_info(params, param_file, self.save_directory)
        super().__init__(self.save_directory / "log.txt")
        if debug_log:
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s %(name)-15s %(levelname)-8s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                filename=str(self.save_directory / "log_debug.txt"),
                filemode="w",
                force=True, # existing handlers attached to the root logger are removed and closed
            )
            # jax's DEBUG logging is very verbose, reduce this to WARNING.
            if logging.getLogger('jax') is not None:
                logging.getLogger('jax').setLevel(logging.WARNING)

    def __enter__(self):
        return super().__enter__()

    def __exit__(self, exc_type, exc_value, tb):
        super().__exit__(exc_type, exc_value, tb)
