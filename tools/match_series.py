#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import yaml
import numpy as np
from skimage import color
from cerberus import Validator
from humanize import naturalsize
import psutil
import imagesize
import argparse
from msiplib.registration import (
    ParametricAffineDeformation,
    ParametricRigidBodyMotion,
    ParametricTranslation,
    parametric_registration_mld,
    create_template_filename_list,
)
from msiplib.io import read_image, write_image
from msiplib.io.terminal import set_terminal_title
from msiplib.misc import StandardLogger


def get_common_support_center_per_image(grid_shape, par_def_params):
    h = max(grid_shape) - 1

    origin = np.array([0, 0])
    top_right = np.array([grid_shape[0] - 1, grid_shape[1] - 1])

    translations = -par_def_params[:, :2][:, ::-1]
    pixel_translation = translations * h

    origins = origin[np.newaxis, :] + pixel_translation
    top_rights = top_right[np.newaxis, :] + pixel_translation

    # Get the common support bounding box
    min_x = np.amax(np.minimum(origins[:, 0], top_rights[:, 0]))
    min_y = np.amax(np.minimum(origins[:, 1], top_rights[:, 1]))
    max_x = np.amin(np.maximum(origins[:, 0], top_rights[:, 0]))
    max_y = np.amin(np.maximum(origins[:, 1], top_rights[:, 1]))

    center = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])

    centers = center[np.newaxis, :] - pixel_translation
    return centers


def compute_series_crop_parameters(grid_shape, par_def_params):
    centers = get_common_support_center_per_image(grid_shape, par_def_params)

    centers_int = np.rint(centers).astype(int)

    min_centers = np.amin(centers_int, axis=0)
    max_centers = np.amax(centers_int, axis=0)

    crop_width = np.array(
        [
            min(min_centers[0], grid_shape[0] - 1 - max_centers[0]),
            min(min_centers[1], grid_shape[1] - 1 - max_centers[1]),
        ]
    )
    crop_size = 2 * crop_width

    crop_start = centers_int - crop_width[np.newaxis, :]
    return crop_start, crop_size


def load_and_prepare_image(filename, as_gray):
    print(f"Reading {filename} ", end="", flush=True)
    image = read_image(filename)
    print("done")
    if as_gray and (image.ndim == 3):
        image = color.rgb2gray(image)
    return image


def save_image(filename, image):
    print(f"Writing {filename} ", end="", flush=True)
    write_image(filename, image)
    print("done")


def reduce_translations(vs_accumulated, def_type, reference_shape, template_filenames, save_directory, params):
    vs_accumulated = np.array(vs_accumulated)
    translations = vs_accumulated[:, :2]
    mean_translation = np.mean(translations, axis=0)
    vs_accumulated[:, :2] = translations - mean_translation

    crop_series = def_type == ParametricTranslation
    if crop_series:
        crop_start, crop_size = compute_series_crop_parameters(reference_shape, vs_accumulated)

    for i in range(len(template_filenames)):
        filename = template_filenames[i]
        template = load_and_prepare_image(filename, params["convertInputToGrayscale"])
        phi = def_type(vs_accumulated[i, :], reference_shape)
        deformed_image = phi.deform_image(template, cval=template.mean())
        ext = os.path.splitext(filename)[1]
        basename = (
            f"{params['saveBaseName']}_{i:02d}"
            if "saveBaseName" in params
            else os.path.basename(os.path.splitext(filename)[0])
        )
        save_image(save_directory + basename + "_reduced" + ext, deformed_image.astype(template.dtype))
        if crop_series:
            save_image(
                save_directory + basename + "_cropped_reduced" + ext,
                template[
                    crop_start[i, 0] : crop_start[i, 0] + crop_size[0],
                    crop_start[i, 1] : crop_start[i, 1] + crop_size[1],
                ].astype(template.dtype),
            )


def main():
    from PIL import Image as PILImage

    PILImage.MAX_IMAGE_PIXELS = 333287001

    parser = argparse.ArgumentParser(description="Matching of image series.")
    parser.add_argument("-p", "--param_file", default="match_series.yaml", type=str, help="Parameter file name.")
    args = parser.parse_args()

    params = yaml.load(open(args.param_file, "r"), Loader=yaml.SafeLoader)
    schema = {
        "deformationModel": {
            "required": True,
            "type": "integer",
            "anyof": [{"min": 9, "max": 9}, {"min": 12, "max": 12}, {"min": 20, "max": 20}],
        },
        "templateNamePattern": {
            "dependencies": ["templateNumOffset", "templateNumStep", "numTemplates"],
            "type": "string",
            "excludes": "templates",
        },
        "templateNumOffset": {
            "dependencies": ["templateNamePattern"],
            "type": "integer",
        },
        "templateNumStep": {
            "dependencies": ["templateNamePattern"],
            "type": "integer",
        },
        "numTemplates": {
            "dependencies": ["templateNamePattern"],
            "type": "integer",
            "min": 2,
        },
        "templates": {
            "type": "list",
            "schema": {"type": "string"},
            "excludes": "templateNamePattern",
            "dependencies": ["templatesDirectory"],
        },
        "templatesDirectory": {
            "dependencies": ["templates"],
            "type": "string",
        },
        "saveDirectory": {
            "required": True,
            "type": "string",
        },
        "saveBaseName": {
            "type": "string",
        },
        "numLevels": {
            "required": True,
            "type": "integer",
            "min": 1,
        },
        "numRefineLevels": {
            "required": True,
            "type": "integer",
            "min": 0,
        },
        "earlyStopLevel": {
            "required": True,
            "type": "integer",
            "min": 0,
        },
        "stopEpsilon": {
            "required": True,
            "type": "float",
            "min": 0,
        },
        "maxIterations": {
            "required": True,
            "type": "integer",
            "min": 0,
        },
        "minimizer": {
            "required": True,
            "type": "string",
            "allowed": ["GaussNewton", "trf", "L-BFGS-B", "GradientDescent"],
        },
        "interpolation": {
            "required": True,
            "type": "string",
            "allowed": ["ScipySpline", "bilinear"],
        },
        "useCorrelationToInitTranslation": {
            "required": True,
            "type": "boolean",
        },
        "correlationChannel": {
            "type": "integer",
            "min": 0,
        },
        "correlationOverlapRatio": {
            "type": "float",
            "min": 0,
            "max": 1,
        },
        "preSmoothSigma": {
            "required": True,
            "type": "float",
            "min": 0,
        },
        "reduceTranslations": {
            "required": True,
            "type": "boolean",
        },
        "computeAverage": {
            "required": True,
            "type": "boolean",
        },
        "alignToAverage": {
            "oneof": [{"max": 0}, {"min": 1, "dependencies": {"computeAverage": True, "numRefineLevels": 0}}],
            "type": "boolean",
        },
        "averageSaveIncrement": {
            "dependencies": {"computeAverage": True},
            "type": "integer",
            "min": 1,
        },
        "convertInputToGrayscale": {
            "required": True,
            "type": "boolean",
        },
    }
    v = Validator(schema)
    if not v.validate(params, schema):
        print("Validating parameter file failed.")
        print(v.errors)
        return

    if ("templateNamePattern" not in params) and ("templates" not in params):
        print("Either 'templateNamePattern' or 'templates' must be specified.")
        return

    template_filenames = create_template_filename_list(params)

    save_directory = os.path.expandvars(params["saveDirectory"])

    with StandardLogger(save_directory, params, args.param_file):
        num_levels = params["numLevels"]

        deformation_models = {9: ParametricAffineDeformation, 12: ParametricTranslation, 20: ParametricRigidBodyMotion}
        def_type = deformation_models.get(params["deformationModel"], None)
        vs = []
        v_accumulated = def_type.get_identity_deformation_params()
        vs_accumulated = [v_accumulated]

        # imagesize.get only works on local files.
        if not template_filenames[0].startswith("https://"):
            sizes = []
            for i in range(len(template_filenames)):
                sizes.append(imagesize.get(template_filenames[i]))
            if not all(elem == sizes[0] for elem in sizes):
                for i in range(len(template_filenames)):
                    print(template_filenames[i], sizes[i])
                raise ValueError("All images in the series must be of the same size!")

        reference = load_and_prepare_image(template_filenames[0], params["convertInputToGrayscale"])

        compute_average = params["computeAverage"]
        if compute_average:
            average = reference.astype(np.float64)
            num_samples = np.ones(reference.shape)

        align_to_average = ("alignToAverage" in params) and params["alignToAverage"]

        for i in range(1, len(template_filenames)):
            set_terminal_title(os.path.basename(template_filenames[i]))
            template = load_and_prepare_image(template_filenames[i], params["convertInputToGrayscale"])
            v = parametric_registration_mld(
                template,
                reference,
                def_type,
                num_levels,
                early_stop_level=params["earlyStopLevel"],
                stop_eps=params["stopEpsilon"],
                max_iter=params["maxIterations"],
                minimizer=params["minimizer"],
                return_spline_and_grid=False,
                use_correlation_to_init_translation=params["useCorrelationToInitTranslation"],
                interpolation=params["interpolation"],
                pre_smooth_sigma=params["preSmoothSigma"],
                phase_cross_correlation_channel=params["correlationChannel"]
                if "correlationChannel" in params
                else None,
                phase_cross_correlation_overlap_ratio=params["correlationOverlapRatio"]
                if "correlationOverlapRatio" in params
                else None,
            )
            vs.append(v)

            if align_to_average:
                v_accumulated = vs[i - 1]
            else:
                v_accumulated = def_type.concatenate_deformation_params(v_accumulated, vs[i - 1])

            if (i > 1) and (params["numRefineLevels"] > 0):
                reference = load_and_prepare_image(template_filenames[0], params["convertInputToGrayscale"])
                v_accumulated = parametric_registration_mld(
                    template,
                    reference,
                    def_type,
                    params["numRefineLevels"],
                    early_stop_level=params["earlyStopLevel"],
                    stop_eps=params["stopEpsilon"],
                    max_iter=params["maxIterations"],
                    minimizer=params["minimizer"],
                    return_spline_and_grid=False,
                    interpolation=params["interpolation"],
                    v0=v_accumulated,
                    pre_smooth_sigma=params["preSmoothSigma"],
                )

            vs_accumulated.append(v_accumulated)
            print("Deforming template image ... ", end="", flush=True)
            phi = def_type(v_accumulated, reference.shape)
            deformed_image = phi.deform_image(template.astype(np.float64), cval=np.inf)
            print("done")
            if compute_average:
                num_samples += np.isfinite(deformed_image)
                average += np.ma.masked_invalid(deformed_image).filled(0)
            basename = os.path.basename(template_filenames[i])
            deformed_image = np.ma.masked_invalid(deformed_image).astype(template.dtype).filled(template.mean())
            save_image(save_directory + basename, deformed_image)
            if (
                compute_average
                and ("averageSaveIncrement" in params)
                and (((i + 1) % params["averageSaveIncrement"]) == 0)
            ):
                save_image(save_directory + f"average_{i:03d}.nc", average / num_samples)
            np.save(save_directory + os.path.splitext(basename)[0] + "_v.npy", v, allow_pickle=False)
            np.save(
                save_directory + os.path.splitext(basename)[0] + "_v_accumulated.npy", v_accumulated, allow_pickle=False
            )

            if align_to_average:
                reference = (average / num_samples).astype(template.dtype)
            else:
                reference = template
            del deformed_image

            print("Used memory:", naturalsize(psutil.Process().memory_info().rss))
            print("Available memory:", naturalsize(psutil.virtual_memory().available))

        if compute_average:
            save_image(save_directory + "numSamples.nc", num_samples)
            save_image(save_directory + "average.nc", average / num_samples)

        print(vs)

        if params["reduceTranslations"]:
            reduce_translations(vs_accumulated, def_type, reference.shape, template_filenames, save_directory, params)


if __name__ == "__main__":
    main()
