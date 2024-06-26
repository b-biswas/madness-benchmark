"""Run Scarlet on test data."""

import logging
import math
import os
import pickle
import sys
import time

import galcheat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scarlet
import scarlet.psf
import sep
import yaml

from madness_benchmark.metrics import (
    compute_aperture_photometry,
    compute_pixel_cosdist,
    compute_shapes,
)
from madness_benchmark.utils import get_benchmark_config_path

# logging level set to INFO
logging.basicConfig(format="%(message)s", level=logging.INFO)
LOG = logging.getLogger(__name__)

with open(get_benchmark_config_path()) as f:
    benchmark_config = yaml.safe_load(f)

survey_name = benchmark_config["survey_name"]
if survey_name not in ["LSST", "HSC"]:
    raise ValueError(
        "survey should be one of: LSST or HSC"
    )  # other surveys to be added soon!
survey = galcheat.get_survey(survey_name)

LOG.info(f"Running tests with scarlet for {survey_name}")

density = sys.argv[1]
if density not in ["high", "low"]:
    raise ValueError("The second argument should be either isolated or blended")

num_repetations = 300
simulation_path = os.path.join(benchmark_config["TEST_DATA_PATH"][survey_name], density)
results_path = benchmark_config["RESULTS_PATH"][survey_name]
density_level = density + "_density"

psf_fwhm = []
for band in survey.available_filters:
    filt = survey.get_filter(band)
    psf_fwhm.append(filt.psf_fwhm.value)


# Define function to make predictions with scarlet
def predict_with_scarlet(
    image, x_pos, y_pos, show_scene, show_sources, filters, obs_psf
):
    """Deblend using the SCARLET deblender.

    Parameters
    ----------
    image: array
        field to be deblended.
    x_pos: array
        x positions of detections.
    y_pos: array
        y position of detection.
    show_scene: bool
        To run scarlet.display.show_scene or not.
    show_sources: bool
        To run scarlet.display.show_sources or not.
    filters: list of hashable elements
        Names/identifiers of spectral channels
    obs_psf: scarlet.ImagePSF
        observed PSF.

    Returns
    -------
    predicted_sources:
        array with reconstructions predicted by SCARLET

    """
    weights = np.ones_like(image)
    bkg = np.array(
        [
            galcheat.utilities.mean_sky_level(survey, f).to_value("electron")
            for f in filters
        ]
    )
    weights = np.ones(image.shape) / bkg.reshape((-1, 1, 1))
    observation = scarlet.Observation(
        image, psf=obs_psf, weights=weights, channels=filters
    )

    model_psf = scarlet.GaussianPSF(
        sigma=(0.7,) * len(filters)
    )  # Setting this value according to the standard configuration of scarlet
    model_frame = scarlet.Frame(image.shape, psf=model_psf, channels=filters)

    observation = observation.match(model_frame)
    sources = []
    centers = [(x_pos[i], y_pos[i]) for i in range(len(x_pos))]

    sources, _ = scarlet.initialization.init_all_sources(
        model_frame,
        centers,
        observation,
        max_components=2,
        min_snr=50,
        thresh=1,
        fallback=True,
        silent=True,
        set_spectra=True,
    )

    scarlet_blend = scarlet.Blend(sources, observation)

    t0 = time.time()
    scarlet_blend.fit(200, e_rel=1e-6)
    t1 = time.time()

    LOG.info("SCARLET TIME: " + str(t1 - t0))
    # print(f"scarlet ran for {it} iterations to logL = {logL}")
    # scarlet.display.show_likelihood(scarlet_blend)
    # plt.show()

    if show_scene:
        scarlet.display.show_scene(
            sources,
            norm=None,
            observation=observation,
            show_rendered=True,
            show_observed=True,
            show_residual=True,
        )
        plt.show()

    if show_sources:
        scarlet.display.show_sources(
            sources,
            norm=None,
            observation=observation,
            show_rendered=True,
            show_observed=True,
            add_boxes=True,
        )
        plt.show()

    # scarlet.display.show_likelihood(scarlet_blend)
    # plt.show()

    predicted_sources = []
    for src in sources:
        predicted_sources.append(observation.render(src.get_model(frame=model_frame)))
    # print(np.shape(src.get_model(frame=model_frame)))
    return predicted_sources


for file_num in range(num_repetations):
    LOG.info(f"\n\n######### Processing file: {file_num} #########")
    file_name = os.path.join(
        simulation_path,
        str(file_num) + ".pkl",
    )
    with open(file_name, "rb") as f:
        blend = pickle.load(f)

    field_images = blend.blend_images
    isolated_images = blend.isolated_images

    psf = blend.get_numpy_psf()
    obs_psf = scarlet.ImagePSF(psf)
    bands = [f for f in survey._filters]

    x_pos = blend.catalog_list[0]["y_peak"]
    y_pos = blend.catalog_list[0]["x_peak"]

    scarlet_results = []
    scarlet_photometry = []

    # Get Scarlet Predictions

    for field_num in range(len(blend.catalog_list)):
        scarlet_current_predictions = []
        image = field_images[field_num]
        x_pos = blend.catalog_list[field_num]["y_peak"]
        y_pos = blend.catalog_list[field_num]["x_peak"]
        LOG.info(blend.catalog_list[field_num])
        scarlet_current_predictions = predict_with_scarlet(
            image,
            x_pos=x_pos,
            y_pos=y_pos,
            show_scene=False,
            show_sources=False,
            filters=bands,
            obs_psf=obs_psf,
        )

        num_galaxies = len(blend.catalog_list[field_num])

        isolated_images = blend.isolated_images[field_num][0:num_galaxies]

        scarlet_current_res = compute_pixel_cosdist(
            scarlet_current_predictions,
            isolated_images,
            blend.blend_images[field_num],
            survey=survey,
        )

        # scarlet_current_res["images"] = scarlet_current_predictions
        scarlet_current_res["field_num"] = [field_num] * num_galaxies
        scarlet_current_res["file_num"] = [file_num] * num_galaxies
        if "r_band_snr" not in blend.catalog_list[field_num].columns:
            scarlet_current_res["r_band_snr"] = 0
        else:
            scarlet_current_res["r_band_snr"] = blend.catalog_list[field_num][
                "r_band_snr"
            ]
        # make this a table

        # scarlet_results.append(scarlet_current_res)

        bkg_rms = {}
        for band in range(len(survey.available_filters)):
            bkg_rms[band] = sep.Background(
                blend.blend_images[field_num][band]
            ).globalrms

        if survey_name == "LSST":

            a = blend.catalog_list[field_num]["a_d"].value
            b = blend.catalog_list[field_num]["b_d"].value
            theta = blend.catalog_list[field_num]["pa_disk"].value

            cond = (
                blend.catalog_list[field_num]["a_d"]
                < blend.catalog_list[field_num]["a_b"]
            )
            a = np.where(cond, blend.catalog_list[field_num]["a_b"].value, a)
            b = np.where(cond, blend.catalog_list[field_num]["b_b"].value, b)
            theta = np.where(
                cond, blend.catalog_list[field_num]["pa_bulge"].value, theta
            )

            theta = theta % 180
            theta = theta * math.pi / 180

            theta = np.where(theta > math.pi / 2, theta - math.pi, theta)
            scarlet_current_res["size"] = (a * b) ** 0.5

        if survey_name == "HSC":
            a = (
                blend.catalog_list[field_num]["flux_radius"]
                * blend.catalog_list[field_num]["PIXEL_SCALE"]
            )
            b = a
            theta = [0] * len(blend.catalog_list[field_num])
            scarlet_current_res["size"] = a

        scarlet_photometry_current = compute_aperture_photometry(
            field_image=blend.blend_images[field_num],
            predictions=scarlet_current_predictions,
            xpos=blend.catalog_list[field_num]["x_peak"],
            ypos=blend.catalog_list[field_num]["y_peak"],
            a=a / survey.pixel_scale.value,
            b=b / survey.pixel_scale.value,
            theta=theta,
            psf_fwhm=np.array(psf_fwhm) / survey.pixel_scale.value,
            bkg_rms=bkg_rms,
            survey=survey,
        )
        scarlet_current_res.update(scarlet_photometry_current)

        scarlet_shapes_current = compute_shapes(
            field_image=blend.blend_images[field_num],
            predictions=scarlet_current_predictions,
            xpos=blend.catalog_list[field_num]["x_peak"],
            ypos=blend.catalog_list[field_num]["y_peak"],
            survey=survey,
        )
        scarlet_current_res.update(scarlet_shapes_current)

        scarlet_current_res = pd.DataFrame.from_dict(scarlet_current_res)
        scarlet_results.append(scarlet_current_res)
    # scarlet_results = vstack(scarlet_results)
    # scarlet_results = hstack([scarlet_results, vstack(blend.catalog_list)])
    # scarlet_results = hstack(scarlet_results,vstack(scarlet_photometry))

    scarlet_results = pd.concat(scarlet_results)

    save_file_name = os.path.join(
        results_path,
        density_level,
        "scarlet_results",
        str(file_num) + ".pkl",
    )

    scarlet_results.to_pickle(save_file_name)
    # np.save(
    #     save_file_name,
    #     scarlet_results.to_records(),
    # )

    # hickle.dump(scarlet_results, save_file_name, mode="w", compression='gzip')
