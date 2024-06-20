import os

import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import astropy.table
import btk
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from galcheat.utilities import mag2counts, mean_sky_level
from numba import jit

import matplotlib as mpl

from madness_benchmark.utils import get_benchmark_config_path

mpl.rcParams["text.usetex"] = True

with open(get_benchmark_config_path()) as f:
    benchmark_config = yaml.safe_load(f)
btksims_config = benchmark_config["btksims"]

CATSIM_CATALOG_PATH = btksims_config["ORIGINAL_CAT_PATH"][
    benchmark_config["survey_name"]
]

_, ext = os.path.splitext(CATSIM_CATALOG_PATH)
fmt = "fits" if ext.lower() == ".fits" else "ascii.basic"
catalog = astropy.table.Table.read(CATSIM_CATALOG_PATH, format=fmt)

survey = btk.survey.get_surveys("LSST")
r_filter = survey.get_filter("r")

snr = []

# For calculation of source counts (C) and mean_sky_level (B) refer to https://github.com/aboucaud/galcheat/blob/main/galcheat/utilities.py
# For Computaion of SNR, refer to: https://smtn-002.lsst.io/

# background noise
B = mean_sky_level("LSST", "r").to_value("electron")

# Assume gain = 1
g = 1

# compute sigma_instr https://smtn-002.lsst.io/
dark_current = 0.2
exp_time = 15  # 2 * 15 sec exposures
num_exp = 184 * 2  # 184 visits from: https://arxiv.org/pdf/0805.2366.pdf
read_noise = 8.8

sig_instr = np.sqrt((read_noise**2 + dark_current * exp_time) * num_exp)
# But BTK does not add instr noise (https://github.com/LSSTDESC/BlendingToolKit/blob/main/btk/draw_blends.py lines: 418-432)
sig_instr = 0  # Setting it to 0, because no instr noise


@jit
def computeFWHM(hlr_a, hlr_b):
    return 2 * np.sqrt(
        hlr_a * hlr_b
    )  # fwhm = 2*hlr from: https://www.researchgate.net/publication/1778778_Accurate_photometry_of_extended_spherically_symmetric_sources/figures?lo=1


@jit
def compute_snr(C, B, FWHM_gal, FWHM_PSF=r_filter.psf_fwhm.value, sig_instr=sig_instr):

    # convolve with PSF with FWHM of galaxy
    FWHM = np.sqrt(FWHM_gal**2 + FWHM_PSF**2)  # convolution of 2 PSF

    # Source footprint https://smtn-002.lsst.io/
    n_eff = 2.266 * ((FWHM / 0.2) ** 2)

    # compute_snr
    snr = C / np.sqrt(
        C + (B / g + sig_instr**2) * n_eff
    )  # from : https://smtn-002.lsst.io/
    return snr


for row in catalog:
    # Source counts
    C = mag2counts(row["r_ab"], "LSST", "r").to_value("electron")  # converting to electron count

    # FWHM of galaxy
    FWHM_row_d = computeFWHM(row["a_d"], row["b_d"])
    FWHM_row_b = computeFWHM(row["a_b"], row["b_b"])
    FWHM_gal = FWHM_row_d if FWHM_row_d > FWHM_row_b else FWHM_row_b

    current_snr = compute_snr(C=C, B=B, FWHM_gal=FWHM_gal)

    snr.append(current_snr)