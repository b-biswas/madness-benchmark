"""First call to generate tf datasets."""

import os

import yaml

from madness_benchmark.dataset_generator import loadCATSIMDataset
from madness_benchmark.utils import get_benchmark_config_path

with open(get_benchmark_config_path()) as f:
    benchmark_config = yaml.safe_load(f)

btksims_config = benchmark_config["btksims"]
survey_name = benchmark_config["survey_name"]

loadCATSIMDataset(
    train_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "blended_training",
    ),
    val_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "blended_validation",
    ),
    output_dir=os.path.join(
        benchmark_config["TF_DATASET_PATH"][survey_name], "blended_tfDataset"
    ),
)

loadCATSIMDataset(
    train_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "isolated_training",
    ),
    val_data_dir=os.path.join(
        btksims_config["TRAIN_DATA_SAVE_PATH"][survey_name],
        "isolated_validation",
    ),
    output_dir=os.path.join(
        benchmark_config["TF_DATASET_PATH"][survey_name],
        "isolated_tfDataset",
    ),
)
