"""Util functions."""
import os 

def get_benchmark_config_path():
    """Fetch path to madness_deblender config yaml file.

    Returns
    -------
    data_dir: str
        path to data folder

    """
    curdir = os.path.dirname(os.path.abspath(__file__))
    benchmark_config_path = os.path.join(
        curdir, "benchmark_config.yaml"
    )

    return benchmark_config_path