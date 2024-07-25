"""
Read config files.

This file contains the following functions:
    * get_config

Version: 1.0
"""

import yaml


CONFIG_PATH = "config/config.yml"
TEXTY_PATH = "config/texty.yml"
DEFAULTS_PATH = "config/defaults.yml"
PLOTS_PATH = "config/plots.yml"
FITTING_PATH = "config/fitting.yml"


def get_config(cfg_type: str = "params") -> dict | None:
    """
    Returns configuration data.

    Parameters
    ----------
    cfg_type: str
        One of ['params', 'texty', 'defaults']

    Returns
    -------
    config: dict | None
    """
    if cfg_type == "params":
        path = CONFIG_PATH
    elif cfg_type == "texty":
        path = TEXTY_PATH
    elif cfg_type == "defaults":
        path = DEFAULTS_PATH
    elif cfg_type == "plots":
        path = PLOTS_PATH
    elif cfg_type == "fitting":
        path = FITTING_PATH
    else:
        return None
    with open(path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
