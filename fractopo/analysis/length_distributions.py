import powerlaw
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.axes
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Union

POWERLAW = "powerlaw"
LOGNORMAL = "lognormal"
EXPONENTIAL = "exponential"


def determine_fit(
    length_array: np.ndarray, cut_off: Union[None, float] = None
) -> powerlaw.Fit:
    """
    Determines powerlaw (along other) length distribution fits for given data.
    """
    fit = (
        powerlaw.Fit(length_array, xmin=cut_off)
        if cut_off is not None
        else powerlaw.Fit(length_array)
    )
    return fit


def plot_length_data_on_ax(
    ax: matplotlib.axes.Axes,
    length_array: np.ndarray,
    ccm_array: np.ndarray,
    label: str,
):
    ax.scatter(
        x=length_array,
        y=ccm_array,
        s=50,
        logx=True,
        logy=True,
        label=label,
    )


def plot_fit_on_ax(
    ax: matplotlib.axes.Axes,
    fit: powerlaw.Fit,
    fit_distribution: str,
):
    if fit_distribution == POWERLAW:
        fit.power_law.plot_ccdf(ax=ax, label="Powerlaw", linestyle="--", color="red")
    elif fit_distribution == LOGNORMAL:
        fit.lognormal.plot_ccdf(ax=ax, label="Lognormal", linestyle="--", color="lime")
    elif fit_distribution == EXPONENTIAL:
        fit.exponential.plot_ccdf(
            ax=ax, label="Exponential", linestyle="--", color="blue"
        )
