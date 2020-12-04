from enum import Enum, unique
import powerlaw
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.axes
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Union, Optional, Literal

import fractopo.analysis.tools as tools


@unique
class Dist(Enum):
    POWERLAW = "powerlaw"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"


def determine_fit(
    length_array: np.ndarray, cut_off: Optional[float] = None
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
        label=label,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_fit_on_ax(
    ax: matplotlib.axes.Axes,
    fit: powerlaw.Fit,
    fit_distribution: Literal[Dist.EXPONENTIAL, Dist.LOGNORMAL, Dist.POWERLAW],
):
    if fit_distribution == Dist.POWERLAW:
        fit.power_law.plot_ccdf(ax=ax, label="Powerlaw", linestyle="--", color="red")
    elif fit_distribution == Dist.LOGNORMAL:
        fit.lognormal.plot_ccdf(ax=ax, label="Lognormal", linestyle="--", color="lime")
    elif fit_distribution == Dist.EXPONENTIAL:
        fit.exponential.plot_ccdf(
            ax=ax, label="Exponential", linestyle="--", color="blue"
        )
    return


def _setup_length_plot_axlims(
    ax: matplotlib.axes.Axes,
    length_array: np.ndarray,
    ccm_array: np.ndarray,
    cut_off: float,
):
    truncated_length_array = (
        length_array[length_array > cut_off] if cut_off is not None else length_array
    )

    left = length_array.min() / 10
    right = length_array.max() * 10
    bottom = ccm_array.min() / 10
    top = ccm_array.max() * 10
    try:
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
    except ValueError:
        # Don't try setting if it errors
        pass


def plot_distribution_fits(
    length_array: np.ndarray, label: str, cut_off: Optional[float] = None
) -> Tuple[powerlaw.Fit, matplotlib.figure.Figure, matplotlib.axes.Axes]:  # type: ignore
    # Determine powerlaw, exponential, lognormal fits
    fit = determine_fit(length_array, cut_off)
    # Get fit xmin
    xmin = xmin if isinstance((xmin := fit.xmin), (int, float)) else 0.0
    # Create figure, ax
    fig, ax = plt.subplots(figsize=(7, 7))
    # Get the x, y data from fit
    truncated_length_array, ccm_array = fit.ccdf()
    # Plot length scatter plot
    plot_length_data_on_ax(ax, truncated_length_array, ccm_array, label)
    # Plot the actual fits (powerlaw, exp...)
    for fit_distribution in (Dist.EXPONENTIAL, Dist.LOGNORMAL, Dist.POWERLAW):
        plot_fit_on_ax(ax, fit, fit_distribution)
    # Setup of ax appearance and axlims
    tools.setup_ax_for_ld(ax, using_branches=False)
    _setup_length_plot_axlims(
        ax=ax,
        length_array=truncated_length_array,
        ccm_array=ccm_array,
        cut_off=xmin,  # type: ignore
    )
    return fit, fig, ax
