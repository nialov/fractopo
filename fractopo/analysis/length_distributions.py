"""
Utilities for analyzing and plotting length distributions for line data.
"""
from enum import Enum, unique
from textwrap import wrap
from typing import Literal, Optional, Tuple

import matplotlib
import matplotlib.axes
import numpy as np
import powerlaw
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ALPHA = "alpha"
EXPONENT = "exponent"
CUT_OFF = "cut-off"


@unique
class Dist(Enum):

    """
    Enums of powerlaw model types.
    """

    POWERLAW = "powerlaw"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"


def determine_fit(
    length_array: np.ndarray, cut_off: Optional[float] = None
) -> powerlaw.Fit:
    """
    Determine powerlaw (along other) length distribution fits for given data.
    """
    fit = (
        powerlaw.Fit(length_array, xmin=cut_off)
        if cut_off is not None
        else powerlaw.Fit(length_array)
    )
    return fit


def plot_length_data_on_ax(
    ax: Axes,
    length_array: np.ndarray,
    ccm_array: np.ndarray,
    label: str,
):
    """
    Plot length data on given ax.

    Sets ax scales to logarithmic.
    """
    ax.scatter(
        x=length_array,
        y=ccm_array,
        s=50,
        label=label,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_fit_on_ax(
    ax: Axes,
    fit: powerlaw.Fit,
    fit_distribution: Literal[Dist.EXPONENTIAL, Dist.LOGNORMAL, Dist.POWERLAW],
) -> None:
    """
    Plot powerlaw model to ax.
    """
    if fit_distribution == Dist.POWERLAW:
        fit.power_law.plot_ccdf(ax=ax, label="Powerlaw", linestyle="--", color="red")
    elif fit_distribution == Dist.LOGNORMAL:
        fit.lognormal.plot_ccdf(ax=ax, label="Lognormal", linestyle="--", color="lime")
    elif fit_distribution == Dist.EXPONENTIAL:
        fit.exponential.plot_ccdf(
            ax=ax, label="Exponential", linestyle="--", color="blue"
        )
    else:
        raise ValueError(f"Expected fit_distribution to be one of {list(Dist)}")
    return


def _setup_length_plot_axlims(
    ax: Axes,
    length_array: np.ndarray,
    ccm_array: np.ndarray,
    cut_off: float,
):
    """
    Set ax limits for length plotting.
    """
    # truncated_length_array = (
    #     length_array[length_array > cut_off] if cut_off is not None else length_array
    # )

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
    length_array: np.ndarray,
    label: str,
    cut_off: Optional[float] = None,
    fit: Optional[powerlaw.Fit] = None,
) -> Tuple[powerlaw.Fit, Figure, Axes]:
    """
    Plot length distribution and `powerlaw` fits.

    If a powerlaw.Fit is not given it will be automatically determined (using
    the optionally given cut_off).
    """
    if fit is None:
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
    setup_ax_for_ld(ax, using_branches=False)
    _setup_length_plot_axlims(
        ax=ax,
        length_array=truncated_length_array,
        ccm_array=ccm_array,
        cut_off=xmin,  # type: ignore
    )
    return fit, fig, ax


def setup_ax_for_ld(ax_for_setup, using_branches, indiv_fit=False):
    """
    Configure ax for length distribution plots.

    :param ax_for_setup: Ax to setup.
    :type ax_for_setup: Axes
    :param using_branches: Are the lines branches or traces.
    :type using_branches: bool
    """
    #
    ax = ax_for_setup
    # LABELS
    label = "Branch length $(m)$" if using_branches else "Trace Length $(m)$"
    ax.set_xlabel(
        label,
        fontsize="xx-large",
        fontfamily="DejaVu Sans",
        style="italic",
        labelpad=16,
    )
    # Individual powerlaw fits are not normalized to area because they aren't
    # multiscale
    ccm_unit = r"$(\frac{1}{m^2})$" if not indiv_fit else ""
    ax.set_ylabel(
        "Complementary Cumulative Number " + ccm_unit,
        fontsize="xx-large",
        fontfamily="DejaVu Sans",
        style="italic",
    )
    # TICKS
    plt.xticks(color="black", fontsize="x-large")
    plt.yticks(color="black", fontsize="x-large")
    plt.tick_params(axis="both", width=1.2)
    # LEGEND
    handles, labels = ax.get_legend_handles_labels()
    labels = ["\n".join(wrap(l, 13)) for l in labels]
    lgnd = plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(1.37, 1.02),
        ncol=2,
        columnspacing=0.3,
        shadow=True,
        prop={"family": "DejaVu Sans", "weight": "heavy", "size": "large"},
    )
    for lh in lgnd.legendHandles:
        # lh._sizes = [750]
        lh.set_linewidth(3)
    ax.grid(zorder=-10, color="black", alpha=0.5)


def describe_powerlaw_fit(fit: powerlaw.Fit, label: Optional[str] = None):
    """
    Return short dict of fit powerlaw attributes.
    """
    base = {ALPHA: fit.alpha, EXPONENT: -(fit.alpha - 1), CUT_OFF: fit.xmin}
    if label is None:
        return base
    else:
        return {f"{label} {key}": value for key, value in base.items()}
