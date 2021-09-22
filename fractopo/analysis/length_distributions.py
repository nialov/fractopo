"""
Utilities for analyzing and plotting length distributions for line data.
"""
from dataclasses import dataclass
from enum import Enum, unique
from itertools import cycle
from textwrap import wrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import powerlaw
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ALPHA = "alpha"
EXPONENT = "exponent"
CUT_OFF = "cut-off"
KOLM_DIST = "Kolmogorov-Smirnov distance D"
SIGMA = "sigma"
MU = "mu"
LAMBDA = "lambda"
LOGLIKELIHOOD = "loglikelihood"


@dataclass
class LengthDistribution:

    """
    Dataclass for length distributions.
    """

    name: str
    lengths: np.ndarray
    area_value: float


@unique
class Dist(Enum):

    """
    Enums of powerlaw model types.
    """

    POWERLAW = "power_law"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    TRUNCATED_POWERLAW = "truncated_power_law"


def determine_fit(
    length_array: np.ndarray, cut_off: Optional[float] = None
) -> powerlaw.Fit:
    """
    Determine powerlaw (along other) length distribution fits for given data.
    """
    fit = (
        powerlaw.Fit(length_array, xmin=cut_off, verbose=False)
        if cut_off is not None
        else powerlaw.Fit(length_array, verbose=False)
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
    fit_distribution: Dist,
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


def _setup_length_plot_axlims(
    ax: Axes,
    length_array: np.ndarray,
    ccm_array: np.ndarray,
    # cut_off: float,
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
    labels = ["\n".join(wrap(label, 13)) for label in labels]
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


def distribution_compare_dict(fit: powerlaw.Fit) -> Dict[str, float]:
    """
    Compose a dict of length distribution fit comparisons.
    """
    compare_dict = dict()
    for dist_enum_pairs in [
        (Dist.POWERLAW, Dist.LOGNORMAL),
        (Dist.POWERLAW, Dist.EXPONENTIAL),
        (Dist.LOGNORMAL, Dist.EXPONENTIAL),
        (Dist.POWERLAW, Dist.TRUNCATED_POWERLAW),
    ]:
        first, second = dist_enum_pairs[0].value, dist_enum_pairs[1].value
        r, p = fit.distribution_compare(first, second, normalized_ratio=True)
        compare_dict[f"{first} vs. {second} R"] = r
        compare_dict[f"{first} vs. {second} p"] = p
    return compare_dict


def all_fit_attributes_dict(fit: powerlaw.Fit) -> Dict[str, float]:
    """
    Collect 'all' fit attributes into a dict.
    """
    return {
        # **describe_powerlaw_fit(fit),
        # Attributes for remaking fits
        Dist.LOGNORMAL.value + " " + SIGMA: fit.lognormal.sigma,
        Dist.LOGNORMAL.value + " " + MU: fit.lognormal.mu,
        Dist.EXPONENTIAL.value + " " + LAMBDA: fit.exponential.Lambda,
        Dist.TRUNCATED_POWERLAW.value + " " + LAMBDA: fit.truncated_power_law.Lambda,
        Dist.TRUNCATED_POWERLAW.value + " " + ALPHA: fit.truncated_power_law.alpha,
        Dist.TRUNCATED_POWERLAW.value
        + " "
        + EXPONENT: -(fit.truncated_power_law.alpha - 1),
        # Fit statistics
        Dist.LOGNORMAL.value + " " + LOGLIKELIHOOD: fit.lognormal.loglikelihood,
        Dist.EXPONENTIAL.value + " " + LOGLIKELIHOOD: fit.exponential.loglikelihood,
        Dist.TRUNCATED_POWERLAW.value
        + " "
        + LOGLIKELIHOOD: fit.truncated_power_law.loglikelihood,
    }


def describe_powerlaw_fit(
    fit: powerlaw.Fit, label: Optional[str] = None
) -> Dict[str, float]:
    """
    Compose dict of fit powerlaw attributes and comparisons between fits.
    """
    base = {
        **distribution_compare_dict(fit),
        Dist.POWERLAW.value + " " + KOLM_DIST: fit.power_law.D,
        Dist.EXPONENTIAL.value + " " + KOLM_DIST: fit.exponential.D,
        Dist.LOGNORMAL.value + " " + KOLM_DIST: fit.lognormal.D,
        Dist.TRUNCATED_POWERLAW.value + " " + KOLM_DIST: fit.truncated_power_law.D,
        Dist.POWERLAW.value + " " + ALPHA: fit.alpha,
        Dist.POWERLAW.value + " " + EXPONENT: -(fit.alpha - 1),
        Dist.POWERLAW.value + " " + CUT_OFF: fit.xmin,
        Dist.POWERLAW.value + " " + SIGMA: fit.power_law.sigma,
        **all_fit_attributes_dict(fit),
    }
    if label is None:
        return base
    return {f"{label} {key}": value for key, value in base.items()}


def polyfit_to_multi_scale_lengths(
    all_ccms_concat: np.ndarray, all_length_arrs_concat: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Fit np.polyfit to multiscale length distributions.

    Returns the fitted values, exponent and constant of fit.
    """
    # Find sorted order of lengths
    length_sort_permutation = all_length_arrs_concat.argsort()

    # Apply sorted order of lengths to both lenghts and ccms
    all_length_arrs_concat_sorted_log = np.log(
        all_length_arrs_concat[length_sort_permutation]
    )
    all_ccms_concat_log = np.log(all_ccms_concat[length_sort_permutation])

    # Fit numpy polyfit to data
    fit_vals = np.polyfit(all_length_arrs_concat_sorted_log, all_ccms_concat_log, 1)

    if len(fit_vals) == 2:
        m, c = fit_vals[0], fit_vals[1]
    else:
        raise ValueError("Expected two values from np.polyfit.")

    # Calculate the fitted values of y
    y_fit = np.exp(m * all_length_arrs_concat_sorted_log + c)
    return y_fit, m, c


def multi_scale_length_distribution_fit(
    distributions: List[LengthDistribution],
    auto_cut_off: bool,
    using_branches: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Plot multi scale length distributions and their fit.
    """
    # Gather variations of length distribution lengths and ccm
    truncated_length_arrays = []
    ccm_arrays_normed = []
    names = []

    # Iterate over LengthDistributions
    for length_distribution in distributions:

        # Determine fit for distribution
        fit = determine_fit(
            length_distribution.lengths, cut_off=None if auto_cut_off else 1e-8
        )

        # Get the full length data along with full ccm using the original
        # data instead of fitted (should be same with cut_off==0.0)
        full_length_array, full_ccm_array = fit.ccdf(original_data=True)

        # Get boolean array where length is over cut_off
        are_over_cut_off = full_length_array > fit.xmin

        # Cut lengths and corresponding ccm to indexes where are over cut off
        truncated_length_array = full_length_array[are_over_cut_off]
        ccm_array = full_ccm_array[are_over_cut_off]

        # Normalize ccm with area value
        ccm_array_normed = ccm_array / length_distribution.area_value

        # Gather results
        truncated_length_arrays.append(truncated_length_array)
        ccm_arrays_normed.append(ccm_array_normed)
        names.append(length_distribution.name)

    # Create concatenated version of collected lengths and ccms
    ccm_arrays_normed_concat = np.concatenate(ccm_arrays_normed)
    truncated_length_arrays_concat = np.concatenate(truncated_length_arrays)

    # Determine polyfit to multiscale distributions
    # Last variable is constant
    y_fit, m, _ = polyfit_to_multi_scale_lengths(
        all_ccms_concat=ccm_arrays_normed_concat,
        all_length_arrs_concat=truncated_length_arrays_concat,
    )

    # Start making plot
    fig, ax = plt.subplots(figsize=(7, 7))
    setup_ax_for_ld(ax_for_setup=ax, using_branches=using_branches)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_facecolor("oldlace")
    ax.set_title(f"$Exponent = {m}$")

    # Make color cycle
    color_cycle = cycle(sns.color_palette("dark", 5))

    # Plot length distributions
    for name, truncated_length_array, ccm_array_normed in zip(
        names, truncated_length_arrays, ccm_arrays_normed
    ):
        ax.scatter(
            x=truncated_length_array,
            y=ccm_array_normed,
            s=25,
            label=name,
            marker="X",
            color=next(color_cycle),
        )

    # Plot polyfit
    ax.plot(
        (truncated_length_arrays_concat[0], truncated_length_arrays_concat[-1]),
        (y_fit[0], y_fit[-1]),
        label="Polyfit",
        linestyle="dashed",
    )

    # Add legend
    plt.legend()

    return fig, ax
