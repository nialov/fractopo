"""
Utilities for analyzing and plotting length distributions for line data.
"""
import logging
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from itertools import chain, cycle
from textwrap import wrap
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
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


@dataclass
class MultiLengthDistribution:

    """
    Multi-scale length distribution.
    """

    distributions: List[LengthDistribution]
    cut_distributions: bool
    using_branches: bool

    def __hash__(self) -> int:
        """
        Implement hashing for MultiLengthDistribution.
        """
        all_lengths = tuple(chain(*[ld.lengths for ld in self.distributions]))
        all_lengths_str = tuple(map(str, all_lengths))
        all_area_values_str = tuple(map(str, [ld.name for ld in self.distributions]))
        return hash(
            (
                all_lengths_str,
                self.cut_distributions,
                self.using_branches,
                all_area_values_str,
            )
        )

    @lru_cache(maxsize=None)
    def create_normalized_distributions(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create normalized ccm of distributions.
        """
        return create_normalized_distributions(
            distributions=self.distributions, cut_distributions=self.cut_distributions
        )

    @property
    def truncated_length_array_all(self) -> List[np.ndarray]:
        """
        Get truncated length array by cut-off.
        """
        return self.create_normalized_distributions()[0]

    @property
    def ccm_array_normed_all(self) -> List[np.ndarray]:
        """
        Get truncated ccm array by cut-off.
        """
        return self.create_normalized_distributions()[1]

    @property
    def concatted_lengths(self) -> np.ndarray:
        """
        Concat lengths into single array.
        """
        return np.concatenate(self.truncated_length_array_all)

    @property
    def concatted_ccm(self) -> np.ndarray:
        """
        Concat ccm into single array.
        """
        return np.concatenate(self.ccm_array_normed_all)

    @lru_cache(maxsize=None)
    def fit_to_multi_scale_lengths(self) -> Tuple[np.ndarray, float, float]:
        """
        Fit np.polyfit to multi-scale lengths.
        """
        return fit_to_multi_scale_lengths(
            lengths=self.concatted_lengths, ccm=self.concatted_ccm
        )

    @property
    def fitted_y_values(self) -> np.ndarray:
        """
        Get fitted y values.
        """
        return self.fit_to_multi_scale_lengths()[0]

    @property
    def m_value(self) -> float:
        """
        Get fitted m value.
        """
        return self.fit_to_multi_scale_lengths()[1]

    @property
    def constant(self) -> float:
        """
        Get fitted constant value.
        """
        return self.fit_to_multi_scale_lengths()[2]

    @property
    def names(self) -> List[str]:
        """
        Get length distribution names.
        """
        return [ld.name for ld in self.distributions]

    def plot_multi_length_distributions(self) -> Tuple[Figure, Axes]:
        """
        Plot multi-scale length distribution.
        """
        return plot_multi_distributions_and_fit(
            truncated_length_array_all=self.truncated_length_array_all,
            concatted_lengths=self.concatted_lengths,
            ccm_array_normed_all=self.ccm_array_normed_all,
            names=self.names,
            y_fit=self.fitted_y_values,
            m_value=self.m_value,
            using_branches=self.using_branches,
        )


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


def sort_and_log_lengths_and_ccm(
    lengths: np.ndarray, ccm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess lengths and ccm.

    Sorts them and calculates their natural logarithmic.
    """
    # Find sorted order of lengths
    length_sort_permutation = lengths.argsort()

    # Apply sorted order of lengths to both lengths and ccms
    log_lengths_sorted = np.log(lengths[length_sort_permutation])
    log_ccm_sorted = np.log(ccm[length_sort_permutation])

    return log_lengths_sorted, log_ccm_sorted


def calculate_fitted_values(
    log_lengths: np.ndarray, m_value: float, constant: float
) -> np.ndarray:
    """
    Calculate fitted values of y.
    """
    # Calculate the fitted values of y
    y_fit = np.exp(m_value * log_lengths + constant)

    return y_fit


def fit_to_multi_scale_lengths(
    ccm: np.ndarray,
    lengths: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit np.polyfit to multiscale length distributions.

    Returns the fitted values, exponent and constant of fit.
    """
    log_lengths_sorted, log_ccm_sorted = sort_and_log_lengths_and_ccm(
        lengths=lengths, ccm=ccm
    )

    # Fit numpy polyfit to data
    fit_vals = np.polyfit(log_lengths_sorted, log_ccm_sorted, 1)

    assert len(fit_vals) == 2

    m_value, constant = fit_vals

    # Calculate the fitted values of y
    y_fit = calculate_fitted_values(
        log_lengths=log_lengths_sorted,
        m_value=m_value,
        constant=constant,
    )
    return y_fit, m_value, constant


# def multi_scale_length_distribution_fit(
#     distributions: List[LengthDistribution],
#     auto_cut_off: bool,
#     using_branches: bool = False,
# ) -> Tuple[Figure, Axes]:
#     """
#     Plot multi scale length distributions and their fit.
#     """
#     # Gather variations of length distribution lengths and ccm
#     truncated_length_arrays = []
#     ccm_arrays_normed = []
#     names = []

#     # Iterate over LengthDistributions
#     for length_distribution in distributions:

#         # Determine fit for distribution
#         fit = determine_fit(
#             length_distribution.lengths, cut_off=None if auto_cut_off else 1e-8
#         )

#         # Get the full length data along with full ccm using the original
#         # data instead of fitted (should be same with cut_off==0.0)
#         full_length_array, full_ccm_array = fit.ccdf(original_data=True)

#         # Get boolean array where length is over cut_off
#         are_over_cut_off = full_length_array > fit.xmin

#         # Cut lengths and corresponding ccm to indexes where are over cut off
#         truncated_length_array = full_length_array[are_over_cut_off]
#         ccm_array = full_ccm_array[are_over_cut_off]

#         # Normalize ccm with area value
#         ccm_array_normed = ccm_array / length_distribution.area_value

#         # Gather results
#         truncated_length_arrays.append(truncated_length_array)
#         ccm_arrays_normed.append(ccm_array_normed)
#         names.append(length_distribution.name)

#     # Create concatenated version of collected lengths and ccms
#     ccm_arrays_normed_concat = np.concatenate(ccm_arrays_normed)
#     truncated_length_arrays_concat = np.concatenate(truncated_length_arrays)

#     # Determine polyfit to multiscale distributions
#     # Last variable is constant
#     y_fit, m, _ = polyfit_lengths(
#         ccm=ccm_arrays_normed_concat,
#         lengths=truncated_length_arrays_concat,
#     )

#     # Start making plot
#     fig, ax = plt.subplots(figsize=(7, 7))
#     setup_ax_for_ld(ax_for_setup=ax, using_branches=using_branches)
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_facecolor("oldlace")
#     ax.set_title(f"$Exponent = {m}$")

#     # Make color cycle
#     color_cycle = cycle(sns.color_palette("dark", 5))

#     # Plot length distributions
#     for name, truncated_length_array, ccm_array_normed in zip(
#         names, truncated_length_arrays, ccm_arrays_normed
#     ):
#         ax.scatter(
#             x=truncated_length_array,
#             y=ccm_array_normed,
#             s=25,
#             label=name,
#             marker="X",
#             color=next(color_cycle),
#         )

#     # Plot polyfit
#     ax.plot(
#         (truncated_length_arrays_concat[0], truncated_length_arrays_concat[-1]),
#         (y_fit[0], y_fit[-1]),
#         label="Polyfit",
#         linestyle="dashed",
#     )

#     # Add legend
#     plt.legend()

#     return fig, ax


def normalize_fit_to_area(
    fit: powerlaw.Fit, length_distribution: LengthDistribution
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize powerlaw.fit ccdf to area value.
    """
    # Get the full length data along with full ccm using the original
    # data instead of fitted (should be same with cut_off==0.0)
    full_length_array, full_ccm_array = fit.ccdf(original_data=True)

    # Get boolean array where length is over cut_off
    are_over_cut_off = full_length_array > fit.xmin

    assert isinstance(are_over_cut_off, np.ndarray)
    assert sum(are_over_cut_off) > 0

    # Cut lengths and corresponding ccm to indexes where are over cut off
    truncated_length_array = full_length_array[are_over_cut_off]
    ccm_array = full_ccm_array[are_over_cut_off]

    area_value = length_distribution.area_value
    assert area_value > 0
    # Normalize ccm with area value
    logging.info(
        "Normalizing ccm with area_value.",
        extra=dict(
            area_value=area_value,
            ccm_array_description=pd.Series(ccm_array).describe().to_dict(),
        ),
    )
    ccm_array_normed = ccm_array / area_value

    logging.info(
        "Normalized fit ccm.",
        extra=dict(
            sum_are_over_cut_off=sum(are_over_cut_off),
            fit_xmin=fit.xmin,
            amount_filtered=len(full_length_array) - len(truncated_length_array),
            length_distribution_area_value=area_value,
        ),
    )

    return truncated_length_array, ccm_array_normed


def create_normalized_distributions(
    distributions: List[LengthDistribution],
    cut_distributions: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create normalized ccms for all distributions.
    """
    truncated_length_array_all = []
    ccm_array_normed_all = []

    for length_distribution in distributions:
        no_cut_off_value = 1e-18
        assert no_cut_off_value < length_distribution.lengths.min()
        fit = determine_fit(
            length_array=length_distribution.lengths,
            cut_off=None if cut_distributions else no_cut_off_value,
        )

        truncated_length_array, ccm_array_normed = normalize_fit_to_area(
            fit=fit, length_distribution=length_distribution
        )
        logging.info(
            "Determined fit and normalized it.",
            extra=dict(
                fit_xmin=fit.xmin,
                normalized_ccm_description=pd.DataFrame(
                    {
                        "truncated_length_array": truncated_length_array,
                        "ccm_array_normed": ccm_array_normed,
                    }
                )
                .describe()
                .to_dict(),
            ),
        )
        truncated_length_array_all.append(truncated_length_array)
        ccm_array_normed_all.append(ccm_array_normed)

    return truncated_length_array_all, ccm_array_normed_all


def plot_multi_distributions_and_fit(
    truncated_length_array_all: List[np.ndarray],
    concatted_lengths: np.ndarray,
    ccm_array_normed_all: List[np.ndarray],
    names: List[str],
    y_fit: np.ndarray,
    m_value: float,
    using_branches: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Plot multi-scale length distribution.
    """
    # Start making plot
    fig, ax = plt.subplots(figsize=(7, 7))
    setup_ax_for_ld(ax_for_setup=ax, using_branches=using_branches)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_facecolor("oldlace")
    ax.set_title(f"$Exponent = {m_value}$")

    # Make color cycle
    color_cycle = cycle(sns.color_palette("dark", 5))

    # Plot length distributions
    for name, truncated_length_array, ccm_array_normed in zip(
        names, truncated_length_array_all, ccm_array_normed_all
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
        (concatted_lengths[0], concatted_lengths[-1]),
        (y_fit[0], y_fit[-1]),
        label="Polyfit",
        linestyle="dashed",
    )

    # Add legend
    plt.legend()

    return fig, ax
