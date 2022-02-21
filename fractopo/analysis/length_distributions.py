"""
Utilities for analyzing and plotting length distributions for line data.
"""
import logging
from dataclasses import dataclass
from enum import Enum, unique
from itertools import chain, cycle
from textwrap import wrap
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import powerlaw
import seaborn as sns
import sklearn.metrics as sklm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

from fractopo import general

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

    TODO: Move functionality from MultiLengthDistribution back here.
    """

    name: str
    lengths: np.ndarray
    area_value: float
    using_branches: bool

    def __post_init__(self):
        """
        Filter lengths lower than general.MINIMUM_LINE_LENGTH.

        Also log the creation parameters.
        """
        # Filter lengths lower than general.MINIMUM_LINE_LENGTH.
        # Lengths lower than the value can cause runtime issues.
        filtered_lengths = self.lengths[self.lengths > general.MINIMUM_LINE_LENGTH]

        # Calculate proportion for logging purposes
        filtered_proportion = (len(self.lengths) - len(filtered_lengths)) / len(
            self.lengths
        )
        logging.info(
            "Created LengthDistribution instance.",
            extra=dict(
                LengthDistribution_name=self.name,
                min_length=self.lengths.min(),
                max_length=self.lengths.max(),
                area_value=self.area_value,
                using_branches=self.using_branches,
                filtered_proportion=filtered_proportion,
            ),
        )
        self.lengths = filtered_lengths


@unique
class Dist(Enum):

    """
    Enums of powerlaw model types.
    """

    POWERLAW = "power_law"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    TRUNCATED_POWERLAW = "truncated_power_law"


class SilentFit(powerlaw.Fit):

    """
    Wrap powerlaw.Fit for the singular purpose of silencing output.

    Silences output both to stdout and stderr.
    """

    def __init__(
        self,
        data,
        discrete=False,
        xmin=None,
        xmax=None,
        verbose=True,
        fit_method="Likelihood",
        estimate_discrete=True,
        discrete_approximation="round",
        sigma_threshold=None,
        parameter_range=None,
        fit_optimizer=None,
        xmin_distance="D",
        xmin_distribution="power_law",
        **kwargs,
    ):
        """
        Override Fit.__init__ to silence output.
        """
        with general.silent_output("__init__"):
            super().__init__(
                data,
                discrete=discrete,
                xmin=xmin,
                xmax=xmax,
                verbose=verbose,
                fit_method=fit_method,
                estimate_discrete=estimate_discrete,
                discrete_approximation=discrete_approximation,
                sigma_threshold=sigma_threshold,
                parameter_range=parameter_range,
                fit_optimizer=fit_optimizer,
                xmin_distance=xmin_distance,
                xmin_distribution=xmin_distribution,
                **kwargs,
            )

    def __getattribute__(self, name):
        """
        Get attribute with silent output.

        Also wraps all callables (~instance methods) with silent_output. The
        stdout and stderr is reported with logging.info so it is not lost.
        """
        with general.silent_output("__getattribute__"):
            attribute = super().__getattribute__(name)
        if callable(attribute):
            return general.wrap_silence(attribute)
        return attribute


def numpy_polyfit(log_lengths: np.ndarray, log_ccm: np.ndarray) -> Tuple[float, float]:
    """
    Fit numpy polyfit to data.
    """
    vals = np.polyfit(log_lengths, log_ccm, 1)
    assert len(vals) == 2
    return vals


def scikit_linear_regression(
    log_lengths: np.ndarray, log_ccm: np.ndarray
) -> Tuple[float, float]:
    """
    Fit using scikit LinearRegression.
    """
    model = LinearRegression().fit(log_lengths.reshape((-1, 1)), log_ccm)
    coefs = model.coef_
    intercept = model.intercept_

    assert len(coefs) == 1
    assert isinstance(intercept, float)
    m_value = coefs[0]
    assert isinstance(m_value, float)

    return m_value, intercept


@dataclass
class MultiLengthDistribution:

    """
    Multi-scale length distribution.
    """

    distributions: List[LengthDistribution]
    cut_distributions: bool
    using_branches: bool
    fitter: Callable[[np.ndarray, np.ndarray], Tuple[float, float]] = numpy_polyfit

    _fit_to_multi_scale_lengths: Optional[Tuple[np.ndarray, float, float]] = None
    _create_normalized_distributions: Optional[
        Tuple[List[np.ndarray], List[np.ndarray]]
    ] = None

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
                self.fitter.__name__,
            )
        )

    def create_normalized_distributions(
        self,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create normalized ccm of distributions.
        """
        if self._create_normalized_distributions is None:

            (
                truncated_length_array_all,
                ccm_array_normed_all,
            ) = create_normalized_distributions(
                distributions=self.distributions,
                cut_distributions=self.cut_distributions,
            )
            self._create_normalized_distributions = (
                truncated_length_array_all,
                ccm_array_normed_all,
            )
        return self._create_normalized_distributions

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
        concatted = np.concatenate(self.truncated_length_array_all)
        assert all(isinstance(value, float) for value in concatted)
        return concatted

    @property
    def concatted_ccm(self) -> np.ndarray:
        """
        Concat ccm into single array.
        """
        return np.concatenate(self.ccm_array_normed_all)

    def fit_to_multi_scale_lengths(self) -> Tuple[np.ndarray, float, float]:
        """
        Fit np.polyfit to multi-scale lengths.
        """
        if self._fit_to_multi_scale_lengths is None:

            y_fit, m_value, constant = fit_to_multi_scale_lengths(
                lengths=self.concatted_lengths,
                ccm=self.concatted_ccm,
                fitter=self.fitter,
            )
            self._fit_to_multi_scale_lengths = y_fit, m_value, constant
        return self._fit_to_multi_scale_lengths

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
        SilentFit(length_array, xmin=cut_off, verbose=False)
        if cut_off is not None
        else SilentFit(length_array, verbose=False)
    )
    return fit


def plot_length_data_on_ax(
    ax: Axes,
    length_array: np.ndarray,
    ccm_array: np.ndarray,
    label: str,
    truncated: bool = True,
):
    """
    Plot length data on given ax.

    Sets ax scales to logarithmic.
    """
    ax.scatter(
        x=length_array,
        y=ccm_array,
        s=25,
        label=label if truncated else None,
        alpha=1.0 if truncated else 0.02,
        color="black" if truncated else "brown",
        marker="x",
    )


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

    if len(length_array) == 0:
        logging.error(
            "Empty length array passed into plot_distribution_fits. "
            "Fit and plot will be invalid."
        )
        return fit, fig, ax

    # Get the x, y data from fit
    truncated_length_array, ccm_array = fit.ccdf()
    full_length_array, full_ccm_array = fit.ccdf(original_data=True)

    # Normalize full_ccm_array to the truncated ccm_array
    full_ccm_array = full_ccm_array / (
        full_ccm_array[len(full_ccm_array) - len(ccm_array)] / ccm_array.max()
    )

    # Plot length scatter plot
    plot_length_data_on_ax(ax, truncated_length_array, ccm_array, label)
    plot_length_data_on_ax(
        ax, full_length_array, full_ccm_array, label, truncated=False
    )

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
    rounded_exponent = round(calculate_exponent(fit.alpha), 3)

    # Set title with exponent
    ax.set_title(f"Power-law Exponent = ${rounded_exponent}$")

    plot_axvline = cut_off is None or cut_off != 0.0
    if plot_axvline:
        # Indicate cut-off if cut-off is not given as 0.0
        ax.axvline(
            truncated_length_array.min(),
            linestyle="dotted",
            color="black",
            alpha=0.8,
            label="Cut-Off",
        )
        ax.text(
            truncated_length_array.min(),
            ccm_array.min(),
            f"{round(truncated_length_array.min(), 2)} m",
            rotation=90,
            horizontalalignment="right",
            fontsize="small",
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
        labelpad=14,
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
        loc="upper right",
        # bbox_to_anchor=(1.37, 1.02),
        ncol=2,
        columnspacing=0.3,
        # shadow=True,
        # prop={"family": "DejaVu Sans", "weight": "heavy", "size": "large"},
        framealpha=0.8,
        facecolor="white",
    )
    for lh in lgnd.legendHandles:
        # lh._sizes = [750]
        lh.set_linewidth(3)
    ax.grid(zorder=-10, color="black", alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")


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
        Dist.TRUNCATED_POWERLAW.value + " "
        # + EXPONENT: -(fit.truncated_power_law.alpha - 1),
        + EXPONENT: calculate_exponent(fit.truncated_power_law.alpha),
        # Fit statistics
        Dist.LOGNORMAL.value + " " + LOGLIKELIHOOD: fit.lognormal.loglikelihood,
        Dist.EXPONENTIAL.value + " " + LOGLIKELIHOOD: fit.exponential.loglikelihood,
        Dist.TRUNCATED_POWERLAW.value
        + " "
        + LOGLIKELIHOOD: fit.truncated_power_law.loglikelihood,
    }


def calculate_exponent(alpha: float):
    """
    Calculate exponent from powerlaw.alpha.
    """
    return -(alpha - 1)


def cut_off_proportion_of_data(fit: powerlaw.Fit, length_array: np.ndarray) -> float:
    """
    Get the proportion of data cut off by `powerlaw` cut off.

    If no fit is passed the cut off is the one used in `automatic_fit`.
    """
    arr_less_than = length_array < fit.xmin
    assert isinstance(arr_less_than, np.ndarray)
    return sum(arr_less_than) / len(length_array) if len(length_array) > 0 else 0.0


def describe_powerlaw_fit(
    fit: powerlaw.Fit, length_array: np.ndarray, label: Optional[str] = None
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
        Dist.POWERLAW.value + " " + EXPONENT: calculate_exponent(fit.alpha),
        Dist.POWERLAW.value + " " + CUT_OFF: fit.xmin,
        Dist.POWERLAW.value + " " + SIGMA: fit.power_law.sigma,
        **all_fit_attributes_dict(fit),
        "lengths cut off proportion": cut_off_proportion_of_data(
            fit=fit, length_array=length_array
        ),
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
    fitter: Callable[[np.ndarray, np.ndarray], Tuple[float, float]] = numpy_polyfit,
) -> Tuple[np.ndarray, float, float]:
    """
    Fit np.polyfit to multiscale length distributions.

    Returns the fitted values, exponent and constant of fit.
    """
    log_lengths_sorted, log_ccm_sorted = sort_and_log_lengths_and_ccm(
        lengths=lengths, ccm=ccm
    )

    # Fit numpy polyfit to data
    fit_vals = fitter(log_lengths_sorted, log_ccm_sorted)

    assert len(fit_vals) == 2

    m_value, constant = fit_vals
    logging.info(
        "Fitted with fitter.",
        extra=dict(fitter=fitter, m_value=m_value, constant=constant),
    )

    # Calculate the fitted values of y
    y_fit = calculate_fitted_values(
        log_lengths=log_lengths_sorted,
        m_value=m_value,
        constant=constant,
    )

    return y_fit, m_value, constant


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

    TODO: Duplicate method name in MultiLengthDistribution.
    """
    truncated_length_array_all = []
    ccm_array_normed_all = []

    for length_distribution in distributions:
        # no_cut_off_value = 1e-18
        assert general.MINIMUM_LINE_LENGTH < length_distribution.lengths.min()
        fit = determine_fit(
            length_array=length_distribution.lengths,
            cut_off=None if cut_distributions else general.MINIMUM_LINE_LENGTH,
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
        assert all(isinstance(length, float) for length in truncated_length_array)
        assert all(isinstance(value, (int, float)) for value in ccm_array_normed)
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


def apply_cut_off(length_array: np.ndarray, ccm_array: np.ndarray, cut_off: float):
    """
    Apply cut-off to lengths and associated ccm array.
    """
    are_truncated = length_array < cut_off
    return length_array[~are_truncated], ccm_array[~are_truncated]


def apply_cut_offs(full_length_arrays, full_ccm_arrays, cut_offs):
    """
    Apply cut-offs to length data.
    """
    cut_lengths_all, cut_ccm_all = [], []
    for arr, ccm, cut_off in zip(full_length_arrays, full_ccm_arrays, cut_offs):
        cut_lengths, cut_ccm = apply_cut_off(
            length_array=arr, ccm_array=ccm, cut_off=cut_off
        )
        cut_lengths_all.append(cut_lengths)
        cut_ccm_all.append(cut_ccm)

    return cut_lengths_all, cut_ccm_all


def _objective_function(
    cut_offs: np.ndarray,
    full_length_arrays=List[np.ndarray],
    full_ccm_arrays=List[np.ndarray],
) -> Tuple[np.ndarray, float, float, float]:
    cut_length_arrays, cut_ccm_arrays = apply_cut_offs(
        full_length_arrays=full_length_arrays,
        full_ccm_arrays=full_ccm_arrays,
        cut_offs=cut_offs,
    )

    length_array_concat = np.concatenate(cut_length_arrays)
    ccm_array_concat = np.concatenate(cut_ccm_arrays)

    y_fit, m_value, constant = fit_to_multi_scale_lengths(
        ccm=ccm_array_concat, lengths=length_array_concat
    )
    msle = sklm.mean_squared_log_error(ccm_array_concat, y_fit)
    return y_fit, m_value, constant, msle


def objective_function(
    cut_offs: np.ndarray,
    full_length_arrays=List[np.ndarray],
    full_ccm_arrays=List[np.ndarray],
) -> float:
    """
    Optimize multiscale fit.
    """
    y_fit, m_value, constant, msle = _objective_function(
        cut_offs=cut_offs,
        full_length_arrays=full_length_arrays,
        full_ccm_arrays=full_ccm_arrays,
    )
    return msle


def optimize_multi_scale_fit(
    full_length_arrays: List[np.ndarray],
    area_values: List[float],
    names: List[str],
    using_branches: bool,
):
    """
    Create optimized powerlaw fit for multi-scale data.
    """
    distributions = [
        LengthDistribution(
            name=name,
            lengths=lengths,
            area_value=area_value,
            using_branches=using_branches,
        )
        for name, lengths, area_value in zip(names, full_length_arrays, area_values)
    ]

    truncated_length_array_all, ccm_array_normed_all = create_normalized_distributions(
        distributions=distributions, cut_distributions=False
    )

    fits = [determine_fit(length_array=arr) for arr in truncated_length_array_all]
    xmins = [fit.xmin for fit in fits]

    bounds = [(min(arr), max(arr)) for arr in truncated_length_array_all]

    res = minimize(
        objective_function,
        x0=xmins,
        args=(truncated_length_array_all, ccm_array_normed_all),
        bounds=bounds,
    )

    y_fit, m_value, constant, msle = _objective_function(
        res.x,
        full_length_arrays=truncated_length_array_all,
        full_ccm_arrays=ccm_array_normed_all,
    )

    fig, axes = plot_multi_distributions_and_fit(
        truncated_length_array_all=truncated_length_array_all,
        ccm_array_normed_all=ccm_array_normed_all,
        concatted_lengths=np.concatenate(truncated_length_array_all),
        names=names,
        m_value=m_value,
        y_fit=y_fit,
    )

    return res, y_fit, m_value, constant, msle, fig, axes
