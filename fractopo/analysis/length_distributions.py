"""
Utilities for analyzing and plotting length distributions for line data.
"""
import logging
from dataclasses import dataclass
from enum import Enum, unique
from itertools import chain, cycle
from textwrap import wrap
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import powerlaw
import seaborn as sns
import sklearn.metrics as sklm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.optimize import OptimizeResult, shgo
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


@unique
class Dist(Enum):

    """
    Enums of powerlaw model types.
    """

    POWERLAW = "power_law"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    TRUNCATED_POWERLAW = "truncated_power_law"


def numpy_polyfit(log_lengths: np.ndarray, log_ccm: np.ndarray) -> Tuple[float, float]:
    """
    Fit numpy polyfit to data.
    """
    vals = np.polyfit(log_lengths, log_ccm, 1)
    assert len(vals) == 2
    return vals


@dataclass
class LengthDistribution:

    """
    Dataclass for length distributions.
    """

    lengths: np.ndarray
    area_value: float
    using_branches: bool = False
    name: str = ""

    _automatic_fit: Optional[powerlaw.Fit] = None

    def __post_init__(self):
        """
        Filter lengths lower than general.MINIMUM_LINE_LENGTH.

        Also log the creation parameters.
        """
        assert self.area_value > 0
        # Filter lengths lower than general.MINIMUM_LINE_LENGTH.
        # Lengths lower than the value can cause runtime issues.
        filtered_lengths = self.lengths[self.lengths > general.MINIMUM_LINE_LENGTH]

        # Calculate proportion for logging purposes
        filtered_proportion = (len(self.lengths) - len(filtered_lengths)) / len(
            self.lengths
        )
        high_filtered = filtered_proportion > 0.1
        logging_func = logging.warning if high_filtered else logging.info
        logging_func(
            "Created LengthDistribution instance."
            + (" High filter proportion!" if high_filtered else ""),
            extra=dict(
                LengthDistribution_name=self.name,
                min_length=self.lengths.min(),
                max_length=self.lengths.max(),
                area_value=self.area_value,
                using_branches=self.using_branches,
                filtered_proportion=filtered_proportion,
            ),
        )

        # Set preprocessed lengths back into lengths attribute
        self.lengths = filtered_lengths

    @property
    def automatic_fit(self) -> powerlaw.Fit:
        """
        Get automatic powerlaw Fit.
        """
        if self._automatic_fit is None:
            self._automatic_fit = determine_fit(length_array=self.lengths)
        return self._automatic_fit

    def manual_fit(self, cut_off: float):
        """
        Get manual powerlaw Fit.
        """
        return determine_fit(length_array=self.lengths, cut_off=cut_off)

    def generate_distributions(self, cut_off: float = general.MINIMUM_LINE_LENGTH):
        """
        Generate ccdf and truncated length data with cut_off.
        """
        lengths, ccm = sorted_lengths_and_ccm(
            lengths=self.lengths, area_value=self.area_value
        )
        lengths, ccm = apply_cut_off(lengths=lengths, ccm=ccm, cut_off=cut_off)
        return lengths, ccm


class Polyfit(NamedTuple):

    """
    Results of a polyfit to length data.
    """

    y_fit: np.ndarray
    m_value: float
    constant: float
    score: float


class MultiScaleOptimizationResult(NamedTuple):

    """
    Results of scipy.optimize.shgo on length data.
    """

    polyfit: Polyfit
    cut_offs: np.ndarray
    optimize_result: OptimizeResult
    x0: np.ndarray
    bounds: np.ndarray
    proportions_of_data: List[float]


@dataclass
class MultiLengthDistribution:

    """
    Multi length distribution.
    """

    distributions: List[LengthDistribution]
    using_branches: bool
    fitter: Callable[[np.ndarray, np.ndarray], Tuple[float, float]] = numpy_polyfit
    cut_offs: Optional[List[float]] = None

    # Private caching attributes
    _fit_to_multi_scale_lengths: Optional[Tuple[np.ndarray, float, float]] = None
    _normalized_distributions: Optional[
        Tuple[List[np.ndarray], List[np.ndarray]]
    ] = None
    _optimized: bool = False

    def __hash__(self) -> int:
        """
        Implement hashing for MultiLengthDistribution.
        """
        all_lengths = tuple(chain(*[ld.lengths for ld in self.distributions]))
        all_lengths_str = tuple(map(str, all_lengths))
        all_area_values_str = tuple(map(str, [ld.name for ld in self.distributions]))
        cut_offs_tuple = (
            self.cut_offs if self.cut_offs is None else tuple(self.cut_offs)
        )
        return hash(
            (
                all_lengths_str,
                cut_offs_tuple,
                self.using_branches,
                all_area_values_str,
                self.fitter.__name__,
            )
        )

    def normalized_distributions(
        self, automatic_cut_offs: bool
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Create normalized and truncated lengths and ccms.
        """
        # Collect length and ccm arrays after their normalization
        truncated_length_array_all, ccm_array_normed_all = [], []

        # Iterate over distributions
        for idx, distribution in enumerate(self.distributions):

            # Default cut_off is a static value to reduce errors
            # due to very low decimal numbers
            cut_off = general.MINIMUM_LINE_LENGTH
            if automatic_cut_offs:

                # Use automatic cut off determined by powerlaw
                cut_off = distribution.automatic_fit.xmin
            elif self.cut_offs is not None:

                # Use manual cut offs given as inputs
                cut_off = self.cut_offs[idx]

            assert isinstance(cut_off, float)

            # Resolve the lengths and ccm
            (
                truncated_length_array,
                ccm_array_normed,
            ) = distribution.generate_distributions(cut_off=cut_off)

            # Collect
            truncated_length_array_all.append(truncated_length_array)
            ccm_array_normed_all.append(ccm_array_normed)

        # Return all length arrays and all ccm arrays
        return truncated_length_array_all, ccm_array_normed_all

    @property
    def names(self) -> List[str]:
        """
        Get length distribution names.
        """
        return [ld.name for ld in self.distributions]

    def plot_multi_length_distributions(
        self,
        automatic_cut_offs: bool,
        scorer: Callable[[np.ndarray, np.ndarray], float] = sklm.mean_squared_log_error,
    ) -> Tuple[Polyfit, Figure, Axes]:
        """
        Plot multi-scale length distribution.
        """
        # Get length arrays and ccm arrays
        (
            truncated_length_array_all,
            ccm_array_normed_all,
        ) = self.normalized_distributions(automatic_cut_offs=automatic_cut_offs)

        # Concatenate
        lengths_concat = np.concatenate(truncated_length_array_all)
        ccm_concat = np.concatenate(ccm_array_normed_all)

        # Fit a powerlaw to the multi dataset values
        polyfit = fit_to_multi_scale_lengths(
            ccm=ccm_concat, lengths=lengths_concat, fitter=self.fitter, scorer=scorer
        )
        fig, ax = plot_multi_distributions_and_fit(
            truncated_length_array_all=truncated_length_array_all,
            ccm_array_normed_all=ccm_array_normed_all,
            names=self.names,
            polyfit=polyfit,
            using_branches=self.using_branches,
        )
        return polyfit, fig, ax

    def optimize_cut_offs(
        self,
        shgo_kwargs: Dict[str, Any] = dict(),
        scorer: Callable[[np.ndarray, np.ndarray], float] = sklm.mean_squared_log_error,
    ) -> Tuple[MultiScaleOptimizationResult, "MultiLengthDistribution"]:
        """
        Get cut-off optimized MultiLengthDistribution.
        """
        opt_result = self.optimized_multi_scale_fit(
            shgo_kwargs=shgo_kwargs, scorer=scorer
        )
        optimized_mld = MultiLengthDistribution(
            distributions=self.distributions,
            using_branches=self.using_branches,
            cut_offs=list(opt_result.cut_offs),
            _optimized=True,
            fitter=self.fitter,
        )
        return opt_result, optimized_mld

    def optimized_multi_scale_fit(
        self,
        scorer: Callable[[np.ndarray, np.ndarray], float],
        shgo_kwargs: Dict[str, Any] = dict(),
    ) -> MultiScaleOptimizationResult:
        """
        Use scipy.optimize.shgo to optimize fit.
        """
        # Use automatic powerlaw fitting to get first guesses for optimization
        xmins = [ld.automatic_fit.xmin for ld in self.distributions]
        # xmins = [1.0 for ld in self.distributions]

        # Get all arrays of lengths and use them to define
        # lower and upper bounds for cut-offs
        truncated_length_array_all, _ = self.normalized_distributions(
            automatic_cut_offs=False
        )

        # Bound to upper 99 quantile
        bounds = [np.quantile(arr, (0.0, 1.0)) for arr in truncated_length_array_all]
        # bounds = [(0.0, 1.0) for _ in truncated_length_array_all]

        def constrainer(
            cut_offs: np.ndarray, distributions: List[LengthDistribution]
        ) -> float:
            """
            Constrain sample count to 2 or more.
            """
            lens = 0
            for dist, cut_off in zip(distributions, cut_offs):
                ls, _ = dist.generate_distributions(cut_off=cut_off)
                lens += len(ls)

            # sample count should be 2 or more
            return lens - 1.5

        constraints = {
            "type": "ineq",
            "fun": constrainer,
            "args": (self.distributions,),
        }

        # Run the shgo global optimization to find optimal cut-off values
        # that result in the minimum value of score (loss function).
        # Values will be stored in the x attribute of optimize_result
        optimize_result = shgo(
            optimize_cut_offs,
            args=(self.distributions, self.fitter, scorer),
            bounds=bounds,
            constraints=constraints,
            **shgo_kwargs,
        )
        assert isinstance(optimize_result, OptimizeResult)

        # Use the optimized cut-offs to determine the resulting exponent
        polyfit, proportions = _optimize_cut_offs(
            optimize_result.x,
            distributions=self.distributions,
            fitter=self.fitter,
            scorer=scorer,
        )

        # Return a composed object of the results for easier parsing downstream
        return MultiScaleOptimizationResult(
            polyfit=polyfit,
            cut_offs=optimize_result.x,
            optimize_result=optimize_result,
            x0=np.array(xmins),
            bounds=np.array(bounds),
            proportions_of_data=proportions,
        )


def determine_fit(
    length_array: np.ndarray, cut_off: Optional[float] = None
) -> powerlaw.Fit:
    """
    Determine powerlaw (along other) length distribution fits for given data.
    """
    fit = (
        # Use cut-off if it is given
        SilentFit(length_array, xmin=cut_off, verbose=False)
        # Otherwise powerlaw determines it automatically using the data
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
    scorer: Callable[[np.ndarray, np.ndarray], float] = sklm.mean_squared_log_error,
) -> Polyfit:
    """
    Fit np.polyfit to multiscale length distributions.

    Returns the fitted values, exponent and constant of fit
    within a ``Polyfit`` instance.
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
    score = scorer(ccm, y_fit)
    assert isinstance(score, float)

    return Polyfit(y_fit=y_fit, m_value=m_value, constant=constant, score=score)


def r2_scorer(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Score fit with r2 metric.

    Changes the scoring to be best at a value of 0.
    """
    score = abs(1 - sklm.r2_score(y_true=y_true, y_pred=y_predicted))
    assert isinstance(score, float)
    return score


def plot_multi_distributions_and_fit(
    truncated_length_array_all: List[np.ndarray],
    ccm_array_normed_all: List[np.ndarray],
    names: List[str],
    polyfit: Polyfit,
    using_branches: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Plot multi-scale length distribution.
    """
    # Start making plot
    fig, ax = plt.subplots(figsize=(7, 7))
    setup_ax_for_ld(ax_for_setup=ax, using_branches=using_branches)
    ax.set_facecolor("oldlace")
    ax.set_title(f"$Exponent = {polyfit.m_value:.2f} Score = {polyfit.score:.2f}$")

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
    y_fit = polyfit.y_fit
    concatted_lengths = np.concatenate(truncated_length_array_all)
    ax.plot(
        (concatted_lengths[0], concatted_lengths[-1]),
        (y_fit[0], y_fit[-1]),
        label="Polyfit",
        linestyle="dashed",
    )

    # Add legend
    plt.legend()

    return fig, ax


def _optimize_cut_offs(
    cut_offs: np.ndarray,
    distributions: List[LengthDistribution],
    fitter: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
    scorer: Callable[[np.ndarray, np.ndarray], float],
) -> Tuple[Polyfit, List[float]]:

    # Each length and associated ccm must be collected
    cut_length_arrays, cut_ccm_arrays, proportions = [], [], []

    # Iterate over given distributions and cut-offs
    for dist, cut_off in zip(distributions, cut_offs):

        # Use LengthDistribution.apply_cut_off to get truncated and normalized
        # length and ccm data
        cut_lengths, cut_ccm = dist.generate_distributions(cut_off=cut_off)
        cut_off_proportion = len(cut_lengths) / len(dist.lengths)
        cut_length_arrays.append(cut_lengths)
        cut_ccm_arrays.append(cut_ccm)
        proportions.append(cut_off_proportion)

    # Concatenate length and ccm data into single arrays
    length_array_concat = np.concatenate(cut_length_arrays)
    ccm_array_concat = np.concatenate(cut_ccm_arrays)

    # Fit a regression fit to the concatenated arrays
    polyfit = fit_to_multi_scale_lengths(
        ccm=ccm_array_concat, lengths=length_array_concat, fitter=fitter, scorer=scorer
    )

    return polyfit, proportions


def optimize_cut_offs(
    cut_offs: np.ndarray,
    distributions: List[LengthDistribution],
    fitter: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
    scorer: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """
    Optimize multiscale fit.

    Requirements for the optimization function are that the function must take
    one argument of 1-d array and return a single float. It can take
    static arguments (distributions, fitter).
    """
    polyfit, _ = _optimize_cut_offs(
        cut_offs=cut_offs,
        distributions=distributions,
        fitter=fitter,
        scorer=scorer,
    )
    return polyfit.score


def scikit_linear_regression(
    log_lengths: np.ndarray, log_ccm: np.ndarray
) -> Tuple[float, float]:
    """
    Fit using scikit LinearRegression.
    """
    # Fit the regression
    model = LinearRegression().fit(log_lengths.reshape((-1, 1)), log_ccm)

    # Get coefficients and intercept (==m_value and constant)
    coefs = model.coef_
    intercept = model.intercept_

    assert len(coefs) == 1
    assert isinstance(intercept, float)
    m_value = coefs[0]
    assert isinstance(m_value, float)

    return m_value, intercept


def sorted_lengths_and_ccm(
    lengths: np.ndarray, area_value: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get (normalized) complementary cumulative number array.

    Give area_value as None to **not** normalize.

    >>> lengths = np.array([2, 4, 8, 16, 32])
    >>> area_value = 10.0
    >>> sorted_lengths_and_ccm(lengths, area_value)
    (array([ 2,  4,  8, 16, 32]), array([0.1 , 0.08, 0.06, 0.04, 0.02]))

    >>> lengths = np.array([2, 4, 8, 16, 32])
    >>> area_value = None
    >>> sorted_lengths_and_ccm(lengths, area_value)
    (array([ 2,  4,  8, 16, 32]), array([1. , 0.8, 0.6, 0.4, 0.2]))
    """
    # use powerlaw.Fit to determine ccm
    lengths, ccm = SilentFit(data=lengths, xmin=general.MINIMUM_LINE_LENGTH).ccdf(
        original_data=True
    )

    if area_value is not None:

        # Normalize with area
        ccm = ccm / area_value

    return lengths, ccm


def apply_cut_off(
    lengths: np.ndarray, ccm: np.ndarray, cut_off: float = general.MINIMUM_LINE_LENGTH
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply cut-off to length data and associated ccm.

    >>> lengths = np.array([2, 4, 8, 16, 32])
    >>> ccm = np.array([1. , 0.8, 0.6, 0.4, 0.2])
    >>> cut_off = 4.5
    >>> apply_cut_off(lengths, ccm, cut_off)
    (array([ 8, 16, 32]), array([0.6, 0.4, 0.2]))
    """
    are_over_cut_off = lengths > cut_off

    return lengths[are_over_cut_off], ccm[are_over_cut_off]
