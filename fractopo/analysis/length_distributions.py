"""
Utilities for analyzing and plotting length distributions for line data.
"""
import logging
from dataclasses import dataclass
from enum import Enum, unique
from itertools import chain, cycle
from textwrap import fill
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

SCORER_NAMES: Dict[Callable, str] = {
    sklm.mean_squared_log_error: "MSLE",
    sklm.r2_score: "$R^2$",
    general.r2_scorer: "$R^2$ (inv.)",
}


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
    if isinstance(vals, tuple):
        return vals
    elif isinstance(vals, np.ndarray):
        return vals[0], vals[1]
    else:
        raise TypeError(
            "Expected np.polyfit results to be a tuple or an array."
            f" Got {vals} with type: {type(vals)}."
        )


@dataclass
class LengthDistribution:

    """
    Dataclass for length distributions.
    """

    lengths: np.ndarray
    area_value: float
    using_branches: bool
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
        filtered_proportion = (
            ((len(self.lengths) - len(filtered_lengths)) / len(self.lengths))
            if len(self.lengths) > 0
            else 0.0
        )
        high_filtered = filtered_proportion > 0.1
        logging_func = logging.warning if high_filtered else logging.info
        logging_func(
            "Created LengthDistribution instance."
            + (" High filter proportion!" if high_filtered else "")
            # extra=dict(
            #     LengthDistribution_name=self.name,
            #     min_length=self.lengths.min() if len(self.lengths) > 0 else None,
            #     max_length=self.lengths.max() if len(self.lengths) > 0 else None,
            #     is_empty=len(self.lengths) == 0,
            #     area_value=self.area_value,
            #     using_branches=self.using_branches,
            #     filtered_proportion=filtered_proportion,
            # ),
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
    scorer: Callable[[np.ndarray, np.ndarray], float]


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
    ) -> Tuple[
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        """
        Create normalized and truncated lengths and ccms.
        """
        # Collect length and ccm arrays after their normalization
        (
            truncated_length_array_all,
            ccm_array_normed_all,
            full_length_array_all,
            full_ccm_array_normed_all,
            cut_offs,
        ) = ([], [], [], [], [])

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
            (
                full_length_array,
                full_ccm_array_normed,
            ) = distribution.generate_distributions(cut_off=general.MINIMUM_LINE_LENGTH)

            # Collect
            truncated_length_array_all.append(truncated_length_array)
            ccm_array_normed_all.append(ccm_array_normed)
            full_length_array_all.append(full_length_array)
            full_ccm_array_normed_all.append(full_ccm_array_normed)

            cut_offs.append(cut_off)

        # Return all length arrays and all ccm arrays
        return (
            truncated_length_array_all,
            ccm_array_normed_all,
            full_length_array_all,
            full_ccm_array_normed_all,
            cut_offs,
        )

    @property
    def names(self) -> List[str]:
        """
        Get length distribution names.
        """
        return [ld.name for ld in self.distributions]

    def plot_multi_length_distributions(
        self,
        automatic_cut_offs: bool,
        plot_truncated_data: bool,
        scorer: Callable[[np.ndarray, np.ndarray], float] = sklm.mean_squared_log_error,
    ) -> Tuple[Polyfit, Figure, Axes]:
        """
        Plot multi-scale length distribution.
        """
        # Get length arrays and ccm arrays
        (
            truncated_length_array_all,
            ccm_array_normed_all,
            full_length_array_all,
            full_ccm_array_normed_all,
            cut_offs,
        ) = self.normalized_distributions(automatic_cut_offs=automatic_cut_offs)

        # Concatenate
        lengths_concat = np.concatenate(truncated_length_array_all)
        ccm_concat = np.concatenate(ccm_array_normed_all)
        # full_lengths_concat = np.concatenate(full_length_array_all)
        # full_ccm_concat = np.concatenate(full_ccm_array_normed_all)

        # Fit a powerlaw to the multi dataset values
        polyfit = fit_to_multi_scale_lengths(
            ccm=ccm_concat, lengths=lengths_concat, fitter=self.fitter, scorer=scorer
        )
        fig, ax = plot_multi_distributions_and_fit(
            truncated_length_array_all=truncated_length_array_all,
            ccm_array_normed_all=ccm_array_normed_all,
            full_length_array_all=full_length_array_all,
            full_ccm_array_normed_all=full_ccm_array_normed_all,
            cut_offs=cut_offs,
            names=self.names,
            polyfit=polyfit,
            using_branches=self.using_branches,
            plot_truncated_data=plot_truncated_data,
        )
        return polyfit, fig, ax

    def optimize_cut_offs(
        self,
        shgo_kwargs: Optional[Dict[str, Any]] = None,
        scorer: Callable[[np.ndarray, np.ndarray], float] = sklm.mean_squared_log_error,
    ) -> Tuple[MultiScaleOptimizationResult, "MultiLengthDistribution"]:
        """
        Get cut-off optimized MultiLengthDistribution.
        """
        opt_result = self.optimized_multi_scale_fit(
            shgo_kwargs=shgo_kwargs if shgo_kwargs is not None else {}, scorer=scorer
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
        shgo_kwargs: Dict[str, Any],
    ) -> MultiScaleOptimizationResult:
        """
        Use scipy.optimize.shgo to optimize fit.
        """
        # Use automatic powerlaw fitting to get first guesses for optimization
        xmins = [ld.automatic_fit.xmin for ld in self.distributions]
        # xmins = [1.0 for ld in self.distributions]

        # Get all arrays of lengths and use them to define
        # lower and upper bounds for cut-offs
        truncated_length_array_all, _, _, _, _ = self.normalized_distributions(
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


@general.JOBLIB_CACHE.cache
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


def plot_fit_on_ax(
    ax: Axes,
    fit: powerlaw.Fit,
    fit_distribution: Dist,
    use_probability_density_function: bool,
) -> None:
    """
    Plot powerlaw model to ax.
    """
    plot_func = "plot_ccdf" if not use_probability_density_function else "plot_pdf"
    if fit_distribution == Dist.POWERLAW:
        getattr(fit.power_law, plot_func)(
            ax=ax, label="Powerlaw", linestyle="--", color="red"
        )
    elif fit_distribution == Dist.LOGNORMAL:
        getattr(fit.lognormal, plot_func)(
            ax=ax, label="Lognormal", linestyle="--", color="lime"
        )
    elif fit_distribution == Dist.EXPONENTIAL:
        getattr(fit.exponential, plot_func)(
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

    # TODO: Anomalous very low value lengths can mess up xlims
    left = np.quantile(length_array, 0.01) / 10
    # left = length_array.min() / 10
    right = length_array.max() * 10
    bottom = ccm_array.min() / 20
    top = ccm_array.max() * 100
    try:
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
    except ValueError:
        logging.error("Failed to set up x and y limits.", exc_info=True)
        # Don't try setting if it errors
        pass


def plot_distribution_fits(
    length_array: np.ndarray,
    label: str,
    using_branches: bool,
    use_probability_density_function: bool,
    cut_off: Optional[float] = None,
    fit: Optional[powerlaw.Fit] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    fits_to_plot: Tuple[Dist, ...] = (Dist.EXPONENTIAL, Dist.LOGNORMAL, Dist.POWERLAW),
) -> Tuple[powerlaw.Fit, Figure, Axes]:
    """
    Plot length distribution and `powerlaw` fits.

    If a powerlaw.Fit is not given it will be automatically determined (using
    the optionally given cut_off).
    """
    if fit is None:
        # Determine powerlaw, exponential, lognormal fits
        fit = determine_fit(length_array, cut_off)

    if fig is None:
        if ax is None:
            # Create figure, ax
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig_maybe = ax.get_figure()
            assert isinstance(fig_maybe, Figure)
            fig = fig_maybe

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert ax is not None
    assert fit is not None

    if len(length_array) == 0:
        logging.error(
            "Empty length array passed into plot_distribution_fits. "
            "Fit and plot will be invalid."
        )
        return fit, fig, ax

    # Get the x, y data from fit
    # y values are either the complementary cumulative distribution function
    # or the probability density function
    # Depends on use_probability_density_function boolean
    cut_off_is_lower = fit.xmin < length_array.min()
    if not use_probability_density_function:
        # complementary cumulative distribution
        truncated_length_array, y_array = fit.ccdf()
        full_length_array, full_y_array = fit.ccdf(original_data=True)
    else:
        # probability density function
        bin_edges, y_array = fit.pdf()

        # Use same bin_edges and y_array if xmin/cut-off is lower than
        # smallest line length
        full_bin_edges, full_y_array = fit.pdf(original_data=True)

        # fit.pdf returns the bin edges. These need to be transformed to
        # centers for plotting.
        truncated_length_array = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        full_length_array = (full_bin_edges[:-1] + full_bin_edges[1:]) / 2.0

        # filter out zeros
        is_zero_y = np.isclose(y_array, 0.0)
        is_zero_full_y = np.isclose(full_y_array, 0.0)
        truncated_length_array = truncated_length_array[~is_zero_y]
        full_length_array = full_length_array[~is_zero_full_y]
        y_array = y_array[~is_zero_y]
        full_y_array = full_y_array[~is_zero_full_y]

    assert len(truncated_length_array) == len(y_array)
    assert len(full_length_array) == len(full_y_array)

    # Plot truncated length scatter plot
    ax.scatter(
        x=truncated_length_array,
        y=y_array,
        s=25,
        label=label,
        alpha=1.0,
        color="black",
        marker="x",
    )

    if not cut_off_is_lower:
        # Normalize full_ccm_array to the truncated ccm_array
        full_y_array = full_y_array / (
            full_y_array[len(full_y_array) - len(y_array)] / y_array.max()
        )
        # Plot full length scatter plot with different color and transparency
        ax.scatter(
            x=full_length_array,
            y=full_y_array,
            s=3,
            # label=f"{label} (cut)",
            alpha=0.5,
            color="gray",
            marker="x",
            zorder=-10,
        )

    # Plot the actual fits (powerlaw, exp...)
    for fit_distribution in fits_to_plot:
        plot_fit_on_ax(
            ax,
            fit,
            fit_distribution,
            use_probability_density_function=use_probability_density_function,
        )

    # Plot cut-off if applicable
    plot_axvline = cut_off is None or cut_off != 0.0
    if plot_axvline:
        # Indicate cut-off if cut-off is not given as 0.0
        truncated_length_array_min = min((truncated_length_array.min(), fit.xmin))
        ax.axvline(
            truncated_length_array_min,
            linestyle="dotted",
            color="black",
            alpha=0.6,
            label="Cut-Off",
            linewidth=3.5,
        )
        ax.text(
            truncated_length_array_min,
            y_array.min(),
            f"{round(truncated_length_array_min, 2)} m",
            rotation=90,
            horizontalalignment="right",
            fontsize="xx-large",
        )

    # Set title with exponent
    rounded_exponent = round(calculate_exponent(fit.alpha), 3)
    target = "Branches" if using_branches else "Traces"
    ax.set_title(
        f"{label}\nPower-law Exponent ({target}) = ${rounded_exponent}$",
        fontdict=dict(fontsize="xx-large"),
    )

    # Setup of ax appearance and axlims
    setup_ax_for_ld(
        ax,
        using_branches=using_branches,
        indiv_fit=True,
        use_probability_density_function=use_probability_density_function,
    )
    _setup_length_plot_axlims(
        ax=ax,
        length_array=truncated_length_array,
        ccm_array=y_array,
    )

    return fit, fig, ax


def setup_length_dist_legend(ax_for_setup: Axes):
    """
    Set up legend for length distribution plots.

    Used for both single and multi distribution plots.
    """
    # Setup legend
    handles, labels = ax_for_setup.get_legend_handles_labels()
    labels = [fill(label, 16) for label in labels]
    lgnd = ax_for_setup.legend(
        handles,
        labels,
        loc="upper right",
        # bbox_to_anchor=(1.37, 1.02),
        ncol=2,
        columnspacing=0.3,
        # shadow=True,
        prop={"size": "xx-large"},
        framealpha=0.6,
        facecolor="white",
        # fontsize=40,
    )

    # Setup legend line widths larger
    for lh in lgnd.legendHandles:
        # lh._sizes = [750]
        lh.set_linewidth(3)


def setup_ax_for_ld(
    ax_for_setup: Axes,
    using_branches: bool,
    indiv_fit: bool,
    use_probability_density_function: bool,
):
    """
    Configure ax for length distribution plots.

    :param ax_for_setup: Ax to setup.
    :param using_branches: Are the lines in the axis branches or traces.
    :param indiv_fit: Is the plot single-scale or multi-scale.
    :param use_probability_density_function: Whether to use complementary
           cumulative distribution function
    """
    # LABELS
    label = "Branch Length $(m)$" if using_branches else "Trace Length $(m)$"
    font_props = dict(
        fontsize="xx-large",
        fontfamily="DejaVu Sans",
        style="italic",
        fontweight="bold",
    )
    ax_for_setup.set_xlabel(
        label,
        labelpad=14,
        **font_props,
    )
    # Individual powerlaw fits are not normalized to area because they aren't
    # multiscale
    ccm_unit = r"$(\frac{1}{m^2})$" if not indiv_fit else ""
    prefix = "" if indiv_fit else "AN"
    function_name = "CCM" if not use_probability_density_function else "PDF"
    ax_for_setup.set_ylabel(
        f"{prefix}{function_name} {ccm_unit}",
        # prefix + function_name + ccm_unit,
        **font_props,
    )

    # Setup x and y axis ticks and their labels
    plt.xticks(color="black", fontsize="x-large")
    plt.yticks(color="black", fontsize="x-large")
    plt.tick_params(axis="both", width=1.2)

    # Setup legend
    setup_length_dist_legend(ax_for_setup=ax_for_setup)

    # Setup grid
    ax_for_setup.grid(zorder=-10, color="black", alpha=0.25)

    # Change x and y scales to logarithmic
    ax_for_setup.set_xscale("log")
    ax_for_setup.set_yscale("log")

    # Set facecolor depending on using_branches
    # facecolor = "oldlace" if using_branches else "gray"
    # ax_for_setup.set_facecolor(facecolor)


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

    return Polyfit(
        y_fit=y_fit, m_value=m_value, constant=constant, score=score, scorer=scorer
    )


def plot_multi_distributions_and_fit(
    truncated_length_array_all: List[np.ndarray],
    ccm_array_normed_all: List[np.ndarray],
    full_length_array_all: List[np.ndarray],
    full_ccm_array_normed_all: List[np.ndarray],
    cut_offs: List[np.ndarray],
    names: List[str],
    polyfit: Polyfit,
    using_branches: bool,
    plot_truncated_data: bool,
) -> Tuple[Figure, Axes]:
    """
    Plot multi-scale length distribution.
    """
    # Start making plot
    fig, ax = plt.subplots(figsize=(7, 7))
    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)

    # Determine scorer name
    # If not found in SCORER_NAMES, the __name__ of the scorer function is used
    try:
        scorer_str = SCORER_NAMES[polyfit.scorer]
    except KeyError:
        logging.warning(
            "Expected to find name string for polyfit scorer: "
            f"{polyfit.scorer} in SCORER_NAMES."
        )
        scorer_str = polyfit.scorer.__name__

    # Determine score string (float or scientific notation)
    is_very_low_score = polyfit.score < 0.01
    score_descriptive = (
        f"{polyfit.score:.2e}" if is_very_low_score else str(round(polyfit.score, 2))
    )

    # Set title with powerlaw exponent and score
    ax.set_title(
        (
            f"Exponent = {polyfit.m_value:.2f} and"
            f" Score ({scorer_str}) = {score_descriptive}"
        ),
        fontsize="x-large",
    )

    full_ccm_array_normed_concat = np.concatenate(full_ccm_array_normed_all)

    # Make color cycle
    color_cycle = cycle(sns.color_palette("dark", 5))

    # Cycle for cut-off label placement
    label_cycle = cycle([True, False])
    # Plot length distributions
    included_cut_off_label = False
    for (
        name,
        truncated_length_array,
        ccm_array_normed,
        full_length_array,
        full_ccm_array,
        cut_off,
    ) in zip(
        names,
        truncated_length_array_all,
        ccm_array_normed_all,
        full_length_array_all,
        full_ccm_array_normed_all,
        cut_offs,
    ):
        # Get color for distribution
        point_color = next(color_cycle)

        if len(truncated_length_array) > 0:
            # Indicate the cut-off with a dotted vertical line
            # truncated_length_array_min = truncated_length_array.min()
            cut_off_legend_kwarg = dict(label="Cut-Off(s)")
            ax.axvline(
                cut_off,
                linestyle="dotted",
                color="black",
                alpha=0.6,
                linewidth=3.5,
                **(cut_off_legend_kwarg if not included_cut_off_label else dict()),
            )
            included_cut_off_label = True

            # Also indicate cut-off value, embedded into plot
            plot_low = next(label_cycle)
            ax.text(
                cut_off,
                ccm_array_normed.min() if plot_low else ccm_array_normed.max(),
                f"{round(cut_off, 2)} m",
                rotation=90,
                horizontalalignment="right",
                fontsize="xx-large",
                verticalalignment="center" if plot_low else "bottom",
            )

        # Plot main data, not truncated
        ax.scatter(
            x=truncated_length_array,
            y=ccm_array_normed,
            s=25,
            label=name,
            marker="X",
            color=point_color,
        )
        if plot_truncated_data:
            # Plot truncated data part
            ax.scatter(
                x=full_length_array,
                y=full_ccm_array,
                s=3,
                marker="x",
                color="gray",
                alpha=0.5,
                zorder=-10,
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

    # Setup axes
    setup_ax_for_ld(
        ax_for_setup=ax,
        using_branches=using_branches,
        indiv_fit=False,
        use_probability_density_function=False,
    )
    _setup_length_plot_axlims(
        ax=ax,
        length_array=np.concatenate(full_length_array_all),
        ccm_array=full_ccm_array_normed_concat,
    )

    # Add legend
    # setup_length_dist_legend(ax_for_setup=ax)

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
    # TODO: extra args are passed in conda ci-tests.
    #    def function_wrapper(x, *wrapper_args):
    # ncalls[0] += 1
    # # A copy of x is sent to the user function (gh13740)
    # >       fx = function(np.copy(x), *(wrapper_args + args))
    # E       TypeError: optimize_cut_offs() takes 4 positional arguments but 7 were given
    *_,
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
