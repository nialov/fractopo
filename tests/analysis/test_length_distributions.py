"""
Tests for length distributions utilities.
"""
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import pytest
from hypothesis import given, settings
from hypothesis.extra import numpy
from hypothesis.strategies import floats
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import tests
from fractopo.analysis import length_distributions
from tests import Helpers


@given(
    numpy.arrays(dtype=float, shape=1),
    floats(min_value=0, allow_infinity=False, allow_nan=False),
)
@settings(max_examples=25)
def test_determine_fit(length_array: np.ndarray, cut_off: float):
    """
    Test determine_fit.
    """
    fit = length_distributions.determine_fit(length_array, cut_off)
    assert isinstance(fit, powerlaw.Fit)


@pytest.mark.parametrize("lengths,label", Helpers.test_describe_powerlaw_fit_params)
def test_describe_powerlaw_fit(lengths, label):
    """
    Test describe_powerlaw_fit.
    """
    fit = length_distributions.determine_fit(lengths)
    result = length_distributions.describe_powerlaw_fit(
        fit=fit, label=label, length_array=lengths
    )
    assert isinstance(result, dict)


@pytest.mark.parametrize("lengths,_", Helpers.test_describe_powerlaw_fit_params)
def test_all_fit_attributes_dict(lengths, _):
    """
    Test describe_powerlaw_fit.
    """
    fit = length_distributions.determine_fit(lengths)
    result = length_distributions.all_fit_attributes_dict(fit)
    assert isinstance(result, dict)


@pytest.mark.parametrize(
    "list_of_length_arrays,list_of_area_values,names",
    [
        (
            [tests.kb11_traces_lengths(), tests.kb11_traces_lengths()],
            [
                tests.kb11_area_value(),
                tests.kb11_area_value() * 10,
            ],
            ["kb11_1", "kb11_2"],
        ),
        (
            [
                tests.kb11_traces_lengths(),
                tests.hastholmen_traces_lengths(),
            ],
            [
                tests.kb11_area_value(),
                tests.hastholmen_area_value(),
            ],
            ["kb11", "hastholmen"],
        ),
    ],
)
@pytest.mark.parametrize("automatic_cut_offs", [True, False])
@pytest.mark.parametrize("using_branches", [True, False])
def test_multilengthdistribution_plot(
    list_of_length_arrays,
    list_of_area_values,
    names,
    automatic_cut_offs,
    using_branches,
    num_regression,
):
    """
    Test MultiLengthDistribution plot_multi_length_distributions.
    """
    assert isinstance(list_of_length_arrays, list)
    assert isinstance(list_of_area_values, list)
    assert isinstance(names, list)
    assert len(list_of_length_arrays) == len(list_of_area_values) == len(names)
    multi_length_distribution = length_distributions.MultiLengthDistribution(
        distributions=[
            length_distributions.LengthDistribution(
                name=name,
                lengths=lengths,
                area_value=area_value,
                using_branches=using_branches,
            )
            for name, lengths, area_value in zip(
                names, list_of_length_arrays, list_of_area_values
            )
        ],
        using_branches=using_branches,
    )

    polyfit, fig, ax = multi_length_distribution.plot_multi_length_distributions(
        automatic_cut_offs=automatic_cut_offs, plot_truncated_data=True
    )
    plt.close()

    mld = multi_length_distribution

    (
        truncated_length_array_all,
        ccm_array_normed_all,
        _,
        _,
    ) = mld.normalized_distributions(
        automatic_cut_offs=automatic_cut_offs,
    )

    assert isinstance(fig, Figure) and isinstance(ax, Axes)
    assert isinstance(truncated_length_array_all, list)
    assert isinstance(ccm_array_normed_all, list)

    num_regression.check(
        {
            "concatted_lengths": np.concatenate(truncated_length_array_all),
            "concatted_ccm": np.concatenate(ccm_array_normed_all),
            "fitted_y_values": polyfit.y_fit,
            "m_value": np.array([polyfit.m_value]),
            "constant": np.array([polyfit.constant]),
        },
        default_tolerance=dict(atol=1e-4, rtol=1e-4),
    )


@pytest.mark.parametrize(
    "length_distribution", tests.test_normalize_fit_to_area_params()
)
def test_normalize_fit_to_area(
    length_distribution: length_distributions.LengthDistribution,
    num_regression,
):
    """
    Test normalize_fit_to_area.
    """
    # fit = powerlaw.Fit(length_distribution.lengths)
    # fit = length_distributions.SilentFit(length_distribution.lengths)

    fit = length_distribution.automatic_fit

    (
        truncated_length_array,
        ccm_array_normed,
    ) = length_distribution.generate_distributions(cut_off=fit.xmin)

    assert isinstance(truncated_length_array, np.ndarray)
    assert isinstance(ccm_array_normed, np.ndarray)

    assert len(truncated_length_array) == len(ccm_array_normed)
    assert all(truncated_length_array > fit.xmin)
    assert all(1.0 >= ccm_array_normed) and all(ccm_array_normed > 0.0)

    num_regression.check(
        {
            "truncated_length_array": truncated_length_array,
            "ccm_array_normed": ccm_array_normed,
        },
        default_tolerance=dict(atol=1e-4, rtol=1e-4),
    )


@pytest.mark.parametrize("automatic_cut_offs", [True, False])
@pytest.mark.parametrize(
    "distributions", tests.test_fit_to_multi_scale_lengths_params()
)
def test_fit_to_multi_scale_lengths_fitter_comparisons(
    distributions: List[length_distributions.LengthDistribution],
    automatic_cut_offs: bool,
):
    """
    Test fit_to_multi_scale_lengths.
    """
    mld = length_distributions.MultiLengthDistribution(
        distributions=distributions, using_branches=False
    )

    (
        truncated_length_array_all,
        ccm_array_normed_all,
        # All length and ccm data ignored
        _,
        _,
    ) = mld.normalized_distributions(automatic_cut_offs=automatic_cut_offs)
    concatted_lengths, concatted_ccm = np.concatenate(
        truncated_length_array_all
    ), np.concatenate(ccm_array_normed_all)

    # Fit with np.polyfit
    numpy_polyfit_vals = length_distributions.fit_to_multi_scale_lengths(
        lengths=concatted_lengths,
        ccm=concatted_ccm,
        fitter=length_distributions.numpy_polyfit,
    )

    # Fit with scikit LinearRegression
    linear_regression_vals = length_distributions.fit_to_multi_scale_lengths(
        lengths=concatted_lengths,
        ccm=concatted_ccm,
        fitter=length_distributions.scikit_linear_regression,
    )

    for val in [*numpy_polyfit_vals, *linear_regression_vals]:
        # Polyfit attributes should be floats, array and callables (scorer func)
        assert isinstance(val, (np.ndarray, float)) or callable(val)


@pytest.mark.parametrize(
    "distributions", tests.test_fit_to_multi_scale_lengths_params()
)
def test_plot_mld_optimized(distributions):
    """
    Test plot_mld_optimized.
    """
    mld = length_distributions.MultiLengthDistribution(
        distributions=distributions, using_branches=False
    )

    opt_result, opt_mld = mld.optimize_cut_offs()

    polyfit, fig, ax = opt_mld.plot_multi_length_distributions(
        automatic_cut_offs=False, plot_truncated_data=True
    )

    assert isinstance(opt_result, length_distributions.MultiScaleOptimizationResult)
    assert isinstance(polyfit, length_distributions.Polyfit)
    assert isinstance(fig, Figure) and isinstance(ax, Axes)

    plt.close()


@pytest.mark.parametrize("using_branches", (True, False))
@pytest.mark.parametrize("use_probability_density_function", (True, False))
@pytest.mark.parametrize(
    "length_array,label", [*Helpers.test_describe_powerlaw_fit_params]
)
def test_plot_distribution_fits(
    length_array: np.ndarray,
    label: str,
    using_branches: bool,
    use_probability_density_function: bool,
):
    """
    Test plot_distribution_fits.
    """
    fit, fig, ax = length_distributions.plot_distribution_fits(
        length_array=length_array,
        label=label,
        using_branches=using_branches,
        use_probability_density_function=use_probability_density_function,
    )
    assert isinstance(fit, length_distributions.SilentFit)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
