"""
Tests for multi_network.py.
"""

import logging
from pathlib import Path

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from ternary.ternary_axes_subplot import TernaryAxesSubplot

import tests
from fractopo import MultiNetwork, Network, general
from fractopo.analysis import length_distributions, subsampling


@pytest.mark.parametrize(
    "network_params,samples,min_radii", tests.test_multinetwork_params()
)
def test_multinetwork_subsample(network_params, samples: int, min_radii: float):
    """
    Test MultiNetwork.subsample.
    """
    assert min_radii > 0.0
    assert samples > 0
    networks = [Network(**kwargs) for kwargs in network_params]
    multi_network = MultiNetwork(networks=tuple(networks))

    subsamples = multi_network.subsample(min_radii=min_radii, samples=samples)

    assert isinstance(subsamples, list)
    assert len(networks) * samples == len(subsamples)

    for subsample in subsamples:
        if subsample.error:
            raise subsample.result
    identifiers = [subsample.identifier for subsample in subsamples]
    assert not any(subsample.error for subsample in subsamples)

    for network in networks:
        assert network.name in identifiers

    gathered = subsampling.gather_subsample_descriptions(subsample_results=subsamples)

    assert isinstance(gathered, list)

    assert len(gathered) > 0


def multinetwork_plot_multi_length_distribution(
    network_params, using_branches=False, automatic_cut_offs=True
):
    """
    Create and test multi_network plot_multi_length_distribution.
    """
    # using_branches = False
    # automatic_cut_offs = True
    networks = [
        Network(**kwargs, determine_branches_nodes=using_branches)
        for kwargs in network_params
    ]
    multi_network = MultiNetwork(networks=tuple(networks))

    mld = multi_network.multi_length_distributions(using_branches=using_branches)

    assert len(mld.distributions) == len(network_params)

    mld, polyfit, fig, ax = multi_network.plot_multi_length_distribution(
        using_branches=using_branches, automatic_cut_offs=automatic_cut_offs
    )

    assert isinstance(mld, length_distributions.MultiLengthDistribution)
    assert isinstance(polyfit, length_distributions.Polyfit)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    return mld, fig, ax


@pytest.mark.parametrize(
    "network_params",
    tests.test_multinetwork_plot_multi_length_distribution_slow_params(),
)
def test_multinetwork_plot_multi_length_distribution_slow(network_params):
    """
    Test MultiNetwork.plot_multi_length_distribution with slow data.
    """
    mld, fig, ax = multinetwork_plot_multi_length_distribution(
        network_params=network_params, using_branches=False, automatic_cut_offs=True
    )
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(mld, length_distributions.MultiLengthDistribution)


@pytest.mark.parametrize("automatic_cut_offs", [True, False])
@pytest.mark.parametrize("using_branches", [False, True])
@pytest.mark.parametrize(
    "network_params",
    tests.test_multinetwork_plot_multi_length_distribution_fast_params(),
)
def test_multinetwork_plot_multi_length_distribution_fast(
    network_params, using_branches, automatic_cut_offs
):
    """
    Test MultiNetwork.plot_multi_length_distribution with fast data.
    """
    multinetwork_plot_multi_length_distribution(
        network_params=network_params,
        using_branches=using_branches,
        automatic_cut_offs=automatic_cut_offs,
    )


@pytest.mark.parametrize(
    "network_params",
    tests.test_multinetwork_plot_multi_length_distribution_fast_params(),
)
def test_multinetwork_plot_ternary(network_params, tmp_path):
    """
    Test MultiNetwork.plot_xyi and plot_branch.
    """

    networks = [
        Network(**params, determine_branches_nodes=True) for params in network_params
    ]

    multi_network = MultiNetwork(tuple(networks))

    for plot_func in ("plot_xyi", "plot_branch"):

        fig, ax, tax = getattr(multi_network, plot_func)()

        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert isinstance(tax, TernaryAxesSubplot)

        tmp_path = Path(tmp_path)
        tmp_path.mkdir(exist_ok=True)
        plot_path = tmp_path / f"{plot_func}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        logging.info(f"Saved {plot_func} plot to {plot_path.absolute()}.")


@pytest.mark.parametrize(
    "automatic_cut_offs",
    [True, False],
)
@pytest.mark.parametrize(
    "using_branches",
    [True, False],
)
@pytest.mark.parametrize(
    "network_params",
    tests.test_multinetwork_plot_azimuth_set_lengths_params(),
)
def test_multinetwork_plot_azimuth_set_lengths(
    network_params, using_branches, automatic_cut_offs, tmp_path: Path
):
    """
    Test MultiNetwork._plot_azimuth_set_lengths.
    """

    networks = [
        Network(**params, determine_branches_nodes=using_branches)
        for params in network_params
    ]

    multi_network = MultiNetwork(tuple(networks))

    mlds, figs, axes = multi_network._plot_azimuth_set_lengths(
        automatic_cut_offs=automatic_cut_offs, using_branches=using_branches
    )

    for name, fig in zip(mlds, figs):
        output_path = general.save_fig(fig=fig, results_dir=tmp_path, name=name)
        logging.info(f"Saved plot to {output_path}.")

    assert isinstance(mlds, dict)
    assert isinstance(figs, list)
    assert isinstance(axes, list)
    assert isinstance(figs[0], Figure)
    assert isinstance(axes[0], Axes)
