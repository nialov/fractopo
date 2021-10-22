"""
Tests for multi_network.py.
"""

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import tests
from fractopo import MultiNetwork, Network
from fractopo.analysis import subsampling


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


@pytest.mark.parametrize("cut_distributions", [True, False])
@pytest.mark.parametrize("using_branches", [False, True])
@pytest.mark.parametrize(
    "network_params", [param[0] for param in tests.test_multinetwork_params()]
)
def test_multinetwork(network_params, using_branches, cut_distributions):
    """
    Test MultiNetwork.subsample.
    """
    networks = [
        Network(**kwargs, determine_branches_nodes=using_branches)
        for kwargs in network_params
    ]
    multi_network = MultiNetwork(networks=tuple(networks))

    mld = multi_network.multi_length_distributions(
        using_branches=using_branches, cut_distributions=cut_distributions
    )

    assert len(mld.distributions) == len(network_params)

    fig, ax = multi_network.plot_multi_length_distribution(
        using_branches=using_branches, cut_distributions=cut_distributions
    )

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
