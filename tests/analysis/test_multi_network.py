"""
Tests for multi_network.py.
"""

import pandas as pd
import pytest

import tests
from fractopo import MultiNetwork, Network
from fractopo.analysis import subsampling


@pytest.mark.parametrize("network_params,samples", tests.test_multinetwork_params())
def test_multinetwork_subsample(network_params, samples: int):
    """
    Test MultiNetwork.subsample.
    """
    networks = [Network(**kwargs) for kwargs in network_params]
    multi_network = MultiNetwork(networks=networks)

    subsamples = multi_network.subsample(min_radius=5.0, samples=samples)

    assert isinstance(subsamples, list)
    assert len(networks) * samples == len(subsamples)

    identifiers = [subsample.identifier for subsample in subsamples]
    assert not any(subsample.error for subsample in subsamples)

    for network in networks:
        assert network.name in identifiers

    gathered = subsampling.gather_subsample_descriptions(subsample_results=subsamples)

    assert isinstance(gathered, pd.DataFrame)

    assert gathered.shape[0] > 0
