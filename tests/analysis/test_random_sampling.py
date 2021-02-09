"""
Tests for NetworkRandomSampler.
"""
import numpy as np
import pytest

from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import NetworkRandomSampler
from fractopo.general import Error_branch
from tests import Helpers


@pytest.mark.parametrize(
    "trace_gdf,area_gdf,min_radius,snap_threshold,samples",
    Helpers.test_network_random_sampler_params,
)
def test_network_random_sampler(
    trace_gdf, area_gdf, min_radius, snap_threshold, samples
):
    """
    Test NetworkRandomSampler sampling.
    """
    sampler = NetworkRandomSampler(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        min_radius=min_radius,
        snap_threshold=snap_threshold,
    )
    assert sampler.max_radius > min_radius
    for _ in range(100):
        (
            random_target_circle,
            random_target_centroid,
            radius,
        ) = sampler.random_target_circle()
        assert min_radius <= radius < sampler.max_radius
        assert random_target_centroid.within(sampler.target_circle)
        assert random_target_circle.area < sampler.target_circle.area

    for _ in range(samples):
        network, target_centroid, radius = sampler.random_network_sample()
        assert isinstance(network, Network)
        if not network.trace_gdf.shape[0] <= trace_gdf.shape[0]:
            assert np.isclose(radius, sampler.max_radius)
        assert target_centroid.within(sampler.target_circle)
        assert (
            sum(network.branch_types == Error_branch) / len(network.branch_types)
            < 0.001
        )
