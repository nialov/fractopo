"""
Tests for NetworkRandomSampler.
"""
import numpy as np
import pytest

from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import NetworkRandomSampler, RandomChoice
from fractopo.general import Error_branch
from tests import Helpers


def test_network_random_sampler_manual():
    """
    Test NetworkRandomSampler sampling manually.
    """
    trace_gdf = Helpers.geta_1_traces
    area_gdf = Helpers.geta_1_1_area
    min_radius = 10
    snap_threshold = 0.001
    samples = 1
    sampler = NetworkRandomSampler(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        min_radius=min_radius,
        snap_threshold=snap_threshold,
        random_choice=RandomChoice.radius,
        name="geta1",
    )
    circle_samples = []
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
        circle_samples.append(random_target_circle)

    for _ in range(samples):
        random_sample = sampler.random_network_sample()
        network, target_centroid, radius = (
            random_sample.network_maybe,
            random_sample.target_centroid,
            random_sample.radius,
        )
        assert isinstance(network, Network)
        if not network.trace_gdf.shape[0] <= trace_gdf.shape[0]:
            assert np.isclose(radius, sampler.max_radius)
        assert target_centroid.within(sampler.target_circle)

        # TODO: Errors are possible and will randomly occur.
        assert (
            sum(network.branch_types == Error_branch) / len(network.branch_types) < 0.01
        )
    return circle_samples


@pytest.mark.parametrize(
    "determine_branches_nodes",
    [True, False],
)
@pytest.mark.parametrize(
    "trace_gdf,area_gdf,min_radius,snap_threshold,samples,random_choice",
    Helpers.test_network_random_sampler_params,
)
def test_network_random_sampler(
    trace_gdf,
    area_gdf,
    min_radius,
    snap_threshold,
    samples,
    random_choice,
    determine_branches_nodes,
):
    """
    Test NetworkRandomSampler sampling.
    """
    sampler = NetworkRandomSampler(
        trace_gdf=trace_gdf,
        area_gdf=area_gdf,
        min_radius=min_radius,
        snap_threshold=snap_threshold,
        random_choice=random_choice,
        name="sampler",
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
        random_sample = sampler.random_network_sample(
            determine_branches_nodes=determine_branches_nodes
        )
        network, target_centroid, radius = (
            random_sample.network_maybe,
            random_sample.target_centroid,
            random_sample.radius,
        )
        assert isinstance(network, Network)
        if not network.trace_gdf.shape[0] <= trace_gdf.shape[0]:
            assert np.isclose(radius, sampler.max_radius)
        assert target_centroid.within(sampler.target_circle)
        if determine_branches_nodes and len(network.branch_types) > 200:
            assert (
                sum(network.branch_types == Error_branch) / len(network.branch_types)
                < 0.01
            )
