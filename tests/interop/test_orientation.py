from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from fractopo.analysis.network import Network
from fractopo.interop.orientation import (
    complete_network_orientations,
    dip_direction_to_pole,
    pole_to_orientation,
)


def test_near_vertical_opposite_dip_directions_are_close_in_pole_space():
    pole_a = dip_direction_to_pole(80.0, 180.0)
    pole_b = dip_direction_to_pole(80.0, 0.0)
    assert np.rad2deg(np.arccos(abs(np.dot(pole_a, pole_b)))) == pytest.approx(20)
    aligned_b = pole_b * np.sign(np.dot(pole_a, pole_b))
    assert pole_to_orientation(np.mean([pole_a, aligned_b], axis=0))[
        0
    ] == pytest.approx(90)


def test_complete_network_orientations_fills_pairs_and_is_reproducible():
    network = SimpleNamespace(
        trace_gdf=gpd.GeoDataFrame(
            {"dip": [80.0, 80.0, np.nan], "dip_dir": [180.0, 0.0, np.nan]},
            geometry=[LineString([(0, 0), (1, 0)])] * 3,
        ),
        trace_data=SimpleNamespace(azimuth_set_array=np.array(["a", "a", "a"])),
        azimuth_set_names=("a",),
    )

    first, diagnostics = complete_network_orientations(
        network, rng=np.random.default_rng(12)
    )
    second, _ = complete_network_orientations(network, rng=np.random.default_rng(12))

    assert first.loc[0, "dip"] == 80.0
    assert first.loc[1, "dip_dir"] == 0.0
    assert np.allclose(
        first.loc[2, ["dip", "dip_dir"]].astype(float),
        second.loc[2, ["dip", "dip_dir"]].astype(float),
    )
    assert np.allclose(
        first.loc[2, ["pole_east", "pole_north", "pole_up"]].astype(float).to_numpy()
        @ first.loc[2, ["pole_east", "pole_north", "pole_up"]].astype(float).to_numpy(),
        1.0,
    )
    assert diagnostics["a"]["measured_count"] == 2
    assert diagnostics["a"]["source"] == "set"
    np.testing.assert_allclose(
        dip_direction_to_pole(first.loc[2, "dip"], first.loc[2, "dip_dir"]),
        first.loc[2, ["pole_east", "pole_north", "pole_up"]].to_numpy(dtype=float),
    )
    assert np.allclose(
        np.linalg.norm(
            first[["pole_east", "pole_north", "pole_up"]].to_numpy(), axis=1
        ),
        1.0,
    )


def test_incomplete_orientation_is_replaced_as_pair_and_fallback_is_used():
    network = SimpleNamespace(
        trace_gdf=gpd.GeoDataFrame(
            {"dip": [45.0], "dip_dir": [np.nan]},
            geometry=[LineString([(0, 0), (1, 0)])],
        ),
        trace_data=SimpleNamespace(azimuth_set_array=np.array(["a"])),
        azimuth_set_names=("a",),
    )
    with pytest.raises(ValueError, match="fallback"):
        complete_network_orientations(network)

    result, diagnostics = complete_network_orientations(
        network,
        rng=np.random.default_rng(2),
        fallback_dip=60.0,
        fallback_dip_dir=120.0,
        fallback_angular_std=0.0,
    )
    assert result.loc[0, "dip"] == pytest.approx(60.0, abs=0.02)
    assert result.loc[0, "dip_dir"] == pytest.approx(120.0, abs=0.02)
    assert diagnostics["a"]["source"] == "fallback"


def test_unlisted_non_null_labels_are_filled():
    network = SimpleNamespace(
        trace_gdf=gpd.GeoDataFrame(
            {"dip": [np.nan, 20.0, 30.0], "dip_dir": [np.nan, 90.0, 100.0]},
            geometry=[LineString([(0, 0), (1, 0)])] * 3,
        ),
        trace_data=SimpleNamespace(
            azimuth_set_array=np.array(["auto", "measured", "measured"])
        ),
        azimuth_set_names=("measured",),
    )
    result, diagnostics = complete_network_orientations(
        network, rng=np.random.default_rng(3)
    )
    assert np.isfinite(result.loc[0, "dip"])
    assert diagnostics["auto"]["source"] == "global"


def test_null_set_is_only_filled_when_requested():
    network = SimpleNamespace(
        trace_gdf=gpd.GeoDataFrame(
            {"dip": [30.0, 40.0], "dip_dir": [90.0, 90.0]},
            geometry=[LineString([(0, 0), (1, 0)])] * 2,
        ),
        trace_data=SimpleNamespace(azimuth_set_array=np.array(["a", "-1"])),
        azimuth_set_names=("a",),
    )
    result, _ = complete_network_orientations(
        network,
        rng=np.random.default_rng(1),
        fill_null_set=True,
    )
    assert np.isfinite(result.loc[1, "dip"])
    assert np.isfinite(result.loc[1, "pole_up"])


def test_network_method_updates_trace_gdf_without_replacing_trace_data():
    network = object.__new__(Network)
    network.trace_gdf = gpd.GeoDataFrame(
        {"dip": [45.0, np.nan], "dip_dir": [90.0, np.nan]},
        geometry=[LineString([(0, 0), (1, 0)])] * 2,
    )
    network.trace_data = SimpleNamespace(azimuth_set_array=np.array(["a", "a"]))
    network.azimuth_set_names = ("a",)
    trace_data = network.trace_data

    diagnostics = network.complete_trace_orientations(
        rng=np.random.default_rng(4),
        fallback_dip=60.0,
        fallback_dip_dir=90.0,
        fallback_angular_std=10.0,
    )

    assert network.trace_data is trace_data
    assert np.isfinite(network.trace_gdf.loc[1, "dip"])
    assert diagnostics["a"]["filled_count"] == 1


def test_undersampled_set_uses_global_model_and_invalid_pair_is_replaced():
    network = SimpleNamespace(
        trace_gdf=gpd.GeoDataFrame(
            {
                "dip": [30.0, 40.0, 91.0, np.nan],
                "dip_dir": [90.0, 100.0, 200.0, 270.0],
            },
            geometry=[LineString([(0, 0), (1, 0)])] * 4,
        ),
        trace_data=SimpleNamespace(azimuth_set_array=np.array(["a", "a", "b", "b"])),
        azimuth_set_names=("a", "b"),
    )
    result, diagnostics = complete_network_orientations(
        network, rng=np.random.default_rng(5)
    )
    assert diagnostics["b"]["source"] == "global"
    assert result.loc[2, "dip"] != 91.0
    assert np.isfinite(result.loc[3, "dip"])
    assert np.isfinite(result.loc[3, "dip_dir"])
