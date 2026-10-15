from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csc_matrix, csr_matrix
from shapely.geometry import LineString

from fractopo.interop.porepy import (
    _extract_network_fracture_set_samples,
    check_porepy_2d_csv_format,
    convert_azimuth_to_strike,
    determine_azimuth,
    export_network_to_porepy_3d_csv_format,
    export_structural_measurements_to_porepy_3d_csv_format,
    export_traces_to_porepy_3d_csv_format,
    exterior_triangle_connectivity,
    porepy_csv_fracture_diameters,
    scale_geometries_to_local,
    summarize_empirical_fracture_lengths,
    summarize_porepy_mesh,
    triangle_cell_connectivity,
)

EXAMPLE_POREPY_2D_CSV_WITH_COMMENTS = """# Domain X_MIN, Y_MIN, X_MAX, Y_MAX
-54900, 6730000, -52000, 6733000
# OBJECTID,START_X,START_Y,END_X,END_Y
-5.361122869999999966e+04,6.732113743700000457e+06,-5.466355320000000211e+04,6.730962761900000274e+06
"""

EXAMPLE_POREPY_2D_CSV_WITHOUT_COMMENTS = """-54900, 6730000, -52000, 6733000
-5.361122869999999966e+04,6.732113743700000457e+06,-5.466355320000000211e+04,6.730962761900000274e+06
"""


def test_summarize_empirical_fracture_lengths_and_csv_diameters():
    network = SimpleNamespace(
        azimuth_set_names=("north", "south", "-1"),
        trace_data=SimpleNamespace(
            azimuth_set_array=np.array(["north", "north", "south", "-1"]),
            azimuth_array=np.array([10.0, 12.0, 40.0, 50.0]),
            length_array=np.array([2.0, 4.0, 8.0, 10.0]),
        ),
        trace_gdf={
            "dip": np.array([30.0, 40.0, 60.0, 50.0]),
            "dip_dir": np.array([100.0, 110.0, 130.0, 120.0]),
        },
    )
    summary = summarize_empirical_fracture_lengths(
        network, "0,0,0,1,1,1\n0,0,0,1,1,0,0,0\n0,0,0,2,2,0,0,0"
    )
    assert summary["by_set"]["north"] == {
        "count": 2,
        "min": 2.0,
        "median": 3.0,
        "mean": 3.0,
        "max": 4.0,
    }
    assert summary["by_set"]["south"]["mean"] == 8.0
    assert summary["generated"]["mean"] == 3.0
    np.testing.assert_array_equal(
        porepy_csv_fracture_diameters("0,0,0,1,1,1\n0,0,0,2,2,0,0,0"), [4.0]
    )
    with pytest.raises(ValueError, match="invalid PorePy"):
        porepy_csv_fracture_diameters("domain\nbad")


def test_summarize_porepy_mesh_and_triangle_connectivity():
    summary = summarize_porepy_mesh(np.array([3, 2, 2]), np.array([4, 5, 6]), 2)
    assert summary == {
        "subdomain grids": 3,
        "subdomain cells": 15,
        "cells by dimension": {2: 11, 3: 4},
        "interfaces": 2,
    }
    with pytest.raises(ValueError, match="aligned"):
        summarize_porepy_mesh(np.array([3]), np.array([4, 5]), 0)

    nodes_by_face = csc_matrix(
        np.array(
            [
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0, 1, 1],
            ]
        )
    )
    cell_by_face = csr_matrix(np.array([[1, 0], [1, 1], [0, 1]]))
    expected_triangles = [[0, 1, 2], [0, 2, 3]]
    np.testing.assert_array_equal(
        exterior_triangle_connectivity(nodes_by_face, cell_by_face),
        expected_triangles,
    )
    np.testing.assert_array_equal(
        triangle_cell_connectivity(nodes_by_face),
        [[0, 1, 2], [1, 2, 3], [0, 2, 3]],
    )


def test_extract_network_fracture_set_samples_filters_and_counts():
    network = SimpleNamespace(
        azimuth_set_names=("north", "south"),
        trace_data=SimpleNamespace(
            azimuth_set_array=np.array(["north", "north", "south", "-1", "south"]),
            azimuth_array=np.array([10.0, np.nan, 40.0, 50.0, 60.0]),
            length_array=np.array([2.0, 3.0, 4.0, 5.0, 0.0]),
        ),
        trace_gdf={"dip": np.array([20.0, 30.0, 40.0, 50.0, 100.0])},
    )

    samples = _extract_network_fracture_set_samples(network)

    np.testing.assert_array_equal(samples.azimuth["north"], [10.0])
    np.testing.assert_array_equal(samples.dip["south"], [40.0])
    np.testing.assert_array_equal(samples.length["south"], [4.0])
    assert samples.proportions == {"north": 0.5, "south": 0.5}


def test_extract_network_fracture_set_samples_rejects_all_invalid():
    network = SimpleNamespace(
        azimuth_set_names=("north",),
        trace_data=SimpleNamespace(
            azimuth_set_array=np.array(["-1"]),
            azimuth_array=np.array([10.0]),
            length_array=np.array([1.0]),
        ),
        trace_gdf={"dip": np.array([20.0]), "dip_dir": np.array([np.nan])},
    )
    with pytest.raises(ValueError, match="usable.*set"):
        _extract_network_fracture_set_samples(network)


@pytest.mark.parametrize(
    "csv_text",
    [
        pytest.param(
            EXAMPLE_POREPY_2D_CSV_WITH_COMMENTS,
            id="comments",
        ),
        pytest.param(
            EXAMPLE_POREPY_2D_CSV_WITHOUT_COMMENTS,
            id="without-comments",
        ),
    ],
)
def test_check_porepy_2d_csv_format_passes(csv_text):
    assert check_porepy_2d_csv_format(csv_text=csv_text)


def test_export_network_to_porepy_3d_csv_format():
    network = SimpleNamespace(
        azimuth_set_names=("manual", "automatic", "unused"),
        trace_data=SimpleNamespace(
            azimuth_set_array=np.array(["manual", "manual", "automatic", "-1"]),
            azimuth_array=np.array([10.0, 11.0, 20.0, 30.0]),
            length_array=np.array([4.0, 6.0, 8.0, 10.0]),
        ),
        trace_gdf={
            "dip": np.array([20.0, 30.0, 40.0, 50.0]),
            "dip_dir": np.array([100.0, 101.0, 110.0, 120.0]),
        },
    )
    bounds = (0.0, 10.0, 20.0, 1.0, 11.0, 21.0)
    first = export_network_to_porepy_3d_csv_format(
        network, 4, bounds, np.random.default_rng(7)
    )
    second = export_network_to_porepy_3d_csv_format(
        network, 4, bounds, np.random.default_rng(7)
    )
    assert first == second
    lines = first.splitlines()
    assert [float(value) for value in lines[0].split(",")] == list(bounds)
    assert len(lines) == 5
    for line in lines[1:]:
        fields = np.asarray([float(value) for value in line.split(",")])
        assert fields.size == 8 and np.all(np.isfinite(fields))
        assert fields[3] == fields[4] > 0
        assert 0 <= fields[0] <= 1 and 10 <= fields[1] <= 11
        assert 20 <= fields[2] <= 21
    # Counts are 2:1, so largest remainder gives one automatic sample;
    # its source strike/dip and semi-axis are all observable in the output.
    assert any(
        np.isclose(float(row.split(",")[6]), np.deg2rad(20)) for row in lines[1:]
    )
    assert any(
        np.isclose(float(row.split(",")[7]), np.deg2rad(40)) for row in lines[1:]
    )
    assert all(float(row.split(",")[3]) in (2.0, 3.0, 4.0) for row in lines[1:])
    expected_strikes = {
        np.deg2rad(20.0): np.deg2rad(10.0),
        np.deg2rad(30.0): np.deg2rad(11.0),
        np.deg2rad(40.0): np.deg2rad(20.0),
    }
    assert all(
        np.isclose(row[6], expected_strikes[row[7]])
        for row in (
            np.asarray([float(value) for value in line.split(",")])
            for line in lines[1:]
        )
    )


def test_network_export_preserves_opposite_dip_direction_sides():
    network = SimpleNamespace(
        azimuth_set_names=("set",),
        trace_data=SimpleNamespace(
            azimuth_set_array=np.array(["set", "set"]),
            azimuth_array=np.array([90.0, 90.0]),
            length_array=np.array([2.0, 2.0]),
        ),
        trace_gdf={"dip": np.array([80.0, 80.0]), "dip_dir": np.array([0.0, 180.0])},
    )
    rows = export_network_to_porepy_3d_csv_format(
        network,
        fracture_count=50,
        bounds=(0, 0, 0, 1, 1, 1),
        rng=np.random.default_rng(0),
    ).splitlines()[1:]
    strike_dips = sorted(
        (
            round(np.rad2deg(float(row.split(",")[6]))),
            round(np.rad2deg(float(row.split(",")[7]))),
        )
        for row in rows
    )
    assert all(dip == 80 for _, dip in strike_dips)
    assert {strike for strike, _ in strike_dips} == {90, 270}


@pytest.mark.parametrize(
    "count,bounds,error",
    [
        (0, (0, 0, 0, 1, 1, 1), "positive integer"),
        (1, (0, 0, 0, 0, 1, 1), "strictly increasing"),
        (1, (0, 0, np.inf, 1, 1, 1), "finite"),
    ],
)
def test_export_network_to_porepy_3d_csv_format_validates(count, bounds, error):
    with pytest.raises(ValueError, match=error):
        export_network_to_porepy_3d_csv_format(SimpleNamespace(), count, bounds)


@pytest.mark.parametrize(
    "traces,dip_values,y_scale,z_values",
    [
        pytest.param(
            [LineString([(0, 0), (4, 0)]), LineString([(0, 0), (0, 3)])],
            np.array([45.0, 60.0]),
            None,
            None,
            id="unscaled-traces",
        ),
        pytest.param(
            [LineString([(1, 2), (2, 5)])],
            np.array([30.0]),
            10,
            None,
            id="scaled-trace-without-z-values",
        ),
        pytest.param(
            [LineString([(1, 2), (2, 5)])],
            np.array([30.0]),
            10,
            np.array([5.0]),
            id="scaled-trace-with-z-values",
        ),
    ],
)
def test_export_traces_to_porepy_3d_csv_format(traces, dip_values, y_scale, z_values):
    csv_out = export_traces_to_porepy_3d_csv_format(
        traces, dip_values, y_scale, z_values
    )
    lines = [line for line in csv_out.strip().split("\n")]
    assert len(lines) == len(traces)
    if y_scale is not None:
        expected_traces, *_ = scale_geometries_to_local(traces, y_scale)
    else:
        expected_traces = traces

    for line, expected_dip, trace in zip(lines, dip_values, expected_traces):
        fields = [float(x) for x in line.split(",")]
        assert len(fields) == 8
        # PorePy axes are (possibly scaled) trace semi-lengths.
        assert np.allclose(
            fields[:2], trace.interpolate(0.5, normalized=True).coords[0]
        )
        scaled_semi_length = trace.length / 2
        assert abs(fields[3] - fields[4]) < 1e-8
        assert np.isclose(fields[3], scaled_semi_length, atol=1e-8)
        expected_strike = convert_azimuth_to_strike(
            determine_azimuth(trace, halved=True)
        )
        assert np.isclose(fields[6], np.deg2rad(expected_strike), atol=1e-8)
        # Dip angle (rad) must match input dip within numerical conversion to radians
        assert np.isclose(fields[7], np.deg2rad(expected_dip), atol=1e-8)


def test_export_structural_measurements_to_porepy_3d_csv_format():
    csv_out = export_structural_measurements_to_porepy_3d_csv_format(
        dip_values=np.array([30.0]),
        dip_direction_values=np.array([120.0]),
        length_values=np.array([10.0]),
        measurement_points=[LineString([(2, 3), (2, 3)]).centroid],
        y_scale=None,
        z_values=np.array([7.0]),
    )

    fields = [float(value) for value in csv_out.split(",")]
    assert len(fields) == 8
    assert np.allclose(fields[:3], [2.0, 3.0, 7.0])
    assert np.allclose(fields[3:5], [5.0, 5.0])
    assert np.isclose(fields[5], 0.0)
    assert np.isclose(fields[6], np.deg2rad(30.0))
    assert np.isclose(fields[7], np.deg2rad(30.0))
