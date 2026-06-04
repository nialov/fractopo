import numpy as np
import pytest
from shapely.geometry import LineString

from fractopo.interop.porepy import (
    check_porepy_2d_csv_format,
    export_traces_to_porepy_3d_csv_format,
    scale_geometries_to_local,
)

EXAMPLE_POREPY_2D_CSV_WITH_COMMENTS = """# Domain X_MIN, Y_MIN, X_MAX, Y_MAX
-54900, 6730000, -52000, 6733000
# OBJECTID,START_X,START_Y,END_X,END_Y
-5.361122869999999966e+04,6.732113743700000457e+06,-5.466355320000000211e+04,6.730962761900000274e+06
"""

EXAMPLE_POREPY_2D_CSV_WITHOUT_COMMENTS = """-54900, 6730000, -52000, 6733000
-5.361122869999999966e+04,6.732113743700000457e+06,-5.466355320000000211e+04,6.730962761900000274e+06
"""


@pytest.mark.parametrize(
    "csv_text",
    [EXAMPLE_POREPY_2D_CSV_WITH_COMMENTS, EXAMPLE_POREPY_2D_CSV_WITHOUT_COMMENTS],
)
def test_check_porepy_2d_csv_format_passes(csv_text):
    assert check_porepy_2d_csv_format(csv_text=csv_text)


@pytest.mark.parametrize(
    "traces,dip_values,y_scale,z_values",
    [
        (
            [LineString([(0, 0), (4, 0)]), LineString([(0, 0), (0, 3)])],
            np.array([45.0, 60.0]),
            None,
            None,
        ),
        (
            [LineString([(1, 2), (2, 5)])],
            np.array([30.0]),
            10,
            None,
        ),
        (
            [LineString([(1, 2), (2, 5)])],
            np.array([30.0]),
            10,
            np.array([5.0]),
        ),
    ],
)
def test_export_traces_to_porepy_3d_csv_format(traces, dip_values, y_scale, z_values):
    csv_out = export_traces_to_porepy_3d_csv_format(
        traces, dip_values, y_scale, z_values
    )
    lines = [line for line in csv_out.strip().split("\n")]
    assert len(lines) == len(traces)
    # If scaling is applied, need to get scaling factor

    if y_scale is not None:
        *_, scale = scale_geometries_to_local(traces, y_scale)
    else:
        scale = 1.0

    for line, expected_dip, trace in zip(lines, dip_values, traces):
        fields = [float(x) for x in line.split(",")]
        assert len(fields) == 8
        # Major/minor axes equal to (possibly scaled) trace length
        scaled_length = trace.length * scale
        assert abs(fields[3] - fields[4]) < 1e-8
        assert np.isclose(fields[3], scaled_length, atol=1e-8)
        # Dip angle (rad) must match input dip within numerical conversion to radians
        assert np.isclose(fields[7], np.deg2rad(expected_dip), atol=1e-8)
