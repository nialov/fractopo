import os
import sqlite3
import subprocess
import sys
from functools import partial
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

import fractopo.tval.trace_validation
import tests
from marimos.utils import _resolve_layer_name

SAMPLE_DATA_DIR = Path(__file__).parent.parent.joinpath("sample_data/")
VALIDATION_NOTEBOOK = Path(__file__).parent.parent.parent.joinpath(
    "marimos/validation.py"
)
NETWORK_NOTEBOOK = Path(__file__).parent.parent.parent.joinpath("marimos/network.py")
PYTHON_INTERPRETER = sys.executable

check_python_call = partial(
    subprocess.check_call,
    env={
        **os.environ,
        "PYTHONPATH": str.join(
            ":",
            [
                str(Path(__file__).parent.parent.parent),
                *sys.path,
            ],
        ),
    },
)

pytest_mark_xfail_windows_flaky = pytest.mark.xfail(
    sys.platform == "win32",
    reason="Subprocess call is flaky in Windows",
    raises=subprocess.CalledProcessError,
)


@pytest_mark_xfail_windows_flaky
@pytest.mark.parametrize(
    "traces_path,area_path,name",
    [
        (
            SAMPLE_DATA_DIR.joinpath("hastholmen_traces.geojson").as_posix(),
            SAMPLE_DATA_DIR.joinpath("hastholmen_area.geojson").as_posix(),
            "hastholmen",
        ),
        (
            tests.kb7_trace_100_path.as_posix(),
            tests.kb7_area_path.as_posix(),
            "kb7",
        ),
    ],
)
def test_validation_cli(traces_path: str, area_path: str, name: str):
    args = [
        "--traces-path",
        traces_path,
        "--area-path",
        area_path,
        "--name",
        name,
    ]

    check_python_call(
        [PYTHON_INTERPRETER, VALIDATION_NOTEBOOK.as_posix(), *args],
    )


@pytest_mark_xfail_windows_flaky
@pytest.mark.parametrize(
    "args,raises",
    [
        (["--wrong-arg"], pytest.raises(subprocess.CalledProcessError)),
    ],
)
def test_validation_cli_args(args, raises):
    with raises:
        check_python_call([PYTHON_INTERPRETER, VALIDATION_NOTEBOOK.as_posix(), *args])


@pytest_mark_xfail_windows_flaky
@pytest.mark.parametrize(
    "snap_threshold", [fractopo.tval.trace_validation.Validation.SNAP_THRESHOLD]
)
@pytest.mark.parametrize(
    "traces_path,area_path,name",
    [
        (
            tests.kb7_trace_100_path.as_posix(),
            tests.kb7_area_path.as_posix(),
            "kb7",
        ),
    ],
)
def test_network_cli(
    traces_path: str,
    area_path: str,
    name: str,
    snap_threshold: float,
):
    args = [
        "--traces-path",
        traces_path,
        "--area-path",
        area_path,
        "--name",
        name,
        "--snap-threshold",
        str(snap_threshold),
    ]

    check_python_call(
        [PYTHON_INTERPRETER, NETWORK_NOTEBOOK.as_posix(), *args],
    )


@pytest_mark_xfail_windows_flaky
@pytest.mark.parametrize(
    "notebook_path",
    [VALIDATION_NOTEBOOK, NETWORK_NOTEBOOK],
)
def test_empty_traces_cli(notebook_path: Path, tmp_path: Path):
    """Test that an empty traces file produces an explicit error."""
    # Create an empty GeoJSON file (valid GeoJSON, but 0 features)
    empty_geojson = tmp_path / "empty_traces.geojson"
    empty_geojson.write_text('{"type": "FeatureCollection", "features": []}')

    # Use a valid area file
    area_path = SAMPLE_DATA_DIR.joinpath("hastholmen_area.geojson").as_posix()

    args = [
        "--traces-path",
        empty_geojson.as_posix(),
        "--area-path",
        area_path,
        "--name",
        "empty_test",
    ]

    with pytest.raises(subprocess.CalledProcessError):
        check_python_call(
            [PYTHON_INTERPRETER, notebook_path.as_posix(), *args],
        )


class TestResolveLayerName:
    """Tests for _resolve_layer_name helper in marimos/utils.py."""

    def test_user_specified_layer_returned_unchanged(self, tmp_path: Path):
        """When a layer name is provided, return it as-is without inspecting the file."""
        gpkg_path = tmp_path / "data.gpkg"
        gdf = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        gdf.to_file(gpkg_path, layer="my_layer", driver="GPKG")

        assert _resolve_layer_name(gpkg_path, "my_layer") == "my_layer"

    def test_auto_select_single_spatial_layer(self, tmp_path: Path):
        """Auto-select when exactly one spatial layer exists."""
        gpkg_path = tmp_path / "data.gpkg"
        gdf = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        gdf.to_file(gpkg_path, layer="traces", driver="GPKG")

        assert _resolve_layer_name(gpkg_path, None) == "traces"

    def test_gpkg_with_layer_styles_ignored(self, tmp_path: Path):
        """The layer_styles table should be filtered out, leaving one spatial layer."""
        gpkg_path = tmp_path / "data.gpkg"
        gdf = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        gdf.to_file(gpkg_path, layer="traces", driver="GPKG")

        # Add a QGIS-style layer_styles table registered in gpkg_contents
        conn = sqlite3.connect(gpkg_path)
        try:
            conn.execute(
                "CREATE TABLE layer_styles (id INTEGER PRIMARY KEY, f_table_name TEXT)"
            )
            conn.execute(
                "INSERT INTO gpkg_contents (table_name, data_type, identifier) "
                "VALUES ('layer_styles', 'attributes', 'layer_styles')"
            )
            conn.commit()
        finally:
            conn.close()

        assert _resolve_layer_name(gpkg_path, None) == "traces"

    def test_no_spatial_layers_raises(self, tmp_path: Path):
        """Raise ValueError when no spatial layers exist."""
        gpkg_path = tmp_path / "empty.gpkg"
        # First create a valid GPKG with a spatial layer, then remove the
        # spatial layer so that only non-spatial metadata tables remain.
        gdf = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")
        gdf.to_file(gpkg_path, layer="to_remove", driver="GPKG")

        conn = sqlite3.connect(gpkg_path)
        try:
            # Add a non-spatial table
            conn.execute("CREATE TABLE nonspatial (id INTEGER PRIMARY KEY, val TEXT)")
            conn.execute(
                "INSERT INTO gpkg_contents (table_name, data_type, identifier) "
                "VALUES ('nonspatial', 'attributes', 'nonspatial')"
            )
            # Remove the spatial layer
            conn.execute("DROP TABLE IF EXISTS to_remove")
            conn.execute("DELETE FROM gpkg_contents WHERE table_name = 'to_remove'")
            conn.execute(
                "DELETE FROM gpkg_geometry_columns WHERE table_name = 'to_remove'"
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(ValueError, match="No spatial layers found"):
            _resolve_layer_name(gpkg_path, None)

    def test_multiple_spatial_layers_raises(self, tmp_path: Path):
        """Raise ValueError when multiple spatial layers exist and none specified."""
        gpkg_path = tmp_path / "multi.gpkg"
        traces = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326"
        )
        areas = gpd.GeoDataFrame(
            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs="EPSG:4326"
        )
        traces.to_file(gpkg_path, layer="traces", driver="GPKG")
        areas.to_file(gpkg_path, layer="areas", driver="GPKG")

        with pytest.raises(ValueError, match="Multiple spatial layers found"):
            _resolve_layer_name(gpkg_path, None)


@pytest_mark_xfail_windows_flaky
@pytest.mark.parametrize(
    "notebook_path",
    [VALIDATION_NOTEBOOK, NETWORK_NOTEBOOK],
)
def test_gpkg_with_layer_styles_cli(notebook_path: Path, tmp_path: Path):
    """CLI should succeed when a GeoPackage contains a layer_styles table."""
    traces_gdf = gpd.read_file(tests.kb7_trace_100_path)
    area_gdf = gpd.read_file(tests.kb7_area_path)

    traces_gpkg = tmp_path / "traces.gpkg"
    area_gpkg = tmp_path / "area.gpkg"
    traces_gdf.to_file(traces_gpkg, layer="traces", driver="GPKG")
    area_gdf.to_file(area_gpkg, layer="area", driver="GPKG")

    # Inject layer_styles table into both files
    for gpkg_path in (traces_gpkg, area_gpkg):
        conn = sqlite3.connect(gpkg_path)
        try:
            conn.execute(
                "CREATE TABLE layer_styles (id INTEGER PRIMARY KEY, f_table_name TEXT)"
            )
            conn.execute(
                "INSERT INTO gpkg_contents (table_name, data_type, identifier) "
                "VALUES ('layer_styles', 'attributes', 'layer_styles')"
            )
            conn.commit()
        finally:
            conn.close()

    args = [
        "--traces-path",
        traces_gpkg.as_posix(),
        "--area-path",
        area_gpkg.as_posix(),
        "--name",
        "layer_styles_test",
    ]

    check_python_call(
        [PYTHON_INTERPRETER, notebook_path.as_posix(), *args],
    )
