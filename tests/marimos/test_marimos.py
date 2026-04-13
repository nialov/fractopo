import sqlite3
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

from marimos.utils import _resolve_layer_name

SAMPLE_DATA_DIR = Path(__file__).parent.parent.joinpath("sample_data/")
VALIDATION_NOTEBOOK = Path(__file__).parent.parent.parent.joinpath(
    "marimos/validation.py"
)
NETWORK_NOTEBOOK = Path(__file__).parent.parent.parent.joinpath("marimos/network.py")


@pytest.mark.parametrize(
    "notebook_path",
    [VALIDATION_NOTEBOOK, NETWORK_NOTEBOOK],
    ids=["validation", "network"],
)
def test_notebook_pytest(notebook_path: Path):
    """Run pytest on the marimo notebook to execute its embedded test cells."""
    exit_code = pytest.main(
        [
            str(notebook_path),
            "-x",
            "--no-header",
            "-rN",
            "--override-ini=addopts=",
        ]
    )
    assert exit_code == pytest.ExitCode.OK, (
        f"pytest failed on {notebook_path.name} with exit code {exit_code}"
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
