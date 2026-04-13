"""Pytest conftest for marimo notebook integration tests."""

from pathlib import Path

import geopandas as gpd
import pytest

SAMPLE_DATA_DIR = Path("tests/sample_data")


@pytest.fixture(
    params=[
        (
            SAMPLE_DATA_DIR / "hastholmen_traces.geojson",
            SAMPLE_DATA_DIR / "hastholmen_area.geojson",
            "hastholmen",
        ),
        (
            SAMPLE_DATA_DIR / "KB7" / "KB7_tulkinta_100.geojson",
            SAMPLE_DATA_DIR / "KB7" / "KB7_tulkinta_alue.geojson",
            "kb7",
        ),
    ],
    ids=["hastholmen", "kb7"],
)
def traces_area_name(request):
    """Fixture providing (traces_path, area_path, name) tuples."""
    return request.param


@pytest.fixture()
def traces_gdf(traces_area_name):
    """Load traces GeoDataFrame from the parameterized test data."""
    traces_path, _, _ = traces_area_name
    return gpd.read_file(traces_path)


@pytest.fixture()
def area_gdf(traces_area_name):
    """Load area GeoDataFrame from the parameterized test data."""
    _, area_path, _ = traces_area_name
    return gpd.read_file(area_path)


@pytest.fixture()
def dataset_name(traces_area_name):
    """Return the dataset name from the parameterized test data."""
    _, _, name = traces_area_name
    return name
