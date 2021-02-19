"""
Test trace Validators.
"""
import geopandas as gpd
import pytest
from shapely.geometry import LineString

from fractopo import general
from fractopo.tval.trace_validation import Validation
from fractopo.tval.trace_validators import TargetAreaSnapValidator, VNodeValidator
from tests import Helpers
from tests.sample_data.py_samples.samples import results_in_multijunction_list_of_ls


@pytest.mark.parametrize(
    "endpoint_nodes, snap_threshold,snap_threshold_error_multiplier,assumed_result",
    Helpers.test_determine_v_nodes_params,
)
def test_determine_v_nodes(
    endpoint_nodes, snap_threshold, snap_threshold_error_multiplier, assumed_result
):
    result = VNodeValidator.determine_v_nodes(
        endpoint_nodes, snap_threshold, snap_threshold_error_multiplier
    )
    if assumed_result is not None:
        assert result == assumed_result


def test_determine_faulty_junctions_with_known_false_pos():
    false_positive_linestrings = results_in_multijunction_list_of_ls
    fpls = gpd.GeoDataFrame(geometry=gpd.GeoSeries(false_positive_linestrings))
    area = general.bounding_polygon(fpls)
    area_gdf = gpd.GeoDataFrame({"geometry": [area]})
    validation = Validation(
        traces=fpls,
        area=area_gdf,
        name="teest",
        allow_fix=True,
        SHARP_AVG_THRESHOLD=80.0,
        SHARP_PREV_SEG_THRESHOLD=70.0,
    )
    result = validation.run_validation()
    assert set(result[Validation.ERROR_COLUMN].astype(str)) == {"['SHARP TURNS']", "[]"}
    # faulty_junctions = validation.faulty_junctions
    # endpoint_nodes, intersect_nodes = (
    #     validation.endpoint_nodes,
    #     validation.intersect_nodes,
    # )
    # return fpls, intersect_nodes, endpoint_nodes, faulty_junctions


class TestTargetAreaSnapValidator:

    """
    Tests for TargetAreaSnapValidator.
    """

    @staticmethod
    @pytest.mark.parametrize(
        "geom,area,snap_threshold,"
        "snap_threshold_error_multiplier,area_edge_snap_multiplier,"
        "assume_result",
        Helpers.test_testtargetareasnapvalidator_validation_method,
    )
    def test_validation_method(
        geom: LineString,
        area: gpd.GeoDataFrame,
        snap_threshold: float,
        snap_threshold_error_multiplier: float,
        area_edge_snap_multiplier: float,
        assume_result: bool,
    ):
        """
        Test validation_method.
        """
        assert assume_result == TargetAreaSnapValidator.validation_method(
            geom=geom,
            area=area,
            snap_threshold=snap_threshold,
            snap_threshold_error_multiplier=snap_threshold_error_multiplier,
            area_edge_snap_multiplier=area_edge_snap_multiplier,
        )
