import pytest
import geopandas as gpd

from tests import Helpers
from tests.sample_data.py_samples.samples import results_in_multijunction_list_of_ls

from fractopo.tval.trace_validation import (
    BaseValidator,
    GeomNullValidator,
    GeomTypeValidator,
    SimpleGeometryValidator,
    MultiJunctionValidator,
    VNodeValidator,
    MultipleCrosscutValidator,
    UnderlappingSnapValidator,
    TargetAreaSnapValidator,
    StackedTracesValidator,
    SharpCornerValidator,
)
import fractopo.general as general
from fractopo.tval.executor_v2 import Validation


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
    validation = Validation(traces=fpls, area=area_gdf, name="teest", allow_fix=True)
    result = validation.run_validation()
    assert set(result[Validation.ERROR_COLUMN].astype(str)) == {"['SHARP TURNS']", "[]"}
    # faulty_junctions = validation.faulty_junctions
    # endpoint_nodes, intersect_nodes = (
    #     validation.endpoint_nodes,
    #     validation.intersect_nodes,
    # )
    # return fpls, intersect_nodes, endpoint_nodes, faulty_junctions
