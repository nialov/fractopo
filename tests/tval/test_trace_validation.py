import pytest
import geopandas as gpd

from tests import Helpers
from tests.sample_data.py_samples import results_in_multijunction_list_of_ls

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
    fpls = gpd.GeoSeries(false_positive_linestrings)
    intersect_nodes, _ = general.determine_general_nodes(fpls, snap_threshold=0.01)

    faulty_junctions = MultiJunctionValidator.determine_faulty_junctions(
        intersect_nodes, snap_threshold=0.01, snap_threshold_error_multiplier=1.1
    )
    return fpls, intersect_nodes, faulty_junctions
