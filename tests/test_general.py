import fractopo.general as general
from shapely.geometry import Point
import geopandas as gpd

from hypothesis import given
from tests import Helpers
import pytest


@pytest.mark.parametrize(
    "first,second,same,from_first,is_none", Helpers.test_match_crs_params
)
def test_match_crs(first, second, same: bool, from_first: bool, is_none: bool):
    if from_first:
        original_crs = first.crs
    else:
        original_crs = second.crs
    first_matched, second_matched = general.match_crs(first, second)
    if same:
        assert first_matched.crs == original_crs
        assert second_matched.crs == original_crs
    else:
        assert first_matched.crs != second_matched.crs
    if is_none:
        assert first_matched.crs is None
        assert second_matched.crs is None


# @given()
# def test_determine_set(value, value_range, loop_around):
#     result = general.determine_set(value, value_range, loop_around)
#     assert isinstance(result, bool)
