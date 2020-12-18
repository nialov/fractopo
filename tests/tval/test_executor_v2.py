from typing import List, Optional

import pytest
import geopandas as gpd

from tests import Helpers
from fractopo.tval.executor_v2 import Validation


@pytest.mark.parametrize(
    "validator, geom, current_errors, allow_fix,assumed_result",
    Helpers.test__validate_params,
)
def test__validate(validator, geom, current_errors, allow_fix, assumed_result):
    geom, current_errors, ignore_geom = Validation._validate(
        geom, validator, current_errors, allow_fix
    )
    if assumed_result is not None:
        assert geom == assumed_result[0]
        assert current_errors == assumed_result[1]
        assert ignore_geom == assumed_result[2]


@pytest.mark.parametrize(
    "traces,area,name,allow_fix,assume_errors",
    Helpers.test_validation_params,
)
def test_validation(traces, area, name, allow_fix, assume_errors: Optional[List[str]]):
    validated_gdf = Validation(traces, area, name, allow_fix).run_validation()
    assert isinstance(validated_gdf, gpd.GeoDataFrame)
    assert Validation.ERROR_COLUMN in validated_gdf.columns.values
    if assume_errors is not None:
        for assumed_error in assume_errors:
            flat_validated_gdf_errors = [
                val
                for subgroup in validated_gdf[Validation.ERROR_COLUMN].values
                for val in subgroup
            ]
            assert assumed_error in flat_validated_gdf_errors
    validated_gdf[Validation.ERROR_COLUMN] = validated_gdf[
        Validation.ERROR_COLUMN
    ].astype(str)
    return validated_gdf
