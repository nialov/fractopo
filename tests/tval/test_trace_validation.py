from typing import List, Optional

import geopandas as gpd
import pytest
from shapely.wkt import loads

from fractopo.tval.trace_validation import Validation
from fractopo.tval.trace_validators import MultiJunctionValidator, SharpCornerValidator
from tests import Helpers, ValidationHelpers


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
    additional_kwargs = dict()
    if assume_errors and SharpCornerValidator.ERROR in assume_errors:
        additional_kwargs = dict(
            SHARP_AVG_THRESHOLD=80.0, SHARP_PREV_SEG_THRESHOLD=70.0
        )

    validated_gdf = Validation(
        traces, area, name, allow_fix, **additional_kwargs
    ).run_validation()
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


@pytest.mark.parametrize(
    "traces,area,name,allow_fix,assume_errors,error_amount,false_positive",
    ValidationHelpers.get_all_errors(),
)
def test_validation_known(
    traces,
    area,
    name,
    allow_fix,
    assume_errors: Optional[List[str]],
    error_amount,
    false_positive: bool,
):
    # Thresholds are passed explicitly to avoid error changes with default
    # threshold changes.
    validated_gdf = Validation(
        traces,
        area,
        name,
        allow_fix,
        SNAP_THRESHOLD=0.01,
        SNAP_THRESHOLD_ERROR_MULTIPLIER=1.1,
        AREA_EDGE_SNAP_MULTIPLIER=1.1,
        TRIANGLE_ERROR_SNAP_MULTIPLIER=10.0,
        OVERLAP_DETECTION_MULTIPLIER=50.0,
        SHARP_AVG_THRESHOLD=135.0,
        SHARP_PREV_SEG_THRESHOLD=100.0,
    ).run_validation()
    assert isinstance(validated_gdf, gpd.GeoDataFrame)
    assert Validation.ERROR_COLUMN in validated_gdf.columns.values
    if assume_errors is not None:
        for assumed_error in assume_errors:
            flat_validated_gdf_errors = [
                val
                for subgroup in validated_gdf[Validation.ERROR_COLUMN].values
                for val in subgroup
            ]
            if false_positive:
                assert assumed_error not in flat_validated_gdf_errors
            else:
                assert assumed_error in flat_validated_gdf_errors
                assert (
                    sum([err == assumed_error for err in flat_validated_gdf_errors])
                    == error_amount
                )
