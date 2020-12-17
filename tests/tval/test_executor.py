import geopandas as gpd
from pathlib import Path
from fractopo.tval import executor
import pytest
from fractopo.tval.trace_validator import BaseValidator
import fractopo.tval.trace_validation as trace_validation
import fractopo.tval.trace_builder as trace_builder
from fractopo.tval.executor import Validation


from timeit import default_timer as timer


from tests import (
    Helpers,
    SNAP_THRESHOLD,
    SNAP_THRESHOLD_ERROR_MULTIPLIER,
    AREA_EDGE_SNAP_MULTIPLIER,
    GEOMETRY_COLUMN,
    ERROR_COLUMN,
    BaseValidator,
    GeomTypeValidator,
    MultiJunctionValidator,
    VNodeValidator,
    MultipleCrosscutValidator,
    TargetAreaSnapValidator,
    UnderlappingSnapValidator,
    GeomNullValidator,
    StackedTracesValidator,
)

BaseValidator.set_snap_threshold_and_multipliers(
    SNAP_THRESHOLD, SNAP_THRESHOLD_ERROR_MULTIPLIER, AREA_EDGE_SNAP_MULTIPLIER
)

test_save_validated_gdfs_params = [
    {"gdf1": Helpers.valid_gdf_get()},
    {"gdf2": Helpers.valid_gdf_get()},
]

# (traces, areas, names, auto_fix):
test_main_params = [
    (
        [Helpers.valid_gdf_get(), Helpers.invalid_gdf_get()],
        [Helpers.valid_area_gdf_get(), Helpers.invalid_area_gdf_get()],
        ["valid", "invalid"],
        True,
    ),
    (
        [Helpers.valid_gdf_get(), Helpers.invalid_gdf_get()],
        [Helpers.valid_area_gdf_get(), Helpers.invalid_area_gdf_get()],
        ["valid", "invalid"],
        False,
    ),
]


def test_assemble_error_column():
    original_column = [["a"], ["b"], ["v"]]
    new_columns = (
        [[1, 2], [2, 5, 6], [36, 123]],
        [[1, 2], [2, 5, 6], [36, 123]],
    )
    result = executor.assemble_error_column(original_column, new_columns)
    for og, res, nc_0, nc_1 in zip(
        original_column, result, new_columns[0], new_columns[1]
    ):
        assert all([n in res for n in og])
        assert all([n in res for n in nc_0])
        assert all([n in res for n in nc_1])


def main_test(trace_gdf, area_gdf):
    result = executor.main([trace_gdf], [area_gdf])
    assert len(result) != 0
    return result


@pytest.mark.parametrize("gdfs", test_save_validated_gdfs_params)
def test_save_validated_gdfs(gdfs, tmp_path):
    executor.save_validated_gdfs(gdfs, tmp_path)
    executor.save_validated_gdfs(gdfs, tmp_path / Path("notafolder"))
    assert (tmp_path / Path("notafolder")).exists()
    assert (tmp_path / Path("notafolder")).is_dir()


@pytest.mark.parametrize(
    "traces, areas, names, auto_fix",
    test_main_params,
)
def test_main(
    data_regression,
    traces,
    areas,
    names,
    auto_fix,
):
    # TODO: Invalid doesnt work because MultiLineString cannot be fixed
    try:
        gdfs = executor.main(traces, areas, names, auto_fix)
    except AssertionError:
        if "invalid" in names:
            from warnings import warn

            warn(
                "test_main: Invalid doesnt work because MultiLineString cannot be fixed"
            )
            return
        else:
            raise
    for key in gdfs:
        gdf = gdfs[key]
        assert ERROR_COLUMN in gdf.columns
        if key == "valid":
            assert all([len(e) == 0 for e in gdf[ERROR_COLUMN]])
        data_regression.check(gdf[ERROR_COLUMN].to_dict())
        if auto_fix:
            gdf[ERROR_COLUMN] = gdf[ERROR_COLUMN].astype(str)
            gdf["error_column_backup"] = gdf["error_column_backup"].astype(str)
            print(gdf.columns)
            gdf.to_file(f"dev/test_gdfs/{key}.gpkg", driver="GPKG")


# def test_main_threaded():
#     trace_path = Path("tests/sample_data/KB7/KB7_tulkinta.shp")
#     area_path = Path("tests/sample_data/KB7/KB7_tulkinta_alue.shp")
#     result = executor.main_threaded([trace_path], [area_path])
#     non_threaded_result = main_test(trace_path, area_path)
#     assert isinstance(result, dict) and isinstance(non_threaded_result, dict)
#     for key, val in result.items():
#         assert key in non_threaded_result.keys()
#         result_gdf = result[key]
#         non_threaded_gdf = non_threaded_result[key]
#         for idxrow1, idxrow2 in zip(result_gdf.iterrows(), non_threaded_gdf.iterrows()):
#             for err in idxrow1[1][BaseValidator.ERROR_COLUMN]:
#                 assert err in idxrow2[1][BaseValidator.ERROR_COLUMN]


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
