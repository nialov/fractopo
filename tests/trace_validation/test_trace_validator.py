import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, LineString, MultiLineString
import shapely.wkt as wkt
import numpy as np
import hypothesis
import pytest
from hypothesis.strategies import (
    booleans,
    floats,
    sets,
    lists,
    tuples,
    one_of,
    text,
    integers,
)
from hypothesis import given
from hypothesis_geometry import planar

from tests import (
    trace_validator,
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
    EmptyGeometryValidator,
    SimpleGeometryValidator,
    SharpCornerValidator,
)
from tests.sample_data import stacked_test
from tests.sample_data.py_samples.stacked_traces_sample import stacked_traces_ls

BaseValidator.set_snap_threshold_and_multipliers(
    SNAP_THRESHOLD, SNAP_THRESHOLD_ERROR_MULTIPLIER, AREA_EDGE_SNAP_MULTIPLIER
)


class TestBaseValidator:
    def test_set_snap_threshold_and_multipliers(self):
        assert BaseValidator.SNAP_THRESHOLD == SNAP_THRESHOLD
        assert (
            BaseValidator.SNAP_THRESHOLD_ERROR_MULTIPLIER
            == SNAP_THRESHOLD_ERROR_MULTIPLIER
        )
        assert BaseValidator.AREA_EDGE_SNAP_MULTIPLIER == AREA_EDGE_SNAP_MULTIPLIER

    def test_execute(self):
        try:
            BaseValidator.execute(
                Helpers.valid_gdf_get(), None, auto_fix=False, parallel=False
            )
        except NotImplementedError:
            pass

    def test_zip_equal(self):
        try:
            _ = BaseValidator.zip_equal([1, 2, 3], [1, 2])
        except ValueError:
            pass
        for a, b in BaseValidator.zip_equal([1, 2], [1, 2]):

            assert isinstance(a, int)
            assert isinstance(b, int)

    def test_handle_error_column(self):
        orig_cols = Helpers.valid_gdf_get().columns
        valid_result = BaseValidator.handle_error_column(Helpers.valid_gdf_get())
        assert len(valid_result.columns) == len(orig_cols)
        assert isinstance(valid_result, gpd.GeoDataFrame)
        assert len(valid_result) == len(Helpers.valid_gdf_get())
        assert all(valid_result[ERROR_COLUMN] == Helpers.valid_gdf_get()[ERROR_COLUMN])

        copy_gdf = Helpers.valid_gdf_with_faulty_error_col_get().copy()
        invalid_result = BaseValidator.handle_error_column(copy_gdf)
        assert isinstance(invalid_result, gpd.GeoDataFrame)
        assert len(invalid_result) == len(Helpers.valid_gdf_with_faulty_error_col_get())
        assert not all(
            invalid_result[ERROR_COLUMN]
            == Helpers.valid_gdf_with_faulty_error_col_get()[ERROR_COLUMN]
        )

        for idx, row in invalid_result.iterrows():
            assert isinstance(row[ERROR_COLUMN], list)
            assert row[ERROR_COLUMN] == []

    @given(one_of([floats(), integers(), lists(elements=text()), booleans(), text()]))
    def test_error_test(self, anything):
        assert isinstance(BaseValidator.error_test(anything), list)

    @given(Helpers.get_multi_polyline_strategy())
    def test_determine_nodes_hypothesis(self, multi_polyline_strategy):
        gdf = gpd.GeoDataFrame(
            {
                "geometry": gpd.GeoSeries(
                    [LineString(polyline) for polyline in multi_polyline_strategy]
                )
            }
        )
        nodes_of_interaction, node_id_data = BaseValidator.determine_nodes(gdf)


class TestGeomTypeValidator(TestBaseValidator):
    def test_validation_function(self):
        valid_result = GeomTypeValidator.validation_function(Helpers.valid_geom)
        invalid_result = GeomTypeValidator.validation_function(
            Helpers.invalid_geom_multilinestring
        )

        assert valid_result
        assert not invalid_result

    def test_fix_function(self):
        # Contains no errors in ERROR_COLUMN and the geom is a LineString
        row = Helpers.valid_gdf_get().iloc[0]
        try:
            result = GeomTypeValidator.fix_function(row)
        except TypeError:
            # Raises ValueError because cannot linemerge LineString.
            pass
        # Geometry is a MultiLineString and error column contains right error.
        multilinestring_row = pd.Series(
            {
                GEOMETRY_COLUMN: Helpers.invalid_geom_multilinestring,
                ERROR_COLUMN: [GeomTypeValidator.ERROR],
            }
        )
        mls_result = GeomTypeValidator.fix_function(multilinestring_row)
        assert isinstance(mls_result[GEOMETRY_COLUMN], shapely.geometry.MultiLineString)
        # Check that Error was not fixed
        # (the passed MultiLineString is unmergeable)
        assert GeomTypeValidator.ERROR in mls_result[ERROR_COLUMN]

    def test_validate(self):
        valid_gdf = Helpers.valid_gdf_get()
        result_gdf = GeomTypeValidator.validate(valid_gdf)
        assert all(
            [
                isinstance(geom, shapely.geometry.LineString)
                or GeomTypeValidator.ERROR in err
                for geom, err in zip(
                    result_gdf[GEOMETRY_COLUMN], result_gdf[ERROR_COLUMN]
                )
            ]
        )
        invalid_gdf = Helpers.invalid_gdf_get()
        checked_result_gdf = GeomTypeValidator.validate(invalid_gdf)
        # All are not LineStrings
        assert not all(
            [
                isinstance(geom, shapely.geometry.LineString)
                for geom in checked_result_gdf[GEOMETRY_COLUMN]
            ]
        )
        # MultiLineString typeerror has been caught
        assert any(
            [
                GeomTypeValidator.ERROR in errors
                for errors in checked_result_gdf[ERROR_COLUMN]
            ]
        )
        for idx, row in checked_result_gdf.iterrows():
            if isinstance(row[GEOMETRY_COLUMN], shapely.geometry.MultiLineString):
                assert GeomTypeValidator.ERROR in row[ERROR_COLUMN]
        return checked_result_gdf

    def test_fix(self):
        invalid_gdf = Helpers.invalid_gdf_get()
        checked_result_gdf = GeomTypeValidator.validate(invalid_gdf)
        print(checked_result_gdf)
        fixed_result_gdf = GeomTypeValidator.fix(checked_result_gdf)

        print(fixed_result_gdf)
        assert all(
            [
                isinstance(geom, shapely.geometry.LineString)
                or GeomTypeValidator.ERROR in err
                for geom, err in zip(
                    fixed_result_gdf[GEOMETRY_COLUMN], fixed_result_gdf[ERROR_COLUMN]
                )
            ]
        )
        # MultiLineString typeerror has been caught and fixed
        # It is no longer in error list
        assert len(fixed_result_gdf) > 0 and len(fixed_result_gdf) == len(
            checked_result_gdf
        )
        assert GEOMETRY_COLUMN in fixed_result_gdf.columns
        assert ERROR_COLUMN in fixed_result_gdf.columns

    def test_execute(self):
        valid_gdf_orig_cols = Helpers.valid_gdf_get().columns
        valid_result = GeomTypeValidator.execute(
            Helpers.valid_gdf_get(), None, auto_fix=False, parallel=False
        )
        assert all([col in valid_result.columns for col in valid_gdf_orig_cols])
        valid_result_fix = GeomTypeValidator.execute(
            Helpers.valid_gdf_get(), None, auto_fix=True, parallel=False
        )

        invalid_result = GeomTypeValidator.execute(
            Helpers.invalid_gdf_get(), None, auto_fix=False, parallel=False
        )
        invalid_result_fix = GeomTypeValidator.execute(
            Helpers.invalid_gdf_get(), None, auto_fix=True, parallel=False
        )
        assert len(valid_result) == len(valid_result_fix)
        assert len(invalid_result) == len(invalid_result_fix)
        assert all(
            [
                isinstance(geom, shapely.geometry.LineString)
                for geom in valid_result_fix[GEOMETRY_COLUMN]
            ]
        )
        # Either geom is a LineString or error has been reported.
        assert all(
            [
                isinstance(geom, shapely.geometry.LineString)
                or GeomTypeValidator.ERROR in err
                for geom, err in zip(
                    invalid_result_fix[GEOMETRY_COLUMN],
                    invalid_result_fix[ERROR_COLUMN],
                )
            ]
        )


class TestMultiJunctionValidator:
    def test_validate(self):
        valid_result = MultiJunctionValidator.validate(Helpers.valid_gdf_get())
        invalid_result = MultiJunctionValidator.validate(Helpers.invalid_gdf_get())
        # print(valid_result)
        # print(invalid_result)
        assert (
            MultiJunctionValidator.ERROR in all_errors
            for all_errors in invalid_result[BaseValidator.ERROR_COLUMN]
        )

    def test_determine_intersections(self):
        nodes_of_interaction, node_id_data = MultiJunctionValidator.determine_nodes_old(
            Helpers.valid_gdf_get()
        )
        (
            invalid_nodes_of_interaction,
            invalid_node_id_data,
        ) = MultiJunctionValidator.determine_nodes_old(Helpers.invalid_gdf_get())
        assert len(nodes_of_interaction) > 0 and len(node_id_data) > 0
        assert len(invalid_nodes_of_interaction) > 0 and len(invalid_node_id_data) > 0
        (
            optimized_valid_result_nodes_of_interaction,
            optimized_valid_result_node_id_data,
        ) = MultiJunctionValidator.get_nodes(Helpers.valid_gdf_get())
        (
            optimized_invalid_result_nodes_of_interaction,
            optimized_invalid_result_node_id_data,
        ) = MultiJunctionValidator.get_nodes(Helpers.invalid_gdf_get())

        assert len(optimized_valid_result_nodes_of_interaction) == len(
            nodes_of_interaction
        )
        assert len(optimized_valid_result_node_id_data) == len(node_id_data)
        assert all(
            [
                val in optimized_valid_result_nodes_of_interaction
                for val in nodes_of_interaction
            ]
        )

        BaseValidator.nodes_of_interaction_both, BaseValidator.node_id_data_both = (
            optimized_valid_result_nodes_of_interaction,
            optimized_valid_result_node_id_data,
        )
        (
            from_class_nodes_of_interaction,
            from_class_node_id_data,
        ) = MultiJunctionValidator.get_nodes(Helpers.valid_gdf_get())
        assert all(
            [
                val in from_class_nodes_of_interaction
                for val in optimized_valid_result_nodes_of_interaction
            ]
        )
        assert all(
            [
                val in from_class_node_id_data
                for val in optimized_valid_result_node_id_data
            ]
        )

    def test_validation_function(self):
        point_ids = (0, 1)
        faulty_junctions = Helpers.valid_traces
        assert isinstance(faulty_junctions.index, pd.RangeIndex)
        assert not MultiJunctionValidator.validation_function(
            point_ids, faulty_junctions
        )
        assert MultiJunctionValidator.validation_function(
            point_ids, faulty_junctions[2:]
        )

    def test_determine_faulty_junctions(self):
        stacked_points = [
            shapely.geometry.Point(0, 0),
            shapely.geometry.Point(0, 0),
            shapely.geometry.Point(0, 0),
            shapely.geometry.Point(0, 0),
        ]
        stacked_result = MultiJunctionValidator.determine_faulty_junctions(
            stacked_points
        )
        assert isinstance(stacked_result, gpd.GeoSeries)
        assert len(stacked_points) == len(stacked_result)

        non_stacked_points = [
            shapely.geometry.Point(0, 0),
            shapely.geometry.Point(2, 0),
            shapely.geometry.Point(0, 2),
            shapely.geometry.Point(0, 3),
        ]
        non_stacked_result = MultiJunctionValidator.determine_faulty_junctions(
            non_stacked_points
        )
        assert isinstance(non_stacked_result, gpd.GeoSeries)
        assert len(non_stacked_result) == 0


class TestVNodeValidator:
    def test_determine_v_nodes(self):
        stacked_points = [
            shapely.geometry.Point(0, 0),
            shapely.geometry.Point(0, 0),
        ]
        stacked_result = VNodeValidator.determine_v_nodes(stacked_points)
        assert isinstance(stacked_result, gpd.GeoSeries)
        assert len(stacked_points) == len(stacked_result)

        non_stacked_points = [
            shapely.geometry.Point(0, 0),
            shapely.geometry.Point(1, 1),
        ]
        non_stacked_result = VNodeValidator.determine_v_nodes(non_stacked_points)
        assert isinstance(non_stacked_result, gpd.GeoSeries)
        assert len(non_stacked_points) != len(non_stacked_result)
        assert len(non_stacked_result) == 0

    def test_validate(self):
        pass


class TestMultipleCrosscutValidator:
    def test_validation_function(self):
        assert MultipleCrosscutValidator.validation_function(0, [1, 2, 3])
        assert not MultipleCrosscutValidator.validation_function(0, [0, 2, 3])

    def test_validate(self):
        invalid_result = MultipleCrosscutValidator.validate(Helpers.invalid_gdf_get())
        print(invalid_result[MultipleCrosscutValidator.ERROR_COLUMN])
        assert MultipleCrosscutValidator.ERROR in str(
            invalid_result[MultipleCrosscutValidator.ERROR_COLUMN]
        )

    def test_determine_stacked_traces(self):
        invalid_rows_with_stacked = MultipleCrosscutValidator.determine_stacked_traces(
            Helpers.invalid_gdf_get()
        )
        assert len(invalid_rows_with_stacked) > 1

    def test_determine_stacked_traces_old(self):
        invalid_rows_with_stacked = (
            MultipleCrosscutValidator.determine_stacked_traces_old(
                Helpers.invalid_gdf_get()
            )
        )
        assert len(invalid_rows_with_stacked) > 1


class TestUnderlappingSnapValidator:
    # TODO
    def test_determine_underlapping(self):
        valid_result = UnderlappingSnapValidator.determine_underlapping(
            Helpers.valid_gdf_get()
        )
        assert len(valid_result) == 0

        invalid_result = UnderlappingSnapValidator.determine_underlapping(
            Helpers.invalid_gdf_get()
        )
        assert len(invalid_result) != 0


class TestTargetAreaSnapValidator:
    def test_determine_area_underlapping(self):
        valid_result = TargetAreaSnapValidator.determine_area_underlapping(
            Helpers.valid_gdf_get(), Helpers.valid_area_gdf_get()
        )
        assert len(valid_result) == 0

        invalid_result = TargetAreaSnapValidator.determine_area_underlapping(
            Helpers.invalid_gdf_get(), Helpers.invalid_area_gdf_get()
        )
        assert len(invalid_result) != 0

    def test_validate(self):
        valid_result = TargetAreaSnapValidator.validate(
            Helpers.valid_gdf_get(), Helpers.valid_area_gdf_get()
        )
        assert TargetAreaSnapValidator.ERROR not in str(valid_result[ERROR_COLUMN])

        invalid_result = TargetAreaSnapValidator.validate(
            Helpers.invalid_gdf_get(), Helpers.invalid_area_gdf_get()
        )
        assert TargetAreaSnapValidator.ERROR in str(invalid_result[ERROR_COLUMN])


class TestGeomNullValidator:
    def test_execute(self):
        invalid_result = GeomNullValidator.execute(
            Helpers.invalid_gdf_null_get(),
            Helpers.valid_area_gdf_get(),
            auto_fix=True,
            parallel=False,
        )
        assert len(Helpers.invalid_gdf_null_get()) > len(invalid_result)
        assert len(invalid_result) == 0


class TestAllValidators:
    def test_all(self):
        """
        Each validator is tested individually. Same gdf is not chained from
        one to the next
        """

        for validator in Helpers.iterate_validators():
            assert issubclass(validator, BaseValidator)
            valid_result = validator.execute(
                Helpers.valid_gdf_get(),
                Helpers.valid_area_gdf_get(),
                auto_fix=False,
                parallel=False,
            )
            valid_result_fix = validator.execute(
                Helpers.valid_gdf_get(),
                Helpers.valid_area_gdf_get(),
                auto_fix=True,
                parallel=False,
            )
        assert all(
            [col in valid_result.columns for col in Helpers.valid_gdf_get().columns]
        )
        assert all(
            [col in valid_result_fix.columns for col in Helpers.valid_gdf_get().columns]
        )
        # After iterating and executing once through all validators
        # the nodes should have been calculated with parallel=True:
        for validator in Helpers.iterate_validators():
            assert issubclass(validator, BaseValidator)
            _ = validator.execute(
                Helpers.valid_gdf_get(),
                Helpers.valid_area_gdf_get(),
                auto_fix=False,
                parallel=True,
            )
        for validator in Helpers.iterate_validators():
            assert validator.nodes_calculated

    def test_all_chained(self):
        """
        The same gdf is passed through all validators in chain.
        """
        valid_result = Helpers.valid_gdf_get()
        valid_result_cols = valid_result.columns
        valid_result_fix = Helpers.valid_gdf_get()
        valid_result_fix_cols = valid_result_fix.columns
        for validator in Helpers.iterate_validators():
            valid_result = validator.execute(
                valid_result,
                Helpers.valid_area_gdf_get(),
                auto_fix=False,
                parallel=False,
            )
            valid_result_fix = validator.execute(
                valid_result_fix,
                Helpers.valid_area_gdf_get(),
                auto_fix=True,
                parallel=False,
            )
            assert all([col in valid_result.columns for col in valid_result_cols])
            assert all(
                [col in valid_result_fix.columns for col in valid_result_fix_cols]
            )

    def test_all_chained_invalid(self):
        """
        Invalid gdf passed through all validators.
        """
        invalid_result = Helpers.invalid_gdf_get()
        invalid_result_fix = Helpers.invalid_gdf_get()
        invalid_result_null_fix = Helpers.invalid_gdf_null_get()
        invalid_result_faulty_error_fix = Helpers.valid_gdf_with_faulty_error_col_get()
        all_iterated = False
        for validator in Helpers.iterate_validators():
            invalid_result = validator.execute(
                invalid_result,
                Helpers.invalid_area_gdf_get(),
                auto_fix=False,
                parallel=False,
            )
            invalid_result_fix = validator.execute(
                invalid_result_fix,
                Helpers.invalid_area_gdf_get(),
                auto_fix=True,
                parallel=False,
            )
            invalid_result_faulty_error_fix = validator.execute(
                invalid_result_faulty_error_fix,
                Helpers.valid_area_gdf_get(),
                auto_fix=True,
                parallel=False,
            )
            invalid_result_null_fix = validator.execute(
                invalid_result_null_fix,
                Helpers.invalid_area_gdf_get(),
                auto_fix=True,
                parallel=False,
            )
            # TargetAreaSnapValidator currently last in iterables.
            # If it is reached probably all others are as well.
            if validator == TargetAreaSnapValidator:
                all_iterated = True
        assert all_iterated


class TestStackedTracesValidator:
    def test_determine_overlapping_traces(self):
        valid_result = StackedTracesValidator.determine_overlapping_traces(
            Helpers.valid_gdf_get()
        )
        assert len(valid_result) == 0

        invalid_result = StackedTracesValidator.determine_overlapping_traces(
            Helpers.invalid_gdf_get()
        )
        assert len(invalid_result) != 0

    def test_validate(self):
        valid_result = StackedTracesValidator.validate(
            Helpers.valid_gdf_get(), Helpers.valid_area_gdf_get()
        )
        assert StackedTracesValidator.ERROR not in str(valid_result[ERROR_COLUMN])

        invalid_result = StackedTracesValidator.validate(
            Helpers.invalid_gdf_get(), Helpers.invalid_area_gdf_get()
        )
        assert StackedTracesValidator.ERROR in str(invalid_result[ERROR_COLUMN])
        assert (
            len(
                [
                    err
                    for err in invalid_result[ERROR_COLUMN]
                    if StackedTracesValidator.ERROR in err
                ]
            )
            >= 3
        )

    def test_segment_within_buffer(self):
        linestring = LineString([Point(7, -4), Point(7, 4)])
        linestring_that_overlaps = MultiLineString(
            [
                LineString(
                    [
                        Point(7 + SNAP_THRESHOLD * 0.5, -3.5),
                        Point(7 + SNAP_THRESHOLD * 0.5, -3),
                    ]
                )
            ]
        )
        result = StackedTracesValidator.segment_within_buffer(
            linestring, linestring_that_overlaps
        )
        assert result

    def test_segmentize_linestring(self):
        linestring = LineString(
            [
                (0, 0),
                (10, 0),
            ]
        )
        linestring2 = LineString(
            [
                (0, 0),
                (10, 0),
                (11, 0),
                (12, 0),
                (21, 0),
            ]
        )
        result = StackedTracesValidator.segmentize_linestring(linestring, 1)
        result2 = StackedTracesValidator.segmentize_linestring(linestring, 3)
        result3 = StackedTracesValidator.segmentize_linestring(linestring2, 1)
        result4 = StackedTracesValidator.segmentize_linestring(linestring2, 3)
        assert len(result2) < len(result4)
        assert len(result) < len(result3)

    def test_validate_with_known_data(self):
        linestrings = stacked_test.stacked_linestrings
        trace_gdf = gpd.GeoDataFrame(
            {GEOMETRY_COLUMN: linestrings, ERROR_COLUMN: [[] for _ in linestrings]}
        )
        result = StackedTracesValidator.validate(trace_gdf)
        assert StackedTracesValidator.ERROR not in str(result[ERROR_COLUMN])
        overlap_detection_multiplier = (
            StackedTracesValidator.OVERLAP_DETECTION_MULTIPLIER
        )
        # Check if the fix I implemented really fixes the issue ->
        # With OVERLAP_DETECTION_MULTIPLIER as 0, the function works
        # as if no fix has been implemented.
        StackedTracesValidator.OVERLAP_DETECTION_MULTIPLIER = 0.0
        result = StackedTracesValidator.validate(trace_gdf)
        assert StackedTracesValidator.ERROR in str(result[ERROR_COLUMN])
        StackedTracesValidator.OVERLAP_DETECTION_MULTIPLIER = (
            overlap_detection_multiplier
        )

    def test_validate_with_clear_overlap(self):
        # BaseValidator.set_snap_threshold_and_multipliers(
        #     SNAP_THRESHOLD * 10,
        #     SNAP_THRESHOLD_ERROR_MULTIPLIER,
        #     AREA_EDGE_SNAP_MULTIPLIER,
        # )
        trace_gdf = gpd.GeoDataFrame(
            {
                GEOMETRY_COLUMN: stacked_test.overlaps_and_cuts_self_linestrings,
                ERROR_COLUMN: [
                    [] for _ in stacked_test.overlaps_and_cuts_self_linestrings
                ],
            }
        )
        result = StackedTracesValidator.validate(trace_gdf)
        assert StackedTracesValidator.ERROR in str(result[ERROR_COLUMN])
        # BaseValidator.set_snap_threshold_and_multipliers(
        #     SNAP_THRESHOLD, SNAP_THRESHOLD_ERROR_MULTIPLIER, AREA_EDGE_SNAP_MULTIPLIER
        # )

    @staticmethod
    def test_with_known_non_stacked():
        stacked_traces_gdf = gpd.GeoDataFrame(geometry=stacked_traces_ls).set_crs(3067)
        result_gdf = StackedTracesValidator.execute(
            stacked_traces_gdf, area_geodataframe=None, auto_fix=False, parallel=False
        )
        assert BaseValidator.ERROR_COLUMN in result_gdf.columns
        assert StackedTracesValidator.ERROR not in str(
            result_gdf[BaseValidator.ERROR_COLUMN]
        )

    @staticmethod
    def test_determine_middle_in_triangle():
        segments = [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),  # middle
            LineString([(2, 2), (3, 3)]),
        ]
        result = StackedTracesValidator.determine_middle_in_triangle(segments)
        assert result.wkt == segments[1].wkt


class TestEmptyGeometryValidator:
    def test_validate(self):
        invalid_result = EmptyGeometryValidator.validate(
            gpd.GeoDataFrame({GEOMETRY_COLUMN: [LineString()], ERROR_COLUMN: [[]]})
        )
        assert EmptyGeometryValidator.ERROR in str(invalid_result[ERROR_COLUMN])
        valid_result = EmptyGeometryValidator.validate(
            gpd.GeoDataFrame(
                {GEOMETRY_COLUMN: [LineString([(1, 1), (2, 2)])], ERROR_COLUMN: [[]]}
            )
        )
        assert EmptyGeometryValidator.ERROR not in str(valid_result[ERROR_COLUMN])


class TestSimpleGeometryValidator:
    def test_validate(self):
        valid_result = SimpleGeometryValidator.validate(Helpers.valid_gdf_get())
        assert SimpleGeometryValidator.ERROR not in str(valid_result[ERROR_COLUMN])
        invalid_result = SimpleGeometryValidator.validate(
            gpd.GeoDataFrame(
                {
                    GEOMETRY_COLUMN: [wkt.loads(stacked_test.non_simple_geometry)],
                    ERROR_COLUMN: [[]],
                }
            )
        )
        assert SimpleGeometryValidator.ERROR in str(invalid_result[ERROR_COLUMN])


class TestSharpCornerValidator:
    def test_validation_function(self):
        ls = LineString(
            [
                Point(-2, 0),
                Point(-2, 10),
                Point(-3, 0),
            ]
        )
        assert not SharpCornerValidator.validation_function(ls)
        ls = LineString(
            [
                Point(-2, 0),
                Point(-3, 0),
                Point(-4, 0),
            ]
        )
        assert SharpCornerValidator.validation_function(ls)

    def test_with_very_edgy_and_not_very_edgy(self):
        very_edgy_linestrings = stacked_test.very_edgy_linestring_list
        assert not any(
            list(map(SharpCornerValidator.validation_function, very_edgy_linestrings))
        )
        not_edgy = stacked_test.not_so_edgy_linestrings_list
        # First wkt LineString will not pass. It was too edgy all along.
        assert sum(list(map(SharpCornerValidator.validation_function, not_edgy))) == 3

    @given(
        planar.polylines(Helpers.nice_integer_coordinates)
        .map(LineString)
        .filter(lambda polyline: polyline.is_valid)
    )
    def test_validation_function_hypothesis(self, trace):
        result = SharpCornerValidator.validation_function(trace)
        if len(trace_validator.get_trace_coord_points(trace)) == 2:
            assert result


def test_point_to_xy():
    point = Point(1, 0)
    x, y = trace_validator.point_to_xy(point)
    assert np.isclose(x, 1)
    assert np.isclose(y, 0)
