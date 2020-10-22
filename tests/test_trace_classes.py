# import pandas as pd
# import geopandas as gpd
# import shapely
# from shapely.geometry import LineString, MultiLineString, Point
# import numpy as np
# import hypothesis
# from pathlib import Path

# from tval import trace_classes, trace_builder
# from tval.trace_builder import line_generator
# from tval.trace_classes import TraceSeries, TraceDataFrame
# from tval.trace_validator import BaseValidator


# class Helpers:

#     snap_threshold = 0.01
#     snap_threshold_error_multiplier = 1.1
#     area_edge_snap_multiplier = 5
#     BaseValidator.set_snap_threshold_and_multipliers(
#         snap_threshold=snap_threshold,
#         snap_threshold_error_multiplier=snap_threshold_error_multiplier,
#         area_edge_snap_multiplier=area_edge_snap_multiplier,
#     )
#     valid_data = [
#         LineString([(0, 0), (1, 1)]),
#         LineString([(0, 0), (2, 1)]),
#         LineString([(0, 0), (3, 1)]),
#         LineString([(0, 0), (4, 1)]),
#     ]

#     @classmethod
#     def get_valid_data(cls):
#         return cls.valid_data.copy()

#     invalid_data = [Point(1, 1), None, np.nan, LineString()]

#     @classmethod
#     def get_invalid_data(cls):
#         return cls.invalid_data.copy()

#     valid_path_kb7 = Path("tests/sample_data/KB7/KB7_tulkinta.shp")

#     valid_traces, invalid_traces, valid_areas, invalid_areas = trace_builder.main(
#         snap_threshold=snap_threshold,
#         snap_threshold_error_multiplier=snap_threshold_error_multiplier,
#     )

#     @classmethod
#     def get_valid_traces_and_areas(cls):
#         return cls.valid_traces.copy(), cls.valid_areas.copy()

#     @classmethod
#     def get_valid_traces_and_areas_traceseries(cls):
#         return TraceSeries(cls.valid_traces.copy()), cls.valid_areas.copy()

#     @classmethod
#     def get_invalid_traces_and_areas(cls):
#         return cls.invalid_traces.copy(), cls.invalid_areas.copy()

#     @classmethod
#     def get_invalid_traces_and_areas_traceseries(cls):
#         return TraceSeries(cls.invalid_traces.copy()), cls.invalid_areas.copy()


# def test_trace_series():
#     trace_series = TraceSeries(Helpers.get_valid_data())
#     try:
#         trace_series = TraceSeries(Helpers.get_invalid_data())
#         assert False
#     except TypeError:
#         pass


# def test_trace_geodataframe():
#     trace_dataframe = TraceDataFrame(
#         {
#             "geometry": Helpers.get_valid_data(),
#             "test_col": ["aa" for _ in Helpers.get_invalid_data()],
#         }
#     )
#     try:
#         trace_dataframe = TraceDataFrame({"geometry": Helpers.get_invalid_data(),})
#         assert False
#     except TypeError:
#         pass
#     trace_dataframe_kb7 = TraceDataFrame(gpd.read_file(Helpers.valid_path_kb7))
#     assert isinstance(trace_dataframe_kb7["geometry"], TraceSeries)
#     assert isinstance(trace_dataframe_kb7.geometry, TraceSeries)
#     assert isinstance(trace_dataframe["geometry"], TraceSeries)
#     assert isinstance(trace_dataframe.geometry, TraceSeries)


# def test_determine_nodes():
#     valid_traces, _ = Helpers.get_valid_traces_and_areas()
#     valid_traceseries = TraceSeries(valid_traces)

#     nodes_of_interaction, node_id_data = trace_classes.determine_nodes(
#         valid_traceseries, snap_threshold=Helpers.snap_threshold
#     )
#     nodes_of_interaction_old, node_id_data_old = BaseValidator.determine_nodes(
#         gpd.GeoDataFrame({"geometry": valid_traces})
#     )
#     assert len(nodes_of_interaction) > 0 and len(node_id_data) > 0
#     for i1, i2 in zip(nodes_of_interaction, nodes_of_interaction_old):
#         assert i1 == i2
#     for i1, i2 in zip(node_id_data, node_id_data_old):
#         assert i1 == i2


# def test_read_path_to_tracedataframe():
#     invalid_filepath = Path("this/file/really/doesnt/exist")
#     valid_filepath = Path("tests/sample_data/KB7/KB7_tulkinta.shp")
#     try:
#         trace_classes.read_path_to_tracedataframe(invalid_filepath)
#     except FileNotFoundError:
#         pass
#     result = trace_classes.read_path_to_tracedataframe(valid_filepath)
#     assert isinstance(result, TraceDataFrame)


# def test_check_for_line_geometry():
#     valid_traces, _ = Helpers.get_valid_traces_and_areas()
#     invalid_traces, _ = Helpers.get_invalid_traces_and_areas()
#     trace_classes.check_for_line_geometry(valid_traces)
#     try:
#         trace_classes.check_for_line_geometry(invalid_traces)
#     except TypeError:
#         pass
#     try:
#         trace_classes.check_for_line_geometry(gpd.GeoSeries())
#     except ValueError:
#         pass


# def test_get_trace_endpoints():
#     valid_result = trace_classes.get_trace_endpoints(LineString([(0, 0), (1, 1)]))
#     mls = trace_builder.multi_line_generator(
#         [(Point(3, -4), Point(3, -1)), (Point(3, 0), Point(3, 4))]
#     )
#     try:
#         invalid_result = trace_classes.get_trace_endpoints(mls)
#     except NotImplementedError:
#         pass


# def test_get_nodes():
#     valid_traces, _ = Helpers.get_valid_traces_and_areas_traceseries()
#     assert isinstance(valid_traces, TraceSeries)
#     nodes_of_interaction_both, node_id_data_both = valid_traces.get_nodes(
#         snap_threshold=Helpers.snap_threshold, interactions=True, endpoints=True
#     )
#     assert len(nodes_of_interaction_both) > 0 and len(node_id_data_both) > 0
#     # Test that trying to get nodes raises an error when they're not determined.
#     try:
#         valid_traces_copy = valid_traces.copy()
#         valid_traces_copy.nodes_of_interaction_both = None
#         valid_traces_copy.nodes_determined = True
#         valid_traces_copy.get_nodes(snap_threshold=Helpers.snap_threshold)
#     except AttributeError:
#         pass
