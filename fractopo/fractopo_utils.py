"""
Miscellaneous utilities and scripts of fractopo.
"""
from itertools import count
from typing import List, Union

import geopandas as gpd
from fractopo.general import (
    compare_unit_vector_orientation,
    create_unit_vector,
    get_trace_endpoints,
)
from shapely.geometry import LineString


class LineMerge:
    @staticmethod
    def conditional_linemerge(
        first: LineString, second: LineString, tolerance: float, buffer_value: float
    ) -> Union[None, LineString]:
        """
        Conditionally merge two LineStrings (first and second).

        Merge occurs if:
            1. Their endpoints are within buffer_value of each other.
            2. Their total orientations are within tolerance (degrees) of each
               other.

        Merges by joining their coordinates. The endpoint
        (that is within buffer_value of endpoint of first) of the second
        LineString is trimmed from the resulting coordinates.
        """
        assert isinstance(first, LineString) and isinstance(second, LineString)
        # Get trace endpoints
        first_start, first_end = get_trace_endpoints(first)
        second_start, second_end = get_trace_endpoints(second)

        # Get unit vectors from endpoints
        first_unit_vector = create_unit_vector(first_start, first_end)
        second_unit_vector = create_unit_vector(second_start, second_end)

        # Check if unit vectors are close in orientation
        are_close = compare_unit_vector_orientation(
            first_unit_vector, second_unit_vector, threshold_angle=tolerance
        )
        are_close_reverse = compare_unit_vector_orientation(
            -first_unit_vector, second_unit_vector, threshold_angle=tolerance
        )
        # Get coordinates
        first_coords = first.coords
        second_coords = second.coords
        if (
            first_end.buffer(buffer_value).intersects(second_start.buffer(buffer_value))
            and are_close
        ):
            # First ends in the start of second -> Sequence of coords is correct
            # Do not include first coordinate of second in new
            new_coords = [coord for coord in first_coords] + [
                coord for coord in second_coords
            ][1:]
        elif (
            first_end.buffer(buffer_value).intersects(second_end.buffer(buffer_value))
            and are_close_reverse
        ):
            # First ends in second end
            new_coords = [coord for coord in first_coords] + list(
                reversed([coord for coord in second_coords])
            )[1:]
        elif (
            first_start.buffer(buffer_value).intersects(
                second_start.buffer(buffer_value)
            )
            and are_close_reverse
        ):
            # First starts from the same as second
            new_coords = list(reversed([coord for coord in second_coords]))[:-1] + [
                coord for coord in first_coords
            ]
        elif (
            first_start.buffer(buffer_value).intersects(second_end.buffer(buffer_value))
            and are_close
        ):
            # First starts from end of second
            new_coords = [coord for coord in second_coords][:-1] + [
                coord for coord in first_coords
            ]
        else:
            return None
        return LineString(new_coords)

    @staticmethod
    def conditional_linemerge_collection(
        traces: Union[gpd.GeoDataFrame, gpd.GeoSeries],
        tolerance: float,
        buffer_value: float,
    ):
        spatial_index = traces.sindex
        new_traces = []
        modified_idx = []
        for i, trace in enumerate(traces.geometry):
            trace_candidates_idx: List[int] = list(
                spatial_index.intersection(trace.bounds)
            )
            trace_candidates_idx.remove(i)
            if len(trace_candidates_idx) == 0 or i in modified_idx:
                continue
            for idx in trace_candidates_idx:
                if idx in modified_idx or i in modified_idx:
                    continue
                trace_candidate = traces.geometry.iloc[idx]

                merged = LineMerge.conditional_linemerge(
                    trace,
                    trace_candidate,
                    tolerance=tolerance,
                    buffer_value=buffer_value,
                )
                if merged is not None:
                    new_traces.append(merged)
                    modified_idx.append(i)
                    modified_idx.append(idx)
                    break

        return new_traces, modified_idx

    @staticmethod
    def run_loop(traces: gpd.GeoDataFrame, tolerance: float, buffer_value: float):
        loop_count = count()
        while True:
            new_traces, modified_idx = LineMerge.conditional_linemerge_collection(
                traces, tolerance=tolerance, buffer_value=buffer_value
            )
            if len(modified_idx) == 0:
                return traces
            traces = LineMerge.integrate_replacements(traces, new_traces, modified_idx)
            if next(loop_count) > 100:
                print("loop 100")
                return traces

    @staticmethod
    def integrate_replacements(traces, new_traces, modified_idx):
        unmod_traces = [
            trace
            for idx, trace in enumerate(traces.geometry)
            if idx not in modified_idx
        ]
        all_traces = unmod_traces + new_traces
        return gpd.GeoDataFrame(geometry=all_traces).set_crs(3067)
