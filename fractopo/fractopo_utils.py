"""
Miscellaneous utilities and scripts of fractopo.
"""
from itertools import count
from typing import List, Tuple, Union

import geopandas as gpd
from shapely.geometry import LineString, Point

from fractopo.general import (
    compare_unit_vector_orientation,
    create_unit_vector,
    geom_bounds,
    get_trace_endpoints,
    pygeos_spatial_index,
    safe_buffer,
    spatial_index_intersection,
)


class LineMerge:

    """
    Merge lines conditionally.
    """

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

        E.g. with merging:

        >>> first = LineString([(0, 0), (0, 2)])
        >>> second = LineString([(0, 2.001), (0, 4)])
        >>> tolerance = 5
        >>> buffer_value = 0.01
        >>> LineMerge.conditional_linemerge(first, second, tolerance, buffer_value).wkt
        'LINESTRING (0 0, 0 2, 0 4)'

        Without merging:

        >>> first = LineString([(0, 0), (0, 2)])
        >>> second = LineString([(0, 2.1), (0, 4)])
        >>> tolerance = 5
        >>> buffer_value = 0.01
        >>> LineMerge.conditional_linemerge(
        ...     first, second, tolerance, buffer_value
        ... ) is None
        True

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
        first_coords_list = list(first.coords)
        second_coords_list = list(second.coords)
        if (
            first_end.buffer(buffer_value).intersects(second_start.buffer(buffer_value))
            and are_close
        ):
            # First ends in the start of second -> Sequence of coords is correct
            # Do not include first coordinate of second in new
            new_coords = first_coords_list + second_coords_list[1:]
        elif (
            first_end.buffer(buffer_value).intersects(second_end.buffer(buffer_value))
            and are_close_reverse
        ):
            # First ends in second end
            new_coords = first_coords_list + list(reversed(second_coords_list))[1:]
        elif (
            first_start.buffer(buffer_value).intersects(
                second_start.buffer(buffer_value)
            )
            and are_close_reverse
        ):
            # First starts from the same as second
            new_coords = list(reversed(second_coords_list))[:-1] + first_coords_list
        elif (
            first_start.buffer(buffer_value).intersects(second_end.buffer(buffer_value))
            and are_close
        ):
            # First starts from end of second
            new_coords = second_coords_list[:-1] + first_coords_list
        else:
            return None
        return LineString(new_coords)

    @staticmethod
    def conditional_linemerge_collection(
        traces: Union[gpd.GeoDataFrame, gpd.GeoSeries],
        tolerance: float,
        buffer_value: float,
    ) -> Tuple[List[LineString], List[int]]:
        """
        Conditionally linemerge within a collection of LineStrings.

        Returns the linemerged traces and the idxs of traces that were
        linemerged.

        E.g.


        >>> first = LineString([(0, 0), (0, 2)])
        >>> second = LineString([(0, 2.001), (0, 4)])
        >>> traces = gpd.GeoSeries([first, second])
        >>> tolerance = 5
        >>> buffer_value = 0.01
        >>> new_traces, idx = LineMerge.conditional_linemerge_collection(
        ...     traces, tolerance, buffer_value
        ... )
        >>> [trace.wkt for trace in new_traces], idx
        (['LINESTRING (0 0, 0 2, 0 4)'], [0, 1])

        """
        spatial_index = pygeos_spatial_index(traces)

        new_traces = []
        modified_idx = []
        for i, trace in enumerate(traces.geometry):
            assert isinstance(trace, LineString)
            trace_candidates_idx: List[int] = list(
                spatial_index.intersection(
                    geom_bounds(safe_buffer(trace, buffer_value * 2))
                )
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
    def run_loop(
        traces: gpd.GeoDataFrame, tolerance: float, buffer_value: float
    ) -> gpd.GeoDataFrame:
        """
        Run multiple conditional linemerge iterations for GeoDataFrame.

        This is the main entrypoint.

        GeoDataFrame should contain LineStrings.

        E.g.

        >>> first = LineString([(0, 0), (0, 2)])
        >>> second = LineString([(0, 2.001), (0, 4)])
        >>> traces = gpd.GeoDataFrame(geometry=[first, second])
        >>> tolerance = 5
        >>> buffer_value = 0.01
        >>> LineMerge.run_loop(traces, tolerance, buffer_value)
                                                    geometry
        0  LINESTRING (0.00000 0.00000, 0.00000 2.00000, ...

        """
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
    def integrate_replacements(
        traces: gpd.GeoDataFrame, new_traces: List[LineString], modified_idx: List[int]
    ) -> gpd.GeoDataFrame:
        """
        Add linemerged and remove the parts that were linemerged.

        E.g.


        >>> first = LineString([(0, 0), (0, 2)])
        >>> second = LineString([(0, 2.001), (0, 4)])
        >>> traces = gpd.GeoDataFrame(geometry=[first, second])
        >>> new_traces = [LineString([(0, 0), (0, 2), (0, 4)])]
        >>> modified_idx = [0, 1]
        >>> LineMerge.integrate_replacements(traces, new_traces, modified_idx)
                                                    geometry
        0  LINESTRING (0.00000 0.00000, 0.00000 2.00000, ...

        """
        unmod_traces = [
            trace
            for idx, trace in enumerate(traces.geometry)
            if idx not in modified_idx
        ]
        all_traces = unmod_traces + new_traces
        gdf = gpd.GeoDataFrame(geometry=all_traces)
        if traces.crs is not None:
            gdf = gdf.set_crs(traces.crs)
        return gdf


def remove_identical_sindex(
    geosrs: gpd.GeoSeries, snap_threshold: float
) -> gpd.GeoSeries:
    """
    Remove stacked nodes by using a search buffer the size of snap_threshold.
    """
    geosrs_reset = geosrs.reset_index(inplace=False, drop=True)
    assert isinstance(geosrs_reset, gpd.GeoSeries)
    geosrs = geosrs_reset
    spatial_index = geosrs.sindex
    identical_idxs = []
    point: Point
    for idx, point in enumerate(geosrs.geometry.values):
        if idx in identical_idxs:
            continue
        # point = point.buffer(snap_threshold) if snap_threshold != 0 else point
        p_candidate_idxs = (
            # list(spatial_index.intersection(point.buffer(snap_threshold).bounds))
            spatial_index_intersection(
                spatial_index=spatial_index,
                coordinates=geom_bounds(safe_buffer(geom=point, radius=snap_threshold)),
            )
            if snap_threshold != 0
            else list(spatial_index.intersection(point.coords[0]))
        )
        p_candidate_idxs.remove(idx)
        p_candidates = geosrs.iloc[p_candidate_idxs]
        inter = p_candidates.distance(point) < snap_threshold
        colliding = inter.loc[inter]
        if len(colliding) > 0:
            index_to_list = colliding.index.to_list()
            assert len(index_to_list) > 0
            assert all(isinstance(i, int) for i in index_to_list)
            identical_idxs.extend(index_to_list)
    geosrs_dropped = geosrs.drop(identical_idxs)
    assert isinstance(geosrs_dropped, gpd.GeoSeries)
    return geosrs_dropped
