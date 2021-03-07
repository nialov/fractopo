"""
Do profiling of Network analysis run time with pyinstrument.
"""
from pathlib import Path

import geopandas as gpd

from fractopo.analysis.network import Network


def main():
    """
    Run a sample Network analysis.
    """
    sample_trace_data = Path("tests/sample_data/geta1/Getaberget_20m_1_traces.gpkg")
    sample_area_data = Path("tests/sample_data/geta1/Getaberget_20m_1_1_area.gpkg")
    traces = gpd.read_file(sample_trace_data)
    area = gpd.read_file(sample_area_data)
    assert isinstance(traces, gpd.GeoDataFrame)
    assert isinstance(area, gpd.GeoDataFrame)
    Network(
        trace_gdf=traces,
        area_gdf=area,
        truncate_traces=True,
        determine_branches_nodes=True,
        snap_threshold=0.001,
        circular_target_area=False,
    )


if __name__ == "__main__":
    main()
