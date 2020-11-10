from hypothesis.strategies import floats, lists
from hypothesis import given, settings
import shapely
from shapely.geometry import Point, LineString, Polygon
from shapely.prepared import PreparedGeometry
from shapely import strtree
import numpy as np
import pandas as pd
import geopandas as gpd
import ternary
import powerlaw
import matplotlib.pyplot as plt
import logging
from pathlib import Path

import fractopo.branches_and_nodes as branches_and_nodes
from fractopo.analysis import (
    tools,
    multiple_target_areas,
    config,
    analysis_and_plotting,
    target_area,
    main,
)


def test_main_regression_kb11(tmp_path: Path, only_lds=False, full_data=False):
    """
    Tests that analysis results match between changes in code.
    I.e. no regression occurs.
    """
    trace_data = ["tests/sample_data/KB11_traces.shp"]
    area_data = ["tests/sample_data/KB11_area.shp"]
    traces = [
        gpd.read_file(data_file).set_crs("EPSG:3067", allow_override=True)
        for data_file in trace_data
    ]
    areas = [
        gpd.read_file(data_file).set_crs("EPSG:3067", allow_override=True)
        for data_file in area_data
    ]
    # Half the data for faster analysis (when pytesting)
    if not full_data:
        traces = [trace.iloc[int(len(trace) * 0.90) :] for trace in traces]
    # Each trace dataset must have a target area
    assert len(traces) == len(areas)
    # Assign snap_threshold
    snap_threshold = 0.001

    # Generate branches and nodes for each trace dataset
    branches, nodes = zip(
        *[
            branches_and_nodes.branches_and_nodes(trace, area, snap_threshold)
            for trace, area in zip(traces, areas)
        ]
    )
    analysis_name = "pytest_sample_kb11"
    results_folder = str(tmp_path)

    # All datasets must have names
    names = ["KB11"]

    assert len(names) == len(traces) == len(areas) == len(branches) == len(nodes)

    # Each group must have defined cut offs for traces and branches
    groups = ["KB"]

    cut_offs_traces = [1.8]
    cut_offs_branches = [1.6]

    # Assign group to each dataset
    datasets_grouped = ["KB"]

    assert len(groups) == len(cut_offs_traces) == len(cut_offs_branches)
    assert len(names) == len(datasets_grouped)
    set_names = ["1", "2", "3"]
    set_limits = [
        (0.0, 30.0),
        (30.0, 90.0),
        (90.0, 180.0),
    ]
    length_set_names = ["len1", "len2", "len3"]
    length_set_limits = [
        (0, 10),
        (10, 20),
        (20, 50),
    ]

    assert len(set_names) == len(set_limits)
    # Choose which analyses to perform.
    choose_your_analyses = {
        "Branches": True,
        "LengthDistributions": True,
        "Azimuths": True,
        "XYI": True,
        "BranchClassification": True,
        "Topology": True,
        "Cross-cuttingAbutting": True,
        "Anisotropy": True,
        "Hexbin": True,
    }
    if only_lds:
        choose_your_analyses = {
            "Branches": True,
            "LengthDistributions": True,
            "Azimuths": False,
            "XYI": False,
            "BranchClassification": False,
            "Topology": False,
            "Cross-cuttingAbutting": False,
            "Anisotropy": False,
            "Hexbin": False,
        }

    # Analyze trace, branch and node data
    main.analyze_datasets(
        traces,
        areas,
        branches,
        nodes,
        names,
        groups,
        cut_offs_traces,
        cut_offs_branches,
        datasets_grouped,
        set_names,
        set_limits,
        length_set_names,
        length_set_limits,
        analysis_name,
        results_folder,
        choose_your_analyses,
    )
