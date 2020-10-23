# from hypothesis.strategies import floats, lists
# from hypothesis import given, settings
# import shapely
# from shapely.geometry import Point, LineString, Polygon
# from shapely.prepared import PreparedGeometry
# from shapely import strtree
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# import ternary
# import powerlaw
# import matplotlib.pyplot as plt
# import logging
# from pathlib import Path

# from fracture_analysis_kit import (
#     tools,
#     multiple_target_areas,
#     config,
#     analysis_and_plotting,
#     target_area,
#     main,
# )


# def test_main_regression_kb11(file_regression, tmp_path: Path):
#     """
#     Tests that analysis results match between changes in code.
#     I.e. no regression occurs.
#     """
#     traces_path = Path("tests/sample_data/KB11_traces.shp)")
#     branches_path = Path("tests/sample_data/KB11_branches.shp)")
#     nodes_path = Path("tests/sample_data/KB11_nodes.shp)")
#     areas_path = Path("tests/sample_data/KB11_area.shp)")
#     assert all(
#         [p.exists() for p in (traces_path, branches_path, nodes_path, areas_path)]
#     )
#     traces = gpd.read_file(traces_path)
#     branches = gpd.read_file(branches_path)
#     nodes = gpd.read_file(nodes_path)
#     areas = gpd.read_file(areas_path)
#     names = ["KB11"]
#     groups = ["group1"]
#     cut_off_traces = [1.8]
#     cut_off_branches = [val / 1.1 for val in cut_off_traces]
#     analysis_df = pd.DataFrame(
#         {
#             "traceframe": traces,
#             "branchframe": branches,
#             "nodeframe": nodes,
#             "areaframe": areas,
#             "Name": names,
#             "Group": groups,
#         }
#     )
#     results_folder = str(tmp_path.resolve())
#     analysis_name = "regression_testing"
#     group_names_cutoffs_df = pd.DataFrame(
#         {
#             "Group": groups,
#             "CutOffTraces": cut_off_traces,
#             "CutOffBranches": cut_off_branches,
#         }
#     )
#     set_df = pd.DataFrame(
#         {
#             "Set": ["1", "2", "3", "4", "5"],
#             "SetLimits": [(50, 100), (130, 160), (25, 50), (100, 130), (160, 25)],
#         }
#     )
#     choose_your_analyses = config.choose_your_analyses
#     main.main_multi_target_area(
#         analysis_df,
#         results_folder,
#         analysis_name,
#         group_names_cutoffs_df,
#         set_df,
#         choose_your_analyses,
#     )
