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

from fractopo.analysis import (
    tools,
    multiple_target_areas,
    config,
    analysis_and_plotting,
    target_area,
    main,
)

from tests import Helpers


class TestTools:
    line_float = floats(-1, 1)
    coord_strat_1 = lists(line_float, min_size=2, max_size=2)
    coord_strat_2 = lists(line_float, min_size=2, max_size=2)

    halved_strategy = floats(0, 180)
    length_strategy = floats(0)

    @given(coord_strat_1, coord_strat_2)
    @settings(max_examples=5)
    def test_calc_azimu(self, coords_1, coords_2):
        simple_line = LineString([coords_1, coords_2])
        tools.calc_azimu(simple_line)
        ass = tools.calc_azimu(LineString([(0, 0), (1, 1)]))
        assert np.isclose(ass, 45.0)

    def test_azimuth_plot_attributes(self):
        res = tools.azimuth_plot_attributes(
            pd.DataFrame(
                {"azimu": np.array([0, 45, 90]), "length": np.array([1, 2, 1])}
            )
        )
        assert np.allclose(
            res,
            np.array(
                [
                    33.33333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    33.33333333,
                    0.0,
                    0.0,
                    0.0,
                    33.33333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    33.33333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    33.33333333,
                    0.0,
                    0.0,
                    0.0,
                    33.33333333,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        )

    def test_tern_plot_the_fing_lines(self):
        _, tax = ternary.figure()
        tools.tern_plot_the_fing_lines(tax)

    def test_tern_plot_branch_lines(self):
        _, tax = ternary.figure()
        tools.tern_plot_branch_lines(tax)

    def test_aniso_get_class_as_value(self):
        classes = ("C - C", "C - I", "I - I")
        assert tools.aniso_get_class_as_value(classes[0]) == 1
        assert tools.aniso_get_class_as_value(classes[2]) == 0
        assert tools.aniso_get_class_as_value("asd") == 0

    @given(halved_strategy, length_strategy)
    @settings(max_examples=10)
    def test_aniso_calc_anisotropy_1(self, halved, length):
        c = "C - C"
        result = tools.aniso_calc_anisotropy(halved, c, length)
        for val in result:
            assert val >= 0

    def test_aniso_calc_anisotropy_2(self):
        result = tools.aniso_calc_anisotropy(90, "C - C", 10)
        assert np.allclose(
            result,
            np.array(
                [
                    6.12323400e-16,
                    5.00000000e00,
                    8.66025404e00,
                    1.00000000e01,
                    8.66025404e00,
                    5.00000000e00,
                ]
            ),
        )
        result_0 = tools.aniso_calc_anisotropy(90, "C - I", 10)
        assert np.allclose(result_0, np.array([0, 0, 0, 0, 0, 0]))

    def test_calc_y_distribution(self):
        df = pd.DataFrame({"length": [1, 2, 3, 4, 5, 6, 2.5]})
        area = 10
        result_df = tools.calc_y_distribution(df, area)
        result_df = result_df.reset_index()
        assert np.isclose(result_df.y.iloc[-1], 7 / 10)
        assert result_df.length.max() == result_df.length.iloc[0]

    def test_calc_cut_off_length(self):
        result = tools.calc_cut_off_length(
            pd.DataFrame({"length": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}), 0.55
        )
        assert np.isclose(result, 4.5)

    def test_define_set(self):
        result_1 = tools.define_set(
            50, pd.DataFrame({"Set": [1, 2], "SetLimits": [(10, 90), (100, 170)]})
        )
        assert result_1 == "1"
        result_2 = tools.define_set(
            99, pd.DataFrame({"Set": [1, 2], "SetLimits": [(10, 90), (100, 170)]})
        )
        assert result_2 == "-1"

    @given(coord_strat_1, coord_strat_2)
    @settings(max_examples=10)
    def test_line_end_point(self, coords1, coords2):
        line = LineString([coords1, coords2])
        tools.line_start_point(line)

    def test_get_nodes_intersecting_sets(self):
        points = gpd.GeoDataFrame(
            data={"geometry": [Point(0, 0), Point(1, 1), Point(3, 3)]}
        )
        lines = gpd.GeoDataFrame(
            data={
                "geometry": [
                    LineString([(0, 0), (2, 2)]),
                    LineString([(0, 0), (0.5, 0.5)]),
                ],
                "set": [1, 2],
            }
        )

        result = tools.get_nodes_intersecting_sets(points, lines)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.geometry.iloc[0] == Point(0.00000, 0.00000)

    def test_get_intersect_frame(self):
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1)], "c": ["Y", "X"]}
        )
        traces = gpd.GeoDataFrame(
            data={
                "geometry": [
                    LineString([(-1, -1), (2, 2)]),
                    LineString([(0, 0), (-0.5, 0.5)]),
                    LineString([(2, 0), (0, 2)]),
                ],
                "set": [1, 2, 2],
            }
        )
        traces["startpoint"] = traces.geometry.apply(tools.line_start_point)
        traces["endpoint"] = traces.geometry.apply(tools.line_end_point)
        sets = (1, 2)
        intersect_frame = tools.get_intersect_frame(nodes, traces, sets)
        assert intersect_frame.node.iloc[0] == Point(0, 0)
        assert intersect_frame.sets.iloc[0] == (2, 1)

    # def test_report_powerlaw_fit_statistics(self, tmp_path):
    #     filename = tmp_path / Path("test_logger.txt")
    #     # filename.touch(exist_ok=True)
    #     logging.basicConfig(filename=filename, filemode="w+", level=logging.DEBUG)
    #     name = "testing_reporting"
    #     fit = powerlaw.Fit(Helpers.trace_frame["length"])
    #     logger = logging.getLogger("test_logger")
    #     logger.setLevel(logging.DEBUG)
    #     filehandler = logging.FileHandler(filename)
    #     filehandler.setFormatter(logging.Formatter("%(message)s"))
    #     logger.addHandler(filehandler)

    #     using_branches = True
    #     tools.report_powerlaw_fit_statistics(name, fit, logger, using_branches)
    #     with filename.open(mode="r") as file:
    #         should_be_there = "Powerlaw, lognormal and exp"
    #         file_as_str = file.read()
    #         assert should_be_there in file_as_str

    def test_prepare_geometry_traces(self):
        traceframe = Helpers.trace_frame
        prep_col, trace_col = tools.prepare_geometry_traces(traceframe)
        assert isinstance(trace_col, shapely.geometry.MultiLineString)
        assert isinstance(prep_col, PreparedGeometry)

    def test_make_point_tree(self):
        traceframe = Helpers.trace_frame
        tree = tools.make_point_tree(traceframe)
        assert isinstance(tree, strtree.STRtree)
        # take a point from the input traceframe and see if query from tree returns it
        point_buffer = traceframe.startpoint.iloc[0].buffer(0.2)
        queried = [o.wkt for o in tree.query(point_buffer)]
        # this depends on the setup of 'traceframe' and if it changes it errors
        assert len(queried) >= 1

    # def test_read_and_verify_spatial_data(self):
    #     trace_result = tools.read_and_verify_spatial_data(Helpers.sample_trace_data)
    #     branch_result = tools.read_and_verify_spatial_data(Helpers.sample_branch_data)
    #     area_result = tools.read_and_verify_spatial_data(Helpers.sample_area_data)
    #     assert all(
    #         [
    #             isinstance(result, gpd.GeoDataFrame) and len(result) != 0
    #             for result in (trace_result, branch_result, area_result)
    #         ]
    #     )

    def test_report_powerlaw_fit_statistics_df(self):
        report_df = tools.initialize_report_df()
        fit = powerlaw.Fit(Helpers.trace_frame["length"])
        result_report_df = tools.report_powerlaw_fit_statistics_df(
            "test", fit, report_df, False, Helpers.trace_frame["length"]
        )
        assert len(result_report_df) > len(report_df)
        assert all([col in result_report_df.columns for col in tools.ReportDfCols.cols])
        print(result_report_df[tools.ReportDfCols.powerlaw_manual])
        assert "False" in result_report_df[tools.ReportDfCols.powerlaw_manual].values


class TestMultipleTargetAreas:

    # noinspection PyPep8Naming
    def test_MultiTargetAreaQGIS(self, tmp_path):
        # noinspection PyTypeChecker
        class Dummy(analysis_and_plotting.MultiTargetAreaAnalysis):

            # noinspection PyMissingConstructor
            def __init__(self):
                return

        dummy = Dummy()
        spatial_df = pd.DataFrame(
            columns=[
                "Name",
                "Group",
                "branchframe",
                "traceframe",
                "areaframe",
                "nodeframe",
            ]
        )
        line_1 = LineString([(0, 0), (0.5, 0.5)])
        line_2 = LineString([(0, 0), (0.5, -0.5)])
        line_3 = LineString([(0, 0), (1, 0)])
        branch_frame = gpd.GeoDataFrame(
            {
                "geometry": [line_1, line_2, line_3],
                "Connection": ["C - C", "C - I", "I - I"],
                # "Class": ["X - I", "Y - Y", "I - I"],
            }
        )

        trace_frame = gpd.GeoDataFrame({"geometry": [line_1, line_2]})
        point_1 = Point(0.5, 0.5)
        point_2 = Point(1, 1)
        point_3 = Point(10, 10)
        node_frame = gpd.GeoDataFrame(
            {"geometry": [point_1, point_2, point_3], "Class": ["X", "Y", "I"]}
        )
        area_1 = Polygon([(0, 0), (1, 1), (1, 0)])
        area_frame = gpd.GeoDataFrame({"geometry": [area_1]})
        name = "test1"
        group = "T1"

        set_df = pd.DataFrame({"Set": [1, 2], "SetLimits": [(40, 50), (130, 140)]})
        length_set_df = pd.DataFrame(
            {
                "LengthSet": ["l1", "l2", "l3"],
                "LengthSetLimits": [(0, 1000), (1100, 1200), (1300, 1400)],
            }
        )

        table_append = {
            "Name": name,
            "Group": group,
            "branchframe": branch_frame,
            "traceframe": trace_frame,
            "areaframe": area_frame,
            "nodeframe": node_frame,
        }

        spatial_df = spatial_df.append(table_append, ignore_index=True)

        group_names_cutoffs_df = pd.DataFrame(
            {"Group": [group], "CutOffTraces": [0], "CutOffBranches": [0]}
        )

        dummy.determine_branches = True
        dummy.determine_relationships = False
        dummy.determine_length_relationships = False
        dummy.determine_length_distributions = True
        dummy.determine_azimuths = True
        dummy.determine_xyi = True
        dummy.determine_branch_classification = True
        dummy.determine_topology = True
        dummy.determine_anisotropy = True
        dummy.determine_hexbin = True
        dummy.spatial_df = spatial_df
        dummy.group_names_cutoffs_df = group_names_cutoffs_df
        dummy.set_df = set_df
        dummy.length_set_df = length_set_df
        dummy.logger = logging.getLogger()

        dummy.analysis()

        config.g_list = ["T1"]
        config.ta_list = ["test1"]
        config.n_g = 1
        config.n_ta = 1

        plotting_directory = tools.plotting_directories(str(tmp_path), "TEST")
        dummy.plotting_directory = plotting_directory
        dummy.plot_results()
        assert Path(plotting_directory).exists()
        # If plotting_directory dir names are changed -> These will error
        # Or excel filename
        assert (Path(plotting_directory) / "length_distributions").exists()
        excel_filepath = (
            Path(plotting_directory)
            / "length_distributions"
            / "traces"
            / "report_df_unified_traces.xlsx"
        )

        assert excel_filepath.exists()
        df = pd.read_excel(excel_filepath)
        assert len(df) > 0
        assert all([col in df.columns for col in tools.ReportDfCols.cols])


class TestTargetArea:
    def test_calc_anisotropy(self):

        result = target_area.TargetAreaLines.calc_anisotropy(Helpers.branch_frame)

        assert isinstance(result, np.ndarray)

    def test_plot_xyi_point(self):
        fig, tax = ternary.figure()
        target_area.TargetAreaNodes.plot_xyi_point(
            Helpers.node_frame, name="testing", tax=tax, color_for_plot="red"
        )

    def test_plot_xyi_plot(self):
        nodeframe = Helpers.node_frame
        name = "testing_plotting"
        unified = False
        target_area.TargetAreaNodes.plot_xyi_plot(nodeframe, name, unified)

    def test_plot_length_distribution_fit(self):
        lineframe = Helpers.trace_frame
        power_law_fit = powerlaw.Fit(lineframe["length"])
        for fit_distribution in [config.POWERLAW, config.LOGNORMAL, config.EXPONENTIAL]:
            fig, ax = plt.subplots()
            result = target_area.TargetAreaLines.plot_length_distribution_fit(
                power_law_fit, fit_distribution, ax
            )
            assert result is None


class TestMain:
    def test_initialize_analysis_logging(self, tmp_path):
        logger = main.initialize_analysis_logging(str(tmp_path))
        assert isinstance(logger, logging.Logger)
        logged_str = "HELLO-THIS-SHOULD-BE-IN-LOGGER-FILE"
        logger.info(logged_str)
        with open(tmp_path / "analysis_statistics.txt", mode="r") as logfile:
            log_contents = logfile.read()
            assert logged_str in log_contents
