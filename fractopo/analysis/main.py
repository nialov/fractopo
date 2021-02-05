"""
Main module for the control of analysis and plotting.
"""
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import pandas as pd

from fractopo.analysis import analysis_and_plotting as taaq, config, tools


def main_multi_target_area(
    analysis_df,
    results_folder,
    analysis_name,
    group_names_cutoffs_df,
    set_df,
    choose_your_analyses,
):
    """
    Main method for firstly initializing data analysis, then starting data
    analysis, and then starting plotting of analysis results.  Also initializes
    plotting directories, debugging methods and the plotting config parameters.

    :param analysis_df: DataFrame with trace, branch, node, etc. vector layer data.
    :type analysis_df: pandas.DataFrame
    :param results_folder: Path to folder in which plots_{analysis_name} folder
    will be built to and where plots will be saved to.
    :type results_folder: str
    :param analysis_name: Name for the analysis. Will be used in the
    plots_{analysis_name} folder name.
    :type analysis_name: str
    :param group_names_cutoffs_df: DataFrame with group name and cut-off data
    for both traces and branches.
    :type group_names_cutoffs_df: pandas.DataFrame
    :param set_df: DataFrame with set names and ranges.
    :type set_df: pandas.DataFrame
    :return: Returns analysis object.
    :rtype: taaq.MultiTargetAreaAnalysis
    """

    # SETUP PLOTTING DIRECTORIES
    plotting_directory = tools.plotting_directories(results_folder, analysis_name)

    # SETUP LOGGER
    logger = initialize_analysis_logging(plotting_directory)

    # SETUP CONFIG PARAMETERS
    config.n_ta = len(analysis_df)
    config.n_g = len(group_names_cutoffs_df)
    config.ta_list = analysis_df.Name.tolist()
    config.g_list = group_names_cutoffs_df.Group.tolist()

    # SAVE ANALYSIS SETTINGS AND INPUTS INTO LOGGER
    logger.info("Analysis Settings and Inputs")
    logger.info("-----------------------------------")
    logger.info(f"Layer table DataFrame:\n {analysis_df.to_string()}")
    logger.info(f"Results folder:\n {results_folder}")
    logger.info(f"Analysis name:\n {analysis_name}")
    logger.info(
        f"Group names and Cut-offs DataFrame:\n {group_names_cutoffs_df.to_string()}"
    )
    logger.info(f"Set DataFrame:\n {set_df.to_string()}")
    logger.info(f"Chosen analyses from config.py file:\n {choose_your_analyses}")

    # START __init__

    mta_analysis = taaq.MultiTargetAreaAnalysis(
        analysis_df,
        plotting_directory,
        analysis_name,
        group_names_cutoffs_df,
        set_df,
        choose_your_analyses,
        logger,
    )

    # Start analysis
    mta_analysis.analysis()
    # Start plotting
    mta_analysis.plot_results()


def initialize_analysis_logging(plotting_directory):
    """
    Initializes a logger that collects all analysis statistics some of which
    are not drawn onto the finished plots.

    :param plotting_directory: The logger file is saved into the plotting_directory
    :type plotting_directory: str
    :return:
    :rtype:
    """
    filename = Path(plotting_directory) / Path("analysis_statistics.txt")
    if filename.exists():
        # Should not happen. But remove if exists. TODO: Add qgis message
        filename.unlink()
    # TODO: Is basicConfig needed?
    logging.basicConfig(
        filename=filename,
        filemode="w+",
        format="> %(message)s",
        datefmt="%H:%M:%S'",
        level=logging.WARNING,
    )

    logger = logging.getLogger("Analysis_Statistics")
    logger.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(filename)
    filehandler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(filehandler)

    logger.info("Analysis Statistics logfile initialized.")
    return logger


def analyze_datasets(
    traces: List[gpd.GeoDataFrame],
    areas: List[gpd.GeoDataFrame],
    branches: List[gpd.GeoDataFrame],
    nodes: List[gpd.GeoDataFrame],
    names: List[str],
    groups: List[str],
    cut_offs_traces: List[float],
    cut_offs_branches: List[float],
    datasets_grouped: List[str],
    set_names: List[str],
    set_limits: List[Tuple[float, float]],
    length_set_names: List[str],
    length_set_limits: List[Tuple[float, float]],
    analysis_name: str,
    results_folder: str,
    choose_your_analyses: Dict[str, bool],
):

    # SETUP PLOTTING DIRECTORIES
    plotting_directory = tools.plotting_directories(results_folder, analysis_name)

    # SETUP LOGGER
    logger = initialize_analysis_logging(plotting_directory)

    # SETUP CONFIG PARAMETERS
    config.n_ta = len(traces)
    config.n_g = len(groups)
    config.ta_list = names
    config.g_list = groups

    analysis_df = pd.DataFrame(
        {
            "Name": names,
            "Group": datasets_grouped,
            "traceframe": traces,
            "areaframe": areas,
            "branchframe": branches,
            "nodeframe": nodes,
        }
    )

    group_names_cutoffs_df = pd.DataFrame(
        {
            "Group": groups,
            "CutOffTraces": cut_offs_traces,
            "CutOffBranches": cut_offs_branches,
        }
    )
    # If user gives no input for length_set_names or length_set_limits ->
    # create dummy values
    set_df = pd.DataFrame(
        {
            "Set": set_names,
            "SetLimits": set_limits,
        }
    )
    length_set_df = pd.DataFrame(
        {
            "LengthSet": length_set_names,
            "LengthSetLimits": length_set_limits,
        }
    )

    # SAVE ANALYSIS SETTINGS AND INPUTS INTO LOGGER
    logger.info("Analysis Settings and Inputs")
    logger.info("-----------------------------------")
    logger.info(f"Layer table DataFrame:\n {analysis_df.to_string()}")
    logger.info(f"Results folder:\n {results_folder}")
    logger.info(f"Analysis name:\n {analysis_name}")
    logger.info(
        f"Group names and Cut-offs DataFrame:\n {group_names_cutoffs_df.to_string()}"
    )
    logger.info(f"Set DataFrame:\n {set_df.to_string()}")
    logger.info(f"Chosen analyses from config.py file:\n {choose_your_analyses}")

    # START __init__

    mta_analysis = taaq.MultiTargetAreaAnalysis(
        analysis_df,
        plotting_directory,
        analysis_name,
        group_names_cutoffs_df,
        set_df,
        length_set_df,
        choose_your_analyses,
        logger,
    )

    # Start analysis
    mta_analysis.analysis()
    # Start plotting
    mta_analysis.plot_results()
