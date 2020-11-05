"""
Contains calculation and plotting tools.
"""

import math
from textwrap import wrap
import os
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import shapely
import ternary
from shapely import strtree
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import linemerge
from shapely import prepared
import powerlaw
import logging
from typing import Tuple, Union


# Own code imports
from fractopo.analysis import target_area as ta, config

style = config.styled_text_dict
prop = config.styled_prop

# Values used in spatial estimates of superposition
buffer_value = config.buffer_value
snap_value = config.snap_value

# Columns for report_df


class ReportDfCols:
    name = "name"
    powerlaw_exponent = "powerlaw_exponent"
    powerlaw_sigma = "powerlaw_sigma"
    powerlaw_cutoff = "powerlaw_cutoff"
    powerlaw_manual = "powerlaw_manual"
    lognormal_mu = "lognormal_mu"
    lognormal_sigma = "lognormal_sigma"
    exponential_lambda = "exponential_lambda"
    powerlaw_vs_lognormal_r = "powerlaw_vs_lognormal_r"
    powerlaw_vs_lognormal_p = "powerlaw_vs_lognormal_p"
    powerlaw_vs_exponential_r = "powerlaw_vs_exponential_r"
    powerlaw_vs_exponential_p = "powerlaw_vs_exponential_p"
    lognormal_vs_exponential_r = "lognormal_vs_exponential_r"
    lognormal_vs_exponential_p = "lognormal_vs_exponential_p"
    remaining_percent_of_data = "remaining_percent_of_data"

    cols = [
        name,
        powerlaw_exponent,
        powerlaw_sigma,
        powerlaw_cutoff,
        lognormal_mu,
        lognormal_sigma,
        exponential_lambda,
        powerlaw_vs_lognormal_r,
        powerlaw_vs_lognormal_p,
        powerlaw_vs_exponential_r,
        powerlaw_vs_exponential_p,
        lognormal_vs_exponential_r,
        lognormal_vs_exponential_p,
        remaining_percent_of_data,
    ]


# def initialize_branch_frame(filename):
#     branchframe = gpd.read_file(filename)
#     branchframe['azimu'] = branchframe.geometry.apply(calc_azimu)
#     branchframe['c'] = branchframe.Class.apply(str)
#     branchframe['connection'] = branchframe.Connection.apply(str)
#     branchframe['halved'] = branchframe.azimu.apply(azimu_half)
#     # branchframe = branchframe.dropna()
#     return branchframe


# def initialize_trace_frame(filename):
#     branchframe = gpd.read_file(filename)
#     branchframe['azimu'] = branchframe.geometry.apply(calc_azimu)
#     branchframe['halved'] = branchframe.azimu.apply(azimu_half)
#     # branchframe = branchframe.dropna()
#     return branchframe


# def initialize_node_frame(filename):
#     nodeframe = gpd.read_file(filename)
#     nodeframe['c'] = nodeframe.Class.apply(str)
#     return nodeframe


# def initialize_normalisation(filelist):
#     norm_list = []
#     for i in range(len(filelist)):
#         name = filelist[i]
#         normalisation_str = input("Enter normalisation for " + name + ":")
#         normalisation = float(normalisation_str)
#         norm_list.append(normalisation)
#     return norm_list


# def initialize_area_frame(filename):
#     areaframe = gpd.read_file(filename)
#     return areaframe


def calc_strike(dip_direction):
    """
    Calculates strike from dip direction. Right-handed rule.
    Examples:

    >>> calc_strike(50)
    320

    >>> calc_strike(180)
    90

    :param dip_direction: Dip direction of plane
    :type dip_direction: float
    :return: Strike of plane
    :rtype: float
    """
    strike = dip_direction - 90
    if strike < 0:
        strike = 360 + strike
    return strike


# def initialize_plane_file(filename, declination_fix=9):
#     planeframe = gpd.read_file(filename)
#     planeframe['DIRECTION_OF_DIP_DEC_FIX'] = planeframe.DIRECTION_ - declination_fix
#     planeframe['STRIKE'] = planeframe.DIRECTION_OF_DIP_DEC_FIX.apply(calc_strike)
#     return planeframe


# def get_filenames_planefiles(directory):
#     faultfiles = list(glob.glob(os.path.join(directory, '*_faults_*.shp')))
#     dykefiles = list(glob.glob(os.path.join(directory, '*_layerings_*.shp')))
#     lineationfiles = list(glob.glob(os.path.join(directory, '*_lineations_*.shp')))
#     if len(faultfiles + dykefiles + lineationfiles) == 0:
#         raise Exception('No fault, dyke or lineation files found. Check spelling and directory.')
#     return faultfiles, dykefiles, lineationfiles


# def get_filenames_branches_nodes(directory):
#     branchfiles = list(glob.glob(os.path.join(directory, '*branches.shp')))
#     branchfiles = sorted(branchfiles)
#     nodefiles = list(glob.glob(os.path.join(directory, '*nodes.shp')))
#     nodefiles = sorted(nodefiles)
#     if len(branchfiles) + len(nodefiles) == 0:
#         raise Exception('No branch or nodefiles found. Check spelling and directory.')
#     if len(branchfiles) != len(nodefiles):
#         raise Exception('Unequal amount of nodefiles and branchfiles')
#     return branchfiles, nodefiles


# def get_filenames_branches_or_traces(directory):
#     branchfiles = list(glob.glob(os.path.join(directory, '*branches.shp')))
#     tracefiles = list(glob.glob(os.path.join(directory, '*tulkinta.shp')))
#     files = branchfiles + tracefiles
#     if len(files) == 0:
#         raise Exception('No branch or tracefiles found. Check spelling and directory.')
#     return files


def calc_azimu(line):
    """
    Calculates azimuth of given line.

    e.g.:
    Accepts LineString

    >>> calc_azimu(shapely.geometry.LineString([(0, 0), (1, 1)]))
    45.0

    Accepts mergeable MultiLineString

    >>> calc_azimu(shapely.geometry.MultiLineString([((0, 0), (1, 1)), ((1, 1), (2, 2))]))
    45.0

    Returns np.nan when the line cannot be merged into one continuous line.

    >>> calc_azimu(shapely.geometry.MultiLineString([((0, 0), (1, 1)), ((1.5, 1), (2, 2))]))
    nan

    :param line: Continous line feature (trace, branch, etc.)
    :type line: shapely.geometry.LineString | shapely.geometry.MultiLineString
    :return: Azimuth of line.
    :rtype: float | np.nan
    """
    try:
        coord_list = list(line.coords)
    except NotImplementedError:
        # TODO: Needs more testing?
        line = linemerge(line)
        try:
            coord_list = list(line.coords)
        except NotImplementedError:
            return np.NaN
    start_x = coord_list[0][0]
    start_y = coord_list[0][1]
    end_x = coord_list[-1][0]
    end_y = coord_list[-1][1]
    azimu = 90 - math.degrees(math.atan2((end_y - start_y), (end_x - start_x)))
    if azimu < 0:
        azimu = azimu + 360
    return azimu


def azimu_half(degrees):
    """
    Transforms azimuths from 180-360 range to range 0-180

    :param degrees: Degrees in range 0 - 360
    :type degrees: float
    :return: Degrees in range 0 - 180
    :rtype: float
    """
    if degrees >= 180:
        degrees = degrees - 180
    return degrees


# def get_filenames_branches_traces_and_areas(branchdirectory,
#                                             areadirectory):  # Gets both branches and traces + their areas
#     branchfiles = list(glob.glob(os.path.join(branchdirectory, '*branches.shp')))
#     tracefiles = list(glob.glob(os.path.join(branchdirectory, '*tulkinta.shp')))
#     branchfiles = branchfiles + tracefiles
#     branchfiles = sorted(branchfiles)
#
#     areafiles = list(glob.glob(os.path.join(areadirectory, '*tulkinta_alue.shp')))
#     areafiles = sorted(areafiles)
#     if len(branchfiles) + len(areafiles) == 0:
#         raise Exception('No branch or areafiles found. Check spelling and directories.')
#     if len(branchfiles) != len(areafiles):
#         raise Exception('Unequal amount of areafiles and branchfiles')
#     return branchfiles, areafiles


# def get_filenames_branches_traces_and_areas_coded_old(linedirectory, areadirectory, code):
#     # Gets both branches and traces + their areas
#     branchfiles = list(glob.glob(
#         os.path.join(linedirectory, '*' + code + '*branches.shp')))  # Only gets based on code e.g. 'KL' or 'KB'
#     tracefiles = list(glob.glob(os.path.join(linedirectory, '*' + code + '*tulkinta.shp')))
#     tracefiles_2 = list(glob.glob(os.path.join(linedirectory, '*' + code + '*traces.shp')))
#     linefiles = branchfiles + tracefiles + tracefiles_2
#     linefiles = sorted(linefiles)
#
#     areafiles = list(glob.glob(os.path.join(areadirectory, '*' + code + '*tulkinta_alue.shp')))
#     areafiles_2 = list(glob.glob(os.path.join(areadirectory, '*' + code + '*area_*.shp')))
#     areafiles = areafiles + areafiles_2
#     areafiles = sorted(areafiles)
#     # if len(branchfiles)+len(areafiles) == 0:
#     #   raise Exception('No branch or areafiles found. Check spelling and directories.')
#     if len(linefiles) != len(areafiles):
#         print(linefiles, areafiles)
#         raise Exception('Unequal amount of areafiles and linefiles found')
#     return linefiles, areafiles


# def get_filenames_nodes_coded_old(nodedirectory, code):  # Gets nodefiles using code
#     nodefiles = list(glob.glob(os.path.join(nodedirectory, code + '*nodes.shp')))
#     nodefiles = sorted(nodefiles)
#
#     if len(nodefiles) == 0:
#         raise Exception('No nodefiles found. Check spelling and directories.')
#
#     return nodefiles


# def get_filenames_branches_traces_and_areas_coded(linedirectory, areadirectory, code):
#     # Gets both branches and traces + their areas
#     branchfiles = glob.glob(linedirectory + '/{}_*branches.shp'.format(code)) \
#                   + glob.glob(linedirectory + '/{}[0-9]_*branches.shp'.format(code)) \
#                   + glob.glob(linedirectory + '/{}[0-9][0-9]_*branches.shp'.format(code))
#
#     tracefiles = glob.glob(linedirectory + '/{}_*tulkinta.shp'.format(code)) \
#                  + glob.glob(linedirectory + '/{}[0-9]_*tulkinta.shp'.format(code)) \
#                  + glob.glob(linedirectory + '/{}[0-9][0-9]_*tulkinta.shp'.format(code))
#
#     tracefiles_alt = glob.glob(linedirectory + '/{}_*traces.shp'.format(code)) \
#                      + glob.glob(linedirectory + '/{}[0-9]_*traces.shp'.format(code)) \
#                      + glob.glob(linedirectory + '/{}[0-9][0-9]_*traces.shp'.format(code))
#
#     linefiles = branchfiles + tracefiles + tracefiles_alt
#     if len(linefiles) != len(branchfiles) \
#             and len(linefiles) != len(tracefiles) \
#             and len(linefiles) != len(tracefiles_alt):
#         print('-----------error-----------')
#         print('linefiles:', linefiles)
#         print('--------------------------')
#         print('branchfiles:', branchfiles)
#         print('--------------------------')
#         print('tracefiles_alt:', tracefiles_alt)
#         raise Exception(
#             'Amount of linefiles doesnt equal exactly with the amount of one of either trace- or branchfiles')
#     linefiles = sorted(linefiles)
#
#     areafiles = glob.glob(areadirectory + '/{}_*tulkinta_alue.shp'.format(code)) \
#                 + glob.glob(areadirectory + '/{}[0-9]_*tulkinta_alue.shp'.format(code)) \
#                 + glob.glob(areadirectory + '/{}[0-9][0-9]_*tulkinta_alue.shp'.format(code))
#
#     areafiles_alt = glob.glob(areadirectory + '/{}_*area_*.shp'.format(code)) \
#                     + glob.glob(areadirectory + '/{}[0-9]_*area_*.shp'.format(code)) \
#                     + glob.glob(areadirectory + '/{}[0-9][0-9]_*area_*.shp'.format(code))
#
#     areafiles = areafiles + areafiles_alt
#     areafiles = sorted(areafiles)
#     # if len(branchfiles)+len(areafiles) == 0:
#     #   raise Exception('No branch or areafiles found. Check spelling and directories.')
#     if len(linefiles) != len(areafiles):
#         print(linefiles, areafiles)
#         raise Exception('Unequal amount of areafiles and linefiles found')
#     return linefiles, areafiles


# def get_filenames_nodes_coded(nodedirectory, code):  # Gets nodefiles using code
#     nodefiles = glob.glob(nodedirectory + '/{}_*nodes.shp'.format(code)) \
#                 + glob.glob(nodedirectory + '/{}[0-9]_*nodes.shp'.format(code)) \
#                 + glob.glob(nodedirectory + '/{}[0-9][0-9]_*nodes.shp'.format(code))
#
#     nodefiles = sorted(nodefiles)
#
#     if len(nodefiles) == 0:
#         raise Exception('No nodefiles found. Check spelling and directories.')
#
#     return nodefiles


# def get_filenames_nodes_coded_excp(nodedirectory, code):  # Gets nodefiles using code
#     nodefiles = list(glob.glob(os.path.join(nodedirectory, code + '*nodes.shp')))
#     nodefiles = sorted(nodefiles)
#     return nodefiles


# def get_area_normalisations(areafiles):
#     area_list = []
#     for file in areafiles:
#         areaframe = gpd.read_file(file)
#         area = areaframe['Shape_Area'].iloc[0]
#         area_list.append(area)
#     area_array = np.array(area_list)
#     max_area = area_array.max()
#     norm_array = area_array / max_area
#     norm_list = norm_array.tolist()
#     return norm_list


# def get_area_normalisations_frames(areaframes):  # returns list with normalisations
#     area_list = []  # order important!
#     for frame in areaframes:

#         if isinstance(frame, gpd.GeoDataFrame):
#             area = sum([polygon.area for polygon in frame.geometry])
#         else:
#             try:
#                 area = frame['Shape_Area'].sum()
#             except KeyError:
#                 area = frame['SHAPE_Area'].sum()
#         area_list.append(area)
#     area_array = np.array(area_list)
#     max_area = area_array.max()
#     norm_array = area_array / max_area
#     norm_list = norm_array.tolist()
#     return norm_list


# def unify_frames(frames):  # Appends multiple dataframes together
#     unifiedframe = frames[0]
#     for frame in frames:
#         unifiedframe = unifiedframe.append(frame)
#     return unifiedframe


# def unify(linedirectory, areadirectory, code):
#     linefiles, areafiles = get_filenames_branches_traces_and_areas_coded(linedirectory, areadirectory, code)
#     if len(linefiles) == 0:  # If no such files with certain code
#         lineframe = []  # Returns an empty list
#         areaframe = []
#         return lineframe, areaframe
#     lineframes = []
#     areaframes = []
#     for i in range(len(linefiles)):
#         lineframe = initialize_trace_frame(linefiles[i])
#         areaframe = initialize_area_frame(areafiles[i])
#         lineframes.append(lineframe)
#         areaframes.append(areaframe)
#     lineframe = unify_frames(lineframes)
#     areaframe = unify_frames(areaframes)
#
#     return lineframe, areaframe


# def flatten_list(l):
#     flat_list = []
#     for sublist in l:
#         for item in sublist:
#             flat_list.append(item)
#     return flat_list


# def define_marker_sizes(shp_len):
#     """
#     Custom marker sizes based on length. Size is determined based on distance from median.
#     :param shp_len: Array of lengths
#     :type shp_len: np.ndarray
#     :return: Array of sizes
#     :rtype: np.ndarray
#     """
#     def dist(length, median):
#         if (length - median) == 0:
#             return 0.01
#         distance = np.abs((1 / (median - length)))
#         return distance
#
#     lineframe = pd.DataFrame({'length': shp_len}, columns=['length'])
#     median = lineframe.length.median()
#
#     lineframe['marker_size'] = lineframe.apply(lambda x: dist(x['length'], median), axis=1)
#     sizes = lineframe['marker_size'].values
#     sizes = sizes / np.linalg.norm(sizes)
#     sizes = np.log10(sizes + 1)
#     sizes = sizes * 10000
#     return sizes


def azimuth_plot_attributes(lineframe, weights=False):
    """
    Calculates azimuth bins for plotting azimuth rose plots.

    Examples:

    Non-weighted

    >>> azimuth_plot_attributes(pd.DataFrame({'azimu': np.array([0, 45, 90]), 'length': np.array([1, 2, 1])}))
    array([33.33333333,  0.        ,  0.        ,  0.        ,  0.        ,
    33.33333333,  0.        ,  0.        ,  0.        , 33.33333333,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        , 33.33333333,  0.        ,
    0.        ,  0.        ,  0.        , 33.33333333,  0.        ,
    0.        ,  0.        , 33.33333333,  0.        ,  0.        ,
    0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
    0.        ])

    Weighted

    >>> azimuth_plot_attributes(pd.DataFrame({'azimu': np.array([0, 45, 90]), 'length': np.array([1, 2, 1])}), weights=True)
    array([25.,  0.,  0.,  0.,  0., 50.,  0.,  0.,  0., 25.,  0.,  0.,  0.,
    0.,  0.,  0.,  0.,  0., 25.,  0.,  0.,  0.,  0., 50.,  0.,  0.,
    0., 25.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])


    :param lineframe: DataFrame containing lines with azimu (and length columns)
    :type lineframe: pd.DataFrame
    :param weights: Whether to use weights for bins.
    :type weights: bool
    :return: bin_locs: Locations for each bin, number_of_azimuths: Frequency (length-weighted) of each bin
    :rtype: tuple
    """

    # def get_statistics(lineframe):
    #     df = lineframe
    #     df['halved'] = df.azimu.apply(azimu_half)
    #     df = df.loc[:, 'halved']
    #     stats = df.describe()
    #     count = round(stats[0], 2)
    #     mean = round(stats[1], 2)
    #     std = round(stats[2], 2)
    #     min_len = round(stats[3], 5)
    #     max_len = round(stats[7], 2)
    #     median = round(df.median(), 2)
    #     return {'count': count, 'mean': mean, 'std': std, 'minLen': min_len, 'maxLen': max_len, 'median': median}

    lineframe = lineframe.dropna(subset=["azimu"])
    # stats = get_statistics(lineframe)
    azimuths = lineframe.loc[:, "azimu"].values
    bin_edges = np.arange(-5, 366, 10)
    lengths = None
    if weights:
        lengths = lineframe.loc[:, "length"].values
        number_of_azimuths, bin_edges = np.histogram(
            azimuths, bin_edges, weights=lengths
        )
    else:
        number_of_azimuths, bin_edges = np.histogram(azimuths, bin_edges)
    number_of_azimuths[0] += number_of_azimuths[-1]
    half = np.sum(np.split(number_of_azimuths[:-1], 2), 0)
    two_halves = np.concatenate([half, half])
    if weights:
        azi_count = lengths.sum()
    else:
        azi_count = len(azimuths)
    two_halves = (two_halves / azi_count) * 100
    return two_halves


def azimuth_plot_attributes_experimental(lineframe, weights=False):
    """
    Calculates azimuth bins for plotting azimuth rose plots. Uses HALVED azimuths (i.e. column 'halved' in lineframe)

    :param lineframe: DataFrame containing lines with halved azimuth (and length) columns
    :type lineframe: pd.DataFrame
    :param weights: Whether to use weights for bins.
    :type weights: bool
    :return: bin_width: Azimuth bin array, bin_locs: Bin locs, number_of_azimuths: Azimuth bin frequency (length-weighted)
    :rtype: tuple
    """

    # def get_statistics(lineframe):
    #     df = lineframe
    #     df['halved'] = df.azimu.apply(azimu_half)
    #     df = df.loc[:, 'halved']
    #     stats = df.describe()
    #     count = round(stats[0], 2)
    #     mean = round(stats[1], 2)
    #     std = round(stats[2], 2)
    #     min_len = round(stats[3], 5)
    #     max_len = round(stats[7], 2)
    #     median = round(df.median(), 2)
    #     return {'count': count, 'mean': mean, 'std': std, 'minLen': min_len, 'maxLen': max_len, 'median': median}

    def calc_ideal_bin_width(n, axial=True):
        """
        Calculates ideal bin width. axial or vector data
        Reference:
            Sanderson, D.J., Peacock, D.C.P., 2020.
            Making rose diagrams fit-for-purpose. Earth-Science Reviews. doi:10.1016/j.earscirev.2019.103055

        E.g.

        >>> calc_ideal_bin_width(30)
        28.964681538168897

        >>> calc_ideal_bin_width(90)
        20.08298850246509

        :param n: Sample size
        :type n: int
        :param axial: Whether data is axial or vector
        :type axial: bool
        :return: Bin width in degrees
        :rtype: float
        :raises: ValueError
        """
        if n <= 0:
            raise ValueError("Sample size cannot be 0 or lower")
        if axial:
            degree_range = 180
        else:
            degree_range = 360
        return degree_range / (2 * n ** (1 / 3))

    def calc_bins(ideal_bin_width):
        """
        Calculates bin edges and real bin width

        E.g.

        >>> calc_bins(25.554235)
        (array([  0. ,  22.5,  45. ,  67.5,  90. , 112.5, 135. , 157.5, 180. ]), 22.5)

        :param ideal_bin_width: Ideal bin width
        :type ideal_bin_width: float
        :return: bin_edges: bin edges, bin_width: real bin width (rounded up)
        :rtype: np.ndarray, float
        """

        div = 180 / ideal_bin_width
        rounded_div = math.ceil(div)
        bin_width = 180 / rounded_div

        start = 0
        end = 180 + bin_width * 0.01
        bin_edges = np.arange(start, end, bin_width)
        return bin_edges, bin_width

    def calc_locs(bin_width):
        """
        Calculates bar plot bar locations.
        :param bin_width: Real bin width
        :type bin_width: float
        :return: Array of locations
        :rtype: np.ndarray
        """
        start = bin_width / 2
        end = 180 + bin_width / 2
        locs = np.arange(start, end, bin_width)
        return locs

    lineframe = lineframe.dropna(subset=["halved"])
    # stats = get_statistics(lineframe)
    azimuths = lineframe.loc[:, "halved"].values
    ideal_bin_width = calc_ideal_bin_width(lineframe.shape[0])
    # Half the bin width if set so in config.py
    if config.half_the_bin_width:
        ideal_bin_width = ideal_bin_width / 2
    bin_edges, bin_width = calc_bins(ideal_bin_width)
    bin_locs = calc_locs(bin_width)

    if weights:
        lengths = lineframe.loc[:, "length"].values
        number_of_azimuths, _ = np.histogram(azimuths, bin_edges, weights=lengths)
    else:
        number_of_azimuths, _ = np.histogram(azimuths, bin_edges)

    # bin_edges = np.arange(-5, 366, 10)
    # if weights:
    #     lengths = lineframe.loc[:, 'length'].values
    #     number_of_azimuths, bin_edges = np.histogram(azimuths, bin_edges, weights=lengths)
    # else:
    #     number_of_azimuths, bin_edges = np.histogram(azimuths, bin_edges)

    # number_of_azimuths[0] += number_of_azimuths[-1]
    # half = np.sum(np.split(number_of_azimuths[:-1], 2), 0)
    # two_halves = np.concatenate([half, half])
    # if weights:
    #     azi_count = lengths.sum()
    # else:
    #     azi_count = len(azimuths)
    # two_halves = (two_halves / azi_count) * 100
    return bin_width, bin_locs, number_of_azimuths


def plot_azimuth_plot(two_halves, weights=False):
    """
    Plots an azimuth plot using a bin array. Weighted or non-weighted.

    :param two_halves: Azimuth bin array
    :type two_halves: np.ndarray
    :param weights: Whether to use weights for bins.
    :type weights: bool

    """
    fig = plt.figure(polar=True)
    fig.patch.set_facecolor("#CDFFE6")
    ax = plt.gca()
    ax.bar(
        np.deg2rad(np.arange(0, 360, 10)),
        two_halves,
        width=np.deg2rad(10),
        bottom=0.0,
        color="#F7CECC",
        edgecolor="r",
        alpha=0.75,
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # ax.set_thetagrids(np.arange(0, 360, 45), labels=np.linspace(0, 360, 9).astype('int'), fmt='%d%°', fontweight= 'bold')
    ax.set_thetagrids(np.arange(0, 360, 45), fontweight="bold")
    # ax.set_rgrids(np.arange(0, two_halves.max() + 1, two_halves.max()/2), angle=0, weight= 'black')
    ax.set_rgrids(
        np.linspace(5, 10, num=2), angle=0, weight="black", fmt="%d%%", fontsize=7
    )
    ax.grid(linewidth=1, color="k")
    ax.set_facecolor("#FEF1CD")
    if weights:
        ax.set_title(
            "WEIGHTED",
            x=0.1,
            y=1,
            fontsize=14,
            fontweight="heavy",
            fontfamily="Times New Roman",
        )
    else:
        ax.set_title(
            "NONWEIGHTED",
            x=0.1,
            y=1,
            fontsize=14,
            fontweight="heavy",
            fontfamily="Times New Roman",
        )


# def length_distribution_plot(lineframe):       #Frame with lengths and y (normed)
#    lineframe['logLen'] = lineframe.length.apply(np.log)
#    lineframe['logY'] = lineframe.y.apply(np.log)
#    m, c = np.polyfit(lineframe['logLen'].values, lineframe['logY'].values, 1)       # log(y) = m*log(x) + c fitted     y = c*x^m
#    y_fit = np.exp(m*lineframe['logLen'].values + c) # calculate the fitted values of y
#    lineframe['y_fit'] = y_fit
#    plt.plot(lineframe['logLen'].values, lineframe['y_fit'].values, linewidth=1, c='k')
#    return lineframe


def length_distribution_plot(length_distribution):
    """
    Plots a length distribution plot.

    :param length_distribution: TargetAreaLines Class
    :type length_distribution: TargetAreaLines
    """
    ld = length_distribution
    name = ld.name
    lineframe = ld.lineframe_main
    #    lineframe.to_csv(r'F:\Users\nikke\2D_Fracture_Analysis_Kit\test\{}_non_cut.csv'.format(name), sep=';') DEBUG
    lineframe = pd.DataFrame(lineframe)
    lineframe.plot.scatter(
        x="length",
        y="y",
        logx=True,
        logy=True,
        xlim=(ld.left, ld.right),
        ylim=(ld.bottom, ld.top),
        label=name + "_full",
    )

    lineframe = ld.lineframe_main_cut
    #    lineframe.to_csv(r'F:\Users\nikke\2D_Fracture_Analysis_Kit\test\{}_cut.csv'.format(name), sep=';') DEBUG
    lineframe = pd.DataFrame(lineframe)
    lineframe.plot.scatter(
        x="length",
        y="y",
        logx=True,
        logy=True,
        xlim=(ld.left, ld.right),
        ylim=(ld.bottom, ld.top),
        label=name + "_cut",
    )


def length_distribution_plot_with_fit(length_distribution):
    """
    Plots a length distribution plot with fit.

    :param length_distribution: TargetAreaLines Class
    :type length_distribution: TargetAreaLines
    """
    fig, ax = plt.subplots()

    ld = length_distribution
    name = ld.name
    lineframe = ld.lineframe_main_cut
    lineframe = pd.DataFrame(lineframe)
    lineframe.plot.scatter(
        x="length",
        y="y",
        logx=True,
        logy=True,
        xlim=(ld.left, ld.right),
        ylim=(ld.bottom, ld.top),
        label=name + "_cut",
        ax=ax,
    )
    lineframe["logLen"] = lineframe.length.apply(np.log)
    lineframe["logY"] = lineframe.y.apply(np.log)
    vals = np.polyfit(
        lineframe["logLen"].values, lineframe["logY"].values, 1
    )  # log(y) = m*log(x) + c fitted     y = c*x^m
    if len(vals) == 2:
        m, c = vals[0], vals[1]
    else:
        raise Exception("Too many values from np.polyfit, 2 expected")
    y_fit = np.exp(
        m * lineframe["logLen"].values + c
    )  # calculate the fitted values of y
    lineframe["y_fit"] = y_fit
    lineframe.plot(x="length", y="y_fit", color="k", ax=ax)

    ax.set_xlim(ld.left, ld.right)
    ax.set_ylim(ld.bottom, ld.top)


def length_distribution_plot_sets(length_distribution):
    """
    Plots a length distribution plot using sets.

    :param length_distribution: TargetAreaLines Class
    :type length_distribution: TargetAreaLines
    """
    ld = length_distribution
    name = ld.name
    setframes = ld.setframes
    setframes_cut = ld.setframes_cut
    for setframe in setframes:
        curr_set = setframe.set.iloc[0]
        setframe = pd.DataFrame(setframe)
        setframe.plot.scatter(
            x="length",
            y="y",
            logx=True,
            logy=True,
            xlim=(ld.left, ld.right),
            ylim=(ld.bottom, ld.top),
            label=name + "_full_set_" + str(curr_set),
        )
    for setframe_cut in setframes_cut:
        curr_set = setframe_cut.set.iloc[0]
        setframe_cut = pd.DataFrame(setframe_cut)
        setframe_cut.plot.scatter(
            x="length",
            y="y",
            logx=True,
            logy=True,
            xlim=(ld.left, ld.right),
            ylim=(ld.bottom, ld.top),
            label=name + "_cut_set_" + str(curr_set),
        )


def sd_calc(data):
    """
    TODO: Wrong results atm. Needs to take into account real length, not just orientation of unit vector.
    Calculates standard deviation for radial data (degrees)

    E.g.

    >>> sd_calc(np.array([2, 5, 8]))
    (3.0, 5.00)

    :param data: Array of degrees
    :type data: np.ndarray
    :return: Standard deviation
    :rtype: float
    """

    n = len(data)
    if n <= 1:
        return 0.0
    mean, sd = avg_calc(data), 0.0
    # calculate stan. dev.
    for el in data:
        diff = abs(mean - float(el))
        if diff > 180:
            diff = 360 - diff

        sd += diff ** 2
    sd = math.sqrt(sd / float(n - 1))

    return sd, mean


def avg_calc(data):
    # TODO: Should take length into calculations.......................... not real average atm
    n, mean = len(data), 0.0

    if n <= 1:
        return data[0]
    vectors = []
    # calculate average
    for el in data:
        rad = math.radians(el)
        v = np.array([math.cos(rad), math.sin(rad)])
        vectors.append(v)

    meanv = np.mean(np.array(vectors), axis=0)

    mean = math.degrees(math.atan2(meanv[1], meanv[0]))
    # print(mean)
    if mean < 0:
        mean = 360 + mean
    return mean


# def get_azimu_statistics(lineframe):
#     df = lineframe
#     df["halved"] = df.azimu.apply(azimu_half)
#     median = round(df.halved.median(), 2)
#     df = df.loc[:, "azimu"]

#     stats = df.describe()
#     count = round(stats[0], 2)
#     mean = round(azimu_half(avg_calc(df.values.tolist())), 2)
#     #    mean = round(stats[1], 2)
#     std = round(sd_calc(df.values.tolist()), 2)
#     #    std = round(stats[2], 2)
#     min_az = azimu_half(round(stats[3], 5))
#     max_az = azimu_half(round(stats[7], 2))

#     return {
#         "count": count,
#         "mean": mean,
#         "std": std,
#         "minAz": min_az,
#         "maxAz": max_az,
#         "median": median,
#     }


# def transform_to_ellipse(halved, phi):
#     rotation_of_ellipse = 90 - phi
#     transformed = rotation_of_ellipse - halved
#     return transformed
#
#
# def radius_of_ellipse(a, b, transformed):
#     angle = np.deg2rad(transformed)
#     m = max([a, b])
#     a = a / m
#     b = b / m
#     ab = a * b
#     u_sqrt = (a ** 2) * np.sin(angle) * np.sin(angle) + (b ** 2) * np.cos(angle) * np.cos(angle)
#     radius = ab / np.sqrt(u_sqrt)
#     return radius


# def calc_ellipse_weight_for_row(halved, length, a, b, phi):
#     transformed = transform_to_ellipse(halved, phi)
#     radius = radius_of_ellipse(a, b, transformed)
#     weight = length / radius
#     # print('old length:', length, 'new length:', weight)  # DEBUGGING
#     # print('phi:', phi, 'halved:', halved)  # DEBUGGING
#     return weight


# def calc_ellipse_weight(lineframe, a, b, phi):
#     lineframe['ellipse_weight'] = lineframe.apply(
#         lambda row: calc_ellipse_weight_for_row(row['halved'], row['length'], a, b, phi), axis=1)
#     return lineframe


def tern_yi_func(c, x):
    """
    Function for plotting *Connections per branch* line to branch ternary plot. Absolute values.
    """
    temp = 6 * (1 - 0.5 * c)
    temp2 = 3 - (3 / 2) * c
    temp3 = 1 + c / temp
    y = (c + 3 * c * x) / (temp * temp3) - (4 * x) / (temp2 * temp3)
    i = 1 - x - y
    return x, i, y


def tern_plot_the_fing_lines(tax, cs_locs=(1.3, 1.5, 1.7, 1.9)):
    """
    Plots *connections per branch* parameter to XYI-plot.

    :param tax: Ternary axis to plot to
    :type tax: ternary.TernaryAxesSubplot
    :param cs_locs: Pre-determined locations for lines
    :type cs_locs: tuple
    """

    def tern_find_last_x(c, x_start=0):
        x, i, y = tern_yi_func(c, x_start)
        while y > 0:
            x_start += 0.01
            x, i, y = tern_yi_func(c, x_start)
        return x

    def tern_yi_func_perc(c, x):
        temp = 6 * (1 - 0.5 * c)
        temp2 = 3 - (3 / 2) * c
        temp3 = 1 + c / temp
        y = (c + 3 * c * x) / (temp * temp3) - (4 * x) / (temp2 * temp3)
        i = 1 - x - y
        return 100 * x, 100 * i, 100 * y

    for c in cs_locs:
        last_x = tern_find_last_x(c)
        x1 = 0
        x2 = last_x
        point1 = tern_yi_func_perc(c, x1)
        point2 = tern_yi_func_perc(c, x2)
        tax.line(
            point1,
            point2,
            alpha=0.4,
            color="k",
            zorder=-5,
            linestyle="dashed",
            linewidth=0.5,
        )
        ax = plt.gca()
        rot = 6.5
        rot2 = 4.5
        ax.text(x=55, y=59, s=r"$C_B = 1.3$", fontsize=10, rotation=rot, ha="center")
        ax.text(x=61, y=50, s=r"$C_B = 1.5$", fontsize=10, rotation=rot, ha="center")
        ax.text(
            x=68.5,
            y=36.6,
            s=r"$C_B = 1.7$",
            fontsize=10,
            rotation=rot2 + 1,
            ha="center",
        )
        ax.text(x=76, y=17, s=r"$C_B = 1.9$", fontsize=10, rotation=rot2, ha="center")


def tern_plot_branch_lines(tax):
    """
    Plot line of random assignment of nodes to branches to a given branch ternary tax.
    Line positions taken from NetworkGT open source code.
    Credit to:
    https://github.com/BjornNyberg/NetworkGT

    :param tax: Ternary axis to plot to
    :type tax: ternary.TernaryAxesSubplot
    """
    ax = tax.get_axes()
    tax.boundary()
    points = [
        (0, 1, 0),
        (0.01, 0.81, 0.18),
        (0.04, 0.64, 0.32),
        (0.09, 0.49, 0.42),
        (0.16, 0.36, 0.48),
        (0.25, 0.25, 0.5),
        (0.36, 0.16, 0.48),
        (0.49, 0.09, 0.42),
        (0.64, 0.04, 0.32),
        (0.81, 0.01, 0.18),
        (1, 0, 0),
    ]
    for idx, p in enumerate(points):
        points[idx] = points[idx][0] * 100, points[idx][1] * 100, points[idx][2] * 100

    text_loc = [(0.37, 0.2), (0.44, 0.15), (0.52, 0.088), (0.64, 0.055), (0.79, 0.027)]
    for idx, t in enumerate(text_loc):
        text_loc[idx] = t[0] * 100, t[1] * 100
    text = [r"$C_B = 1.0$", r"$1.2$", r"$1.4$", r"$1.6$", r"$1.8$"]
    rots = [-61, -44, -28, -14, -3]
    # rot = -65
    for t, l, rot in zip(text, text_loc, rots):
        ax.annotate(t, xy=l, fontsize=9, rotation=rot)
        # rot += 17
    tax.plot(
        points,
        linewidth=1.5,
        marker="o",
        color="k",
        linestyle="dashed",
        markersize=3,
        zorder=-5,
        alpha=0.6,
    )


# def match_filenames_based_on_code(linefiles, nodefiles, areafiles):
#     if len(linefiles) == 0 == len(nodefiles) == len(areafiles):
#         print('all 0')  # DEBUGGING
#         raise Exception('All 0 in match_filenames_based_on_code')
#     matched_files = []
#     for lf, nf, af in zip(linefiles, nodefiles, areafiles):
#         found_match = False
#         lf_filename = Path(lf).stem
#         lf_code = lf_filename.split('_')[0]
#         nf_filename = Path(nf).stem
#         nf_code = nf_filename.split('_')[0]
#         af_filename = Path(af).stem
#         af_code = af_filename.split('_')[0]
#         if lf_code == nf_code == af_code:
#             matched_files.append((lf, nf, af))
#             found_match = True
#         else:
#             for nf_2 in nodefiles:
#                 nf_2_code = Path(nf_2).stem.split('_')[0]
#                 if lf_code == nf_2_code:
#                     for af_2 in areafiles:
#                         af_2_code = Path(af_2).stem.split('_')[0]
#                         if lf_code == nf_2_code == af_2_code:
#                             matched_files.append((lf, nf_2, af_2))
#                             found_match = True
#         if not found_match:
#             raise Exception('ERROR: Couldnt match all filenames based on code. Check spelling of filenames.')
#     return matched_files


def aniso_get_class_as_value(c):
    """
    Return value based on branch classification. Only C-C branches have a value, but this can be changed here.
    Classification can differ from ('C - C', 'C - I', 'I - I') (e.g. 'C - E') in which case a value is still returned.

    E.g.

    >>> aniso_get_class_as_value('C - C')
    1

    >>> aniso_get_class_as_value('C - E')
    0

    :param c: Branch classification
    :type c: str
    :return: Value for classification
    :rtype: float
    """
    if c not in ("C - C", "C - I", "I - I"):
        return 0
    if c == "C - C":
        return 1
    elif c == "C - I":
        return 0
    elif c == "I - I":
        return 0
    else:
        return 0


def aniso_calc_anisotropy(halved, c, length):
    """
    Calculates anisotropy of connectivity for a branch based on azimuth, classification and length.
    Value is calculated for preset angles (angles_of_study = np.arange(0, 179, 30))

    E.g.

    Anisotropy for a C-C classified branch:

    >>> aniso_calc_anisotropy(90, 'C - C', 10)
    array([6.12323400e-16, 5.00000000e+00, 8.66025404e+00, 1.00000000e+01,
           8.66025404e+00, 5.00000000e+00])

    Other classification for branch:

    >>> aniso_calc_anisotropy(90, 'C - I', 10)
    array([0, 0, 0, 0, 0, 0])

    :param halved: Azimuth of branch in range 0 - 180
    :type halved: float
    :param c: Branch classification
    :type c: str
    :param length: Branch length
    :type length: float
    :return: Result is given for every angle of study (angles_of_study array from config.py-file)
    :rtype: np.ndarray
    """
    angles_of_study = config.angles_for_examination
    # print(angles_of_study)
    c_value = aniso_get_class_as_value(c)
    # CALCULATION
    results = []
    for angle in angles_of_study:
        if c_value == 0:
            results.append(0)
            continue
        diff = np.abs(angle - halved)
        if diff > 90:
            diff = 180 - max([angle, halved]) + min([angle, halved])
        cos_diff = np.cos(np.deg2rad(diff))
        result = length * c_value * cos_diff
        results.append(result)
    # print(results)
    return np.array(results)


# def calc_sum_isotropy(branchframe):
#     sums = []
#     for _, row in branchframe.iterrows():
#         sum_list = []
#         for a in row.anisotropy:
#             sum_list.append(a)
#         sums.append(np.array(sum_list))
#     for idx, sum_arr in enumerate(sums):
#         if idx == 0:
#             arr_sum = sum_arr
#         else:
#             arr_sum += sum_arr
#     return arr_sum


def calc_y_distribution(lineframe_main: pd.DataFrame, area: float) -> pd.DataFrame:
    """
    Calculates Complementary Cumulative Number (= y) for a length distribution
    and normalises the values based on target area size.

    >>> df = pd.DataFrame({"length": [1, 10, 100, 1000, 4, 2123, 24]})
    >>> area = 234
    >>> calc_y_distribution(df, area)
       length         y
    5    2123  0.004274
    3    1000  0.008547
    2     100  0.012821
    6      24  0.017094
    1      10  0.021368
    4       4  0.025641
    0       1  0.029915

    :param lineframe_main: DataFrame with lengths
    :type lineframe_main: pd.DataFrame
    :param area: Area value
    :type area: float
    :return: DataFrame with y values calculated
    :rtype: pd.DataFrame
    """

    lineframe_main = lineframe_main.sort_values(by=["length"], ascending=False)
    nrows = len(lineframe_main.index)
    y = np.arange(1, nrows + 1, 1) / area
    lineframe_main["y"] = y
    return lineframe_main


def calc_cut_off_length(lineframe_main, cut_off: float):
    """
    Calculates minimum length for distribution with a proportional cut-off
    value. Exact cut-off length is calculated using cut-off value and the
    maximum line length in distribution.

    E.g.

    >>> calc_cut_off_length(pd.DataFrame({'length': np.array([1, 2, 3, 4, 5, 6,
    ... 7, 8, 9, 10])}), 0.55)
    4.5

    >>> calc_cut_off_length(pd.DataFrame({'length': np.array([1, 2, 3, 4, 5, 6,
    ... 7, 8, 9, 10])}), 0.5)
    5.0

    :param lineframe_main: DataFrame with lengths
    :type lineframe_main: pd.DataFrame
    :param cut_off: Proportional cut-off value in range [0, 1]
    :type cut_off: float
    :return: Cut-off length
    :rtype: float
    """

    if cut_off == 1.0:
        min_length = lineframe_main.length.min()
    else:
        largest_length = lineframe_main.length.max()
        min_length = largest_length * (1 - cut_off)

    cut_off_length = min_length
    return cut_off_length


def calc_xlims(lineframe) -> Tuple[float, float]:
    left = lineframe.length.min() / 50
    right = lineframe.length.max() * 50
    return left, right


def calc_ylims(lineframe) -> Tuple[float, float]:
    # TODO: Take y series instead of while dataframe...
    top = lineframe.y.max() * 50
    bottom = lineframe.y.min() / 50
    return top, bottom


def define_set(azimuth, set_df):  # Uses HALVED azimuth: 0-180
    """
    Defines set based on azimuth value in degrees. Uses halved azimuth values
    i.e. azimuth must be: 0 < azimuth < 180.

    E.g. when azimuth is in set 1:

    >>> define_set(50, pd.DataFrame({'Set': [1, 2], 'SetLimits': [(10, 90), (100, 170)]}))
    '1'

    When azimuth isn't within ranges:

    >>> define_set(99, pd.DataFrame({'Set': [1, 2], 'SetLimits': [(10, 90), (100, 170)]}))
    '-1'

    :param azimuth: azimuth as float (degrees)
    :type azimuth: float
    :param set_df: DataFrame with set ranges and set names
    :type set_df: DataFrame
    :return: Set name based on azimuth
    :rtype: str
    """
    # Quick assertation
    if azimuth < 0 or azimuth > 180:
        raise ValueError("Azimuth value wasnt in range 0 - 180.\n" f"Value: {azimuth}")

    # Set_num is -1 when azimuth is in no set range
    set_label = -1
    for _, row in set_df.iterrows():
        set_range = row.SetLimits
        # Checks if azimuth degree value is within the set range and
        # checks whether set range is normal (e.g. 20, 50) or if it extends over 0 (e.g. 171, 15)
        if set_range[0] > set_range[1]:
            if (azimuth >= set_range[0]) | (azimuth <= set_range[1]):
                set_label = row.Set
        else:
            if (azimuth >= set_range[0]) & (azimuth <= set_range[1]):
                set_label = row.Set

    return str(set_label)


def construct_length_distribution_base(
    lineframe: gpd.GeoDataFrame,
    areaframe: gpd.GeoDataFrame,
    name: str,
    group: str,
    cut_off_length=1,
    using_branches=False,
):
    """
    Helper function to construct TargetAreaLines to a pandas DataFrame using apply().
    """
    ld = ta.TargetAreaLines(
        lineframe, areaframe, name, group, using_branches, cut_off_length
    )
    return ld


def construct_node_data_base(nodeframe, name, group):
    """
    Helper function to construct TargetAreaNodes to a pandas DataFrame using apply().
    """
    node_data_base = ta.TargetAreaNodes(nodeframe, name, group)
    return node_data_base


def unify_lds(list_of_lds: list, group: str, cut_off_length: float):
    """
    Unifies/adds multiple TargetAreaLines objects together

    :param list_of_lds: List of TargetAreaLines objects
    :type list_of_lds: list
    :param group: Group name for the objects
    :type group: str
    :param cut_off_length: Cut-off for the group objects.
    :type cut_off_length: float
    :return: Unified TargetAreaLines object
    :rtype: ta.TargetAreaLines
    """
    df = pd.DataFrame(
        columns=["lineframe_main", "lineframe_main_cut", "areaframe", "name"]
    )
    for ld in list_of_lds:
        df = df.append(
            {
                "lineframe_main": ld.lineframe_main,
                "lineframe_main_cut": ld.lineframe_main_cut,
                "areaframe": ld.areaframe,
                "name": ld.name,
            },
            ignore_index=True,
        )
    lineframe_main = pd.concat(df.lineframe_main.tolist(), ignore_index=True)
    lineframe_main_cut = pd.concat(df.lineframe_main_cut.tolist(), ignore_index=True)
    areaframe = pd.concat(df.areaframe.tolist(), ignore_index=True)

    unif_ld_main = ta.TargetAreaLines(
        lineframe_main,
        areaframe,
        group,
        group=group,
        using_branches=list_of_lds[0].using_branches,
        cut_off_length=cut_off_length,
    )
    unif_ld_main.lineframe_main = lineframe_main
    unif_ld_main.lineframe_main_cut = lineframe_main_cut

    try:  # Only succeeds if sets have been defined, otherwise passed
        unif_ld_main.set_list = list_of_lds[0].set_list
    except AttributeError:
        pass

    return unif_ld_main


def unify_nds(list_of_nds: list, group: str):
    """
    Unifies/adds multiple TargetAreaNodes objects together

    >>> from fractopo.analysis.target_area import TargetAreaNodes
    >>> gdf1 = gpd.GeoDataFrame({"geometry": [Point(1, 1), Point(2, 2)], "Class": ["X", "Y"]})
    >>> gdf2 = gpd.GeoDataFrame({"geometry": [Point(0, 1), Point(3, 2)], "Class": ["I", "I"]})
    >>> target_area_nodes1 = TargetAreaNodes(gdf1, "some_name", "some_group")
    >>> target_area_nodes2 = TargetAreaNodes(gdf2, "some_name", "some_group")
    >>> list_of_nds = [target_area_nodes1, target_area_nodes2]
    >>> unify_nds(list_of_nds, "some_grouping")
                    nodeframe:
                                      geometry Class
    0  POINT (1.00000 1.00000)     X
    1  POINT (2.00000 2.00000)     Y
    2  POINT (0.00000 1.00000)     I
    3  POINT (3.00000 2.00000)     I
                    name:
                    some_grouping
                    group:
                    some_grouping

    :param list_of_nds: List of TargetAreaNodes objects
    :type list_of_nds: list
    :param group: Group name
    :type group: str
    :return: Unified TargetAreaNodes object
    :rtype: ta.TargetAreaNodes
    """
    df = pd.DataFrame(columns=["nodeframe", "name"])
    for nd in list_of_nds:
        df = df.append({"nodeframe": nd.nodeframe, "name": nd.name}, ignore_index=True)

    nodeframe = pd.concat(df.nodeframe.tolist(), ignore_index=True)
    unif_nd = ta.TargetAreaNodes(nodeframe, group, group)
    return unif_nd


# def calc_curviness_dataframe(lineframe, idx):
#     lineframe['curviness'] = lineframe.geometry.apply(curviness)
#     return lineframe, idx


def unite_areaframes(list_of_areaframes):
    loa = list_of_areaframes
    united = pd.concat(loa, ignore_index=True)
    return united


def curviness(linestring):
    try:
        coords = list(linestring.coords)
    except NotImplementedError:
        return np.NaN
    df = pd.DataFrame(columns=["azimu", "length"])
    for i in range(len(coords) - 1):
        start = Point(coords[i])
        end = Point(coords[i + 1])
        l = LineString([start, end])
        azimu = calc_azimu(l)
        # halved = tools.azimu_half(azimu)
        length = l.length
        addition = {
            "azimu": azimu,
            "length": length,
        }  # Addition to DataFrame with fit x and fit y values
        df = df.append(addition, ignore_index=True)

    std = sd_calc(df.azimu.values.tolist())
    azimu_std = std

    return azimu_std


def curviness_initialize_sub_plotting(filecount, ncols=4):
    nrows = filecount
    width = 20
    height = (width / 4) * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(width, height),
        gridspec_kw=dict(wspace=0.45, hspace=0.3),
    )
    fig.patch.set_facecolor("#CDFFE6")
    return fig, axes, nrows, ncols


def plot_curv_plot(lineframe, ax=plt.gca(), name=""):
    lineframe["curviness"] = lineframe.geometry.apply(curviness)

    # labels = ["{0} - {1}".format(i, i + 9) for i in range(0, 180, 9)]
    lineframe["group"] = pd.cut(lineframe.halved, range(0, 181, 30), right=False)

    sns.boxplot(data=lineframe, x="curviness", y="group", notch=True, ax=ax)
    ax.set_title(name, fontsize=14, fontweight="heavy", fontfamily="Times New Roman")
    ax.set_ylabel("Set (°)", fontfamily="Times New Roman")
    ax.set_xlabel("Curvature (°)", fontfamily="Times New Roman", style="italic")
    ax.grid(True, color="k", linewidth=0.3)
    # sns.violinplot(data=df, x='curviness', y='group')


# def plot_all_curv_plots(branchdirectory, traces=False):
#     sns.set(style='ticks')
#     branchfiles = get_filenames_branches_or_traces(branchdirectory)
#     filecount = len(branchfiles)
#     fig, axes, nrows, ncols = curviness_initialize_sub_plotting(filecount)
#     row = -1
#     for idx, file in enumerate(branchfiles):
#         name = Path(file).stem
#         if traces:
#             lineframe = initialize_trace_frame(file)
#         else:
#             lineframe = initialize_branch_frame(file)
#         col = idx % ncols
#
#         if col == 0:
#             row = row + 1
#         ax = axes[row][col]
#         plot_curv_plot(lineframe, ax, name)


# def plot_azimuths_sub_plot(lineframe, ax, filename, weights, striations=False, small_text=False,
#                            single_plot=False):
#     lineframe = lineframe.dropna(subset=['azimu'])
#     stats = get_azimu_statistics(lineframe)
#     azimuths = lineframe.loc[:, 'azimu'].values
#     bin_edges = np.arange(-5, 366, 10)
#     lengths = None
#     if weights:
#         lineframe['length'] = lineframe.geometry.length
#         lengths = lineframe.loc[:, 'length'].values
#         number_of_azimuths, bin_edges = np.histogram(azimuths, bin_edges, weights=lengths)
#     # elif ellipse_weights:
#     #     ellipse_weight_values = lineframe.loc[:, 'ellipse_weight'].values
#     #     number_of_azimuths, bin_edges = np.histogram(azimuths, bin_edges, weights=ellipse_weight_values)
#     else:
#         number_of_azimuths, bin_edges = np.histogram(azimuths, bin_edges)
#     number_of_azimuths[0] += number_of_azimuths[-1]
#     half = np.sum(np.split(number_of_azimuths[:-1], 2), 0)
#
#     two_halves = np.concatenate([half, half])
#     if weights:
#         azi_count = lengths.sum()
#     # elif ellipse_weights:
#     #     azi_count = ellipse_weight_values.sum()
#     else:
#         azi_count = len(azimuths)
#     two_halves = (two_halves / azi_count) * 100
#
#     ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, width=np.deg2rad(10), bottom=0.0, color='#F7CECC',
#            edgecolor='r', alpha=0.85, zorder=4)
#     ax.set_theta_zero_location('N')
#     ax.set_theta_direction(-1)
#     # ax.set_thetagrids(np.arange(0, 360, 45), labels=np.linspace(0, 360, 9).astype('int'), fmt='%d%°', fontweight= 'bold')
#     ax.set_thetagrids(np.arange(0, 360, 45), fontweight='bold')
#     # ax.set_rgrids(np.arange(0, two_halves.max() + 1, two_halves.max()/2), angle=0, weight= 'black')
#     ax.set_rgrids(np.linspace(5, 10, num=2), angle=0, weight='black', fmt='%d%%', fontsize=7)
#     ax.grid(linewidth=1, color='k')
#
#     title_props = dict(boxstyle='square', facecolor='wheat', path_effects=[path_effects.withSimplePatchShadow()])
#     title_x = 0.18
#     title_y = 1.25
#     if small_text:
#         fontmult = 0.5
#     else:
#         fontmult = 1
#     if weights:
#         ax.set_title(filename + '\nWEIGHTED', x=title_x, y=title_y, fontsize=fontmult * 20, fontweight='heavy'
#                      , fontfamily='Times New Roman', bbox=title_props, va='top')
#     # elif ellipse_weights:
#     #     ax.set_title(filename + '\nELLIPSE\nWEIGHTED', x=title_x, y=title_y, fontsize=fontmult * 17, fontweight='heavy'
#     #                  , fontfamily='Times New Roman', bbox=title_props, va='center')
#     else:
#         title_x = 0.18
#         title_y = 0.9
#         font_size = 20
#         if single_plot:
#             title_x = 0.14
#             title_y = 1.05
#             font_size = 25
#         ax.set_title(filename, x=title_x, y=title_y, fontsize=fontmult * font_size, fontweight='heavy'
#                      , fontfamily='Times New Roman', bbox=title_props, va='center')
#
#     props = dict(boxstyle='round', facecolor='wheat', path_effects=[path_effects.withSimplePatchShadow()])
#
#     text_x = 0.55
#     text_y = 1.42
#     if not small_text:
#         if striations:
#             text_x = 0.55
#             text_y = 1
#             if single_plot:
#                 text_x = 0.65
#                 text_y = 1.05
#             text = 'Striation count: ' + str(len(lineframe)) \
#                    + '\nMean striation: ' + str(int(stats['mean']))
#             ax.text(text_x, text_y, text, transform=ax.transAxes, fontsize=fontmult * 25, weight='roman'
#                     , bbox=props, fontfamily='Times New Roman', va='center')
#             # TICKLABELS
#             labels = ax.get_xticklabels()
#             for label in labels:
#                 label._y = -0.05
#                 label._fontproperties._size = 24
#                 label._fontproperties._weight = 'bold'
#         else:
#             text = 'n = ' + str(len(lineframe)) + '\n'
#             text = text + create_azimuth_set_text(lineframe)
#             # +'\nMedian Azimuth: '+str(stats['median'])\
#             # +'\nSTD Azimuth: '+str(stats['std'])
#             ax.text(text_x, text_y, text, transform=ax.transAxes, fontsize=fontmult * 20, weight='roman'
#                     , bbox=props, fontfamily='Times New Roman', va='top')
#             # TICKLABELS
#             labels = ax.get_xticklabels()
#             for label in labels:
#                 label._y = -0.05
#                 label._fontproperties._size = 24
#                 label._fontproperties._weight = 'bold'
#
#     if striations:
#         mean_striation = stats['mean']
#         arrow_dir = (180 + (180 - mean_striation))
#         arrow_len = two_halves.max() - 10
#         ax.vlines(np.deg2rad(mean_striation), 0, arrow_len, zorder=5, colors=['blue'], alpha=0.8)
#         ax.vlines(np.deg2rad(mean_striation + 180), 0, arrow_len, zorder=5, colors=['blue'], alpha=0.8)
#         ax.scatter(np.deg2rad(mean_striation), arrow_len, marker=(3, 0, arrow_dir), s=500, zorder=6, color='blue')


def line_end_point(line):
    """
    Determine the end point of a given shapely Line.

    >>> line_end_point(LineString([(0, 0), (1, 1), (2, 2)])).wkt
    'POINT (2 2)'

    :param line:
    :type line:
    :return: Point or np.NaN if unsuccessful.
    :rtype: shapely.geometry.Point | numpy.NaN
    """
    try:
        coord_list = list(line.coords)
    except:
        return np.NaN
    end_x = coord_list[-1][0]
    end_y = coord_list[-1][1]
    return shapely.geometry.Point(end_x, end_y)


def line_start_point(line):
    """
    Determine the start point of a given shapely Line.

    :param line:
    :type line:
    :return: Point or np.NaN if unsuccessful.
    :rtype: shapely.geometry.Point | numpy.NaN
    """
    try:
        coord_list = list(line.coords)
    except:
        return np.NaN
    start_x = coord_list[0][0]
    start_y = coord_list[0][1]
    return shapely.geometry.Point(start_x, start_y)


def prepare_geometry_traces(traceframe):
    """
    Prepare tracefrace geometries for a spatial analysis (faster).
    The try-except clause is required due to QGIS differences in the shapely package

    :param traceframe:
    :type traceframe:
    :return:
    :rtype:
    """
    traces = traceframe.geometry.values
    traces = np.asarray(traces).tolist()
    trace_col = shapely.geometry.MultiLineString(traces)
    prep_col = prepared.prep(trace_col)
    return prep_col, trace_col

    # traces_ls = []
    # for t in traces:
    #     merged_t = shapely.ops.linemerge(t)
    #     if isinstance(merged_t, shapely.geometry.MultiLineString):
    #         print(merged_t)
    #         traces_ls.extend([merged_t])
    #     else:
    #         traces_ls.append(merged_t)
    #
    #
    # trace_col = shapely.geometry.MultiLineString(traces)
    # prep_col = shapely.prepared.prep(trace_col)
    # return prep_col, trace_col


def make_point_tree(traceframe):
    points = []
    for idx, row in traceframe.iterrows():
        sp = row.startpoint
        ep = row.endpoint
        points.extend([sp, ep])
    tree = strtree.STRtree(points)
    return tree


def get_nodes_intersecting_sets(xypointsframe, traceframe):
    """
    Does a spatial intersect between node GeoDataFrame with only X- and Y-nodes and
    the trace GeoDataFrame with only two sets. Returns only nodes that intersect traces from BOTH sets.

    E.g.

    >>> p = gpd.GeoDataFrame(data={'geometry': [Point(0, 0), Point(1, 1), Point(3, 3)]})
    >>> l = gpd.GeoDataFrame(data={'geometry': [LineString([(0, 0), (2, 2)]), LineString([(0, 0), (0.5, 0.5)])], 'set': [1, 2]})
    >>> get_nodes_intersecting_sets(p, l)
    geometry
    0  POINT (0.00000 0.00000)

    :param xypointsframe: GeoDataFrame with only X- and Y-nodes
    :type xypointsframe: GeoDataFrame
    :param traceframe: Trace GeoDataFrame with only two sets
    :type traceframe: GeoDataFrame
    :return: GeoDataFrame with nodes that intersect traces from BOTH sets
    :rtype: GeoDataFrame
    """

    sets = traceframe.set.unique().tolist()
    if len(sets) != 2:
        return xypointsframe.iloc[0:0]
        # raise Exception('get_nodes_intersecting_sets function received traceframe without exactly two sets.')
    set_1_frame = traceframe.loc[traceframe.set == sets[0]]
    set_2_frame = traceframe.loc[traceframe.set == sets[1]]

    prep_traces_1, _ = prepare_geometry_traces(set_1_frame)
    prep_traces_2, _ = prepare_geometry_traces(set_2_frame)

    rows_to_keep = []
    for idx, row in xypointsframe.iterrows():

        point = row.geometry
        if (prep_traces_1.intersects(point.buffer(buffer_value))) and (
            prep_traces_2.intersects(point.buffer(buffer_value))
        ):
            rows_to_keep.append(idx)
    intersecting_nodes_frame = xypointsframe.iloc[rows_to_keep].reset_index(drop=True)
    return intersecting_nodes_frame


def get_intersect_frame(intersecting_nodes_frame, traceframe, set_tuple):
    """
    Does spatial intersects to determine how abutments and crosscuts occur between two sets

    E.g. where Set 2 ends in set 1 and Set 2 crosscuts set 1 (TODO: DECREPID)

    >>> nodes = gpd.GeoDataFrame({'geometry': [Point(0, 0), Point(1, 1)], 'c': ['Y', 'X']})
    >>> traces = gpd.GeoDataFrame(data={'geometry': [
    ... LineString([(-1, -1), (2, 2)]), LineString([(0, 0), (-0.5, 0.5)]), LineString([(2, 0), (0, 2)])]
    ... , 'set': [1, 2, 2]})
    >>> traces['startpoint'] = traces.geometry.apply(line_start_point)
    >>> traces['endpoint'] = traces.geometry.apply(line_end_point)
    >>> sets = (1, 2)
    >>> intersect_frame = get_intersect_frame(nodes, traces, sets)
    >>> intersect_frame
              node nodeclass    sets  error
    0  POINT (0 0)         Y  (2, 1)  False
    1  POINT (1 1)         X  (1, 2)  False


    :param intersecting_nodes_frame: GeoDataFrame with X- and Y-nodes that intersect both sets.
        Contains columns: "c", "geometry"
    :type intersecting_nodes_frame: GeoDataFrame
    :param traceframe: GeoDataFrame of traces.  Contains columns: "set", "geometry", "startpoint", "endpoint"
    :type traceframe: GeoDataFrame
    :param set_tuple: Sets used in the comparison
    :type set_tuple: tuple
    :return: DataFrame with intersect results. Contains columns: 'node', 'nodeclass', 'sets', 'error':
        (node = node as shapely.geometry.Point, nodeclass = classification, i.e. X or Y, sets = tuple (set 1, set 2),
        error = possible error results)
    :rtype: DataFrame
    """

    intersectframe = pd.DataFrame(columns=["node", "nodeclass", "sets", "error"])

    set1, set2 = set_tuple[0], set_tuple[1]  # sets for comparison

    set1frame = traceframe.loc[traceframe.set == set1]  # cut first setframe
    set2frame = traceframe.loc[traceframe.set == set2]  # cut second setframe

    set1_prep, _ = prepare_geometry_traces(set1frame)
    set2_prep, _ = prepare_geometry_traces(set2frame)
    # Creates a rtree from all start- and endpoints of set 1
    # Used in deducting in which set a trace abuts (Y-node)
    set1pointtree = make_point_tree(set1frame)

    if len(intersecting_nodes_frame) == 0:
        # No intersections between sets
        return intersectframe

    for idx, row in intersecting_nodes_frame.iterrows():
        node = row.geometry
        c = row.c

        l1 = set1_prep.intersects(
            node.buffer(buffer_value)
        )  # Checks if node intersects set 1 traces.
        l2 = set2_prep.intersects(
            node.buffer(buffer_value)
        )  # Checks if node intersects set 2 traces.

        if (l1 is False) and (l2 is False):  # DEBUGGING
            raise Exception(
                f"Node {node} does not intersect both sets {set1} and {set2}\n l1 is {l1} and l2 is {l2}"
            )

        # NO RELATIONS FOR NODE IS GIVEN AS ERROR == TRUE (ERROR).
        # addition gets overwritten if there are no errors
        addition = {
            "node": node,
            "nodeclass": c,
            "sets": set_tuple,
            "error": True,
        }  # DEBUGGING

        # ALL X NODE RELATIONS
        if c == "X":
            if (l1 is True) and (l2 is True):  # It's an x-node between sets
                sets = (set1, set2)
                addition = {"node": node, "nodeclass": c, "sets": sets, "error": False}

            if (l1 is True) and (l2 is False):  # It's an x-node inside set 1
                raise Exception(
                    f"Node {node} does not intersect both sets {set1} and {set2}\n l1 is {l1} and l2 is {l2}"
                )
                # sets = (set1, set1)
                # addition = {'node': node, 'nodeclass': c, 'sets': sets}

            if (l1 is False) and (l2 is True):  # It's an x-node inside set 2
                raise Exception(
                    f"Node {node} does not intersect both sets {set1} and {set2}\n l1 is {l1} and l2 is {l2}"
                )
                # sets = (set2, set2)
                # addition = {'node': node, 'nodeclass': c, 'sets': sets}

        # ALL Y NODE RELATIONS
        elif c == "Y":
            if (l1 is True) and (l2 is True):  # It's an y-node between sets
                # p1 == length of list of nodes from set1 traces that intersect with X- or Y-node
                p1 = len(set1pointtree.query(node.buffer(buffer_value)))
                if p1 != 0:  # set 1 ends in set 2
                    sets = (set1, set2)
                else:  # set 2 ends in set 1
                    sets = (set2, set1)
                addition = {"node": node, "nodeclass": c, "sets": sets, "error": False}

            if (l1 is True) and (l2 is False):  # It's a y-node inside set 1
                raise Exception(
                    f"Node {node} does not intersect both sets {set1} and {set2}\n l1 is {l1} and l2 is {l2}"
                )
                # sets = (set1, set1)
                # addition = {'node': node, 'nodeclass': c, 'sets': sets}

            if (l1 is False) and (l2 is True):  # It's a y-node inside set 2
                raise Exception(
                    f"Node {node} does not intersect both sets {set1} and {set2}\n l1 is {l1} and l2 is {l2}"
                )
                # sets = (set2, set2)
                # addition = {'node': node, 'nodeclass': c, 'sets': sets}
        else:
            raise ValueError(f"Node {node} neither X or Y")

        intersectframe = intersectframe.append(
            addition, ignore_index=True
        )  # Append frame with result

    return intersectframe


def initialize_ternary_points(ax, tax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    tax.boundary(linewidth=1.5)
    tax.gridlines(linewidth=0.1, multiple=20, color="grey", alpha=0.6)
    tax.ticks(
        axis="lbr",
        linewidth=1,
        multiple=20,
        offset=0.035,
        tick_formats="%d%%",
        fontsize="small",
    )
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    fontsize = 25
    fdict = {
        "path_effects": [path_effects.withStroke(linewidth=3, foreground="k")],
        "color": "white",
        "family": "Calibri",
        "size": fontsize,
        "weight": "bold",
    }
    ax.text(-0.1, -0.03, "Y", transform=ax.transAxes, fontdict=fdict)
    ax.text(1.03, -0.03, "X", transform=ax.transAxes, fontdict=fdict)
    ax.text(0.5, 1.07, "I", transform=ax.transAxes, fontdict=fdict, ha="center")
    # tax.set_title(name, x=0.1, y=1, fontsize=14, fontweight='heavy', fontfamily='Times New Roman')


def initialize_ternary_branches_points(ax, tax):
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    tax.boundary(linewidth=1.5)
    tax.gridlines(linewidth=0.9, multiple=20, color="black", alpha=0.6)
    tax.ticks(
        axis="lbr",
        linewidth=0.9,
        multiple=20,
        offset=0.03,
        tick_formats="%d%%",
        fontsize="small",
        alpha=0.6,
    )
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis("off")
    fontsize = 20
    fdict = {
        "path_effects": [path_effects.withStroke(linewidth=3, foreground="k")],
        "color": "w",
        "family": "Calibri",
        "size": fontsize,
        "weight": "bold",
    }
    ax.text(-0.1, -0.06, "I - C", transform=ax.transAxes, fontdict=fdict)
    ax.text(1.0, -0.06, "C - C", transform=ax.transAxes, fontdict=fdict)
    ax.text(0.5, 1.07, "I - I", transform=ax.transAxes, fontdict=fdict, ha="center")


# def get_individual_xy_relations(ld, nd, for_ax=False, ax=None):
#     if for_ax:
#         nd.determine_XY_relation(length_distribution_for_relation=ld, for_ax=True, ax=ax)
#     else:
#         nd.determine_XY_relation(length_distribution_for_relation=ld)


def setup_ax_for_ld(ax_for_setup, using_branches, indiv_fit=False):
    """
    Function to setup ax for length distribution plots.

    :param ax_for_setup: Ax to setup.
    :type ax_for_setup: matplotlib.axes.Axes
    :param using_branches: Are the lines branches or traces.
    :type using_branches: bool
    """
    #
    ax = ax_for_setup
    # LABELS
    label = "Branch length $(m)$" if using_branches else "Trace Length $(m)$"
    ax.set_xlabel(
        label,
        fontsize="xx-large",
        fontfamily="Calibri",
        style="italic",
        labelpad=16,
    )
    # Individual powerlaw fits are not normalized to area because they aren't
    # multiscale
    ccm_unit = r"$(\frac{1}{m^2})$" if not indiv_fit else ""
    ax.set_ylabel(
        "Complementary Cumulative Number " + ccm_unit,
        fontsize="xx-large",
        fontfamily="Calibri",
        style="italic",
    )
    # TICKS
    plt.xticks(color="black", fontsize="x-large")
    plt.yticks(color="black", fontsize="x-large")
    plt.tick_params(axis="both", width=1.2)
    # LEGEND
    handles, labels = ax.get_legend_handles_labels()
    labels = ["\n".join(wrap(l, 13)) for l in labels]
    lgnd = plt.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(1.37, 1.02),
        ncol=2,
        columnspacing=0.3,
        shadow=True,
        prop={"family": "Calibri", "weight": "heavy", "size": "large"},
    )
    for lh in lgnd.legendHandles:
        # lh._sizes = [750]
        lh.set_linewidth(3)
    ax.grid(zorder=-10, color="black", alpha=0.5)


def create_azimuth_set_text(lineframe):
    """
    Creates azimuth set statistics for figure.

    >>> lineframe = pd.DataFrame(
    ...     {"length": [1, 2, 35, 7, 8, 6, 12, 5], "set": [1, 2, 1, 2, 1, 2, 1, 2]}
    ...     )
    >>> print(create_azimuth_set_text(lineframe))
    Set 1, FoL = 73.7%
    Set 2, FoL = 26.3%
    """
    sum_length = lineframe.length.sum()
    sets = lineframe.set.unique().tolist()
    sets.sort()
    t = ""
    for i in range(len(sets)):
        total_length = lineframe.loc[lineframe.set == sets[i]].length.sum()
        percent = total_length / sum_length
        # text = 'Set {}, Length (m) = {}'.format(sets[i], int(total_length))
        text = "Set {}, FoL = {}".format(sets[i], "{:.1%}".format(percent))

        if i < len(sets) - 1:
            text = text + "\n"
        t = t + text
    return t


# def report_powerlaw_fit_statistics(
#     name: str, fit: powerlaw.Fit, logger: logging.Logger, using_branches: bool
# ):
#     """
#     Writes powerlaw module Fit object statistics to a given logfile. Included
#     powerlaw, lognormal and exponential fit statistics and parameters.
#     """
#     power_law_exponent = -(fit.power_law.alpha - 1)
#     # logger.info(f"----------------------------------------------------------")
#     traces_or_branches = "branches" if using_branches else "traces"
#     # logger.info(
#     #     f"Powerlaw, lognormal and exponential statistics for {name} {traces_or_branches}."
#     # )
#     logger.info(f"Cut-off: {fit.xmin}")
#     # logger.info(f"Is the cut-off manual?: {fit.fixed_xmin}")
#     # Power-law
#     # logger.info(
#     #     f"Powerlaw Parameters and Statistics.\n"
#     #     f"Power Law Exponent: {power_law_exponent}\n"
#     #     f"Power Law Sigma: {fit.power_law.sigma}\n"
#     #     f"Power Law Cut-Off: {fit.power_law.xmin}. Manually solved cut-off: {fit.fixed_xmin}\n"
#     # )
#     # Lognormal
#     # logger.info(
#     #     f"Lognormal Parameters and Statistics.\n"
#     #     f"Lognormal mu: {fit.lognormal.mu}\n"
#     #     f"Lognormal sigma: {fit.lognormal.sigma}\n"
#     # )
#     # Exponent
#     # logger.info(
#     #     f"Exponential Parameters and Statistics.\n"
#     #     f"Exponential lambda: {fit.exponential.Lambda}\n"
#     # )

#     # Kolmogorov-Smirnov test to assess the best fits to given length distribution.

#     # logger.info(
#     #     "Kolmogorov-Smirnov test to assess the best fits to the given length distribution."
#     # )
#     logger.info("https://github.com/jeffalstott/powerlaw")
#     # logger.info(
#     #     "R is the loglikelihood ratio between the two candidate distributions."
#     #     " This number will be positive if the data is more likely in the first distribution,"
#     #     " and negative if the data is more likely in the second distribution."
#     #     " The significance value for that direction is p."
#     # )

#     # logger.info(f"Powerlaw versus lognormal.")
#     R, p = fit.distribution_compare("power_law", "lognormal")
#     # logger.info(f"R: {R}, p: {p}")

#     # logger.info(f"Powerlaw versus exponential.")
#     R, p = fit.distribution_compare("power_law", "exponential")
#     # logger.info(f"R: {R}, p: {p}")

#     # logger.info(f"Lognormal versus exponential.")
#     R, p = fit.distribution_compare("lognormal", "exponential")
#     # logger.info(f"R: {R}, p: {p}")

#     # logger.info(f"----------------------------------------------------------")


def report_powerlaw_fit_statistics_df(
    name: str,
    fit: powerlaw.Fit,
    report_df: pd.DataFrame,
    using_branches: bool,
    lengths: Union[np.ndarray, pd.Series],
):
    """
    Writes powerlaw module Fit object statistics to a given report_df. Included
    powerlaw, lognormal and exponential fit statistics and parameters.
    """
    powerlaw_exponent = -(fit.power_law.alpha - 1)
    # traces_or_branches = "branches" if using_branches else "traces"

    powerlaw_sigma = fit.power_law.sigma
    powerlaw_cutoff = fit.power_law.xmin
    powerlaw_manual = str(fit.fixed_xmin)
    lognormal_mu = fit.lognormal.mu
    lognormal_sigma = fit.lognormal.sigma
    exponential_lambda = fit.exponential.Lambda
    R, p = fit.distribution_compare("power_law", "lognormal")
    powerlaw_vs_lognormal_r = R
    powerlaw_vs_lognormal_p = p
    R, p = fit.distribution_compare("power_law", "exponential")
    powerlaw_vs_exponential_r = R
    powerlaw_vs_exponential_p = p
    R, p = fit.distribution_compare("lognormal", "exponential")
    lognormal_vs_exponential_r = R
    lognormal_vs_exponential_p = p

    remaining_percent_of_data = (sum(lengths > fit.power_law.xmin) / len(lengths)) * 100

    report_df = report_df.append(
        {
            ReportDfCols.name: name,
            ReportDfCols.powerlaw_exponent: powerlaw_exponent,
            ReportDfCols.powerlaw_sigma: powerlaw_sigma,
            ReportDfCols.powerlaw_cutoff: powerlaw_cutoff,
            ReportDfCols.powerlaw_manual: powerlaw_manual,
            ReportDfCols.lognormal_mu: lognormal_mu,
            ReportDfCols.lognormal_sigma: lognormal_sigma,
            ReportDfCols.exponential_lambda: exponential_lambda,
            ReportDfCols.powerlaw_vs_lognormal_r: powerlaw_vs_lognormal_r,
            ReportDfCols.powerlaw_vs_lognormal_p: powerlaw_vs_lognormal_p,
            ReportDfCols.powerlaw_vs_exponential_r: powerlaw_vs_exponential_r,
            ReportDfCols.powerlaw_vs_exponential_p: powerlaw_vs_exponential_p,
            ReportDfCols.lognormal_vs_exponential_r: lognormal_vs_exponential_r,
            ReportDfCols.lognormal_vs_exponential_p: lognormal_vs_exponential_p,
            ReportDfCols.lognormal_vs_exponential_p: lognormal_vs_exponential_p,
            ReportDfCols.remaining_percent_of_data: remaining_percent_of_data,
        },
        ignore_index=True,
    )
    return report_df


def plotting_directories(results_folder, name):
    """
    Creates plotting directories and handles FileExistsErrors when raised.

    :param results_folder: Base folder to create plots_{name} folder to.
    :type results_folder: str
    :param name: Analysis name.
    :type name: str
    :return: Newly made path to plotting directory where all plots will be saved to.
    :rtype: str
    """
    plotting_directory = f"{results_folder}/plots_{name}"
    try:
        try:
            os.mkdir(Path(plotting_directory))
        except FileExistsError:
            print("Earlier plots exist. Overwriting old ones.")
            return plotting_directory
        os.mkdir(Path(f"{plotting_directory}/age_relations"))
        os.mkdir(Path(f"{plotting_directory}/age_relations/indiv"))
        os.mkdir(Path(f"{plotting_directory}/anisotropy"))
        os.mkdir(Path(f"{plotting_directory}/anisotropy/indiv"))
        os.mkdir(Path(f"{plotting_directory}/azimuths"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_radius"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_radius/traces"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_radius/branches"))

        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_area"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_area/traces"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/equal_area/branches"))

        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_radius"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_radius/traces"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_radius/branches"))

        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_area"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_area/traces"))
        os.mkdir(Path(f"{plotting_directory}/azimuths/indiv/equal_area/branches"))

        os.mkdir(Path(f"{plotting_directory}/branch_class"))
        os.mkdir(Path(f"{plotting_directory}/branch_class/indiv"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/branches"))
        os.mkdir(
            Path(f"{plotting_directory}/length_distributions/branches/predictions")
        )
        os.mkdir(Path(f"{plotting_directory}/length_distributions/traces"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/traces/predictions"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/indiv"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/indiv/branches"))
        os.mkdir(Path(f"{plotting_directory}/length_distributions/indiv/traces"))
        os.mkdir(Path(f"{plotting_directory}/topology"))
        os.mkdir(Path(f"{plotting_directory}/topology/branches"))
        os.mkdir(Path(f"{plotting_directory}/topology/traces"))
        os.mkdir(Path(f"{plotting_directory}/xyi"))
        os.mkdir(Path(f"{plotting_directory}/xyi/indiv"))
        os.mkdir(Path(f"{plotting_directory}/hexbinplots"))

    # Should not be needed (shutil.rmtree(Path(f"{plotting_directory}"))).
    # Would run if only SOME of the above folders are present.
    # i.e. Folder creation has failed and same folder is used again or if some folders have been removed and same
    # plotting directory is used again. Edge cases.
    except FileExistsError:
        raise
        # print("Earlier decrepit directories found. Deleting decrepit result-plots folder in plots and remaking.")
        # shutil.rmtree(Path(f"{plotting_directory}"))

    return plotting_directory


# def read_and_verify_spatial_data(
#     spatial_data_filepath: Path, layer_name=""
# ) -> gpd.GeoDataFrame:
#     read_file_kwargs = (
#         {"filename": spatial_data_filepath, "layer": layer_name}
#         if layer_name != ""
#         else {"filename": spatial_data_filepath}
#     )
#     gdf = gpd.read_file(**read_file_kwargs)
#     original_length = len(gdf)
#     # Check that no geometry is NaN, that it is valid
#     # and that it doesn’t crosscut itself
#     gdf = gdf.loc[~gdf.geometry.isna() & gdf.geometry.is_valid & gdf.geometry.is_simple]
#     if isinstance(gdf.geom_type.iloc[0], shapely.geometry.LineString) or isinstance(
#         gdf.geom_type.iloc[0], shapely.geometry.MultiLineString
#     ):
#         # Trace or branch data
#         # -> Check that no geometry forms a ring
#         gdf = gdf.loc[~gdf.geometry.is_ring]
#     print(
#         f"{spatial_data_filepath}\nInvalid data percent of data: {(original_length - len(gdf)) / original_length} %"
#     )
#     return gdf


def initialize_report_df() -> pd.DataFrame:
    report_df = pd.DataFrame(columns=ReportDfCols.cols)
    return report_df


def setup_powerlaw_axlims(ax: matplotlib.axes.Axes, lineframe_main, powerlaw_cut_off):
    # :type ax: matplotlib.axes.Axes
    # TODO: Very inefficient
    lineframe_main = lineframe_main.copy()
    lineframe_main = lineframe_main.loc[lineframe_main["length"] > powerlaw_cut_off]
    lineframe_main["y"] = lineframe_main["y"] / lineframe_main["y"].max()
    left, right = calc_xlims(lineframe_main)
    top, bottom = calc_ylims(lineframe_main)
    left = left * 5
    right = right / 5
    bottom = bottom * 5
    top = top / 5
    try:
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
    except ValueError:
        # Don't try setting if it errors
        pass
