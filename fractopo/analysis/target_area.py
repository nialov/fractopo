"""
Handles a single target area or a single grouped area. Does not discriminate between a single target area and
grouped target areas.
"""

# Python Windows co-operation imports
from pathlib import Path
from textwrap import wrap

import geopandas as gpd
import matplotlib.patches as patches

# Math and analysis imports
# Plotting imports
# DataFrame analysis imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ternary
import powerlaw

from scipy.interpolate import CubicSpline

# Own code imports
import fractopo.analysis.tools as tools
import fractopo.analysis.config as config

from fractopo.analysis.config import POWERLAW, LOGNORMAL, EXPONENTIAL


# Classes


class TargetAreaLines:
    """
    Class for a target area or a group of target areas that consists of lines (traces or branches) and interpretation
    areas for these lines. Additionally contains cut offs and calculates parameters for analysis and plotting,
    information whether the lines are traces or branches and the name for the target area.
    The name parameter acts as the unique indicator to differentiate these objects. Duplicate target area or group names
    are therefore not allowed.
    """

    def __init__(
        self, lineframe, areaframe, name, group, using_branches: bool, cut_off_length=0
    ):
        """
        Init of TargetAreaLines.

        :param lineframe: GeoDataFrame containing line data
        :type lineframe: gpd.GeoDataFrame
        :param areaframe: GeoDataFrame containing area data
        :type areaframe: gpd.GeoDataFrame
        :param name: Name of the target area (user input)
        :type name: str
        :param group: Group of the target area
        :type group: str
        :param using_branches: Branches or traces
        :type using_branches: bool
        :param cut_off_length: Cut-off value in meters. Default 1.0 meters.
        :type cut_off_length: float
        """

        self.lineframe = lineframe
        self.areaframe = areaframe
        self.group = group
        self.name = name
        self.cut_off_length = cut_off_length

        # Get area using geopandas
        if isinstance(lineframe, gpd.GeoDataFrame):
            self.area = sum([polygon.area for polygon in self.areaframe.geometry])
        else:
            try:
                self.area = self.areaframe["Shape_Area"].sum()
            except KeyError:
                self.area = self.areaframe["SHAPE_Area"].sum()

        self.using_branches = using_branches

        # Get line length using geopandas
        if isinstance(lineframe, gpd.GeoDataFrame):
            self.lineframe["length"] = lineframe.geometry.length
        else:
            try:
                self.lineframe["length"] = (
                    lineframe["Shape_Leng"].astype(str).astype(float)
                )
            except KeyError:
                self.lineframe["length"] = (
                    lineframe["SHAPE_Leng"].astype(str).astype(float)
                )

        # Assign None to later initialized attributes
        # TODO: Add DF columns here?
        self.lineframe_main = gpd.GeoDataFrame()
        self.two_halves = None
        self.left, self.right, self.bottom, self.top = None, None, None, None
        self.lineframe_main_cut = None
        self.set_list = None
        self.setframes = None
        self.setframes_cut = None
        self.anisotropy = None
        self.two_halves_non_weighted = None
        self.two_halves_weighted = None
        self.bin_locs = None
        self.number_of_azimuths = None
        self.bin_width = None
        self.set_df = None
        self.length_set_df = None

    def calc_attributes(self):
        """
        Calculates important attributes of target area.

        """
        self.lineframe_main = self.lineframe.sort_values(by=["length"], ascending=False)
        self.lineframe_main = tools.calc_y_distribution(self.lineframe_main, self.area)

        self.lineframe_main["azimu"] = self.lineframe_main.geometry.apply(
            tools.calc_azimu
        )
        # TODO: TEST AND UNDERSTAND WHY THERE ARE nan in AZIMU CALC. LineString errors?
        self.lineframe_main = self.lineframe_main.dropna(subset=["azimu"])
        self.lineframe_main["halved"] = self.lineframe_main.azimu.apply(
            tools.azimu_half
        )

        # Allows plotting of azimuth
        # self.two_halves_non_weighted = tools.azimuth_plot_attributes(self.lineframe_main, weights=False)
        # self.two_halves_weighted = tools.azimuth_plot_attributes(self.lineframe_main, weights=True)
        # Current azimuth plotting
        (
            self.bin_width,
            self.bin_locs,
            self.number_of_azimuths,
        ) = tools.azimuth_plot_attributes_experimental(
            lineframe=self.lineframe_main, weights=True
        )
        # Length distribution plot limits
        self.left, self.right = tools.calc_xlims(self.lineframe_main)
        self.top, self.bottom = tools.calc_ylims(self.lineframe_main)

        self.lineframe_main_cut = self.lineframe_main.loc[
            self.lineframe_main["length"] >= self.cut_off_length
        ]

    def define_sets(self, set_df, for_length: bool):
        """
        Categorizes both non-cut and cut DataFrames with set limits.

        :param set_df: DataFrame with set limits and set names.
        :type set_df: pd.DataFrame
        """

        if for_length:
            self.lineframe_main["length_set"] = self.lineframe_main.apply(
                lambda x: tools.define_length_set(x["length"], set_df), axis=1
            )
            self.lineframe_main_cut["length_set"] = self.lineframe_main_cut.apply(
                lambda x: tools.define_length_set(x["length"], set_df), axis=1
            )
            self.length_set_df = set_df
            self.set_list = set_df.LengthSet.tolist()
        else:
            self.lineframe_main["set"] = self.lineframe_main.apply(
                lambda x: tools.define_set(x["halved"], set_df), axis=1
            )
            self.lineframe_main_cut["set"] = self.lineframe_main_cut.apply(
                lambda x: tools.define_set(x["halved"], set_df), axis=1
            )
            set_list = set_df.Set.tolist()
            self.set_df = set_df
            self.set_list = set_list

    def calc_curviness(self):
        self.lineframe_main["curviness"] = self.lineframe_main.geometry.apply(
            tools.curviness
        )

    @staticmethod
    def plot_length_distribution(
        lineframe_main,
        name: str,
        using_branches: bool,
        unified: bool,
        color_for_plot="black",
        save=False,
        savefolder="",
        fit=None,
        fit_distribution="",
    ):
        """
        Plots a length distribution to its own figure.

        :param unified: Is data from target area or grouped data?
        :type unified: bool
        :param color_for_plot: Color for scatter plot points.
        :type color_for_plot: str or tuple
        :param save: Whether to save
        :type save: bool
        :param savefolder: Folder to save to
        :type savefolder: str
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        normalize_for_power_law = True if fit is not None else False
        xmin = 0 if fit is None else fit.xmin
        # scatter plot with original length data
        TargetAreaLines.plot_length_distribution_ax(
            lineframe_main,
            name,
            ax,
            color_for_plot=color_for_plot,
            normalize_for_powerlaw=normalize_for_power_law,
            powerlaw_cut_off=xmin,
        )
        if fit is not None:
            # Plot the given fit_distribution if it exists
            TargetAreaLines.plot_length_distribution_fit(fit, fit_distribution, ax)
        tools.setup_ax_for_ld(ax, using_branches, indiv_fit=fit is not None)
        tools.setup_powerlaw_axlims(ax, lineframe_main, powerlaw_cut_off=xmin)

        if save:
            group = "group" if unified else "area"
            cut_off = "manual" if fit.fixed_xmin else "automatic"
            savename = Path(
                savefolder + f"/{name}_{group}_indiv_{cut_off}_{fit_distribution}.svg"
            )
            plt.savefig(savename, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_length_distribution_with_all_fits(
        lineframe_main,
        name: str,
        using_branches: bool,
        unified: bool,
        color_for_plot="black",
        save=False,
        savefolder="",
        fit=None,
        fit_distributions=[],
    ):
        """
        Plots a length distribution to its own figure along with powerlaw, lognormal and exponential fits.

        :param unified: Is data from target area or grouped data?
        :type unified: bool
        :param color_for_plot: Color for scatter plot points.
        :type color_for_plot: str or tuple
        :param save: Whether to save
        :type save: bool
        :param savefolder: Folder to save to
        :type savefolder: str
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        TargetAreaLines.plot_length_distribution_ax(
            lineframe_main,
            name,
            ax,
            color_for_plot=color_for_plot,
            normalize_for_powerlaw=True,
            powerlaw_cut_off=fit.power_law.xmin,
        )
        if not len(fit_distributions) == 0:
            # Plot the given fit_distributions along with the scatter plot with original length data.
            [
                TargetAreaLines.plot_length_distribution_fit(fit, fit_distribution, ax)
                for fit_distribution in fit_distributions
            ]
        tools.setup_ax_for_ld(ax, using_branches, indiv_fit=len(fit_distributions) != 0)
        tools.setup_powerlaw_axlims(
            ax, lineframe_main, powerlaw_cut_off=fit.power_law.xmin
        )
        if save:
            group = "group" if unified else "area"
            cut_off = "manual" if fit.fixed_xmin else "automatic"
            savename = Path(
                savefolder + f"/{name}_{group}_indiv_{cut_off}_all_fits.svg"
            )
            plt.savefig(savename, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_length_distribution_fit(
        fit: powerlaw.Fit, fit_distribution: str, ax: plt.Axes
    ):
        assert isinstance(fit, powerlaw.Fit)
        if fit_distribution == POWERLAW:
            fit.power_law.plot_ccdf(
                ax=ax, label="Power law", linestyle="--", color="red"
            )
        elif fit_distribution == LOGNORMAL:
            fit.lognormal.plot_ccdf(
                ax=ax, label="Lognormal", linestyle="--", color="lime"
            )
        elif fit_distribution == EXPONENTIAL:
            fit.exponential.plot_ccdf(
                ax=ax, label="Exponential", linestyle="--", color="blue"
            )
        else:
            raise ValueError(
                "Given fit_distribution does not fit a distribution string.\n"
                f"fit_distribution: {fit_distribution}"
            )

    @staticmethod
    def plot_length_distribution_ax(
        lineframe,
        name,
        ax,
        color_for_plot="black",
        normalize_for_powerlaw=False,
        powerlaw_cut_off=0,
    ):
        """
        Plots a length distribution to a given ax.
        """

        # Plot length distribution
        # Transformed to pandas DataFrame to allow direct plotting through df class method.
        lineframe = pd.DataFrame(lineframe)
        if normalize_for_powerlaw:
            lineframe = lineframe.loc[lineframe["length"] > powerlaw_cut_off]
            lineframe["y"] = lineframe["y"] / lineframe["y"].max()
        lineframe.plot.scatter(
            x="length",
            y="y",
            s=50,
            logx=True,
            logy=True,
            label=name,
            ax=ax,
            color=color_for_plot,
        )

    # def plot_curviness(self, cut_data=False):
    #     fig = plt.figure()
    #     ax = plt.gca()
    #     if ~cut_data:
    #         lineframe = self.lineframe_main
    #     else:
    #         lineframe = self.lineframe_main_cut
    #     name = self.name
    #     lineframe['set'] = lineframe.set.astype('category')
    #     sns.boxplot(data=lineframe, x='curviness', y='set', notch=True, ax=ax)
    #     ax.set_title(name, fontsize=14, fontweight='heavy', fontfamily='Times New Roman')
    #     ax.set_ylabel('Set (°)', fontfamily='Times New Roman', style='italic')
    #     ax.set_xlabel('Curvature (°)', fontfamily='Times New Roman', style='italic')
    #     ax.grid(True, color='k', linewidth=0.3)
    #     plt.savefig(Path('plots/curviness/{}.png'.format(name)), dpi=100)
    #
    # def plot_curviness_violins(self, cut_data=False):
    #     fig = plt.figure()
    #     ax = plt.gca()
    #     if ~cut_data:
    #         lineframe = self.lineframe_main
    #     else:
    #         lineframe = self.lineframe_main_cut
    #     name = self.name
    #     lineframe['set'] = lineframe.set.astype('category')
    #     sns.violinplot(data=lineframe, x='curviness', y='set', ax=ax)
    #     ax.set_title(name, fontsize=14, fontweight='heavy', fontfamily='Times New Roman')
    #     ax.set_ylabel('Set', fontfamily='Times New Roman', style='italic')
    #     ax.set_xlabel('Curvature (°)', fontfamily='Times New Roman', style='italic')
    #     ax.grid(True, color='k', linewidth=0.3)
    #     ax.set_xlim(left=0)
    #     plt.savefig(Path('plots/curviness/{}_violin.png'.format(name)), dpi=100)

    # def create_setframes(self):
    #     sets = self.lineframe_main.set.unique()
    #     sets.sort()
    #
    #     setframes = []
    #     setframes_cut = []
    #     for s in sets:
    #         setframe = self.lineframe_main.loc[self.lineframe_main.set == s]
    #         setframe_cut = self.lineframe_main_cut.loc[self.lineframe_main_cut.set == s]
    #         setframe = tools.calc_y_distribution(setframe, self.area)
    #         setframe_cut = tools.calc_y_distribution(setframe_cut, self.area)
    #         setframes.append(setframe)
    #         setframes_cut.append(setframe_cut)
    #     self.setframes = setframes
    #     self.setframes_cut = setframes_cut

    # def plot_azimuth(self, rose_type, save=False, savefolder='', branches=False, big_plots=False
    #                  , ax=None, ax_w=None):
    #     """
    #     Plot azimuth to either ax or to its own figure,
    #     in which case both non-weighted and weighted versions area made.
    #
    #     :param rose_type: Whether to plot equal-radius or equal-area rose plot e.g. 'equal-radius' or 'equal-area'
    #     :type rose_type: str
    #     :param save: Whether to save
    #     :type save: bool
    #     :param savefolder: Folder to save to
    #     :type savefolder: str
    #     :param branches: Branches or traces
    #     :type branches: bool
    #     :param big_plots: Plotting to a big plot or to an individual one
    #     :type big_plots: bool
    #     :param ax: Ax to plot to
    #     :type ax: matplotlib.projections.polar.PolarAxes
    #     :param ax_w: Weighted azimuth ax to plot to
    #     :type ax_w: matplotlib.projections.polar.PolarAxes
    #     """
    #     if big_plots:
    #
    #         self.plot_azimuth_ax(ax=ax, name=self.name, weights=False, rose_type=rose_type, font_multiplier=0.5)
    #         self.plot_azimuth_ax(ax=ax_w, name=self.name, weights=True, rose_type=rose_type, font_multiplier=0.5)
    #
    #     else:
    #         # Non-weighted
    #         fig, ax = plt.subplots(subplot_kw=dict(polar=True), constrained_layout=True, figsize=(6.5, 5.1))
    #         self.plot_azimuth_ax(ax=ax, name=self.name, weights=False, rose_type=rose_type, font_multiplier=1)
    #
    #         if save:
    #             if branches:
    #                 savename = Path(savefolder + '/{}_azimuth_branches.png'.format(self.name))
    #             else:
    #                 savename = Path(savefolder + '/{}_azimuth_traces.png'.format(self.name))
    #             plt.savefig(savename, dpi=150)
    #         # Weighted
    #         fig, ax_w = plt.subplots(subplot_kw=dict(polar=True), constrained_layout=True, figsize=(6.5, 5.1))
    #         self.plot_azimuth_ax(ax=ax_w, name=self.name, weights=True, rose_type=rose_type, font_multiplier=1)
    #         if save:
    #             if branches:
    #                 savename = Path(savefolder + '/{}_azimuth_branches_WEIGHTED.png'.format(self.name))
    #             else:
    #                 savename = Path(savefolder + '/{}_azimuth_traces_WEIGHTED.png'.format(self.name))
    #             plt.savefig(savename, dpi=150)
    #
    # def plot_azimuth_ax(self, ax, name, weights, rose_type, font_multiplier=1.0):
    #     """
    #     Plot azimuth to ax. Text size can be changed with a multiplier.
    #
    #     :param ax: Polar axis to plot to.
    #     :type ax: matplotlib.projections.polar.PolarAxes
    #     :param name: Name used
    #     :type name: str
    #     :param weights: Whether to weighted or not
    #     :type weights: bool
    #     :param rose_type: Whether to plot equal-radius or equal-area rose plot e.g. 'equal-radius' or 'equal-area'
    #     :type rose_type: str
    #     :param font_multiplier: Multiplier for font sizes. Optional, 1.0 is default
    #     :type font_multiplier: float
    #     """
    #
    #     if weights:
    #         if rose_type == 'equal-radius':
    #             two_halves = self.two_halves_weighted
    #         elif rose_type == 'equal-area':
    #             two_halves = np.sqrt(self.two_halves_weighted)
    #         else:
    #             raise ValueError('Unknown weighted rose type')
    #     else:
    #         if rose_type == 'equal-radius':
    #             two_halves = self.two_halves_non_weighted
    #         elif rose_type == 'equal-area':
    #             two_halves = np.sqrt(self.two_halves_non_weighted)
    #         else:
    #             raise Exception('Unknown non-weighted rose type')
    #         # two_halves = self.two_halves_non_weighted
    #     # Plot azimuth rose plot
    #     ax.bar(np.deg2rad(np.arange(0, 360, 10)), two_halves, width=np.deg2rad(10), bottom=0.0, color='#F7CECC',
    #            edgecolor='r', alpha=0.85, zorder=4)
    #
    #     # Plot setup
    #     ax.set_theta_zero_location('N')
    #     ax.set_theta_direction(-1)
    #     ax.set_thetagrids(np.arange(0, 360, 45), fontweight='bold')
    #     ax.set_rgrids(np.linspace(5, 10, num=2), angle=0, weight='black', fmt='%d%%', fontsize=7)
    #     ax.grid(linewidth=1, color='k')
    #
    #     title_props = dict(boxstyle='square', facecolor='white', path_effects=[path_effects.withSimplePatchShadow()])
    #     font_size = 20
    #     title_x = 0.18
    #     title_y = 1.25
    #     if weights:
    #         ax.set_title(name + '\nWEIGHTED', x=title_x, y=title_y, fontsize=font_multiplier * font_size,
    #                      fontweight='heavy'
    #                      , fontfamily='Times New Roman', bbox=title_props, va='top')
    #     else:
    #         ax.set_title(name, x=title_x, y=title_y, fontsize=font_multiplier * font_size, fontweight='heavy'
    #                      , fontfamily='Times New Roman', bbox=title_props, va='center')
    #     props = dict(boxstyle='round', facecolor='wheat', path_effects=[path_effects.withSimplePatchShadow()])
    #
    #     text_x = 0.55
    #     text_y = 1.42
    #     text = 'n = ' + str(len(self.lineframe_main)) + '\n'
    #     text = text + tools.create_azimuth_set_text(self.lineframe_main)
    #     ax.text(text_x, text_y, text, transform=ax.transAxes, fontsize=font_multiplier * 20, weight='roman'
    #             , bbox=props, fontfamily='Times New Roman', va='top')
    #     # TickLabels
    #     labels = ax.get_xticklabels()
    #     for label in labels:
    #         label._y = -0.05
    #         label._fontproperties._size = 24
    #         label._fontproperties._weight = 'bold'
    #
    # def plot_azimuth_exp(self, rose_type, save=False, savefolder='', branches=False, big_plots=False
    #                      , ax=None, ax_w=None):
    #     """
    #
    #     :param rose_type: Whether to plot equal-radius or equal-area rose plot: 'equal-radius' or 'equal-area'
    #     :type rose_type: str
    #     :param save: Whether to save
    #     :type save: bool
    #     :param savefolder: Folder to save to
    #     :type savefolder: str
    #     :param branches: Branches or traces
    #     :type branches: bool
    #     :param big_plots: Plotting to a big plot or to an individual one
    #     :type big_plots: bool
    #     :param ax: Ax to plot to
    #     :type ax: matplotlib.projections.polar.PolarAxes
    #     :param ax_w: Weighted azimuth ax to plot to
    #     :type ax_w: matplotlib.projections.polar.PolarAxes
    #     """
    #     if big_plots:
    #
    #         self.plot_azimuth_ax_exp(ax=ax, name=self.name, weights=False, rose_type=rose_type, font_multiplier=0.5)
    #         self.plot_azimuth_ax_exp(ax=ax_w, name=self.name, weights=True, rose_type=rose_type, font_multiplier=0.5)
    #
    #     else:
    #         # Non-weighted
    #         fig, ax = plt.subplots(subplot_kw=dict(polar=True), constrained_layout=True, figsize=(6.5, 5.1))
    #         self.plot_azimuth_ax_exp(ax=ax, name=self.name, weights=False, rose_type=rose_type, font_multiplier=1)
    #
    #         if save:
    #             if branches:
    #                 savename = Path(savefolder + '/{}_exp_azimuth_branches.png'.format(self.name))
    #             else:
    #                 savename = Path(savefolder + '/{}_exp_azimuth_traces.png'.format(self.name))
    #             plt.savefig(savename, dpi=150)
    #         # Weighted
    #         fig, ax_w = plt.subplots(subplot_kw=dict(polar=True), constrained_layout=True, figsize=(6.5, 5.1))
    #         self.plot_azimuth_ax_exp(ax=ax_w, name=self.name, weights=True, rose_type=rose_type, font_multiplier=1)
    #         if save:
    #             if branches:
    #                 savename = Path(savefolder + '/{}_exp_azimuth_branches_WEIGHTED.png'.format(self.name))
    #             else:
    #                 savename = Path(savefolder + '/{}_exp_azimuth_traces_WEIGHTED.png'.format(self.name))
    #             plt.savefig(savename, dpi=150)
    #
    # def plot_azimuth_ax_exp(self, ax, name, weights, rose_type, font_multiplier=1.0):
    #     """
    #     EXPERIMENTAL
    #
    #     :param ax: Polar axis to plot to.
    #     :type ax: matplotlib.projections.polar.PolarAxes
    #     :param name: Name used
    #     :type name: str
    #     :param weights: Whether to weight or not
    #     :type weights: bool
    #     :param rose_type: Whether to plot equal-radius or equal-area rose plot e.g. 'equal-radius' or 'equal-area'
    #     :type rose_type: str
    #     :param font_multiplier: Multiplier for font sizes. Optional, 1.0 is default
    #     :type font_multiplier: float
    #     """
    #
    #     if rose_type == 'equal-radius':
    #         number_of_azimuths = self.number_of_azimuths
    #     elif rose_type == 'equal-area':
    #         number_of_azimuths = np.sqrt(self.number_of_azimuths)
    #     else:
    #         raise Exception('Unknown weighted rose type')
    #     # if weights:
    #     #     if rose_type == 'equal-radius':
    #     #         two_halves = self.two_halves_weighted
    #     #     elif rose_type == 'equal-area':
    #     #         two_halves = np.sqrt(self.two_halves_weighted)
    #     #     else:
    #     #         raise Exception('Unknown weighted rose type')
    #     # else:
    #     #     if rose_type == 'equal-radius':
    #     #         two_halves = self.two_halves_non_weighted
    #     #     elif rose_type == 'equal-area':
    #     #         two_halves = np.sqrt(self.two_halves_non_weighted)
    #     #     else:
    #     #         raise Exception('Unknown non-weighted rose type')
    #     # two_halves = self.two_halves_non_weighted
    #
    #     # Plot azimuth rose plot
    #     ax.bar(np.deg2rad(self.bin_locs), number_of_azimuths, width=np.deg2rad(self.bin_width), bottom=0.0,
    #            color='#F7CECC',
    #            edgecolor='r', alpha=0.85, zorder=4)
    #
    #     # Plot setup
    #     ax.set_theta_zero_location('N')
    #     ax.set_theta_direction(-1)
    #     ax.set_thetagrids(np.arange(0, 181, 45), fontweight='bold')
    #     ax.set_thetamin(0)
    #     ax.set_thetamax(180)
    #     ax.set_rgrids(np.linspace(np.sqrt(number_of_azimuths).mean(), np.sqrt(number_of_azimuths).max() * 1.05, num=2),
    #                   angle=0, weight='black', fmt='%d', fontsize=7)
    #     ax.grid(linewidth=1, color='k')
    #
    #     title_props = dict(boxstyle='square', facecolor='white', path_effects=[path_effects.withSimplePatchShadow()])
    #     font_size = 20
    #     title_x = 0.18
    #     title_y = 1.25
    #     if weights:
    #         ax.set_title(name + '\nWEIGHTED', x=title_x, y=title_y, fontsize=font_multiplier * font_size,
    #                      fontweight='heavy'
    #                      , fontfamily='Times New Roman', bbox=title_props, va='top')
    #     else:
    #         ax.set_title(name, x=title_x, y=title_y, fontsize=font_multiplier * font_size, fontweight='heavy'
    #                      , fontfamily='Times New Roman', bbox=title_props, va='center')
    #     props = dict(boxstyle='round', facecolor='wheat', path_effects=[path_effects.withSimplePatchShadow()])
    #
    #     text_x = 0.55
    #     text_y = 1.42
    #     text = 'n = ' + str(len(self.lineframe_main)) + '\n'
    #     text = text + tools.create_azimuth_set_text(self.lineframe_main)
    #     ax.text(text_x, text_y, text, transform=ax.transAxes, fontsize=font_multiplier * 9, weight='roman'
    #             , bbox=props, fontfamily='Times New Roman', va='top')
    #     # TickLabels
    #     labels = ax.get_xticklabels()
    #     for label in labels:
    #         label._y = -0.05
    #         label._fontproperties._size = 24
    #         label._fontproperties._weight = 'bold'

    def plot_azimuth_weighted(self, rose_type, set_visualization):
        """
        Plot weighted azimuth rose-plot. Type can be 'equal-radius' or 'equal-area'.

        :param rose_type: Whether to plot equal-radius or equal-area rose plot: 'equal-radius' or 'equal-area'
        :type rose_type: str
        :param set_visualization: Whether to visualize sets into the same plot
        :type set_visualization: bool
        """
        fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(6.5, 5.1))
        self.plot_azimuth_ax_weighted(
            set_visualization=set_visualization,
            ax=ax,
            name=self.name,
            rose_type=rose_type,
        )

    def plot_azimuth_ax_weighted(self, set_visualization, ax, name, rose_type):
        """
        Plot weighted azimuth rose-plot to given ax. Type can be 'equal-radius' or 'equal-area'.

        :param set_visualization: Whether to visualize sets into the same plot
        :type set_visualization: bool
        :param ax: Polar axis to plot on.
        :type ax: matplotlib.projections.polar.PolarAxes
        :param name: Name of the target area or group
        :type name: str
        :param rose_type: Type can be 'equal-radius' or 'equal-area'
        :type rose_type: str
        :raise ValueError: When given invalid rose_type string. Valid: 'equal-radius' or 'equal-area'
        """

        if rose_type == "equal-radius":
            number_of_azimuths = self.number_of_azimuths
        elif rose_type == "equal-area":
            number_of_azimuths = np.sqrt(self.number_of_azimuths)
        else:
            raise ValueError(f"Unknown weighted rose type string: {rose_type}")

        # Plot azimuth rose plot
        ax.bar(
            np.deg2rad(self.bin_locs),
            number_of_azimuths,
            width=np.deg2rad(self.bin_width),
            bottom=0.0,
            color="darkgrey",
            edgecolor="k",
            alpha=0.85,
            zorder=4,
        )

        # Plot setup
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids(
            np.arange(0, 181, 45),
            fontweight="bold",
            fontfamily="Calibri",
            fontsize=11,
            alpha=0.95,
        )
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        # The average of number_of_azimuths is displayed as a radial grid-line.
        rlines, _ = ax.set_rgrids(
            radii=[number_of_azimuths.mean()],
            angle=0,
            fmt="",
            fontsize=1,
            alpha=0.8,
            ha="left",
        )
        if isinstance(rlines, list):
            rline: plt.Line2D
            for rline in rlines:
                rline.set_linestyle("dashed")

        ax.grid(linewidth=1, color="k", alpha=0.8)

        # Title is the name of the target area or group
        prop_title = dict(boxstyle="square", facecolor="linen", alpha=1, linewidth=2)
        title = "\n".join(wrap(f"{name}", 10))
        ax.set_title(
            title,
            x=0.94,
            y=0.8,
            fontsize=20,
            fontweight="bold",
            fontfamily="Calibri",
            va="top",
            bbox=prop_title,
            transform=ax.transAxes,
            ha="center",
        )

        # Fractions of length for each set in a separate box
        prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)
        text = "n = " + str(len(self.lineframe_main)) + "\n"
        text = text + tools.create_azimuth_set_text(self.lineframe_main)
        ax.text(
            0.94,
            0.3,
            text,
            transform=ax.transAxes,
            fontsize=12,
            weight="roman",
            bbox=prop,
            fontfamily="Calibri",
            va="top",
            ha="center",
        )

        # Tick labels
        labels = ax.get_xticklabels()
        for label in labels:
            label._y = -0.01
            label._fontproperties._size = 15
            label._fontproperties._weight = "bold"
        # Set ranges visualized if set_visualization is True
        if set_visualization:
            for _, row in self.set_df.iterrows():
                set_range = row.SetLimits
                if set_range[0] < set_range[1]:
                    diff = set_range[1] - set_range[0]
                    set_loc = set_range[0] + diff / 2
                else:
                    diff = 360 - set_range[0] + set_range[1]
                    if 180 - set_range[0] > set_range[1]:
                        set_loc = set_range[0] + diff / 2
                    else:
                        set_loc = set_range[1] - diff / 2

                ax.bar(
                    [np.deg2rad(set_loc)],
                    [number_of_azimuths.mean()],
                    width=np.deg2rad(diff),
                    bottom=0.0,
                    alpha=0.5,
                    label=f"Set {row.Set}",
                )
                ax.legend(
                    loc=(-0.02, 0.41),
                    edgecolor="black",
                    prop={"family": "Calibri", "size": 12},
                )

    def topology_parameters_2d_branches(self, branches=False):
        """
        Gather topology parameters for branch data.

        :param branches: Branches or traces
        :type branches: bool
        :raise AttributeError: When given vector layer doesn't contain valid column names.
            e.g. (Connection: ['C - C', 'C - I', ...]
        """
        # SAME METHOD FOR BOTH TRACES AND BRANCHES.
        # MAKE SURE YOU KNOW WHICH YOU ARE USING.
        fracture_intensity = self.lineframe_main.length.sum() / self.area
        aerial_frequency = len(self.lineframe_main) / self.area
        characteristic_length = self.lineframe_main.length.mean()
        if not np.isclose(
            [characteristic_length], [np.mean(self.lineframe_main.geometry.length)]
        ):
            raise ValueError(
                "Lengths from self.lineframe_main.length.mean()"
                "and np.mean(self.lineframe_main.geometry.length)"
                "are not close.\n"
                f"{self.lineframe_main.length.mean()}"
                f"and {np.mean(self.lineframe_main.geometry.length)}"
            )
        dimensionless_intensity = fracture_intensity * characteristic_length
        number_of_lines = len(self.lineframe_main)
        if branches:
            try:
                connection_dict = (
                    self.lineframe_main.Connection.value_counts().to_dict()
                )
            except AttributeError as e:
                raise e("Given vector layer doesn't contain valid column names.")
            # TODO: This should only be done once -> connection_dict as attribute?
            for connection_type in ("C - C", "C - I", "I - I"):
                if connection_type not in [key for key in connection_dict]:
                    connection_dict[connection_type] = 0

            return (
                fracture_intensity,
                aerial_frequency,
                characteristic_length,
                dimensionless_intensity,
                number_of_lines,
                connection_dict,
            )
        else:
            return (
                fracture_intensity,
                aerial_frequency,
                characteristic_length,
                dimensionless_intensity,
                number_of_lines,
            )

    def plot_branch_ternary_plot(
        self, unified: bool, color_for_plot="black", save=False, savefolder=""
    ):
        """
        Plot a branch classification ternary plot to a new ternary figure. Single point in each figure.

        :param unified: Plot for target area or grouped data
        :type unified: bool
        :param color_for_plot: Color for point in plot.
        :type color_for_plot: str or tuple
        :param save: Save or not
        :type save: bool
        :param savefolder: Folder to save plot to
        :type savefolder: str
        """
        connection_dict = self.lineframe_main.Connection.value_counts().to_dict()
        # TODO: This should only be done once -> connection_dict as attribute?
        for connection_type in ("C - C", "C - I", "I - I"):
            if connection_type not in [key for key in connection_dict]:
                connection_dict[connection_type] = 0
        cc = connection_dict["C - C"]
        ci = connection_dict["C - I"]
        ii = connection_dict["I - I"]
        sumcount = cc + ci + ii
        ccp = 100 * cc / sumcount
        cip = 100 * ci / sumcount
        iip = 100 * ii / sumcount

        point = [(ccp, iip, cip)]

        fig, ax = plt.subplots(figsize=(6.5, 5.1))
        scale = 100
        fig, tax = ternary.figure(ax=ax, scale=scale)
        text = (
            "n: "
            + str(len(self.lineframe_main))
            + "\nC-C branches: "
            + str(cc)
            + "\nC-I branches: "
            + str(ci)
            + "\nI-I branches: "
            + str(ii)
        )
        prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)
        ax.text(
            0.86,
            1.05,
            text,
            transform=ax.transAxes,
            fontsize="medium",
            weight="roman",
            verticalalignment="top",
            bbox=prop,
            fontfamily="Calibri",
            ha="center",
        )
        tools.initialize_ternary_branches_points(ax, tax)
        # tax.scatter(point, marker='X', color='black', alpha=1, zorder=3, s=210)

        tax.scatter(
            point,
            marker="X",
            label=self.name,
            alpha=1,
            zorder=4,
            s=125,
            color=color_for_plot,
        )
        tools.tern_plot_branch_lines(tax)
        tax.legend(
            loc="upper center",
            bbox_to_anchor=(0.1, 1.05),
            prop={"family": "Calibri", "weight": "heavy", "size": "x-large"},
            edgecolor="black",
            ncol=2,
            columnspacing=0.7,
            shadow=True,
        )
        if save:
            if unified:
                savename = Path(
                    savefolder + f"/indiv/{self.name}_group_branch_point.svg"
                )
            else:
                savename = Path(
                    savefolder + f"/indiv/{self.name}_area_branch_point.svg"
                )
            plt.savefig(savename, dpi=150, bbox_inches="tight")
            plt.close()

    def plot_branch_ternary_point(self, tax, color_for_plot="black"):

        """
        Plot a branch classification ternary scatter point to a given tax.

        :param tax: python-ternary AxesSubPlot
        :type tax: ternary.TernaryAxesSubplot
        :param color_for_plot: Color for point in plot.
        :type color_for_plot: str or tuple
        """
        connection_dict = self.lineframe_main.Connection.value_counts().to_dict()
        # TODO: This should only be done once -> connection_dict as attribute?
        for connection_type in ("C - C", "C - I", "I - I"):
            if connection_type not in [key for key in connection_dict]:
                connection_dict[connection_type] = 0
        cc = connection_dict["C - C"]
        ci = connection_dict["C - I"]
        ii = connection_dict["I - I"]
        sumcount = cc + ci + ii
        ccp = 100 * cc / sumcount
        cip = 100 * ci / sumcount
        iip = 100 * ii / sumcount

        point = [(ccp, iip, cip)]
        # tax.scatter(point, marker='X', color='black', alpha=1, zorder=3, s=210)
        tax.scatter(
            point,
            marker="X",
            label=self.name,
            alpha=1,
            zorder=4,
            s=125,
            color=color_for_plot,
        )

    @staticmethod
    def calc_anisotropy(lineframe_main):
        """
        Calculates annisotropy of connectivity for branch DataFrame

        """
        branchframe = lineframe_main

        branchframe["anisotropy"] = branchframe.apply(
            lambda row: tools.aniso_calc_anisotropy(
                row["halved"], row["Connection"], row["length"]
            ),
            axis=1,
        )
        arr_sum = branchframe.anisotropy.sum()

        return arr_sum
        # self.anisotropy_div_area = arr_sum / self.rep_circle_area

    @staticmethod
    def plot_anisotropy_styled(anisotropy, for_ax=False, ax=None):
        """
        Plots a styled anisotropy of connectivity figure.

        Spline done with:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html

        :param for_ax: Whether plotting to a ready-made ax or not
        :type for_ax: bool
        :param ax: ax to plot to (optional)
        :type ax: matplotlib.axes.Axes
        """
        double_anisotropy = np.concatenate([anisotropy, anisotropy])
        angles_of_study = config.angles_for_examination
        opp_angles = [i + 180 for i in angles_of_study]
        angles = list(angles_of_study) + opp_angles
        if for_ax:
            pass
        else:
            fig, ax = plt.subplots(subplot_kw=dict(polar=True))

        # PLOT SETUP
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        max_aniso = max(anisotropy)

        for theta, r in zip(angles, double_anisotropy):
            theta = np.deg2rad(theta)
            arrowstyle = patches.ArrowStyle.CurveB(head_length=1, head_width=0.5)
            ax.annotate(
                "",
                xy=(theta, r),
                xytext=(theta, 0),
                arrowprops=dict(
                    edgecolor="black", facecolor="seashell", arrowstyle=arrowstyle
                ),
            )
        # ax.scatter([np.deg2rad(angles_value) for angles_value in angles], double_anisotropy
        #            , marker='o', color='black', zorder=9, s=20)
        # NO AXES
        ax.axis("off")
        # CREATE CURVED STRUCTURE AROUND SCATTER AND ARROWS
        angles.append(359.999)
        double_anisotropy = np.concatenate([double_anisotropy, double_anisotropy[0:1]])
        angles = np.array(angles)

        # TODO: testing CubicSpline
        # And it works!?
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html
        theta = np.deg2rad(angles)
        cs = CubicSpline(theta, double_anisotropy, bc_type="periodic")
        xnew = np.linspace(theta.min(), theta.max(), 300)
        power_smooth = cs(xnew)
        ax.plot(xnew, power_smooth, linewidth=1.5, color="black")

        # INTERPOLATE BETWEEN CALCULATED POINTS
        # noinspection PyArgumentList
        # spl = make_interp_spline(angles, double_anisotropy, k=3)
        # xnew = np.linspace(angles.min(), angles.max(), 300)
        # power_smooth = spl(xnew)
        # power_smooth = np.concatenate([power_smooth, power_smooth[-1:] + 0.001 * power_smooth[-1]])
        # xnew = np.concatenate([xnew, xnew[0:1] + 0.5])

        # ax.plot(np.deg2rad(xnew), power_smooth, linewidth=1.5, color='black')
        circle = patches.Circle(
            (0, 0),
            0.0025 * max_aniso,
            transform=ax.transData._b,
            edgecolor="black",
            facecolor="gray",
            zorder=10,
        )
        ax.add_artist(circle)


class TargetAreaNodes:
    """
    Class for topological nodes. Used in conjunction with TargetAreaLines.

    """

    def __repr__(self):
        return f"""
                nodeframe:
                {self.nodeframe.head()}
                name:
                {self.name}
                group:
                {self.group}
                """

    def __init__(self, nodeframe, name, group):
        """
        Init of TargetAreaNodes.

        :param nodeframe: DataFrame with node data
        :type nodeframe: gpd.GeoDataFrame
        :param name: Name of the target area
        :type name: str
        :param group: Group of the target area
        :type group: str
        """
        self.nodeframe = nodeframe
        self.name = name
        self.group = group

    @staticmethod
    def plot_xyi_plot(
        nodeframe,
        name,
        unified: bool,
        color_for_plot="black",
        save=False,
        savefolder="",
    ):
        """
        Plot a XYI-node ternary plot to a new ternary figure. Single point in each figure.

        :param unified: Plot for target area or grouped data
        :type unified: bool
        :param color_for_plot: Color for point.
        :type color_for_plot: str or tuple
        :param save: Save or not
        :type save: bool
        :param savefolder: Folder to save plot to
        :type savefolder: str
        """
        xcount = len(nodeframe.loc[nodeframe["c"] == "X"])
        ycount = len(nodeframe.loc[nodeframe["c"] == "Y"])
        icount = len(nodeframe.loc[nodeframe["c"] == "I"])

        sumcount = xcount + ycount + icount

        xp = 100 * xcount / sumcount
        yp = 100 * ycount / sumcount
        ip = 100 * icount / sumcount

        point = [(xp, ip, yp)]

        # Scatter Plot
        scale = 100
        fig, ax = plt.subplots(figsize=(6.5, 5.1))
        fig, tax = ternary.figure(ax=ax, scale=scale)
        tools.initialize_ternary_points(ax, tax)
        tools.tern_plot_the_fing_lines(tax)
        text = (
            "n: "
            + str(len(nodeframe))
            + "\nX-nodes: "
            + str(xcount)
            + "\nY-nodes: "
            + str(ycount)
            + "\nI-nodes: "
            + str(icount)
        )
        prop = dict(boxstyle="square", facecolor="linen", alpha=1, pad=0.45)
        ax.text(
            0.85,
            1.05,
            text,
            transform=ax.transAxes,
            fontsize="medium",
            weight="roman",
            verticalalignment="top",
            bbox=prop,
            fontfamily="Calibri",
            ha="center",
        )

        tax.scatter(
            point, s=50, marker="o", label=name, alpha=1, zorder=4, color=color_for_plot
        )
        tax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            prop={"family": "Calibri", "weight": "heavy", "size": "x-large"},
            edgecolor="black",
            ncol=2,
            columnspacing=0.7,
            shadow=True,
        )
        if save:
            if unified:
                savename = Path(savefolder + f"/indiv/{name}_group_xyi_point.svg")
            else:
                savename = Path(savefolder + f"/indiv/{name}_area_xyi_point.svg")
            plt.savefig(savename, dpi=150, bbox_inches="tight")
            plt.close()

    @staticmethod
    def plot_xyi_point(nodeframe, name, tax, color_for_plot="black"):
        """
        Plot a XYI-node ternary scatter point to a given tax.

        :param tax: python-ternary AxesSubPlot
        :type tax: ternary.TernaryAxesSubplot
        :param color_for_plot: Color for plotting point.
        :type color_for_plot: str or tuple
        """
        # Setup
        xcount = len(nodeframe.loc[nodeframe["c"] == "X"])
        ycount = len(nodeframe.loc[nodeframe["c"] == "Y"])
        icount = len(nodeframe.loc[nodeframe["c"] == "I"])
        sumcount = xcount + ycount + icount
        xp = 100 * xcount / sumcount
        yp = 100 * ycount / sumcount
        ip = 100 * icount / sumcount
        point = [(xp, ip, yp)]
        # Plotting
        tax.scatter(
            point, marker="o", label=name, alpha=1, zorder=4, s=50, color=color_for_plot
        )

    def topology_parameters_2d_nodes(self):
        """
        Gathers topology parameters of nodes

        """
        nodeframe = self.nodeframe
        node_dict = nodeframe.c.value_counts().to_dict()
        return node_dict
