"""
Handles analysis and plotting using a MultiTargetAreaAnalysis class. Analysis (i.e. heavy calculations)
and plotting have been separated into two different methods.
"""
from fractopo.analysis import multiple_target_areas as mta
from fractopo.analysis import tools, config


class MultiTargetAreaAnalysis:
    def __init__(
        self,
        filepath_df,
        plotting_directory,
        analysis_name,
        group_names_cutoffs_df,
        set_df,
        length_set_df,
        choose_your_analyses,
        logger,
    ):
        self.set_df = set_df
        self.length_set_df = length_set_df
        self.group_names_cutoffs_df = group_names_cutoffs_df
        self.spatial_df = filepath_df
        self.plotting_directory = plotting_directory
        self.analysis_name = analysis_name
        self.analysis_traces = None
        self.analysis_branches = None
        self.logger = logger

        # Check if cross-cutting and abutting relationships can be determined
        # TODO: Also check if sets actually exist in lines...
        if len(self.set_df) < 2:
            self.determine_relationships = False
        else:
            self.determine_relationships = choose_your_analyses["Cross-cuttingAbutting"]
        if len(self.length_set_df) < 2:
            self.determine_length_relationships = False
        else:
            self.determine_length_relationships = choose_your_analyses[
                "Cross-cuttingAbutting"
            ]
        # Check which analyses to perform
        self.determine_branches = choose_your_analyses["Branches"]
        self.determine_length_distributions = choose_your_analyses[
            "LengthDistributions"
        ]
        self.determine_azimuths = choose_your_analyses["Azimuths"]
        self.determine_xyi = choose_your_analyses["XYI"]
        self.determine_branch_classification = choose_your_analyses[
            "BranchClassification"
        ]
        self.determine_topology = choose_your_analyses["Topology"]
        self.determine_anisotropy = choose_your_analyses["Anisotropy"]
        self.determine_hexbin = choose_your_analyses["Hexbin"]

    def analysis(self):
        """
        Method that runs analysis i.e. heavy calculations for trace, branch and node data.
        """
        # DEBUG
        # self.spatial_df = self.spatial_df.drop(columns=['Trace', 'Branch', 'Area', 'Node'])

        self.analysis_traces = mta.MultiTargetAreaQGIS(
            self.spatial_df,
            self.group_names_cutoffs_df,
            branches=False,
            logger=self.logger,
        )
        # Bar at 25
        if self.determine_branches:
            self.analysis_branches = mta.MultiTargetAreaQGIS(
                self.spatial_df,
                self.group_names_cutoffs_df,
                branches=True,
                logger=self.logger,
            )
        # Bar at 45
        # TRACE DATA SETUP
        self.analysis_traces.calc_attributes_for_all()
        self.analysis_traces.define_sets_for_all(self.set_df)
        self.analysis_traces.define_sets_for_all(self.length_set_df, for_length=True)

        # self.analysis_traces.calc_curviness_for_all()
        self.analysis_traces.unified()

        # TODO: Check setframes
        # self.analysis_traces.create_setframes_for_all_unified()

        if self.determine_topology:
            self.analysis_traces.gather_topology_parameters(unified=False)
            self.analysis_traces.gather_topology_parameters(unified=True)

        # Cross-cutting and abutting relationships
        if self.determine_relationships:
            self.analysis_traces.determine_crosscut_abutting_relationships(
                unified=False
            )
            self.analysis_traces.determine_crosscut_abutting_relationships(unified=True)
        # Cross-cutting and abutting relationships
        if self.determine_length_relationships:
            self.analysis_traces.determine_crosscut_abutting_relationships(
                unified=False, use_length_sets=True
            )
            self.analysis_traces.determine_crosscut_abutting_relationships(
                unified=True, use_length_sets=True
            )

        if self.determine_branches:
            # BRANCH DATA SETUP
            self.analysis_branches.calc_attributes_for_all()

            self.analysis_branches.define_sets_for_all(self.set_df)
            self.analysis_branches.define_sets_for_all(
                self.length_set_df, for_length=True
            )

            self.analysis_branches.unified()

            if self.determine_topology:
                self.analysis_branches.gather_topology_parameters(unified=False)
                self.analysis_branches.gather_topology_parameters(unified=True)

            # Anisotropy
            if self.determine_anisotropy:
                self.analysis_branches.calc_anisotropy(unified=False)
                self.analysis_branches.calc_anisotropy(unified=True)

    def plot_results(self):
        """
        Method that runs plotting based on analysis results for trace, branch and node data.
        """
        if self.determine_branches:
            # ___________________BRANCH DATA_______________________
            config.styling_plots("branches")
            # Length distributions
            if self.determine_length_distributions:
                self.analysis_branches.plot_lengths(
                    unified=False,
                    save=True,
                    savefolder=self.plotting_directory
                    + "/length_distributions/indiv/branches",
                )
                self.analysis_branches.plot_lengths(
                    unified=True,
                    save=True,
                    savefolder=self.plotting_directory
                    + "/length_distributions/branches",
                )

            # Azimuths
            if self.determine_azimuths:
                self.analysis_branches.plot_azimuths_weighted(
                    unified=False,
                    save=True,
                    savefolder=self.plotting_directory + "/azimuths/indiv",
                )
                self.analysis_branches.plot_azimuths_weighted(
                    unified=True,
                    save=True,
                    savefolder=self.plotting_directory + "/azimuths",
                )
            # XYI
            if self.determine_xyi:
                self.analysis_branches.plot_xyi_ternary(
                    unified=False,
                    save=True,
                    savefolder=self.plotting_directory + "/xyi",
                )
                self.analysis_branches.plot_xyi_ternary(
                    unified=True, save=True, savefolder=self.plotting_directory + "/xyi"
                )

            # Topo parameters
            if self.determine_topology:
                self.analysis_branches.plot_topology(
                    unified=False,
                    save=True,
                    savefolder=self.plotting_directory + "/topology/branches",
                )
                self.analysis_branches.plot_topology(
                    unified=True,
                    save=True,
                    savefolder=self.plotting_directory + "/topology/branches",
                )
            # Hexbinplots
            if self.determine_hexbin:
                self.analysis_branches.plot_hexbin_plot(
                    unified=False,
                    save=True,
                    savefolder=self.plotting_directory + "/hexbinplots",
                )
                self.analysis_branches.plot_hexbin_plot(
                    unified=True,
                    save=True,
                    savefolder=self.plotting_directory + "/hexbinplots",
                )

            # ----------------unique for branches-------------------
            # Branch Classification ternary plot
            if self.determine_branch_classification:
                self.analysis_branches.plot_branch_ternary(
                    unified=False,
                    save=True,
                    savefolder=self.plotting_directory + "/branch_class",
                )
                self.analysis_branches.plot_branch_ternary(
                    unified=True,
                    save=True,
                    savefolder=self.plotting_directory + "/branch_class",
                )

            # Anisotropy
            if self.determine_anisotropy:
                self.analysis_branches.plot_anisotropy(
                    frame=self.analysis_branches.df,
                    using_branches=True,
                    unified=False,
                    save=True,
                    savefolder=self.plotting_directory + "/anisotropy/indiv",
                )
                self.analysis_branches.plot_anisotropy(
                    frame=self.analysis_branches.uniframe,
                    using_branches=True,
                    unified=True,
                    save=True,
                    savefolder=self.plotting_directory + "/anisotropy",
                )

        # __________________TRACE DATA______________________
        config.styling_plots("traces")
        if self.determine_length_distributions:
            self.analysis_traces.plot_lengths(
                unified=False,
                save=True,
                savefolder=self.plotting_directory
                + "/length_distributions/indiv/traces",
            )
            self.analysis_traces.plot_lengths(
                unified=True,
                save=True,
                savefolder=self.plotting_directory + "/length_distributions/traces",
            )

        # Azimuths
        if self.determine_azimuths:
            self.analysis_traces.plot_azimuths_weighted(
                unified=False,
                save=True,
                savefolder=self.plotting_directory + "/azimuths/indiv",
            )
            self.analysis_traces.plot_azimuths_weighted(
                unified=True,
                save=True,
                savefolder=self.plotting_directory + "/azimuths",
            )
        # Topo parameters
        if self.determine_topology:
            self.analysis_traces.plot_topology(
                unified=False,
                save=True,
                savefolder=self.plotting_directory + "/topology/traces",
            )
            self.analysis_traces.plot_topology(
                unified=True,
                save=True,
                savefolder=self.plotting_directory + "/topology/traces",
            )
        # Hexbinplots
        if self.determine_hexbin:
            self.analysis_traces.plot_hexbin_plot(
                unified=False,
                save=True,
                savefolder=self.plotting_directory + "/hexbinplots",
            )
            self.analysis_traces.plot_hexbin_plot(
                unified=True,
                save=True,
                savefolder=self.plotting_directory + "/hexbinplots",
            )

        # ---------------unique for traces-------------------

        # Cross-cutting and abutting relationships
        # TODO: Implement Markov chain analysis?
        # Fracture overprinting history using Markov chain analysis:
        # Windsor-Kennetcook subbasin, Maritimes Basin, Canada
        if self.determine_relationships:
            self.analysis_traces.plot_crosscut_abutting_relationships(
                unified=False,
                save=True,
                savefolder=self.plotting_directory + "/age_relations/indiv",
            )
            self.analysis_traces.plot_crosscut_abutting_relationships(
                unified=True,
                save=True,
                savefolder=self.plotting_directory + "/age_relations",
            )
        if self.determine_length_relationships:
            self.analysis_traces.plot_crosscut_abutting_relationships(
                unified=False,
                save=True,
                savefolder=self.plotting_directory + "/length_age_relations/indiv",
                use_length_sets=True,
            )
            self.analysis_traces.plot_crosscut_abutting_relationships(
                unified=True,
                save=True,
                savefolder=self.plotting_directory + "/length_age_relations",
                use_length_sets=True,
            )
