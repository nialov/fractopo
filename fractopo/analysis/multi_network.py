"""
MultiNetwork implementation for handling multiple network analysis.
"""

import pandas as pd
from beartype import beartype
from beartype.typing import Dict, List, NamedTuple, Optional, Tuple, Type, Union
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fractopo.analysis import length_distributions, parameters, subsampling
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import RandomChoice
from fractopo.general import NAME, Number, SetRangeTuple


class MultiNetwork(NamedTuple):
    """
    Multiple Network analysis.
    """

    networks: Tuple[Network, ...]

    @beartype
    def __hash__(self) -> int:
        """
        Implement hashing for MultiNetwork.
        """
        return hash(tuple(self.networks))

    @beartype
    def subsample(
        self,
        min_radii: Union[float, Dict[str, float]],
        random_choice: RandomChoice = RandomChoice.radius,
        samples: int = 1,
    ) -> List[Optional[Dict[str, Union[Number, str]]]]:
        """
        Subsample all ``Networks`` in ``MultiNetwork``.

        :param min_radii: The minimum radius for
            subsamples can be either given as static or as a mapping that maps
            network names to specific radii.
        :param random_choice: Whether to use radius or area
            as the random choice parameter.
        :param samples: How many subsamples to conduct per network.
        :returns: Subsamples
        """
        return subsampling.subsample_networks(
            networks=self.networks,
            min_radii=min_radii,
            random_choice=random_choice,
            samples=samples,
        )

    @beartype
    def collective_azimuth_sets(self) -> Tuple[Tuple[str, ...], SetRangeTuple]:
        """
        Get collective azimuth set names.

        Checks that all Networks have the same azimuth sets.
        """
        # Check that set_names and set_ranges are the same across all Networks
        set_names = self.networks[0].azimuth_set_names
        set_ranges = self.networks[0].azimuth_set_ranges
        for network in self.networks:
            if (
                set_names != network.azimuth_set_names
                and set_ranges == network.azimuth_set_ranges
            ):
                raise ValueError(
                    "Expected azimuth sets and ranges to be the same in all networks."
                )
        return set_names, set_ranges

    @beartype
    def network_length_distributions(
        self, using_branches: bool, using_azimuth_sets: bool
    ) -> Dict[str, Dict[str, length_distributions.LengthDistribution]]:
        """
        Get length distributions of Networks.
        """
        # Get azimuth sets that are identical for all Networks.
        set_names, _ = self.collective_azimuth_sets()

        # Collect (set-wise) length distributions for all networks
        distributions: Dict[str, Dict[str, length_distributions.LengthDistribution]] = (
            dict()
        )

        # Iterate over the networks
        for network in self.networks:
            # Name is either the network name or if categorizing by set,
            # the set name
            # Only one LengthDistribution is collected if not using the set,
            # otherwise an amount equal to the amount of azimuth sets
            names = [network.name] if not using_azimuth_sets else set_names

            # Collect length distributions (for each set)
            network_distributions = dict()
            for name in names:
                # Label the LengthDistribution with either just network name
                # or with the name and the azimuth set
                azimuth_set = name if using_azimuth_sets else None

                # # Use network name or set name as key in second dictionary
                network_distributions[name] = (
                    network.branch_length_distribution(azimuth_set=azimuth_set)
                    # Use either trace or branch lengths
                    if using_branches
                    else network.trace_length_distribution(azimuth_set=azimuth_set)
                )

            # Use network name as key
            distributions[network.name] = network_distributions
        return distributions

    @beartype
    def multi_length_distributions(
        self, using_branches: bool
    ) -> length_distributions.MultiLengthDistribution:
        """
        Get MultiLengthDistribution of all networks.
        """
        distributions_dict = self.network_length_distributions(
            using_branches=using_branches, using_azimuth_sets=False
        )
        distributions = [
            distributions_dict[network.name][network.name] for network in self.networks
        ]

        multi_distribution = length_distributions.MultiLengthDistribution(
            distributions=distributions,
            using_branches=using_branches,
        )
        return multi_distribution

    @beartype
    def plot_multi_length_distribution(
        self,
        using_branches: bool,
        automatic_cut_offs: bool,
        plot_truncated_data: bool,
        multi_distribution: Optional[
            length_distributions.MultiLengthDistribution
        ] = None,
    ) -> Tuple[
        length_distributions.MultiLengthDistribution,
        length_distributions.Polyfit,
        Figure,
        Axes,
    ]:
        """
        Plot multi-length distribution fit.

        Use ``multi_length_distributions()`` to get most parameters used in
        fitting the multi-scale distribution.
        """
        multi_distribution = (
            self.multi_length_distributions(using_branches=using_branches)
            if multi_distribution is None
            else multi_distribution
        )
        polyfit, fig, ax = multi_distribution.plot_multi_length_distributions(
            automatic_cut_offs=automatic_cut_offs,
            plot_truncated_data=plot_truncated_data,
        )

        return multi_distribution, polyfit, fig, ax

    @beartype
    def plot_xyi(
        self,
        colors: Optional[List[Optional[str]]] = None,
    ):
        """
        Plot multi-network ternary XYI plot.
        """
        node_counts_list = [network.node_counts for network in self.networks]
        labels = [network.name for network in self.networks]
        return parameters.plot_ternary_plot(
            counts_list=node_counts_list,
            labels=labels,
            is_nodes=True,
            colors=colors,
        )

    @beartype
    def plot_branch(
        self,
        colors: Optional[List[Optional[str]]] = None,
    ):
        """
        Plot multi-network ternary branch type plot.
        """
        branch_counts_list = [network.branch_counts for network in self.networks]
        labels = [network.name for network in self.networks]
        return parameters.plot_ternary_plot(
            counts_list=branch_counts_list,
            labels=labels,
            is_nodes=False,
            colors=colors,
        )

    @beartype
    def _plot_azimuth_set_lengths(
        self,
        automatic_cut_offs: bool,
        using_branches: bool,
        plot_truncated_data: bool,
    ) -> Tuple[
        Dict[str, length_distributions.MultiLengthDistribution],
        List[length_distributions.Polyfit],
        List[Figure],
        List[Axes],
    ]:
        """
        Plot multi-network azimuths set lengths with fits.
        """
        mlds = self.set_multi_length_distributions(using_branches=using_branches)

        figs, axes, polyfits = [], [], []
        for mld in mlds.values():
            polyfit, fig, ax = mld.plot_multi_length_distributions(
                automatic_cut_offs=automatic_cut_offs,
                plot_truncated_data=plot_truncated_data,
            )
            figs.append(fig)
            axes.append(ax)
            polyfits.append(polyfit)

        return mlds, polyfits, figs, axes

    @beartype
    def plot_trace_azimuth_set_lengths(
        self,
        automatic_cut_offs: bool,
        plot_truncated_data: bool,
    ) -> Tuple[
        Dict[str, length_distributions.MultiLengthDistribution],
        List[length_distributions.Polyfit],
        List[Figure],
        List[Axes],
    ]:
        """
        Plot multi-network trace azimuths set lengths with fits.
        """
        return self._plot_azimuth_set_lengths(
            automatic_cut_offs=automatic_cut_offs,
            using_branches=False,
            plot_truncated_data=plot_truncated_data,
        )

    @beartype
    def plot_branch_azimuth_set_lengths(
        self,
        automatic_cut_offs: bool,
        plot_truncated_data: bool,
    ) -> Tuple[
        Dict[str, length_distributions.MultiLengthDistribution],
        List[length_distributions.Polyfit],
        List[Figure],
        List[Axes],
    ]:
        """
        Plot multi-network trace azimuths set lengths with fits.
        """
        return self._plot_azimuth_set_lengths(
            automatic_cut_offs=automatic_cut_offs,
            using_branches=True,
            plot_truncated_data=plot_truncated_data,
        )

    @beartype
    def set_multi_length_distributions(
        self, using_branches: bool
    ) -> Dict[str, length_distributions.MultiLengthDistribution]:
        """
        Get set-wise multi-length distributions.
        """
        distributions_dict = self.network_length_distributions(
            using_branches=using_branches, using_azimuth_sets=True
        )

        mlds = dict()
        for set_name in self.networks[0].azimuth_set_names:
            set_lengths: List[length_distributions.LengthDistribution] = []
            for lds in distributions_dict.values():
                set_lengths.append(lds[set_name])

            mld = length_distributions.MultiLengthDistribution(
                distributions=set_lengths,
                using_branches=using_branches,
            )
            mlds[set_name] = mld
        return mlds

    @beartype
    def basic_network_descriptions_df(
        self,
        columns: Dict[str, Tuple[Optional[str], Type]],
    ):
        """
        Create DataFrame useful for basic Network characterization.

        ``columns`` should contain key value pairs where the key is the column
        name in ``numerical_network_description`` dict. Value is a tuple where
        the first member is a new name for the column or alternatively None in
        which case the column name isn't changed. The second member should be
        the type of the column, typically either str, int or float.
        """
        numerical_df = pd.DataFrame(
            [network.numerical_network_description() for network in self.networks]
        )
        # Filter to wanted columns only
        numerical_df = numerical_df[list(columns)]

        assert isinstance(numerical_df, pd.DataFrame)

        for column, (_, column_type) in columns.items():
            # apply_func = (
            #     column_type
            #     # if column_type is not float
            #     # else lambda val: round(column_type(val), decimals)
            # )
            numerical_df[column] = numerical_df[column].apply(column_type)

        # Rename columns
        renames = {
            key: value[0] for key, value in columns.items() if value[0] is not None
        }
        numerical_df = numerical_df.rename(renames, axis="columns")

        assert isinstance(numerical_df, pd.DataFrame)

        # Resolve index name
        index_name = NAME
        if NAME in columns:
            new_name = columns[NAME][0]
            if new_name is not None:
                index_name = new_name
        numerical_df.set_index(index_name, inplace=True)

        numerical_df_transposed = numerical_df.transpose()

        return numerical_df_transposed
