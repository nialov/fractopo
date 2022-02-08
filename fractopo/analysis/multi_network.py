"""
MultiNetwork implementation for handling multiple network analysis.
"""

from typing import Dict, List, NamedTuple, Tuple, Union

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from fractopo.analysis import length_distributions, parameters, subsampling
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import RandomChoice
from fractopo.general import ProcessResult, SetRangeTuple


class MultiNetwork(NamedTuple):

    """
    Multiple Network analysis.
    """

    networks: Tuple[Network, ...]

    def __hash__(self) -> int:
        """
        Implement hashing for MultiNetwork.
        """
        return hash(tuple(self.networks))

    def subsample(
        self,
        min_radii: Union[float, Dict[str, float]],
        random_choice: RandomChoice = RandomChoice.radius,
        samples: int = 1,
    ) -> List[ProcessResult]:
        """
        Subsample all ``Networks`` in ``MultiNetwork``.

        :param min_radii: The minimum radius for
            subsamples can be either given as static or as a mapping that maps
            network names to specific radii.
        :param random_choice: Whether to use radius or area
            as the random choice parameter.
        :param samples: How many subsamples to conduct per network.
        :rtype: Subsamples and information on whether
            subsampling succeeded for network.
        """
        return subsampling.subsample_networks(
            networks=self.networks,
            min_radii=min_radii,
            random_choice=random_choice,
            samples=samples,
        )

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
                    "Expected azimuth sets and ranges to be the same"
                    " in all networks."
                )
        return set_names, set_ranges

    def network_length_distributions(
        self, using_branches: bool, using_azimuth_sets: bool
    ) -> Dict[str, Dict[str, length_distributions.LengthDistribution]]:
        """
        Get length distributions of Networks.
        """
        # Get azimuth sets that are identical for all Networks.
        set_names, _ = self.collective_azimuth_sets()

        # Collect (set-wise) length distributions for all networks
        distributions: Dict[
            str, Dict[str, length_distributions.LengthDistribution]
        ] = dict()

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
                label = name if name == network.name else f"{network.name} {name}"

                # Use either trace or branch lengths
                lengths = (
                    network.trace_length_array
                    if not using_branches
                    else network.branch_length_array
                )

                # Filter by azimuth set
                if using_azimuth_sets:
                    # Filter by set name
                    filterer = (
                        network.trace_azimuth_set_array == name
                        if not using_branches
                        else network.branch_azimuth_set_array == name
                    )
                    lengths = lengths[filterer]

                # Create the LengthDistribution
                distribution = length_distributions.LengthDistribution(
                    name=label,
                    lengths=lengths,
                    area_value=network.total_area,
                    using_branches=using_branches,
                )
                # Use network name or set name as key in dictionary
                network_distributions[name] = distribution

            # Use network name as key
            distributions[network.name] = network_distributions
        return distributions

    def multi_length_distributions(
        self, using_branches: bool = False, cut_distributions: bool = True
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
            cut_distributions=cut_distributions,
        )
        return multi_distribution

    def plot_multi_length_distribution(
        self,
        using_branches: bool,
        cut_distributions: bool,
    ):
        """
        Plot multi-length distribution fit.

        Use ``multi_length_distributions()`` to get most parameters used in
        fitting the multi-scale distribution.
        """
        multi_distribution = self.multi_length_distributions(
            using_branches=using_branches, cut_distributions=cut_distributions
        )
        fig, ax = multi_distribution.plot_multi_length_distributions()

        return fig, ax

    def plot_xyi(
        self,
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
        )

    def plot_branch(
        self,
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
        )

    def _plot_azimuth_set_lengths(
        self,
        cut_distributions: bool,
        using_branches: bool,
    ) -> Tuple[
        Dict[str, length_distributions.MultiLengthDistribution],
        List[Figure],
        List[Axes],
    ]:
        """
        Plot multi-network azimuths set lengths with fits.
        """
        mlds = self.set_multi_length_distributions(
            using_branches=using_branches, cut_distributions=cut_distributions
        )

        figs, axes = [], []
        for mld in mlds.values():
            fig, ax = mld.plot_multi_length_distributions()
            figs.append(fig)
            axes.append(ax)

        return mlds, figs, axes

    def plot_trace_azimuth_set_lengths(
        self,
        cut_distributions: bool,
    ) -> Tuple[
        Dict[str, length_distributions.MultiLengthDistribution],
        List[Figure],
        List[Axes],
    ]:
        """
        Plot multi-network trace azimuths set lengths with fits.
        """
        return self._plot_azimuth_set_lengths(
            cut_distributions=cut_distributions, using_branches=False
        )

    def plot_branch_azimuth_set_lengths(
        self,
        cut_distributions: bool,
    ) -> Tuple[
        Dict[str, length_distributions.MultiLengthDistribution],
        List[Figure],
        List[Axes],
    ]:
        """
        Plot multi-network trace azimuths set lengths with fits.
        """
        return self._plot_azimuth_set_lengths(
            cut_distributions=cut_distributions, using_branches=True
        )

    def set_multi_length_distributions(
        self, using_branches: bool = False, cut_distributions: bool = True
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
            for network_name, lds in distributions_dict.items():
                set_lengths.append(lds[set_name])

            mld = length_distributions.MultiLengthDistribution(
                distributions=set_lengths,
                cut_distributions=cut_distributions,
                using_branches=using_branches,
            )
            mlds[set_name] = mld
        return mlds
