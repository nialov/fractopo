"""
MultiNetwork implementation for handling multiple network analysis.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Union

from fractopo.analysis import length_distributions, subsampling
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import RandomChoice
from fractopo.general import ProcessResult


@dataclass
class MultiNetwork:

    """
    Multiple Network analysis.
    """

    networks: Sequence[Network]

    def __hash__(self) -> int:
        """
        Implement hashing for MultiNetwork.
        """
        return hash(tuple([*self.networks]))

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

    @lru_cache(maxsize=None)
    def multi_length_distributions(
        self, using_branches: bool = False, cut_distributions: bool = True
    ) -> length_distributions.MultiLengthDistribution:
        """
        Get MultiLengthDistribution of all networks.
        """
        distributions = [
            length_distributions.LengthDistribution(
                name=network.name,
                lengths=(
                    network.trace_length_array
                    if not using_branches
                    else network.branch_length_array
                ),
                area_value=network.total_area,
            )
            for network in self.networks
        ]

        multi_distribution = length_distributions.MultiLengthDistribution(
            distributions=distributions,
            using_branches=using_branches,
            cut_distributions=cut_distributions,
        )
        return multi_distribution

    def plot_multi_length_distribution(
        self, using_branches: bool, cut_distributions: bool
    ):
        """
        Plot multi-length distribution fit.
        """
        multi_distribution = self.multi_length_distributions(
            using_branches=using_branches, cut_distributions=cut_distributions
        )
        fig, ax = multi_distribution.plot_multi_length_distributions()

        return fig, ax
