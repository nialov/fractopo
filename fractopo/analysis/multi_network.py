"""
MultiNetwork implementation for handling multiple network analysis.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Union

from fractopo.analysis import subsampling
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import RandomChoice
from fractopo.general import ProcessResult


@dataclass
class MultiNetwork:

    """
    Multiple Network analysis.
    """

    networks: Sequence[Network]

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
