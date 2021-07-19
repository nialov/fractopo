"""
MultiNetwork implementation for handling multiple network analysis.
"""

from dataclasses import dataclass
from typing import Sequence

from fractopo.analysis import subsampling
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import RandomChoice


@dataclass
class MultiNetwork:

    """
    Multiple Network analysis.
    """

    networks: Sequence[Network]

    def subsample(
        self,
        min_radius: float,
        random_choice: RandomChoice = RandomChoice.radius,
        samples: int = 1,
    ):
        """
        Subsample all ``Networks`` in ``MultiNetwork``.
        """
        return subsampling.subsample_networks(
            networks=self.networks,
            min_radius=min_radius,
            random_choice=random_choice,
            samples=samples,
        )
