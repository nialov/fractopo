"""
MultiNetwork implementation for handling multiple network analysis.
"""

from dataclasses import dataclass
from typing import Sequence

from fractopo import general
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import NetworkRandomSampler, RandomChoice


@dataclass
class MultiNetwork:

    """
    Multiple Network analysis.
    """

    networks: Sequence[Network]

    @staticmethod
    def _create_sample(sampler: NetworkRandomSampler):
        """
        Sample with ``NetworkRandomSampler``.
        """
        return sampler.random_network_sample()

    def subsample(
        self,
        min_radius: float,
        random_choice: RandomChoice = RandomChoice.radius,
        samples: int = 1,
    ):
        """
        Subsample all Networks.
        """
        assert isinstance(samples, int)
        assert samples > 0

        subsamplers = [
            NetworkRandomSampler.random_network_sampler(
                network=network, min_radius=min_radius, random_choice=random_choice
            )
            for network in self.networks
        ]

        subsamples = general.multiprocess(
            function_to_call=self._create_sample,
            keyword_arguments=subsamplers,
            arguments_identifier=lambda sampler: sampler.name,
            repeats=samples - 1,
        )

        return subsamples
