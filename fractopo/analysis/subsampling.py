"""
Utilities for Network subsampling.
"""
from typing import List, Sequence

import pandas as pd

from fractopo import general
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import (
    NetworkRandomSampler,
    RandomChoice,
    RandomSample,
)

# from fractopo.general import Number


def create_sample(sampler: NetworkRandomSampler):
    """
    Sample with ``NetworkRandomSampler``.
    """
    return sampler.random_network_sample()


def subsample_networks(
    networks: Sequence[Network],
    min_radius: float,
    random_choice: RandomChoice = RandomChoice.radius,
    samples: int = 1,
) -> List[general.ProcessResult]:
    """
    Subsample given Sequence of Networks.
    """
    assert isinstance(samples, int)
    assert samples > 0

    subsamplers = [
        NetworkRandomSampler.random_network_sampler(
            network=network, min_radius=min_radius, random_choice=random_choice
        )
        for network in networks
    ]

    # Gather subsamples with multiprocessing
    subsamples = general.multiprocess(
        function_to_call=create_sample,
        keyword_arguments=subsamplers,
        arguments_identifier=lambda sampler: sampler.name,
        repeats=samples - 1,
    )

    return subsamples


def gather_subsample_descriptions(
    subsample_results: List[general.ProcessResult],
) -> pd.DataFrame:
    """
    Gather results for a list of subsampling ProcessResults.
    """
    descriptions = []
    for subsample in subsample_results:
        random_sample = subsample.result
        assert isinstance(random_sample, RandomSample)
        if random_sample.network_maybe is None:
            continue
        else:
            description = random_sample.network_maybe.numerical_network_description()

        descriptions.append(description)
    return pd.DataFrame(descriptions)
