"""
Utilities for Network subsampling.
"""
import logging
from typing import List, Sequence

import numpy as np
import pandas as pd

from fractopo import general
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import NetworkRandomSampler, RandomChoice


def create_sample(sampler: NetworkRandomSampler):
    """
    Sample with ``NetworkRandomSampler`` and return ``Network`` description.
    """
    # Set random seed instead of inheriting the seed from parent process
    np.random.seed()
    random_sample = sampler.random_network_sample()
    if random_sample.network_maybe is None:
        return None
    return random_sample.network_maybe.numerical_network_description()


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
    Gather results from a list of subsampling ProcessResults.
    """
    descriptions = []
    for subsample in subsample_results:
        random_sample_description = subsample.result
        if not isinstance(random_sample_description, dict):
            logging.error(
                "Expected result random_sample_description to be a dict.\n"
                f" Got: {type(random_sample_description), random_sample_description}\n"
                f"with subsample: {subsample}."
            )
            continue

        descriptions.append(random_sample_description)
    return pd.DataFrame(descriptions)
