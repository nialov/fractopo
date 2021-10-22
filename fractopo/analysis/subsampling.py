"""
Utilities for Network subsampling.
"""
import logging
import random
from itertools import compress, groupby
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
from numpy.random import seed

from fractopo import general
from fractopo.analysis.network import Network
from fractopo.analysis.random_sampling import NetworkRandomSampler, RandomChoice


def create_sample(
    sampler: NetworkRandomSampler,
) -> Optional[Dict[str, Union[general.Number, str]]]:
    """
    Sample with ``NetworkRandomSampler`` and return ``Network`` description.
    """
    # Set random seed instead of inheriting the seed from parent process
    # as is default for multiprocessing
    seed()
    random_sample = sampler.random_network_sample()
    if random_sample.network_maybe is None:
        return None
    return random_sample.network_maybe.numerical_network_description()


def subsample_networks(
    networks: Sequence[Network],
    min_radii: Union[float, Dict[str, float]],
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
            network=network,
            min_radius=min_radii
            if isinstance(min_radii, float)
            else min_radii[network.name],
            random_choice=random_choice,
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
) -> List[Dict[str, Union[general.Number, str]]]:
    """
    Gather results from a list of subsampling ProcessResults.
    """
    descriptions = []
    for subsample in subsample_results:
        random_sample_description = subsample.result
        if not isinstance(random_sample_description, dict):
            logging.error(
                "Expected result random_sample_description to be a dict.",
                extra=dict(
                    random_sample_description_type=type(random_sample_description),
                    random_sample_description=random_sample_description,
                    subsample=subsample,
                ),
            )
            continue

        descriptions.append(random_sample_description)
    return descriptions


# def filter_samples_by_area(
#     group: pd.DataFrame, min_area: float, max_area: Optional[float], indexes: List[int]
# ):
#     """
#     Filter samples by given min and max area values.
#     """
#     # Get circle areas
#     areas = group["area"].to_numpy()

#     # Solve max_area
#     max_area = np.max(areas) if max_area is None else max_area

#     # Filter out areas that do not fit within the range
#     area_compressor = [min_area <= area <= max_area for area in areas]

#     # Filter out indexes accordingly
#     indexes = list(compress(indexes, area_compressor))

#     return indexes


def choose_sample_from_group(
    group: general.ParameterListType,
) -> general.ParameterValuesType:
    """
    Choose single sample from group DataFrame.
    """
    # Make continous index from 0
    indexes = [idx for idx in range(len(group))]

    assert len(indexes) > 0
    # Choose from indexes
    choice = random.choices(population=indexes, k=1)[0]

    # Get the dict at choice index
    chosen_dict = group[choice]
    assert isinstance(chosen_dict, dict)

    return chosen_dict


def area_weighted_index_choice(
    idxs: List[int], areas: List[float], compressor: List[bool]
) -> int:
    """
    Make area-weighted choce from list of indexes.
    """
    possible_idxs = list(compress(idxs, compressor))
    possible_areas = list(compress(areas, compressor))
    choice = random.choices(population=possible_idxs, weights=possible_areas, k=1)[0]
    return choice


def collect_indexes_of_base_circles(
    idxs: List[int], how_many: int, areas: List[float]
) -> List[int]:
    """
    Collect indexes of base circles, area-weighted and randomly.
    """
    which_idxs = []
    for _ in range(how_many):
        compressor = [idx not in which_idxs for idx in idxs]
        choice = area_weighted_index_choice(
            idxs=idxs, areas=areas, compressor=compressor
        )
        which_idxs.append(choice)

    assert len(which_idxs) == how_many

    return which_idxs


def random_sample_of_circles(
    grouped: Dict[str, general.ParameterListType],
    circle_names_with_diameter: Dict[str, int],
    min_circles: int = 1,
    max_circles: Optional[int] = None,
) -> general.ParameterListType:
    """
    Get a random sample of circles from grouped subsampled data.

    Both the amount of overall circles and which circles within each group
    is random. Data is grouped by target area name.
    """
    if max_circles is not None:
        assert max_circles >= min_circles

    # Area names
    # grouped_keys = grouped.groups.keys()
    # names = [name for name in grouped_keys if isinstance(name, str)]
    # assert len(grouped_keys) == len(names)
    names = list(grouped)

    assert all(name in circle_names_with_diameter for name in names)

    # Get area of the base circles corresponding to area name
    areas = [np.pi * (circle_names_with_diameter[name] / 2) ** 2 for name in names]

    # All indexes
    idxs = list(range(0, len(grouped)))

    # "Randomly" choose how many circles
    # Is constrained by given min_circles and max_circles
    how_many = random.randint(
        min_circles, len(grouped) if max_circles is None else max_circles
    )

    # Collect indexes of base circles
    which_idxs = collect_indexes_of_base_circles(
        idxs=idxs, how_many=how_many, areas=areas
    )

    # Collect the Series that are chosen
    chosen: general.ParameterListType = []

    # Iterate over the DataFrameGroupBy dataframe groups
    for idx, group in enumerate(grouped.values()):

        # Skip if not chosen base circle previously
        if idx not in which_idxs:
            continue

        chosen_dict = choose_sample_from_group(group=group)
        chosen.append(chosen_dict)

    assert len(chosen) == how_many

    # Return chosen subsampled circles from base circles
    return chosen


def aggregate_chosen(
    chosen: general.ParameterListType,
    default_aggregator: general.Aggregator = general.Aggregator.MEAN,
) -> Dict[str, Any]:
    """
    Aggregate a collection of subsampled circles for params.

    Weights averages by the area of each subsampled circle.
    """
    columns = list(chosen[0].keys())

    area_values = [params[general.Param.AREA.value.name] for params in chosen]
    aggregated_values = dict()
    for column in columns:
        aggregator = default_aggregator
        assert isinstance(aggregator, Callable)
        column_values = [params[column] for params in chosen]
        for param in general.Param:
            if column == param.value.name:
                aggregator = param.value.aggregator
                assert isinstance(aggregator, Callable)
                break
        try:
            aggregated = aggregator(values=column_values, weights=area_values)
        except Exception:
            logging.info(
                "Could not aggregate column. Falling back to fallback_aggregation.",
                extra=dict(
                    current_column=column, aggregator=aggregator, columns=columns
                ),
                exc_info=True,
            )
            aggregated = general.fallback_aggregation(values=column_values)
        assert isinstance(aggregated, (int, float, str))
        aggregated_values[column] = aggregated

    return aggregated_values


def groupby_keyfunc(
    item: Dict[str, Union[general.Number, str]],
    groupby_column: str = general.NAME,
) -> str:
    """
    Use groupby_column to group values.
    """
    val = item[groupby_column]
    assert isinstance(val, str)
    return val


def group_gathered_subsamples(
    subsamples: List[Dict[str, Union[general.Number, str]]],
    groupby_column: str = general.NAME,
) -> Dict[str, List[Dict[str, Union[general.Number, str]]]]:
    """
    Group gathered subsamples.

    By default groups by Name column.

    >>> subsamples = [{"param": 2.0, "Name": "myname"}, {"param": 2.0, "Name": "myname"}]
    >>> group_gathered_subsamples(subsamples)
    {'myname': [{'param': 2.0, 'Name': 'myname'}, {'param': 2.0, 'Name': 'myname'}]}

    """
    grouped = groupby(
        subsamples,
        key=lambda item: groupby_keyfunc(item=item, groupby_column=groupby_column),
    )
    # for k, g in grouped:
    #     print(k, list(g))

    grouped = {key: list(vals) for key, vals in grouped if isinstance(key, str)}

    return grouped
