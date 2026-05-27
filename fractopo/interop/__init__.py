"""
General interoperability functions to e.g. fill 3D orientation for traces
"""

import numpy as np
from beartype.typing import Annotated, Callable
from beartype.vale import Is

from fractopo.general import NULL_SET

N_STANDARD_DEVIATIONS = 3


def azimuth_set_dip_assigner_normal_distribution(
    azimuth_set_id: str, azimuth_set_ranges, azimuth_set_names, azimuth_set_dips
):
    """
    Assign dip based on assumption of normal distribution within each azimuth set range
    """


def generate_dip_and_dip_direction(
    average_dip: Annotated[float, Is[lambda value: 0.0 <= value <= 90.0]],
    dip_direction: Annotated[float, Is[lambda value: 0.0 <= value <= 360.0]],
    max_distance: Annotated[float, Is[lambda value: 0.0 <= value <= 90.0]],
    n_standard_deviations: Annotated[
        int, Is[lambda value: value > 0]
    ] = N_STANDARD_DEVIATIONS,
):
    """
    If a value below 0 or over 90 is fetched, this means the dip direction must
    be rotated 180 degrees.
    """

    std_dev = max_distance / n_standard_deviations

    random_dip = np.random.normal(average_dip, std_dev)

    if random_dip < 0.0 or random_dip > 90:
        dip_direction = dip_direction + 180
        if dip_direction > 360.0:
            dip_direction = 360.0 - dip_direction
        random_dip = 90 - (random_dip - 90) if random_dip > 90.0 else -random_dip

    return random_dip, dip_direction


def fill_dip_based_on_known_fracture_sets(
    azimuth_set_id: str,
    azimuth_set_dip_assigner: Callable[[str], float],
    null_set=NULL_SET,
) -> float:
    """
    Assign dip based on azimuth_set_id
    """
    pass
