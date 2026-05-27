"""
General interoperability functions to e.g. fill 3D orientation for traces
"""

from beartype.typing import Callable

from fractopo.general import NULL_SET


def azimuth_set_dip_assigner_normal_distribution(
    azimuth_set_id: str, azimuth_set_ranges, azimuth_set_names
):
    """
    Assign dip based on assumption of normal distribution within each azimuth set range
    """


def fill_dip_based_on_known_fracture_sets(
    azimuth_set_id: str,
    azimuth_set_dip_assigner: Callable[[str], float],
    null_set=NULL_SET,
) -> float:
    """
    Assign dip based on azimuth_set_id
    """
    pass
