import numpy as np
import fractopo.analysis.config as config


def aniso_get_class_as_value(c):
    """
    Return value based on branch classification. Only C-C branches have a
    value, but this can be changed here.
    Classification can differ from ('C - C', 'C - I', 'I - I') (e.g. 'C - E')
    in which case a value is still returned.

    E.g.

    >>> aniso_get_class_as_value('C - C')
    1

    >>> aniso_get_class_as_value('C - E')
    0

    """
    if c not in ("C - C", "C - I", "I - I"):
        return 0
    if c == "C - C":
        return 1
    elif c == "C - I":
        return 0
    elif c == "I - I":
        return 0
    else:
        return 0


def aniso_calc_anisotropy(azimuth: float, branch_type: str, length: float):
    """
    Calculates anisotropy of connectivity for a branch based on azimuth,
    classification and length.
    Value is calculated for preset angles (angles_of_study = np.arange(0, 179,
    30))

    E.g.

    Anisotropy for a C-C classified branch:

    >>> aniso_calc_anisotropy(90, 'C - C', 10)
    array([6.12323400e-16, 5.00000000e+00, 8.66025404e+00, 1.00000000e+01,
           8.66025404e+00, 5.00000000e+00])

    Other classification for branch:

    >>> aniso_calc_anisotropy(90, 'C - I', 10)
    array([0, 0, 0, 0, 0, 0])

    """
    angles_of_study = config.angles_for_examination
    # print(angles_of_study)
    c_value = aniso_get_class_as_value(branch_type)
    # CALCULATION
    results = []
    for angle in angles_of_study:
        if c_value == 0:
            results.append(0)
            continue
        diff = np.abs(angle - azimuth)
        if diff > 90:
            diff = 180 - max([angle, azimuth]) + min([angle, azimuth])
        cos_diff = np.cos(np.deg2rad(diff))
        result = length * c_value * cos_diff
        results.append(result)
    # print(results)
    return np.array(results)
