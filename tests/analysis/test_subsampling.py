import numpy as np
import pytest

import tests
from fractopo import general
from fractopo.analysis import subsampling


@pytest.mark.parametrize(
    "chosen,value_type,assumed_results", tests.test_aggregate_chosen_params()
)
def test_aggregate_chosen(
    chosen: general.ParameterListType, value_type: type, assumed_results: dict
):
    """
    Test aggregate_chosen.
    """
    assert isinstance(chosen, list)
    if len(chosen) > 0:
        assert isinstance(chosen[0], dict)
    result = subsampling.aggregate_chosen(chosen)

    assert isinstance(result, dict)

    for value in result.values():
        assert isinstance(value, value_type)

    for col in assumed_results:
        assert col in result
        result_value = result[col]
        assumed_result_value = assumed_results[col]
        if isinstance(result_value, str):
            assert result_value == assumed_result_value
        else:
            assert np.isclose(result_value, assumed_result_value)


@pytest.mark.parametrize(
    "grouped,circle_names_with_diameter,min_circles,max_circles",
    tests.test_random_sample_of_circles_params(),
)
def test_random_sample_of_circles(
    grouped, circle_names_with_diameter, min_circles, max_circles
):
    """
    Test random_sample_of_circles.
    """
    assert isinstance(grouped, dict)
    assert isinstance(grouped[list(grouped.keys())[0]], list)
    result = subsampling.random_sample_of_circles(
        grouped, circle_names_with_diameter, min_circles, max_circles
    )

    assert isinstance(result, list)

    if len(grouped) == 1:
        assert len(result) == 1

    assert len(grouped) >= min_circles


@pytest.mark.parametrize(
    "idxs,how_many,areas", tests.test_collect_indexes_of_base_circles_params()
)
def test_collect_indexes_of_base_circles(idxs, how_many, areas):
    """
    Test collect_indexes_of_base_circles.
    """
    result = subsampling.collect_indexes_of_base_circles(idxs, how_many, areas)
    assert isinstance(result, list)
    if how_many > 0:
        assert isinstance(result[0], int)
    assert all(idx in idxs for idx in result)
